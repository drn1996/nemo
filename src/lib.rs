use std::cell::RefCell;
use std::fmt::Write;
use std::rc::Rc;
use std::time::Duration;

use js_sys::{Array, Object, Reflect};
use slint::{ModelRc, SharedString, VecModel};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    AnalyserNode, AudioBuffer, AudioBufferSourceNode, AudioContext, BlobEvent, MediaDeviceInfo,
    MediaDeviceKind, MediaRecorder, MediaRecorderOptions, MediaStream, MediaStreamAudioSourceNode,
    MediaStreamConstraints, MediaStreamTrack,
};

slint::include_modules!();

const FFT: usize = 2048;
const SIGNAL_THRESHOLD: u8 = 15;

const TRACK_COLORS: &[(u8, u8, u8)] = &[
    (0, 255, 100),
    (0, 200, 255),
    (255, 80, 200),
    (255, 200, 0),
    (255, 120, 0),
    (120, 120, 255),
    (0, 255, 200),
    (255, 100, 100),
];

// ── types ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Device {
    id: String,
    label: String,
}

struct Audio {
    ctx: AudioContext,
    analyser: AnalyserNode,
    source: MediaStreamAudioSourceNode,
    stream: MediaStream,
}

struct LoopState {
    tracks: Vec<AudioBuffer>,
    track_paths: Vec<String>, // cached filled-waveform SVG paths per completed track
    loop_duration: f64,
    // recording
    recorder: Option<MediaRecorder>,
    rec_chunks: Vec<web_sys::Blob>,
    rec_start_time: f64,
    rec_peaks: Vec<u8>, // accumulated peak amplitudes during recording
    armed: bool,
    // playback
    sources: Vec<AudioBufferSourceNode>,
    playing: bool,
    play_start_time: f64,
}

type SharedAudio = Rc<RefCell<Option<Audio>>>;
type SharedDevices = Rc<RefCell<Vec<Device>>>;
type SharedLoop = Rc<RefCell<LoopState>>;

// ── helpers ─────────────────────────────────────────────────────────────────

fn make_constraints(device_id: Option<&str>) -> Result<MediaStreamConstraints, JsValue> {
    let obj = Object::new();
    match device_id {
        Some(id) => {
            let audio_obj = Object::new();
            Reflect::set(&audio_obj, &"deviceId".into(), &id.into())?;
            Reflect::set(&obj, &"audio".into(), &audio_obj)?;
        }
        None => {
            Reflect::set(&obj, &"audio".into(), &JsValue::TRUE)?;
        }
    }
    Reflect::set(&obj, &"video".into(), &JsValue::FALSE)?;
    Ok(obj.unchecked_into())
}

/// Build a filled waveform envelope from peak amplitude data.
/// Draws top edge then mirrored bottom edge, creating a filled shape.
fn build_peaks_path(peaks: &[u8], w: f32, h: f32) -> String {
    if peaks.is_empty() || w < 1.0 {
        return String::new();
    }
    let len = peaks.len();
    let mid = h / 2.0;
    let mut path = String::with_capacity(len * 30 + 50);

    // Top edge
    for i in 0..len {
        let x = if len > 1 {
            i as f32 / (len - 1) as f32 * w
        } else {
            0.0
        };
        let amp = (peaks[i] as f32 / 128.0).min(1.0);
        let y = mid - amp * mid * 0.85;
        if i == 0 {
            let _ = write!(path, "M {:.0} {:.0}", x, y);
        } else {
            let _ = write!(path, " L {:.0} {:.0}", x, y);
        }
    }

    // Bottom edge (mirrored, right to left)
    for i in (0..len).rev() {
        let x = if len > 1 {
            i as f32 / (len - 1) as f32 * w
        } else {
            0.0
        };
        let amp = (peaks[i] as f32 / 128.0).min(1.0);
        let y = mid + amp * mid * 0.85;
        let _ = write!(path, " L {:.0} {:.0}", x, y);
    }

    let _ = write!(path, " Z");
    path
}

/// Extract peak amplitudes from a decoded AudioBuffer.
fn audio_buffer_to_peaks(buffer: &AudioBuffer, num_points: usize) -> Vec<u8> {
    let data = match buffer.get_channel_data(0) {
        Ok(d) => d,
        Err(_) => return vec![0; num_points],
    };
    let len = data.len();
    if len == 0 {
        return vec![0; num_points];
    }
    let chunk_size = (len / num_points).max(1);
    let mut peaks = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(len);
        let peak = data[start..end]
            .iter()
            .map(|&s| (s.abs() * 128.0) as u8)
            .max()
            .unwrap_or(0);
        peaks.push(peak);
    }
    peaks
}

fn signal_above_threshold(data: &[u8]) -> bool {
    data.iter().any(|&s| s.abs_diff(128) > SIGNAL_THRESHOLD)
}

fn format_time(secs: f64) -> String {
    let s = secs.max(0.0) as u32;
    format!("{}:{:02}", s / 60, s % 60)
}

fn track_color(i: usize) -> slint::Color {
    let (r, g, b) = TRACK_COLORS[i % TRACK_COLORS.len()];
    slint::Color::from_rgb_u8(r, g, b)
}

/// Rebuild the Slint tracks model from current LoopState.
fn rebuild_tracks_model(ls: &LoopState, w: f32, h: f32, rec_progress: f32) -> Vec<TrackInfo> {
    let mut infos = Vec::new();

    // Completed tracks
    for (i, path) in ls.track_paths.iter().enumerate() {
        infos.push(TrackInfo {
            label: format!("{}", i + 1).into(),
            waveform: SharedString::from(path.as_str()),
            color: track_color(i),
            is_armed: false,
        });
    }

    // Currently recording track
    if ls.recorder.is_some() && !ls.rec_peaks.is_empty() {
        let idx = ls.tracks.len();
        let fill_w = w * rec_progress;
        let waveform = build_peaks_path(&ls.rec_peaks, fill_w, h);
        infos.push(TrackInfo {
            label: format!("{}", idx + 1).into(),
            waveform: waveform.into(),
            color: track_color(idx),
            is_armed: false,
        });
    }

    // Armed track (empty, waiting)
    if ls.armed {
        let idx = ls.tracks.len();
        infos.push(TrackInfo {
            label: format!("{}", idx + 1).into(),
            waveform: "".into(),
            color: track_color(idx),
            is_armed: true,
        });
    }

    infos
}

fn update_ui_state(weak: &slint::Weak<App>, ls: &LoopState) {
    if let Some(app) = weak.upgrade() {
        app.set_track_count(ls.tracks.len() as i32);
    }
}

// ── audio init ──────────────────────────────────────────────────────────────

async fn init_audio() -> Result<(Audio, Vec<Device>), JsValue> {
    let media_devices = web_sys::window().unwrap().navigator().media_devices()?;
    let constraints = make_constraints(None)?;
    let promise = media_devices.get_user_media_with_constraints(&constraints)?;
    let stream: MediaStream = JsFuture::from(promise).await?.dyn_into()?;

    let promise = media_devices.enumerate_devices()?;
    let arr: js_sys::Array = JsFuture::from(promise).await?.dyn_into()?;
    let mut devices = Vec::new();
    for i in 0..arr.length() {
        let info: MediaDeviceInfo = arr.get(i).dyn_into()?;
        if info.kind() == MediaDeviceKind::Audioinput {
            devices.push(Device {
                id: info.device_id(),
                label: info.label(),
            });
        }
    }

    let audio_ctx = AudioContext::new()?;
    let analyser = audio_ctx.create_analyser()?;
    analyser.set_fft_size(FFT as u32);
    let source = audio_ctx.create_media_stream_source(&stream)?;
    source.connect_with_audio_node(&analyser)?;

    Ok((
        Audio {
            ctx: audio_ctx,
            analyser,
            source,
            stream,
        },
        devices,
    ))
}

async fn switch_device(
    audio: &SharedAudio,
    devices: &SharedDevices,
    idx: usize,
) -> Result<(), JsValue> {
    let device_id = {
        let devs = devices.borrow();
        if idx >= devs.len() {
            return Ok(());
        }
        devs[idx].id.clone()
    };
    {
        let a = audio.borrow();
        if let Some(ref au) = *a {
            let tracks = au.stream.get_audio_tracks();
            for i in 0..tracks.length() {
                if let Ok(t) = tracks.get(i).dyn_into::<MediaStreamTrack>() {
                    t.stop();
                }
            }
        }
    }
    let media_devices = web_sys::window().unwrap().navigator().media_devices()?;
    let constraints = make_constraints(Some(&device_id))?;
    let promise = media_devices.get_user_media_with_constraints(&constraints)?;
    let new_stream: MediaStream = JsFuture::from(promise).await?.dyn_into()?;
    let mut a = audio.borrow_mut();
    if let Some(ref mut au) = *a {
        let _ = au.source.disconnect();
        let new_source = au.ctx.create_media_stream_source(&new_stream)?;
        new_source.connect_with_audio_node(&au.analyser)?;
        au.source = new_source;
        au.stream = new_stream;
    }
    Ok(())
}

// ── recording ───────────────────────────────────────────────────────────────

fn begin_recording(audio: &SharedAudio, ls: &SharedLoop) -> Result<(), JsValue> {
    let a = audio.borrow();
    let au = a.as_ref().ok_or("No audio")?;

    let opts = MediaRecorderOptions::new();
    opts.set_mime_type("audio/webm;codecs=opus");
    let recorder =
        MediaRecorder::new_with_media_stream_and_media_recorder_options(&au.stream, &opts)?;

    {
        let mut l = ls.borrow_mut();
        l.rec_chunks.clear();
        l.rec_peaks.clear();
        l.rec_start_time = au.ctx.current_time();
        l.armed = false;
    }

    let ls_clone = ls.clone();
    let on_data = Closure::<dyn FnMut(BlobEvent)>::new(move |e: BlobEvent| {
        if let Some(blob) = e.data() {
            if blob.size() > 0.0 {
                ls_clone.borrow_mut().rec_chunks.push(blob);
            }
        }
    });
    recorder.set_ondataavailable(Some(on_data.as_ref().unchecked_ref()));
    on_data.forget();

    recorder.start()?;
    ls.borrow_mut().recorder = Some(recorder);
    Ok(())
}

fn finish_recording(audio: &SharedAudio, ls: &SharedLoop, weak: slint::Weak<App>) {
    // Stop playback of previous tracks (if playing during overdub)
    stop_playback(ls);

    let recorder = ls.borrow_mut().recorder.take();
    if let Some(recorder) = recorder {
        let ls_clone = ls.clone();
        let audio_clone = audio.clone();
        let weak2 = weak.clone();
        let on_stop = Closure::<dyn FnMut()>::new(move || {
            let ls2 = ls_clone.clone();
            let au2 = audio_clone.clone();
            let w2 = weak2.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Err(e) = decode_and_add_track(&au2, &ls2, &w2).await {
                    web_sys::console::log_1(&e);
                }
            });
        });
        recorder.set_onstop(Some(on_stop.as_ref().unchecked_ref()));
        on_stop.forget();
        let _ = recorder.stop();
    }

    if let Some(app) = weak.upgrade() {
        app.set_recording(false);
        app.set_armed(false);
        app.set_playing(false);
        app.set_playhead_pos(-1.0);
    }
}

async fn decode_and_add_track(
    audio: &SharedAudio,
    ls: &SharedLoop,
    weak: &slint::Weak<App>,
) -> Result<(), JsValue> {
    let chunks = { ls.borrow().rec_chunks.clone() };
    if chunks.is_empty() {
        return Err("No recorded data".into());
    }

    let parts = Array::new();
    for chunk in &chunks {
        parts.push(chunk);
    }
    let merged_blob = web_sys::Blob::new_with_blob_sequence(&parts)?;
    let promise = merged_blob.array_buffer();
    let array_buffer: js_sys::ArrayBuffer = JsFuture::from(promise).await?.dyn_into()?;

    let ctx = {
        let a = audio.borrow();
        let au = a.as_ref().ok_or("No audio")?;
        au.ctx.clone()
    };
    let promise = ctx.decode_audio_data(&array_buffer)?;
    let audio_buffer: AudioBuffer = JsFuture::from(promise).await?.dyn_into()?;

    // Compute and cache the waveform path
    let w = weak
        .upgrade()
        .map(|a| a.get_wave_width().max(100.0) - 80.0)
        .unwrap_or(600.0);
    let peaks = audio_buffer_to_peaks(&audio_buffer, 400);
    let path = build_peaks_path(&peaks, w, 56.0);

    {
        let mut l = ls.borrow_mut();
        if l.loop_duration == 0.0 {
            l.loop_duration = audio_buffer.duration();
        }
        l.tracks.push(audio_buffer);
        l.track_paths.push(path);
        l.rec_peaks.clear();
    }

    update_ui_state(weak, &ls.borrow());

    // Rebuild tracks model with completed tracks only (no recording/armed)
    if let Some(app) = weak.upgrade() {
        let l = ls.borrow();
        let infos = rebuild_tracks_model(&l, w, 56.0, 1.0);
        app.set_tracks(ModelRc::new(VecModel::from(infos)));
    }

    Ok(())
}

// ── playback ────────────────────────────────────────────────────────────────

fn start_playback(audio: &SharedAudio, ls: &SharedLoop) -> Result<(), JsValue> {
    let a = audio.borrow();
    let au = a.as_ref().ok_or("No audio")?;
    let l = ls.borrow();

    let mut new_sources = Vec::with_capacity(l.tracks.len());
    for track in &l.tracks {
        let source = au.ctx.create_buffer_source()?;
        source.set_buffer(Some(track));
        source.set_loop(false); // one-shot, no auto-repeat
        source.connect_with_audio_node(&au.ctx.destination())?;
        let sched: &web_sys::AudioScheduledSourceNode = source.as_ref();
        let _ = sched.start();
        new_sources.push(source);
    }

    let start_time = au.ctx.current_time();
    drop(l);

    let mut l = ls.borrow_mut();
    l.sources = new_sources;
    l.playing = true;
    l.play_start_time = start_time;

    Ok(())
}

fn stop_playback(ls: &SharedLoop) {
    let mut l = ls.borrow_mut();
    for source in &l.sources {
        let sched: &web_sys::AudioScheduledSourceNode = source.as_ref();
        let _ = sched.stop();
    }
    l.sources.clear();
    l.playing = false;
}

fn clear_all(ls: &SharedLoop, weak: &slint::Weak<App>) {
    stop_playback(ls);
    let mut l = ls.borrow_mut();
    if let Some(recorder) = l.recorder.take() {
        let _ = recorder.stop();
    }
    l.tracks.clear();
    l.track_paths.clear();
    l.loop_duration = 0.0;
    l.rec_chunks.clear();
    l.rec_peaks.clear();
    l.armed = false;

    if let Some(app) = weak.upgrade() {
        app.set_recording(false);
        app.set_armed(false);
        app.set_playing(false);
        app.set_track_count(0);
        app.set_time_display("".into());
        app.set_playhead_pos(-1.0);
        app.set_tracks(ModelRc::new(VecModel::<TrackInfo>::default()));
    }
}

// ── entry point ─────────────────────────────────────────────────────────────

#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen(start))]
pub fn main() {
    let app = App::new().unwrap();
    let audio: SharedAudio = Rc::new(RefCell::new(None));
    let devices: SharedDevices = Rc::new(RefCell::new(Vec::new()));
    let ls: SharedLoop = Rc::new(RefCell::new(LoopState {
        tracks: Vec::new(),
        track_paths: Vec::new(),
        loop_duration: 0.0,
        recorder: None,
        rec_chunks: Vec::new(),
        rec_start_time: 0.0,
        rec_peaks: Vec::new(),
        armed: false,
        sources: Vec::new(),
        playing: false,
        play_start_time: 0.0,
    }));

    // Start button
    {
        let audio = audio.clone();
        let devices = devices.clone();
        let weak = app.as_weak();
        app.on_start_clicked(move || {
            let audio = audio.clone();
            let devices = devices.clone();
            let weak = weak.clone();
            if let Some(app) = weak.upgrade() {
                app.set_status_text("Requesting audio access...".into());
            }
            wasm_bindgen_futures::spawn_local(async move {
                match init_audio().await {
                    Ok((audio_state, device_list)) => {
                        if let Some(app) = weak.upgrade() {
                            let names: Vec<SharedString> = device_list
                                .iter()
                                .enumerate()
                                .map(|(i, d)| {
                                    if d.label.is_empty() {
                                        format!("Input {}", i + 1).into()
                                    } else {
                                        SharedString::from(d.label.as_str())
                                    }
                                })
                                .collect();
                            app.set_device_names(ModelRc::new(VecModel::from(names)));
                            app.set_started(true);
                            app.set_status_text("".into());
                        }
                        *devices.borrow_mut() = device_list;
                        *audio.borrow_mut() = Some(audio_state);
                    }
                    Err(e) => {
                        web_sys::console::log_1(&e);
                        if let Some(app) = weak.upgrade() {
                            app.set_status_text("".into());
                        }
                    }
                }
            });
        });
    }

    // Device selection
    {
        let audio = audio.clone();
        let devices = devices.clone();
        app.on_device_selected(move |idx| {
            let audio = audio.clone();
            let devices = devices.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Err(e) = switch_device(&audio, &devices, idx as usize).await {
                    web_sys::console::log_1(&e);
                }
            });
        });
    }

    // Record
    {
        let audio = audio.clone();
        let ls = ls.clone();
        let weak = app.as_weak();
        app.on_record_clicked(move || {
            let is_armed = ls.borrow().armed;
            let is_recording = ls.borrow().recorder.is_some();
            let has_loop = ls.borrow().loop_duration > 0.0;

            if is_armed {
                // Cancel armed state
                ls.borrow_mut().armed = false;
                if let Some(app) = weak.upgrade() {
                    app.set_armed(false);
                    // Rebuild model without armed track
                    let l = ls.borrow();
                    let w = app.get_wave_width().max(100.0) - 80.0;
                    let infos = rebuild_tracks_model(&l, w, 56.0, 1.0);
                    app.set_tracks(ModelRc::new(VecModel::from(infos)));
                }
            } else if is_recording {
                finish_recording(&audio, &ls, weak.clone());
            } else {
                // Arm for any track — recording starts on signal detection
                ls.borrow_mut().armed = true;
                if let Some(app) = weak.upgrade() {
                    app.set_armed(true);
                    let l = ls.borrow();
                    let w = app.get_wave_width().max(100.0) - 80.0;
                    let infos = rebuild_tracks_model(&l, w, 56.0, 1.0);
                    app.set_tracks(ModelRc::new(VecModel::from(infos)));
                }
            }
        });
    }

    // Play/Stop
    {
        let audio = audio.clone();
        let ls = ls.clone();
        let weak = app.as_weak();
        app.on_play_clicked(move || {
            if ls.borrow().recorder.is_some() || ls.borrow().armed {
                return;
            }
            if ls.borrow().tracks.is_empty() {
                return;
            }
            if ls.borrow().playing {
                stop_playback(&ls);
                if let Some(app) = weak.upgrade() {
                    app.set_playing(false);
                    app.set_playhead_pos(-1.0);
                }
            } else {
                if let Err(e) = start_playback(&audio, &ls) {
                    web_sys::console::log_1(&e);
                    return;
                }
                if let Some(app) = weak.upgrade() {
                    app.set_playing(true);
                }
            }
        });
    }

    // Clear
    {
        let ls = ls.clone();
        let weak = app.as_weak();
        app.on_clear_clicked(move || {
            if ls.borrow().recorder.is_some() || ls.borrow().armed {
                return;
            }
            clear_all(&ls, &weak);
        });
    }

    // Timer
    let timer = slint::Timer::default();
    {
        let audio = audio.clone();
        let ls = ls.clone();
        let weak = app.as_weak();
        timer.start(
            slint::TimerMode::Repeated,
            Duration::from_millis(16),
            move || {
                let a_borrow = audio.borrow();
                let au = match a_borrow.as_ref() {
                    Some(au) => au,
                    None => return,
                };

                let mut buf = vec![128u8; FFT];
                au.analyser.get_byte_time_domain_data(&mut buf);

                let now = au.ctx.current_time();

                // Signal detection for armed state
                let should_start = {
                    let l = ls.borrow();
                    l.armed && signal_above_threshold(&buf)
                };

                // Auto-stop check (track 2+)
                let should_stop = {
                    let l = ls.borrow();
                    l.recorder.is_some()
                        && l.loop_duration > 0.0
                        && l.rec_start_time > 0.0
                        && now - l.rec_start_time >= l.loop_duration
                };

                // Recording: accumulate peak
                let is_recording = ls.borrow().recorder.is_some();
                if is_recording {
                    let peak = buf.iter().map(|&s| s.abs_diff(128)).max().unwrap_or(0);
                    ls.borrow_mut().rec_peaks.push(peak);
                }

                drop(a_borrow);

                if should_stop {
                    finish_recording(&audio, &ls, weak.clone());
                } else if should_start {
                    let has_tracks = !ls.borrow().tracks.is_empty();
                    // Track 2+: start playback simultaneously so harmonies sync
                    if has_tracks {
                        if let Err(e) = start_playback(&audio, &ls) {
                            web_sys::console::log_1(&e);
                        }
                    }
                    if let Err(e) = begin_recording(&audio, &ls) {
                        web_sys::console::log_1(&e);
                    } else if let Some(app) = weak.upgrade() {
                        app.set_recording(true);
                        app.set_armed(false);
                        if has_tracks {
                            app.set_playing(true);
                        }
                    }
                }

                // Update UI
                if let Some(app) = weak.upgrade() {
                    let l = ls.borrow();

                    // Time display
                    if l.recorder.is_some() {
                        let elapsed = now - l.rec_start_time;
                        if l.loop_duration > 0.0 {
                            app.set_time_display(
                                format!(
                                    "REC {} / {}",
                                    format_time(elapsed),
                                    format_time(l.loop_duration)
                                )
                                .into(),
                            );
                        } else {
                            app.set_time_display(format!("REC {}", format_time(elapsed)).into());
                        }
                    } else if l.playing && l.loop_duration > 0.0 {
                        let elapsed = now - l.play_start_time;
                        app.set_time_display(
                            format!(
                                "{} / {}",
                                format_time(elapsed.min(l.loop_duration)),
                                format_time(l.loop_duration)
                            )
                            .into(),
                        );
                    } else if l.loop_duration > 0.0 {
                        app.set_time_display(format_time(l.loop_duration).into());
                    }

                    // Playhead
                    if l.playing && l.loop_duration > 0.0 {
                        let elapsed = now - l.play_start_time;
                        let pos = (elapsed / l.loop_duration).min(1.0) as f32;
                        app.set_playhead_pos(pos);

                        // Auto-stop when playback ends (one-shot)
                        if elapsed >= l.loop_duration && l.recorder.is_none() && !l.armed {
                            drop(l);
                            stop_playback(&ls);
                            app.set_playing(false);
                            app.set_playhead_pos(-1.0);
                        }
                    }

                    // Rebuild tracks model during recording (to show growing waveform)
                    if is_recording {
                        let l = ls.borrow();
                        let w = app.get_wave_width().max(100.0) - 80.0;
                        let rec_progress = if l.loop_duration > 0.0 {
                            ((now - l.rec_start_time) / l.loop_duration).min(1.0) as f32
                        } else {
                            1.0 // track 1: peaks always fill width
                        };
                        let infos = rebuild_tracks_model(&l, w, 56.0, rec_progress);
                        app.set_tracks(ModelRc::new(VecModel::from(infos)));
                    }
                }
            },
        );
    }

    app.run().unwrap();
}
