use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{
    HtmlCanvasElement, KeyboardEvent, MouseEvent, WebGlBuffer, WebGlProgram,
    WebGlRenderingContext as GL,
};

const VERT_SHADER: &str = r#"
    attribute vec3 a_position;
    uniform mat4 u_model;
    uniform mat4 u_projection;
    void main() {
        gl_Position = u_projection * u_model * vec4(a_position, 1.0);
    }
"#;

const FRAG_SHADER: &str = r#"
    precision mediump float;
    void main() {
        gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
    }
"#;

struct State {
    rot_x: f32,
    rot_y: f32,
    auto_rot_x: f32,
    auto_rot_y: f32,
    dragging: bool,
    last_mouse_x: i32,
    last_mouse_y: i32,
}

fn compile_shader(gl: &GL, shader_type: u32, source: &str) -> Result<web_sys::WebGlShader, String> {
    let shader = gl
        .create_shader(shader_type)
        .ok_or("Unable to create shader")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);
    if gl
        .get_shader_parameter(&shader, GL::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(gl
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| "Unknown error".into()))
    }
}

fn link_program(
    gl: &GL,
    vert: &web_sys::WebGlShader,
    frag: &web_sys::WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = gl.create_program().ok_or("Unable to create program")?;
    gl.attach_shader(&program, vert);
    gl.attach_shader(&program, frag);
    gl.link_program(&program);
    if gl
        .get_program_parameter(&program, GL::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| "Unknown error".into()))
    }
}

fn perspective_matrix(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y / 2.0).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) * nf, -1.0,
        0.0, 0.0, 2.0 * far * near * nf, 0.0,
    ]
}

fn rotation_matrix(rx: f32, ry: f32) -> [f32; 16] {
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    [
        cy, sx * sy, -cx * sy, 0.0,
        0.0, cx, sx, 0.0,
        sy, -sx * cy, cx * cy, 0.0,
        0.0, 0.0, -4.0, 1.0,
    ]
}

fn create_cube_buffer(gl: &GL) -> Result<(WebGlBuffer, i32), String> {
    #[rustfmt::skip]
    let vertices: [f32; 72] = [
        // bottom face edges
        -1.0, -1.0, -1.0,  1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,  1.0, -1.0,  1.0,
         1.0, -1.0,  1.0, -1.0, -1.0,  1.0,
        -1.0, -1.0,  1.0, -1.0, -1.0, -1.0,
        // top face edges
        -1.0,  1.0, -1.0,  1.0,  1.0, -1.0,
         1.0,  1.0, -1.0,  1.0,  1.0,  1.0,
         1.0,  1.0,  1.0, -1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0, -1.0,  1.0, -1.0,
        // vertical edges
        -1.0, -1.0, -1.0, -1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,  1.0,  1.0, -1.0,
         1.0, -1.0,  1.0,  1.0,  1.0,  1.0,
        -1.0, -1.0,  1.0, -1.0,  1.0,  1.0,
    ];

    let buffer = gl.create_buffer().ok_or("Failed to create buffer")?;
    gl.bind_buffer(GL::ARRAY_BUFFER, Some(&buffer));

    unsafe {
        let view = js_sys::Float32Array::view(&vertices);
        gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
    }

    Ok((buffer, 24))
}

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()?;

    canvas.set_width(800);
    canvas.set_height(600);

    let gl: GL = canvas.get_context("webgl")?.unwrap().dyn_into()?;

    gl.clear_color(0.0, 0.0, 0.0, 1.0);
    gl.enable(GL::DEPTH_TEST);

    let vert = compile_shader(&gl, GL::VERTEX_SHADER, VERT_SHADER)
        .map_err(|e| JsValue::from_str(&e))?;
    let frag = compile_shader(&gl, GL::FRAGMENT_SHADER, FRAG_SHADER)
        .map_err(|e| JsValue::from_str(&e))?;
    let program = link_program(&gl, &vert, &frag).map_err(|e| JsValue::from_str(&e))?;
    gl.use_program(Some(&program));

    let (buffer, vertex_count) =
        create_cube_buffer(&gl).map_err(|e| JsValue::from_str(&e))?;

    let a_position = gl.get_attrib_location(&program, "a_position") as u32;
    gl.bind_buffer(GL::ARRAY_BUFFER, Some(&buffer));
    gl.vertex_attrib_pointer_with_i32(a_position, 3, GL::FLOAT, false, 0, 0);
    gl.enable_vertex_attrib_array(a_position);

    let u_model = gl.get_uniform_location(&program, "u_model");
    let u_projection = gl.get_uniform_location(&program, "u_projection");

    let proj = perspective_matrix(PI / 4.0, 800.0 / 600.0, 0.1, 100.0);
    gl.uniform_matrix4fv_with_f32_array(u_projection.as_ref(), false, &proj);

    let state = Rc::new(RefCell::new(State {
        rot_x: 0.3,
        rot_y: 0.5,
        auto_rot_x: 0.005,
        auto_rot_y: 0.01,
        dragging: false,
        last_mouse_x: 0,
        last_mouse_y: 0,
    }));

    // Mouse down
    {
        let state = Rc::clone(&state);
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let mut s = state.borrow_mut();
            s.dragging = true;
            s.auto_rot_x = 0.0;
            s.auto_rot_y = 0.0;
            s.last_mouse_x = e.client_x();
            s.last_mouse_y = e.client_y();
        });
        canvas.add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Mouse up
    {
        let state = Rc::clone(&state);
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |_: MouseEvent| {
            state.borrow_mut().dragging = false;
        });
        canvas.add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Mouse move
    {
        let state = Rc::clone(&state);
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let mut s = state.borrow_mut();
            if s.dragging {
                let dx = e.client_x() - s.last_mouse_x;
                let dy = e.client_y() - s.last_mouse_y;
                s.rot_y += dx as f32 * 0.01;
                s.rot_x += dy as f32 * 0.01;
                s.last_mouse_x = e.client_x();
                s.last_mouse_y = e.client_y();
            }
        });
        canvas.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Keyboard
    {
        let state = Rc::clone(&state);
        let cb = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
            let mut s = state.borrow_mut();
            s.auto_rot_x = 0.0;
            s.auto_rot_y = 0.0;
            let step = 0.05;
            match e.key().as_str() {
                "ArrowUp" => s.rot_x -= step,
                "ArrowDown" => s.rot_x += step,
                "ArrowLeft" => s.rot_y -= step,
                "ArrowRight" => s.rot_y += step,
                _ => {}
            }
        });
        document.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // Render loop
    {
        let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
        let g = Rc::clone(&f);

        *g.borrow_mut() = Some(Closure::new(move || {
            {
                let mut s = state.borrow_mut();
                s.rot_x += s.auto_rot_x;
                s.rot_y += s.auto_rot_y;
            }

            let s = state.borrow();
            let model = rotation_matrix(s.rot_x, s.rot_y);
            gl.uniform_matrix4fv_with_f32_array(u_model.as_ref(), false, &model);

            gl.clear(GL::COLOR_BUFFER_BIT | GL::DEPTH_BUFFER_BIT);
            gl.draw_arrays(GL::LINES, 0, vertex_count);

            web_sys::window()
                .unwrap()
                .request_animation_frame(
                    f.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
                )
                .unwrap();
        }));

        window.request_animation_frame(
            g.borrow().as_ref().unwrap().as_ref().unchecked_ref(),
        )?;
    }

    Ok(())
}
