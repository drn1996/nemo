FROM rust:slim

RUN rustup target add wasm32-unknown-unknown \
    && cargo install trunk --locked \
    && rm -rf /usr/local/cargo/registry /usr/local/cargo/git

LABEL org.opencontainers.image.source=https://github.com/drn1996/nemo
