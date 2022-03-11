use dioxus::prelude::*;

use reqwasm::http::{Request, Response};

async fn click() -> Response {
    Request::get("/api/hello")
        .send()
        .await
        .unwrap()
}

fn app(cx: Scope) -> Element {

    let hello = use_future(&cx, (), |_| async move { 
        let res = click().await.text().await;
        match res { 
            Ok(res) => Some(res), 
            _ => None, 
        } 
    });

    let hello = match hello.value() {
        Some(Some(res)) => cx.render(rsx!(div { "{res}" })),
        Some(None) => cx.render(rsx!("loading...")),
        _ => cx.render(rsx!("loading2...")),
    };

    cx.render(rsx! (
        div {
            style: "text-align: center;",
            h1 { "🌗 Dioxus 🚀" }
            h3 { "Frontend that scales." }
            p { "Dioxus is a portable, performant, and ergonomic framework for building cross-platform user interfaces in Rust." }
            hello
        }
    ))
}

fn main() {
    wasm_logger::init(wasm_logger::Config::default());
    console_error_panic_hook::set_once();

    dioxus::web::launch(app); 
}
