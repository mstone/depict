use dioxus::prelude::*;
use dioxus::hooks::suspense::use_suspense;

use reqwasm::http::{Request, Response};

async fn click() -> Response {
    Request::get("/hello")
        .send()
        .await
        .unwrap()
}

fn app(cx: Scope) -> Element {
    let hello = use_suspense(cx, || click(), |cx, res| match res { Ok(res) => rsx!(cx, "OK"), _ => rsx!(cx, "ERR"), });
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
