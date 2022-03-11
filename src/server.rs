use actix_web::{get, web, App, HttpServer, Responder};

use tracing::{instrument, event, Level};
use tracing_error::{InstrumentResult, ExtractSpanTrace};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[get("/hello/{name}")]
async fn greet(name: web::Path<String>) -> impl Responder {
    format!("Hello {name}!")
}

fn main() -> std::io::Result<()> {

    tracing_subscriber::Registry::default()
        .with(tracing_error::ErrorLayer::default())
        .with(tracing_subscriber::fmt::layer())
        .init();


    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            HttpServer::new(|| {
                App::new()
                    .route("/hello", web::get().to(|| async { "Hello World!" }))
                    .service(greet)
            })
            .bind(("127.0.0.1", 8000))?
            .run()
            .await
        })
}