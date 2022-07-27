use std::borrow::Cow;
use std::net::SocketAddr;

use axum::routing::get_service;
use axum::{http::StatusCode, Json, response::IntoResponse, Router, routing::post};

use depict::graph_drawing::error::Error;
use depict::graph_drawing::error::Kind;
use depict::graph_drawing::error::OrErrExt;
use depict::graph_drawing::index::OriginalHorizontalRank;
use depict::graph_drawing::index::VerticalRank;
use depict::graph_drawing::layout::Loc;

use inflector::Inflector;

use tokio::task::JoinError;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{instrument, event, Level};
use tracing_error::{TracedError};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use depict::rest::*;

#[derive(Debug, thiserror::Error)]
#[error("draw error")]
pub struct DrawError {
}

impl From<Error> for DrawError {
    fn from(_source: Error) -> Self {
        Self {}
    }
}

impl From<JoinError> for DrawError {
    fn from(_source: JoinError) -> Self {
        Self {}
    }
}

impl<E> From<TracedError<E>> for DrawError {
    fn from(_source: TracedError<E>) -> Self {
        Self {}
    }
}

impl IntoResponse for DrawError {
    fn into_response(self) -> axum::response::Response {
        let status = StatusCode::INTERNAL_SERVER_ERROR;
        let body = "";
        (status, body).into_response()
    }
}

#[instrument]
async fn draw<'s>(Json(draw_rx): Json<Draw>) -> Result<Json<DrawResp>, DrawError> {
    let data = draw_rx.text.clone();

    tokio::task::spawn_blocking(move || {
        let drawing = depict::graph_drawing::frontend::dom::draw(data)?;
        let drawing = depict::rest::Drawing{
            crossing_number: drawing.crossing_number,
            viewbox_width: drawing.viewbox_width,
            nodes: drawing.nodes.into_iter().map(|n| match n {
                depict::graph_drawing::frontend::dom::Node::Div{key, label, hpos, vpos, width, height, z_index, ..} => {
                    depict::rest::Node::Div{ key, label, hpos, vpos, width, height, z_index }
                },
                depict::graph_drawing::frontend::dom::Node::Svg{ key, path, z_index, rel, label, .. } => {
                    depict::rest::Node::Svg{ key, path, z_index, rel, label: label.map(
                        |depict::graph_drawing::frontend::dom::Label{text, hpos, width, vpos}| {
                            depict::rest::Label{ text, hpos, width, vpos }
                        })
                    }
                },
            }).collect::<Vec<_>>(),
        };

        Result::<Json<DrawResp>, DrawError>::Ok(Json(DrawResp{
            drawing
        }))
    }).await?
}

fn main() -> Result<(), hyper::Error> {
    tracing_subscriber::Registry::default()
        .with(tracing_error::ErrorLayer::default())
        .with(tracing_subscriber::fmt::layer())
        .init();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let app = Router::new()
                .route("/api/draw/v1", post(draw))
                .fallback(
                    get_service(ServeDir::new(std::env::var("WEBROOT").unwrap())).handle_error(|error: std::io::Error| async move {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Unhandled internal error: {}", error),
                        )
                    }),
                )
                .layer(TraceLayer::new_for_http());
            let addr = SocketAddr::from(([127, 0, 0, 1], 8000));
            axum::Server::bind(&addr)
                .serve(app.into_make_service())
                .await
        })
}
