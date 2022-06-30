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
        let render_cell = depict::graph_drawing::frontend::render(Cow::Owned(data))?;
        let depiction = render_cell.borrow_dependent();
        
        let rs = &depiction.geometry_solution.rs;
        let ls = &depiction.geometry_solution.ls;
        let ss = &depiction.geometry_solution.ss;
        let ts = &depiction.geometry_solution.ts;
        let sol_by_loc = &depiction.geometry_problem.sol_by_loc;
        let loc_to_node = &depiction.layout_problem.loc_to_node;
        let vert_node_labels = &depiction.vcg.vert_node_labels;
        let vert_edge_labels = &depiction.vcg.vert_edge_labels;
        let hops_by_edge = &depiction.layout_problem.hops_by_edge;
        let sol_by_hop = &depiction.geometry_problem.sol_by_hop;
        let crossing_number = depiction.layout_solution.crossing_number;
        let condensed = &depiction.cvcg.condensed;

        let height_scale = 80.0;
        let vpad = 50.0;
        let line_height = 20.0;

        let mut texts = vec![];

        let root_n = sol_by_loc[&(VerticalRank(0), OriginalHorizontalRank(0))];
        let root_width = rs[root_n] - ls[root_n];

        for (loc, node) in loc_to_node.iter() {
            let (ovr, ohr) = loc;
            if (*ovr, *ohr) == (VerticalRank(0), OriginalHorizontalRank(0)) { continue; }
            let n = sol_by_loc[&(*ovr, *ohr)];

            let lpos = ls[n];
            let rpos = rs[n];

            let vpos = height_scale * ((*ovr-1).0 as f64) + vpad + ts[*ovr] * line_height;
            let width = (rpos - lpos).round();
            let hpos = lpos.round();

            if let Loc::Node(vl) = node {
                let key = vl.to_string();
                let mut label = vert_node_labels
                    .get(vl)
                    .or_err(Kind::KeyNotFoundError{key: vl.to_string()})?
                    .clone();
                if !label.is_screaming_snake_case() {
                    label = label.to_title_case();
                }
                texts.push(Node::Div{key, label, hpos, vpos, width});
            }
        }

        let mut arrows = vec![];

        for cer in condensed.edge_references() {
            for (m, (vl, wl, ew)) in cer.weight().iter().enumerate() {
                if *vl == "root" { continue; }

                let label_text = vert_edge_labels
                    .get(vl)
                    .and_then(|dsts| dsts
                        .get(wl)
                        .and_then(|rels| rels.get(ew)))
                    .map(|v| v.join("\n"));

                let hops = &hops_by_edge[&(vl.clone(), wl.clone())];
                // eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

                let offset = match ew { 
                    x if x == "actuates" || x == "actuator" => -10.0,
                    x if x == "senses" || x == "sensor" => 10.0,
                    _ => 0.0,
                };

                let mut path = vec![];
                let mut label_hpos = None;
                let mut label_width = None;
                let mut label_vpos = None;
                // use rand::Rng;
                // let mut rng = rand::thread_rng();

                for (n, hop) in hops.iter().enumerate() {
                    let (lvl, (_mhr, nhr)) = hop;
                    let hn = sol_by_hop[&(*lvl+1, *nhr, vl.clone(), wl.clone())];
                    let spos = ss[hn];
                    let hpos = (spos + offset).round(); // + rng.gen_range(-0.1..0.1));
                    let vpos = ((*lvl-1).0 as f64) * height_scale + vpad + ts[*lvl] * line_height;
                    let mut vpos2 = (lvl.0 as f64) * height_scale + vpad + ts[(*lvl+1)] * line_height;

                    if n == 0 {
                        let mut vpos = vpos;
                        if *ew == "senses" {
                            vpos += 33.0; // box height + arrow length
                        } else {
                            vpos += 26.0;
                        }
                        path.push(format!("M {hpos} {vpos}"));
                    }

                    if n == 0 {
                        let n = sol_by_loc[&((*lvl+1), *nhr)];
                        label_hpos = Some(match ew {
                            x if x == "senses" => {
                                // ls[n]
                                hpos
                            },
                            x if x == "actuates" => {
                                // ls[n]
                                hpos
                            },
                            _ => hpos
                        });
                        label_width = Some(rs[n] - ls[n]);
                        label_vpos = Some(((*lvl-1).0 as f64) * height_scale + vpad + ts[*lvl] * line_height);
                    }

                    if n == hops.len() - 1 && *ew == "actuates" { 
                        vpos2 -= 7.0; // arrowhead length
                    }

                    path.push(format!("L {hpos} {vpos2}"));

                }

                let key = format!("{vl}_{wl}_{ew}_{m}");
                let path = path.join(" ");

                let mut label = None;

                if let (Some(label_text), Some(label_hpos), Some(label_width), Some(label_vpos)) = (label_text, label_hpos, label_width, label_vpos) {
                    label = Some(Label{text: label_text, hpos: label_hpos, width: label_width, vpos: label_vpos})
                }
                arrows.push(Node::Svg{key, path, rel: ew.to_string(), label});
            }
        }

        let nodes = texts
            .into_iter()
            .chain(arrows.into_iter())
            .collect::<Vec<_>>();

        event!(Level::TRACE, %root_width, ?nodes, "NODES");

        Result::<Json<DrawResp>, DrawError>::Ok(Json(DrawResp{
            drawing: Drawing{
                crossing_number: Some(crossing_number), 
                viewbox_width: root_width,
                nodes
            }
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
