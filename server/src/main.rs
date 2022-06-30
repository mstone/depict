use std::collections::HashMap;
use std::net::SocketAddr;

use axum::routing::get_service;
use axum::{http::StatusCode, Json, response::IntoResponse, Router, routing::post};

use depict::graph_drawing::error::Error;
use depict::graph_drawing::error::Kind;
use depict::graph_drawing::error::OrErrExt;
use depict::graph_drawing::geometry::GeometryProblem;
use depict::graph_drawing::geometry::GeometrySolution;
use depict::graph_drawing::geometry::calculate_sols;
use depict::graph_drawing::geometry::position_sols;
use depict::graph_drawing::graph::roots;
use depict::graph_drawing::index::OriginalHorizontalRank;
use depict::graph_drawing::index::VerticalRank;
use depict::graph_drawing::layout::eval::eval;
use depict::graph_drawing::layout::{Cvcg, calculate_vcg, Len};
use depict::graph_drawing::layout::Loc;
use depict::graph_drawing::layout::LayoutProblem;
use depict::graph_drawing::layout::Vcg;
use depict::graph_drawing::layout::calculate_locs_and_hops;
use depict::graph_drawing::layout::condense;
use depict::graph_drawing::layout::minimize_edge_crossing;
use depict::graph_drawing::layout::or_insert;
use depict::graph_drawing::layout::rank;

use depict::parser::{Parser, Token, Item};

use inflector::Inflector;
use logos::Logos;
use petgraph::Graph;
use petgraph::dot::Dot;

use tokio::task::JoinError;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{instrument, event, Level};
use tracing_error::{InstrumentResult, TracedError};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use depict::rest::*;

fn estimate_widths<I>(
    vcg: &Vcg<I, I>, 
    cvcg: &Cvcg<I, I>,
    layout_problem: &LayoutProblem<I>,
    geometry_problem: &mut GeometryProblem<I>
) -> Result<(), Error> where
  I: Clone + std::fmt::Debug + Ord + Eq + PartialEq + PartialOrd + std::hash::Hash + std::fmt::Display + Len + PartialEq<&'static str>
{
    // let char_width = 8.67;
    let char_width = 9.0;
    let arrow_width = 40.0;
    
    let vert_node_labels = &vcg.vert_node_labels;
    let vert_edge_labels = &vcg.vert_edge_labels;
    let width_by_loc = &mut geometry_problem.width_by_loc;
    let width_by_hop = &mut geometry_problem.width_by_hop;
    let hops_by_edge = &layout_problem.hops_by_edge;
    let loc_to_node = &layout_problem.loc_to_node;
    let condensed = &cvcg.condensed;
    let condensed_vxmap = &cvcg.condensed_vxmap;
    
    for (loc, node) in loc_to_node.iter() {
        let (ovr, ohr) = loc;
        if let Loc::Node(vl) = node {
            let mut label = vert_node_labels
                .get(vl)
                .or_err(Kind::KeyNotFoundError{key: vl.to_string()})?
                .clone();
            if !label.is_screaming_snake_case() {
                label = label.to_title_case();
            }
            width_by_loc.insert((*ovr, *ohr), char_width * label.len() as f64);
        }
    }

    for ((vl, wl), hops) in hops_by_edge.iter() {
        let mut action_width = 10.0;
        let mut percept_width = 10.0;
        let cex = condensed.find_edge(condensed_vxmap[vl], condensed_vxmap[wl]).unwrap();
        let cew = condensed.edge_weight(cex).unwrap();
        for (vl, wl, ew) in cew.iter() {
            let label_width = vert_edge_labels
                .get(vl)
                .and_then(|dsts| dsts
                    .get(wl)
                    .and_then(|rels| rels.get(ew)))
                    .and_then(|labels| labels
                        .iter()
                        .map(|label| label.len())
                        .max()
                    );

            match ew {
                x if x == &"senses" => {
                    percept_width = label_width.map(|label_width| arrow_width + char_width * label_width as f64).unwrap_or(20.0);
                }
                x if x == &"actuates" => {
                    action_width = label_width.map(|label_width| arrow_width + char_width * label_width as f64).unwrap_or(20.0);
                }
                _ => {}
            }
        }

        for (lvl, (mhr, _nhr)) in hops.iter() {
            width_by_hop.insert((*lvl, *mhr, vl.clone(), wl.clone()), (action_width, percept_width));
            if width_by_loc.get(&(*lvl, *mhr)).is_none() {
                width_by_loc.insert((*lvl, *mhr), action_width + percept_width);
            }
        }
    }

    Ok(())
}

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
        let mut p = Parser::new();
        {
            let lex = Token::lexer(&data);
            let tks = lex.collect::<Vec<_>>();
            event!(Level::TRACE, ?tks, "LEX");
        }
        let mut lex = Token::lexer(&data);
        while let Some(tk) = lex.next() {
            p.parse(tk)
                .map_err(|_| {
                    Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
                })
                .in_current_span()?
        }

        let v: Vec<Item> = p.end_of_input()
            .map_err(|_| {
                Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
            })
            .in_current_span()?;

        event!(Level::TRACE, ?v, "PARSE");
        eprintln!("PARSE {v:#?}");

        let process = eval(&v[..]);

        let vcg = calculate_vcg(process)?;
        let Vcg{vert, vert_vxmap: _, vert_node_labels, vert_edge_labels} = &vcg;

        let cvcg = condense(vert)?;
        let Cvcg{condensed, condensed_vxmap: _} = &cvcg;

        let roots = roots(condensed)?;

        let paths_by_rank = rank(condensed, &roots)?;

        let layout_problem = calculate_locs_and_hops(condensed, &paths_by_rank)?;
        let LayoutProblem{hops_by_level, hops_by_edge, loc_to_node, node_to_loc, ..} = &layout_problem;

        let (crossing_number, solved_locs) = minimize_edge_crossing(&layout_problem)?;

        let mut geometry_problem = calculate_sols(&solved_locs, loc_to_node, hops_by_level, hops_by_edge);

        estimate_widths(&vcg, &cvcg, &layout_problem, &mut geometry_problem)?;

        let GeometrySolution{ls, rs, ss, ts} = position_sols(&vcg, &layout_problem, &solved_locs, &geometry_problem)?;
        
        let GeometryProblem{sol_by_loc, sol_by_hop, ..} = &geometry_problem;

        let mut layout_debug = Graph::<String, String>::new();
        let mut layout_debug_vxmap = HashMap::new();
        for ((vl, wl), hops) in hops_by_edge.iter() {
            if *vl == "root" { continue; }
            let vn = node_to_loc[&Loc::Node(vl.clone())];
            let wn = node_to_loc[&Loc::Node(wl.clone())];
            let vshr = solved_locs[&vn.0][&vn.1];
            let wshr = solved_locs[&wn.0][&wn.1];

            let vx = or_insert(&mut layout_debug, &mut layout_debug_vxmap, format!("{vl} {vshr}"));
            let wx = or_insert(&mut layout_debug, &mut layout_debug_vxmap, format!("{wl} {wshr}"));

            for (n, (lvl, (mhr, nhr))) in hops.iter().enumerate() {
                let shr = solved_locs[lvl][mhr];
                let shrd = solved_locs[&(*lvl+1)][nhr];
                let lvl1 = *lvl+1;
                let vxh = or_insert(&mut layout_debug, &mut layout_debug_vxmap, format!("{vl} {wl} {lvl},{shr}"));
                let wxh = or_insert(&mut layout_debug, &mut layout_debug_vxmap, format!("{vl} {wl} {lvl1},{shrd}"));
                layout_debug.add_edge(vxh, wxh, format!("{lvl},{shr}->{lvl1},{shrd}"));
                if n == 0 {
                    layout_debug.add_edge(vx, vxh, format!("{lvl1},{shrd}"));
                }
                if n == hops.len()-1 {
                    layout_debug.add_edge(wxh, wx, format!("{lvl1},{shrd}"));
                }
            }
        }
        let layout_debug_dot = Dot::new(&layout_debug);
        event!(Level::TRACE, %layout_debug_dot, "LAYOUT GRAPH");

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
                layout_debug,
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
