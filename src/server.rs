use std::collections::HashMap;

use actix_web::{post, ResponseError};
use actix_web::web::Json;
use actix_web::{get, web, App, HttpServer, Responder};

use diagrams::graph_drawing::error::Error;
use diagrams::graph_drawing::error::Kind;
use diagrams::graph_drawing::error::OrErrExt;
use diagrams::graph_drawing::geometry::LayoutProblem;
use diagrams::graph_drawing::geometry::LayoutSolution;
use diagrams::graph_drawing::geometry::calculate_sols;
use diagrams::graph_drawing::geometry::position_sols;
use diagrams::graph_drawing::graph::roots;
use diagrams::graph_drawing::index::OriginalHorizontalRank;
use diagrams::graph_drawing::index::VerticalRank;
use diagrams::graph_drawing::layout::Cvcg;
use diagrams::graph_drawing::layout::Loc;
use diagrams::graph_drawing::layout::Placement;
use diagrams::graph_drawing::layout::Vcg;
use diagrams::graph_drawing::layout::calculate_locs_and_hops;
use diagrams::graph_drawing::layout::calculate_vcg;
use diagrams::graph_drawing::layout::condense;
use diagrams::graph_drawing::layout::minimize_edge_crossing;
use diagrams::graph_drawing::layout::or_insert;
use diagrams::graph_drawing::layout::rank;
use inflector::Inflector;
use petgraph::Graph;
use petgraph::dot::Dot;
use serde::{Deserialize, Serialize};

use tokio::task::JoinError;
use tracing::{instrument, event, Level};
use tracing_error::{InstrumentResult, ExtractSpanTrace, TracedError};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use diagrams::parser::*;
use diagrams::graph_drawing::*;
use diagrams::rest::*;

fn estimate_widths<'s>(
    vcg: &Vcg<&'s str, &'s str>, 
    cvcg: &Cvcg<&'s str, &'s str>,
    placement: &Placement<&'s str>,
    layout_problem: &mut LayoutProblem<&'s str>
) -> Result<(), Error> {
    // let char_width = 8.67;
    let char_width = 9.0;
    let arrow_width = 40.0;
    
    let vert_node_labels = &vcg.vert_node_labels;
    let vert_edge_labels = &vcg.vert_edge_labels;
    let width_by_loc = &mut layout_problem.width_by_loc;
    let width_by_hop = &mut layout_problem.width_by_hop;
    let hops_by_edge = &placement.hops_by_edge;
    let loc_to_node = &placement.loc_to_node;
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

            match *ew {
                "senses" => {
                    percept_width = label_width.map(|label_width| arrow_width + char_width * label_width as f64).unwrap_or(20.0);
                }
                "actuates" => {
                    action_width = label_width.map(|label_width| arrow_width + char_width * label_width as f64).unwrap_or(20.0);
                }
                _ => {}
            }
        }

        for (lvl, (mhr, _nhr)) in hops.iter() {
            width_by_hop.insert((*lvl, *mhr, vl, wl), (action_width, percept_width));
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
    fn from(source: Error) -> Self {
        Self {}
    }
}

impl From<JoinError> for DrawError {
    fn from(source: JoinError) -> Self {
        Self {}
    }
}

impl<E> From<TracedError<E>> for DrawError {
    fn from(source: TracedError<E>) -> Self {
        Self {}
    }
}

impl ResponseError for DrawError {
    fn status_code(&self) -> actix_web::http::StatusCode {
        actix_web::http::StatusCode::INTERNAL_SERVER_ERROR
    }
}

#[post("/draw/v1")]
#[instrument]
async fn draw<'s>(draw_rx: web::Json<Draw>) -> Result<Json<DrawResp>, DrawError> {
    let data = draw_rx.text.clone();
    let v = parse(&data[..])
        .map_err(|e| match e {
            nom::Err::Error(e) => { nom::Err::Error(nom::error::convert_error(&data[..], e)) },
            nom::Err::Failure(e) => { nom::Err::Failure(nom::error::convert_error(&data[..], e)) },
            nom::Err::Incomplete(n) => { nom::Err::Incomplete(n) },
        })
        .in_current_span()?
        .1;

    let vcg = calculate_vcg(&v)?;
    let Vcg{vert, vert_vxmap: _, vert_node_labels, vert_edge_labels} = &vcg;

    // diagrams::graph_drawing::draw(v, &mut vcg)?;


    // let draw_query = Fact::Atom(Ident("draw"));
    // let draw_cmd = diagrams::render::resolve(v.iter(), &draw_query).next().unwrap();
    // diagrams::graph_drawing::draw(&v, draw_cmd, &mut vcg)?;

    // eprintln!("VERT: {:?}", Dot::new(&vert));

    let cvcg = condense(vert)?;
    let Cvcg{condensed, condensed_vxmap: _} = &cvcg;

    let roots = roots(condensed)?;

    let paths_by_rank = rank(condensed, &roots)?;

    let placement = calculate_locs_and_hops(condensed, &paths_by_rank)?;
    let Placement{hops_by_level, hops_by_edge, loc_to_node, node_to_loc, ..} = &placement;

    let (crossing_number, solved_locs) = minimize_edge_crossing(&placement)?;

    let mut layout_problem = calculate_sols(&solved_locs, loc_to_node, hops_by_level, hops_by_edge);

    estimate_widths(&vcg, &cvcg, &placement, &mut layout_problem)?;

    // let LayoutSolution{ls, rs, ss, ts} = tokio::task::spawn_blocking({
    //     let vcg = vcg.clone();
    //     let placement = placement.clone();
    //     let solved_locs = solved_locs.clone();
    //     let layout_problem = layout_problem.clone();
    //     move || {
    //         position_sols(&vcg, &placement, &solved_locs, &layout_problem)
    //     }
    // }).await??;
    let LayoutSolution{ls, rs, ss, ts} = position_sols(&vcg, &placement, &solved_locs, &layout_problem)?;
    
    let LayoutProblem{sol_by_loc, sol_by_hop, ..} = &layout_problem;

    let mut layout_debug = Graph::<String, String>::new();
    let mut layout_debug_vxmap = HashMap::new();
    for ((vl, wl), hops) in hops_by_edge.iter() {
        if *vl == "root" { continue; }
        let vn = node_to_loc[&Loc::Node(*vl)];
        let wn = node_to_loc[&Loc::Node(*wl)];
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
                .get(*vl)
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

            let hops = &hops_by_edge[&(*vl, *wl)];
            // eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

            let offset = match *ew { 
                "actuates" | "actuator" => -10.0,
                "senses" | "sensor" => 10.0,
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
                let hn = sol_by_hop[&(*lvl+1, *nhr, *vl, *wl)];
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
                    label_hpos = Some(match *ew {
                        "senses" => {
                            // ls[n]
                            hpos
                        },
                        "actuates" => {
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
                    .service(draw)
            })
            .bind(("127.0.0.1", 8000))?
            .run()
            .await
        })
}