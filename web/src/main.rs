use std::{default::Default, panic::catch_unwind, collections::HashMap};

use depict::{rest::*, graph_drawing::{error::{Kind, Error, OrErrExt}, layout::{Loc, calculate_vcg2, Vcg, condense, Cvcg, rank, calculate_locs_and_hops, LayoutProblem, minimize_edge_crossing, Len, or_insert, debug::debug}, graph::roots, geometry::{calculate_sols, position_sols, GeometryProblem, GeometrySolution}, index::{LocSol, HopSol, VerticalRank, OriginalHorizontalRank}}, parser::{Parser, Token, Item}};

use dioxus::{prelude::*, core::to_owned};

use futures::StreamExt;
use indoc::indoc;

use logos::Logos;
use petgraph::Graph;
use reqwasm::http::{Request, Response};

use tracing::{event, Level, Instrument};

async fn click(s: String) -> Response {
    let draw_req = serde_json::to_string(&Draw{text: s}).unwrap();
    Request::post("/api/draw/v1")
        .header("Content-Type", "application/json")
        .body(draw_req)
        .send()
        .await
        .unwrap()
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Label {
    pub text: String,
    pub hpos: f64,
    pub width: f64,
    pub vpos: f64,
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Node {
  Div { key: String, label: String, hpos: f64, vpos: f64, width: f64, loc: LocSol, estimated_width: f64 },
  Svg { key: String, path: String, rel: String, label: Option<Label>, hops: Vec<HopSol>, estimated_width: (f64,f64) },
}

#[derive(Clone, Debug)]
pub struct Drawing {
    pub crossing_number: Option<usize>,
    pub viewbox_width: f64,
    pub nodes: Vec<Node>,
}

impl Default for Drawing {
    fn default() -> Self {
        Self { 
            crossing_number: Default::default(), 
            viewbox_width: 1024.0,
            nodes: Default::default() 
        }
    }
}

fn estimate_widths<I>(
    vcg: &Vcg<I, I>, 
    cvcg: &Cvcg<I, I>,
    layout_problem: &LayoutProblem<I>,
    geometry_problem: &mut GeometryProblem<I>
) -> Result<(), Error> where
    I: Clone + std::fmt::Debug + Ord + std::hash::Hash + std::fmt::Display + Len + PartialEq<&'static str>,
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
            let label = vert_node_labels
                .get(vl)
                .or_err(Kind::KeyNotFoundError{key: vl.to_string()})?
                .clone();
            // if !label.is_screaming_snake_case() {
            //     label = label.to_title_case();
            // }
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
                x if *x == "senses" => {
                    percept_width = label_width.map(|label_width| arrow_width + char_width * label_width as f64).unwrap_or(20.0);
                }
                x if *x == "actuates" => {
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

fn draw(data: String) -> Result<Drawing, Error> {

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
            })?
    }

    let v: Vec<Item> = p.end_of_input()
        .map_err(|_| {
            Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
        })?;

    event!(Level::TRACE, ?v, "PARSE");
    eprintln!("PARSE {v:#?}");

    let vcg = calculate_vcg2(&v)?;

    let Vcg{vert, vert_vxmap: _, vert_node_labels, vert_edge_labels} = &vcg;

    // depict::graph_drawing::draw(v, &mut vcg)?;


    // let draw_query = Fact::Atom(Ident("draw"));
    // let draw_cmd = depict::render::resolve(v.iter(), &draw_query).next().unwrap();
    // depict::graph_drawing::draw(&v, draw_cmd, &mut vcg)?;

    // eprintln!("VERT: {:?}", Dot::new(&vert));

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

    debug(&layout_problem, &solved_locs);

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
            let label = vert_node_labels
                .get(vl)
                .or_err(Kind::KeyNotFoundError{key: vl.to_string()})?
                .clone();
            // if !label.is_screaming_snake_case() {
            //     label = label.to_title_case();
            // }
            let estimated_width = geometry_problem.width_by_loc[&(*ovr, *ohr)];
            texts.push(Node::Div{key, label, hpos, vpos, width, loc: n, estimated_width});
        }
    }

    let mut arrows = vec![];

    for cer in condensed.edge_references() {
        let mut prev_vwe = None;
        for (m, (vl, wl, ew)) in cer.weight().iter().enumerate() {
            if *vl == "root" { continue; }

            if prev_vwe == Some((vl, wl, ew)) {
                continue
            } else {
                prev_vwe = Some((vl, wl, ew))
            }

            let label_text = vert_edge_labels
                .get(vl)
                .and_then(|dsts| dsts
                    .get(wl)
                    .and_then(|rels| rels.get(ew)))
                .map(|v| v.join("\n"));

            let hops = &hops_by_edge[&(vl.clone(), wl.clone())];
            // eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

            let offset = match ew { 
                x if x == "actuates" => -10.0,
                x if x == "actuator" => -10.0,
                x if x == "senses" => 10.0,
                x if x == "sensor" => 10.0,
                _ => 0.0,
            };

            let mut path = vec![];
            let mut label_hpos = None;
            let mut label_width = None;
            let mut label_vpos = None;
            // use rand::Rng;
            // let mut rng = rand::thread_rng();
            let mut hn0 = vec![];
            let mut estimated_width0 = None;

            for (n, hop) in hops.iter().enumerate() {
                let (lvl, (mhr, nhr)) = hop;
                // let hn = sol_by_hop[&(*lvl+1, *nhr, vl.clone(), wl.clone())];
                let hn = sol_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())];
                let spos = ss[hn];
                let hnd = sol_by_hop[&(*lvl+1, *nhr, vl.clone(), wl.clone())];
                let sposd = ss[hnd];
                let hpos = (spos + offset).round(); // + rng.gen_range(-0.1..0.1));
                let hposd = (sposd + offset).round();
                let vpos = ((*lvl-1).0 as f64) * height_scale + vpad + ts[*lvl] * line_height;
                let mut vpos2 = (lvl.0 as f64) * height_scale + vpad + ts[(*lvl+1)] * line_height;

                if n == 0 {
                    hn0.push(hn);
                }
                hn0.push(hnd);
                
                if n == 0 {
                    estimated_width0 = Some(geometry_problem.width_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())]);
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
                    // let n = sol_by_loc[&((*lvl), *mhr)];
                    label_hpos = Some(match ew {
                        x if x == "senses" => {
                            // ls[n]
                            hposd
                        },
                        x if x == "actuates" => {
                            // ls[n]
                            hposd
                        },
                        _ => hposd
                    });
                    label_width = Some(match ew {
                        x if x == "senses" => {
                            // ls[n]
                            rs[n] - sposd
                        },
                        x if x == "actuates" => {
                            // ls[n]
                            sposd - ls[n]
                        },
                        _ => rs[n] - ls[n]
                    });
                    label_vpos = Some(((*lvl-1).0 as f64) * height_scale + vpad + ts[*lvl] * line_height);
                }

                if n == hops.len() - 1 && *ew == "actuates" { 
                    vpos2 -= 7.0; // arrowhead length
                }

                path.push(format!("L {hposd} {vpos2}"));

            }

            let key = format!("{vl}_{wl}_{ew}_{m}");
            let path = path.join(" ");

            let mut label = None;

            if let (Some(label_text), Some(label_hpos), Some(label_width), Some(label_vpos)) = (label_text, label_hpos, label_width, label_vpos) {
                label = Some(Label{text: label_text, hpos: label_hpos, width: label_width, vpos: label_vpos})
            }
            arrows.push(Node::Svg{key, path, rel: ew.to_string(), label, hops: hn0, estimated_width: estimated_width0.unwrap()});
        }
    }

    let nodes = texts
        .into_iter()
        .chain(arrows.into_iter())
        .collect::<Vec<_>>();

    event!(Level::TRACE, %root_width, ?nodes, "NODES");
    // println!("NODES: {nodes:#?}");

    Ok(Drawing{
        crossing_number: Some(crossing_number), 
        viewbox_width: root_width,
        nodes
    })
}

const PLACEHOLDER: &str = indoc!("
    person microwave food: open, start, stop / beep : heat
    person food: stir
");

const PLACEHOLDER_DRAWING: &str = r#"
{
    "drawing": {
        "crossing_number": 0,
        "viewbox_width": 412.3232380465066,
        "layout_debug": {
            "nodes": [
                "microwave 0",
                "food 0",
                "microwave food 2,0",
                "microwave food 3,0",
                "person 0",
                "person food 1,0",
                "person food 2,1",
                "person food 3,0",
                "person microwave 1,0",
                "person microwave 2,0"
            ],
            "node_holes": [],
            "edge_property": "directed",
            "edges": [
                [
                    2,
                    3,
                    "2,0->3,0"
                ],
                [
                    0,
                    2,
                    "3,0"
                ],
                [
                    3,
                    1,
                    "3,0"
                ],
                [
                    5,
                    6,
                    "1,0->2,1"
                ],
                [
                    4,
                    5,
                    "2,1"
                ],
                [
                    6,
                    7,
                    "2,1->3,0"
                ],
                [
                    7,
                    1,
                    "3,0"
                ],
                [
                    8,
                    9,
                    "1,0->2,0"
                ],
                [
                    4,
                    8,
                    "2,0"
                ],
                [
                    9,
                    0,
                    "2,0"
                ]
            ]
        },
        "nodes": [
            {
                "Div": {
                    "key": "microwave",
                    "label": "Microwave",
                    "hpos": 0,
                    "vpos": 170,
                    "width": 186
                }
            },
            {
                "Div": {
                    "key": "person",
                    "label": "Person",
                    "hpos": 39,
                    "vpos": 50,
                    "width": 367
                }
            },
            {
                "Div": {
                    "key": "food",
                    "label": "Food",
                    "hpos": 26,
                    "vpos": 250,
                    "width": 387
                }
            },
            {
                "Svg": {
                    "key": "person_food_actuates_0",
                    "path": "M 364 76 L 364 170 L 362 243",
                    "rel": "actuates",
                    "label": {
                        "text": "stir",
                        "hpos": 364,
                        "width": 124.9860715305249,
                        "vpos": 50
                    }
                }
            },
            {
                "Svg": {
                    "key": "person_microwave_actuates_0",
                    "path": "M 136 76 L 136 163",
                    "rel": "actuates",
                    "label": {
                        "text": "open\nstart\nstop",
                        "hpos": 136,
                        "width": 186.00027036035365,
                        "vpos": 50
                    }
                }
            },
            {
                "Svg": {
                    "key": "person_microwave_actuates_1",
                    "path": "M 136 76 L 136 163",
                    "rel": "actuates",
                    "label": {
                        "text": "open\nstart\nstop",
                        "hpos": 136,
                        "width": 186.00027036035365,
                        "vpos": 50
                    }
                }
            },
            {
                "Svg": {
                    "key": "person_microwave_actuates_2",
                    "path": "M 136 76 L 136 163",
                    "rel": "actuates",
                    "label": {
                        "text": "open\nstart\nstop",
                        "hpos": 136,
                        "width": 186.00027036035365,
                        "vpos": 50
                    }
                }
            },
            {
                "Svg": {
                    "key": "person_microwave_senses_3",
                    "path": "M 156 83 L 156 170",
                    "rel": "senses",
                    "label": {
                        "text": "beep",
                        "hpos": 156,
                        "width": 186.00027036035365,
                        "vpos": 50
                    }
                }
            },
            {
                "Svg": {
                    "key": "microwave_food_actuates_0",
                    "path": "M 86 196 L 86 243",
                    "rel": "actuates",
                    "label": {
                        "text": "heat",
                        "hpos": 86,
                        "width": 386.69389586984727,
                        "vpos": 170
                    }
                }
            }
        ]
    }
}
"#;

pub fn render<P>(cx: Scope<P>, drawing: Drawing)-> Option<VNode> {
    let viewbox_width = drawing.viewbox_width;
    let mut nodes = drawing.nodes;
    let viewbox_height = 768;
    let mut children = vec![];
    nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for node in nodes {
        match node {
            Node::Div{key, label, hpos, vpos, width, ..} => {
                children.push(cx.render(rsx! {
                    div {
                        key: "{key}",
                        class: "absolute border border-black text-center z-10 bg-white", // bg-opacity-50
                        top: "{vpos}px",
                        left: "{hpos}px",
                        width: "{width}px",
                        span {
                            "{label}"
                        }
                    }
                }));
            },
            Node::Svg{key, path, rel, label, ..} => {
                let marker_orient = if rel == "actuates" { "auto" } else { "auto-start-reverse" };
                let stroke_dasharray = if rel == "fake" { "5 5" } else { "none" };
                let stroke_color = if rel == "fake" { "hsl(0, 0%, 50%)" } else { "currentColor" };
                children.push(cx.render(rsx!{
                    div {
                        key: "{key}",
                        class: "absolute",
                        svg {
                            fill: "none",
                            stroke: "{stroke_color}",
                            stroke_linecap: "round",
                            stroke_linejoin: "round",
                            stroke_width: "1",
                            view_box: "0 0 {viewbox_width} {viewbox_height}",
                            width: "{viewbox_width}px",
                            height: "{viewbox_height}px",
                            marker {
                                id: "arrowhead",
                                markerWidth: "7",
                                markerHeight: "10",
                                refX: "0",
                                refY: "5",
                                orient: "{marker_orient}",
                                view_box: "0 0 10 10",
                                path {
                                    d: "M 0 0 L 10 5 L 0 10 z",
                                    fill: "#000",
                                }
                            }
                            { 
                                match rel.as_str() {
                                    "actuates" => {
                                        rsx!(path {
                                            d: "{path}",
                                            marker_end: "url(#arrowhead)",
                                        })
                                    },
                                    "senses" => {
                                        rsx!(path {
                                            d: "{path}",
                                            "marker-start": "url(#arrowhead)",
                                            // marker_start: "url(#arrowhead)", // BUG: should work, but doesn't.
                                        })
                                    },
                                    _ => {
                                        rsx!(path {
                                            d: "{path}",
                                            stroke_dasharray: "{stroke_dasharray}",
                                        })
                                    }
                                }
                            }
                        }
                        {match label { 
                            Some(Label{text, hpos, width: _, vpos}) => {
                                let translate = if rel == "actuates" { 
                                    "translate(calc(-100% - 1.5ex))"
                                } else { 
                                    "translate(1.5ex)"
                                };
                                let border = match rel.as_str() { 
                                    "actuates" => "border border-red-300",
                                    "senses" => "border border-blue-300",
                                    _ => "",
                                };
                                rsx!(div {
                                    class: "absolute",
                                    left: "{hpos}px",
                                    // width: "{width}px",
                                    top: "calc({vpos}px + 40px)",
                                    div {
                                        class: "whitespace-pre z-50 bg-white border-box text-sm font-mono {border}",
                                        transform: "{translate}",
                                        "{text}"
                                    }
                                })
                            },
                            _ => rsx!(div {}),
                        }}
                    }
                }));
            },
        }
    }
    // dbg!(cx.render(rsx!(children)))
    cx.render(rsx!(children))
}

pub struct AppProps {
}

pub fn app(cx: Scope<AppProps>) -> Element {

    let model = use_state(&cx, || String::from(PLACEHOLDER));

    // let drawing = use_state(&cx, || serde_json::from_str::<DrawResp>(PLACEHOLDER_DRAWING).unwrap().drawing);
    let drawing = use_state(&cx, || Drawing::default());
    
    let drawing_client = use_coroutine(&cx, |mut rx: UnboundedReceiver<String>| {
        to_owned![drawing];
        async move {
            let mut prev_model: Option<String> = None;
            while let Some(model) = rx.next().await {
                // let res: Result<DrawResp, _> = click(model).await.json().await;
                // if let Ok(drawing_resp) = res {
                //     drawing.set(drawing_resp.drawing);
                // }
                let nodes = if model.trim().is_empty() {
                    Ok(Ok(Drawing::default()))
                } else {
                    catch_unwind(|| {
                        draw(model.clone())
                    })
                };
                match nodes {
                    Ok(Ok(drawing_nodes)) => {
                        prev_model = Some(model);
                        drawing.set(drawing_nodes);
                    },
                    _ => {},
                }
            }
        }
    });

    // let (drawing, nodes) = match drawing_fut.value() {
    //     Some(Some(drawing)) => (Some(drawing), render(cx, drawing.clone())),
    //     Some(None) => (None, cx.render(rsx!("loading..."))),
    //     _ => (None, cx.render(rsx!("error"))),
    // };
    let nodes = render(cx, drawing.get().clone());

    let viewbox_width = drawing.viewbox_width;
    // let viewbox_width = 1024.0;
    // let _crossing_number = cx.render(rsx!(match drawing.get().crossing_number {
    //     Some(cn) => rsx!(span { "{cn}" }),
    //     None => rsx!(div{}),
    // }));

    cx.render(rsx!{
        link { href:"https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css", rel:"stylesheet" }
        div {
            // key: "editor",
            class: "width-full z-20 p-4",
            div {
                class: "max-w-3xl mx-auto flex flex-col",
                div {
                    // key: "editor_label",
                    class: "fs-24",
                    "Model"
                }
                div {
                    // key: "editor_editor",
                    textarea {
                        class: "border",
                        rows: "6",
                        cols: "80",
                        // placeholder: "",
                        oninput: move |e| { 
                            event!(Level::TRACE, "INPUT");
                            drawing_client.send(e.value.clone());
                        },
                        "{model}"
                    }
                }
                div {
                    class: "text-sm text-gray-400 width-full",
                    div {
                        span {
                            class: "text-black",
                            "Syntax: "
                        }
                        span {
                            class: "italic",
                            "node node ... : action action... / percept percept ... : action... / percept..."
                        }
                    }
                    div {
                        span {
                            class: "text-black",
                            "Example: "
                        }
                        span {
                            class: "italic",
                            "person microwave food: open, start, stop / beep : heat. person food: stir"
                        },
                    }
                }
                // div {
                //     class: "text-sm text-gray-400 width-full",
                //     span {
                //         class: "text-black",
                //         "Crossing Number: "
                //     }
                //     span {
                //         class: "italic",
                //         crossing_number
                //     }
                // }
            }
        }
        div {
            class: "width-full",
            div {
                class: "relative mx-auto border border-black",
                width: "{viewbox_width}px",
                nodes
            }
        }
    })   
}

fn main() {
    wasm_logger::init(wasm_logger::Config::default());
    console_error_panic_hook::set_once();

    dioxus::web::launch_with_props(
        app, 
        AppProps { 
        }, 
        |c| c
    ); 
}


