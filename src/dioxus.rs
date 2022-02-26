use std::cell::Cell;
use std::io;
use std::panic::catch_unwind;

use dioxus::core::exports::futures_channel;
use dioxus::prelude::*;

use dioxus_desktop::tao::dpi::{LogicalSize};

use diagrams::parser::{parse};
use diagrams::graph_drawing::*;

use color_spantrace::colorize;
// use dioxus_desktop::use_window;
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender};
use futures::StreamExt;
use indoc::indoc;
use tracing::{instrument, event, Level};
use tracing_error::{InstrumentResult, ExtractSpanTrace};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Node {
  Div { key: String, label: String, hpos: f64, vpos: f64, width: f64 },
  Svg { key: String, path: String },
}

#[instrument(skip(data))]
fn draw(data: String) -> Result<Vec<Node>, Error> {
    let v = parse(&data[..])
        .map_err(|e| match e {
            nom::Err::Error(e) => { nom::Err::Error(nom::error::convert_error(&data[..], e)) },
            nom::Err::Failure(e) => { nom::Err::Failure(nom::error::convert_error(&data[..], e)) },
            nom::Err::Incomplete(n) => { nom::Err::Incomplete(n) },
        })
        .in_current_span()?
        .1;

    let Vcg{vert, v_nodes, h_name} = calculate_vcg(&v)?;

    // diagrams::graph_drawing::draw(v, &mut vcg)?;


    // let draw_query = Fact::Atom(Ident("draw"));
    // let draw_cmd = diagrams::render::resolve(v.iter(), &draw_query).next().unwrap();
    // diagrams::graph_drawing::draw(&v, draw_cmd, &mut vcg)?;

    // eprintln!("VERT: {:?}", Dot::new(&vert));

    let Cvcg{condensed, condensed_vxmap: _} = condense(&vert)?;

    let roots = roots(&condensed)?;

    let paths_by_rank = rank(&condensed, &roots)?;

    let Placement{locs_by_level, hops_by_level, hops_by_edge, loc_to_node, node_to_loc} = calculate_locs_and_hops(&condensed, &paths_by_rank)?;

    let solved_locs = minimize_edge_crossing(&locs_by_level, &hops_by_level)?;

    let layout_problem = calculate_sols(&solved_locs, &loc_to_node, &hops_by_level, &hops_by_edge);

    let LayoutSolution{ls, rs, ss} = position_sols(&vert, &v_nodes, &hops_by_edge, &node_to_loc, &solved_locs, &layout_problem)?;

    let LayoutProblem{sol_by_loc, sol_by_hop, ..} = layout_problem;

    let viewbox_width = 1024;
    let width_scale = 1.0;
    let height_scale = 80.0;
    let vpad = 50.0;

    let mut texts = vec![];

    // use std::collections::hash_map::DefaultHasher;
    // use std::hash::{Hasher, Hash};
    // let mut hasher = DefaultHasher::new();
    // data.hash(&mut hasher);
    // let data_hash = hasher.finish();

    for (loc, node) in loc_to_node.iter() {
        let (ovr, ohr) = loc;
        if (*ovr, *ohr) == (0, 0) { continue; }
        let n = sol_by_loc[&(*ovr, *ohr)];

        let lpos = ls[n];
        let rpos = rs[n];

        let vpos = height_scale * ((ovr-1) as f64) + vpad;
        // let hpos = 1024.0 * ((lpos + rpos) / 2.0);
        let orig_width = (viewbox_width as f64) * (rpos - lpos);
        let width = orig_width * width_scale;
        let hpos = (viewbox_width as f64) * lpos + (0.05 * orig_width);
        // let hpos = 1024.0 * lpos + 0.1 * 1024.0 * (rpos - lpos);
        // let width = 1024.0 * 0.9 * (rpos - lpos);

        if let Loc::Node(vl) = node {
            let key = vl.to_string();
            let label = h_name
                .get(*vl)
                .or_err(Kind::KeyNotFoundError{key: vl.to_string()})?
                .clone();
            texts.push(Node::Div{key, label, hpos, vpos, width});
        }
    }

    let mut arrows = vec![];

    for cer in condensed.edge_references() {
        for (m, (vl, wl, ew)) in cer.weight().iter().enumerate() {
            if *vl == "root" { continue; }
            let hops = &hops_by_edge[&(*vl, *wl)];
            // eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

            let offset = match *ew { 
                "actuates" | "actuator" => 0.0125,
                "senses" | "sensor" => -0.0125,
                _ => 0.0,
            };

            let mut path = vec![];

            for (n, hop) in hops.iter().enumerate() {
                let (lvl, (_mhr, nhr)) = hop;
                let hn = sol_by_hop[&(lvl+1, *nhr, *vl, *wl)];
                let spos = ss[hn];
                let hpos = (viewbox_width as f64) * (spos + offset);
                let vpos = ((lvl-1) as f64) * height_scale + vpad;
                let vpos2 = (*lvl as f64) * height_scale + vpad;

                if n == 0 {
                    path.push(format!("M {hpos} {vpos}"));
                }

                path.push(format!("L {hpos} {vpos2}"));

            }

            let key = format!("{vl}_{wl}_{ew}_{m}");
            let path = path.join(" ");
            arrows.push(Node::Svg{key, path});
        }
    }

    let nodes = texts
        .into_iter()
        .chain(arrows.into_iter())
        .collect::<Vec<_>>();

    event!(Level::TRACE, ?nodes, "NODES");

    Ok(nodes)
}

pub fn render<P>(cx: Scope<P>, mut nodes: Vec<Node>) -> Option<VNode> {
    let viewbox_width = 1024;
    let viewbox_height = 768;
    let mut children = vec![];
    nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for node in nodes {
        match node {
            Node::Div{key, label, hpos, vpos, width} => {
                children.push(rsx! {
                    div {
                        key: "{key}",
                        class: "absolute border border-black text-center z-10 bg-white",
                        top: "{vpos}px",
                        left: "{hpos}px",
                        width: "{width}px",
                        span {
                            "{label}"
                        }
                    }
                });
            },
            Node::Svg{key, path} => {
                children.push(rsx!{
                    div {
                        key: "{key}",
                        class: "absolute",
                        svg {
                            fill: "none",
                            stroke: "currentColor",
                            stroke_linecap: "round",
                            stroke_linejoin: "round",
                            stroke_width: "1",
                            view_box: "0 0 {viewbox_width} {viewbox_height}",
                            width: "{viewbox_width}px",
                            height: "{viewbox_height}px",
                            path {
                                d: "{path}",
                            }
                        }
                    }
                });
            },
        }
    }
    cx.render(rsx!(children))
}

const PLACEHOLDER: &str = indoc!("
driver wheel car: turn / wheel angle
");

// driver accel car: accelerate / accelerator pedal position
// driver brakes car: brake / brake pedal position
// driver screen computer: press screen / read display
// computer thermostat car: set temperature / measure temperature

pub struct AppProps {
    model_sender: Option<UnboundedSender<String>>,
    drawing_receiver: Cell<Option<UnboundedReceiver<Vec<Node>>>>,
}

pub fn app(cx: Scope<AppProps>) -> Element {
    let (model, set_model) = use_state(&cx, || String::from(PLACEHOLDER));
    let (drawing, set_drawing) = use_state(&cx, Vec::new);

    use_coroutine(&cx, || {
        let receiver = cx.props.drawing_receiver.take();
        let set_drawing = set_drawing.to_owned();
        async move {
            if let Some(mut receiver) = receiver {
                while let Some(msg) = receiver.next().await {
                    set_drawing(msg);
                }
            }
        }
    });

    // let desktop = cx.consume_context::<dioxus_desktop::desktop_context::DesktopContext>().unwrap();
    // desktop.devtool();
    // let window = use_window(&cx);
    // window.devtool();

    let nodes = render(cx, drawing.to_owned());
    let model_sender = cx.props.model_sender.clone().unwrap();
    model_sender.unbounded_send(model.clone()).unwrap();

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
                            set_model(e.value.clone());
                            model_sender.unbounded_send(e.value.clone()).unwrap(); 
                        },
                        "{model}"
                    }
                }
            }
        }
        div {
            class: "width-full",
            div {
                class: "relative mx-auto",
                width: "1024px",
                nodes
            }
        }
    })   
}

pub fn main() -> io::Result<()> {    
    tracing_subscriber::Registry::default()
        .with(tracing_error::ErrorLayer::default())
        .with(tracing_subscriber::fmt::layer())
        .init();

    let (model_sender, mut model_receiver) = futures_channel::mpsc::unbounded::<String>();
    let (drawing_sender, drawing_receiver) = futures_channel::mpsc::unbounded::<Vec<Node>>();

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let mut prev_model: Option<String> = None;
                while let Some(model) = model_receiver.next().await {
                    if Some(&model) != prev_model.as_ref() {
                        let nodes = catch_unwind(|| {
                            draw(model.clone())
                        });
                        let model = model.clone();
                        match nodes {
                            Ok(Ok(nodes)) => {
                                prev_model = Some(model);
                                drawing_sender.unbounded_send(nodes).unwrap();
                            },
                            Ok(Err(err)) => {
                                if let Some(st) = err.span_trace() {
                                    let st_col = colorize(st);
                                    event!(Level::ERROR, ?err, %st_col, "DRAWING ERROR SPANTRACE");
                                } else {
                                    event!(Level::ERROR, ?err, "DRAWING ERROR");
                                }
                            }
                            Err(_) => {
                                event!(Level::ERROR, ?nodes, "PANIC");
                            }
                        }
                    }
                }
            });
    });
    
    dioxus::desktop::launch_with_props(app, 
        AppProps { 
            model_sender: Some(model_sender), 
            drawing_receiver: Cell::new(Some(drawing_receiver)) 
        },
        |c| c.with_window(|c| 
            c.with_inner_size(LogicalSize::new(1200.0, 700.0))));

    // let mut vdom = VirtualDom::new(app);
    // let _ = vdom.rebuild();
    // let text = render_vdom(&vdom);
    // eprintln!("{text}");

    Ok(())
}
