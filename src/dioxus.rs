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
use inflector::Inflector;
use tracing::{instrument, event, Level};
use tracing_error::{InstrumentResult, ExtractSpanTrace};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Label {
    pub text: String,
    pub hpos: f64,
    pub vpos: f64,
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Node {
  Div { key: String, label: String, hpos: f64, vpos: f64, width: f64 },
  Svg { key: String, path: String, rel: String, label: Option<Label> },
}

const VIEWBOX_WIDTH: usize = 1024;

#[instrument(skip(data))]
fn draw(data: String) -> Result<(Option<usize>, Vec<Node>), Error> {
    let v = parse(&data[..])
        .map_err(|e| match e {
            nom::Err::Error(e) => { nom::Err::Error(nom::error::convert_error(&data[..], e)) },
            nom::Err::Failure(e) => { nom::Err::Failure(nom::error::convert_error(&data[..], e)) },
            nom::Err::Incomplete(n) => { nom::Err::Incomplete(n) },
        })
        .in_current_span()?
        .1;

    let Vcg{vert, vert_vxmap, vert_node_labels, vert_edge_labels} = calculate_vcg(&v)?;

    // diagrams::graph_drawing::draw(v, &mut vcg)?;


    // let draw_query = Fact::Atom(Ident("draw"));
    // let draw_cmd = diagrams::render::resolve(v.iter(), &draw_query).next().unwrap();
    // diagrams::graph_drawing::draw(&v, draw_cmd, &mut vcg)?;

    // eprintln!("VERT: {:?}", Dot::new(&vert));

    let Cvcg{condensed, condensed_vxmap: _} = condense(&vert)?;

    let roots = roots(&condensed)?;

    let paths_by_rank = rank(&condensed, &roots)?;

    let Placement{locs_by_level, hops_by_level, hops_by_edge, loc_to_node, node_to_loc} = calculate_locs_and_hops(&condensed, &paths_by_rank)?;

    let (crossing_number, solved_locs) = minimize_edge_crossing(&locs_by_level, &hops_by_level)?;

    let layout_problem = calculate_sols(&solved_locs, &loc_to_node, &hops_by_level, &hops_by_edge);

    let LayoutSolution{ls, rs, ss, ts} = position_sols(&vert, &vert_vxmap, &vert_edge_labels, &hops_by_edge, &node_to_loc, &solved_locs, &layout_problem)?;

    let LayoutProblem{sol_by_loc, sol_by_hop, ..} = layout_problem;

    let viewbox_width = VIEWBOX_WIDTH;
    let width_scale = 1.0;
    let height_scale = 80.0;
    let vpad = 50.0;
    let line_height = 20.0;

    let mut texts = vec![];

    for (loc, node) in loc_to_node.iter() {
        let (ovr, ohr) = loc;
        if (*ovr, *ohr) == (0, 0) { continue; }
        let n = sol_by_loc[&(*ovr, *ohr)];

        let lpos = ls[n];
        let rpos = rs[n];

        let vpos = height_scale * ((ovr-1) as f64) + vpad + ts[*ovr] * line_height;
        // let hpos = 1024.0 * ((lpos + rpos) / 2.0);
        let orig_width = (viewbox_width as f64) * (rpos - lpos);
        let width = orig_width * width_scale;
        // let hpos = (viewbox_width as f64) * lpos + (0.05 * orig_width);
        let hpos = (viewbox_width as f64) * lpos;
        // let hpos = 1024.0 * lpos + 0.1 * 1024.0 * (rpos - lpos);
        // let width = 1024.0 * 0.9 * (rpos - lpos);

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
                "actuates" | "actuator" => -0.0125,
                "senses" | "sensor" => 0.0125,
                _ => 0.0,
            };

            let mut path = vec![];
            let mut label_hpos = None;
            let mut label_vpos = None;
            // use rand::Rng;
            // let mut rng = rand::thread_rng();

            for (n, hop) in hops.iter().enumerate() {
                let (lvl, (_mhr, nhr)) = hop;
                let hn = sol_by_hop[&(lvl+1, *nhr, *vl, *wl)];
                let spos = ss[hn];
                let hpos = ((viewbox_width as f64) * (spos + offset)).round(); // + rng.gen_range(-0.1..0.1));
                let vpos = ((lvl-1) as f64) * height_scale + vpad + ts[*lvl] * line_height;
                let mut vpos2 = (*lvl as f64) * height_scale + vpad + ts[lvl+1] * line_height;

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
                    label_hpos = Some(hpos);
                    label_vpos = Some(vpos);
                }

                if n == hops.len() - 1 && *ew == "actuates" { 
                    vpos2 -= 7.0; // arrowhead length
                }

                path.push(format!("L {hpos} {vpos2}"));

            }

            let key = format!("{vl}_{wl}_{ew}_{m}");
            let path = path.join(" ");

            let mut label = None;

            if let (Some(label_text), Some(label_hpos), Some(label_vpos)) = (label_text, label_hpos, label_vpos) {
                label = Some(Label{text: label_text, hpos: label_hpos, vpos: label_vpos})
            }
            arrows.push(Node::Svg{key, path, rel: ew.to_string(), label});
        }
    }

    let nodes = texts
        .into_iter()
        .chain(arrows.into_iter())
        .collect::<Vec<_>>();

    event!(Level::TRACE, ?nodes, "NODES");

    Ok((Some(crossing_number), nodes))
}

pub fn render<P>(cx: Scope<P>, mut nodes: Vec<Node>)-> Option<VNode> {
    let viewbox_width = VIEWBOX_WIDTH;
    let viewbox_height = 768;
    let mut children = vec![];
    nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for node in nodes {
        match node {
            Node::Div{key, label, hpos, vpos, width} => {
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
            Node::Svg{key, path, rel, label} => {
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
                            Some(Label{text, hpos, vpos}) => {
                                let translate = if rel == "actuates" { 
                                    "translate(calc(-100% - 1.5ex))"
                                } else { 
                                    "translate(1.5ex)"
                                };
                                rsx!(div {
                                    class: "absolute",
                                    left: "{hpos}px",
                                    top: "calc({vpos}px + 40px)",
                                    div {
                                        class: "whitespace-pre bg-opacity-50 border-box text-sm",
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
    dbg!(cx.render(rsx!(children)))
}

const PLACEHOLDER: &str = indoc!("
driver wheel car: turn / wheel angle
driver accel car: accelerate / pedal position
driver brakes car: brake /  pedal position
driver screen computer: press screen / read display
computer thermostat car: set temperature / measure temperature
");

pub struct AppProps {
    model_sender: Option<UnboundedSender<String>>,
    #[allow(clippy::type_complexity)]
    drawing_receiver: Cell<Option<UnboundedReceiver<(Option<usize>, Vec<Node>)>>>,
}

pub fn app(cx: Scope<AppProps>) -> Element {
    let model = use_state(&cx, || String::from(PLACEHOLDER));
    let drawing = use_state(&cx, || (None, Vec::new()));

    use_coroutine(&cx, |_: UnboundedReceiver<()>| {
        let receiver = cx.props.drawing_receiver.take();
        let drawing = drawing.to_owned();
        async move {
            if let Some(mut receiver) = receiver {
                while let Some(msg) = receiver.next().await {
                    drawing.set(msg);
                }
            }
        }
    });

    // let desktop = cx.consume_context::<dioxus_desktop::desktop_context::DesktopContext>().unwrap();
    // desktop.devtool();
    // let window = use_window(&cx);
    // window.devtool();

    let nodes = render(cx, drawing.1.to_owned());
    let model_sender = cx.props.model_sender.clone().unwrap();
    model_sender.unbounded_send(model.get().clone()).unwrap();

    let viewbox_width = VIEWBOX_WIDTH;
    let crossing_number = cx.render(rsx!(match drawing.0 {
        Some(cn) => rsx!(span { "{cn}" }),
        None => rsx!(div{}),
    }));

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
                            model.set(e.value.clone());
                            model_sender.unbounded_send(e.value.clone()).unwrap(); 
                        },
                        "{model}"
                    }
                }
                div {
                    class: "text-sm text-gray-400 width-full grid grid-cols-2 gap-4",
                    div {
                        span {
                            class: "text-black",
                            "Syntax: "
                        }
                        span {
                            class: "italic",
                            "node+ : ^:^ ((^,^ action?) (/ (^,^ percept?))?)"
                        }
                    }
                    div {
                        span {
                            class: "text-black",
                            "Example: "
                        }
                        span {
                            class: "italic",
                            "driver brakes car: a / b : c / d, e / f"
                        },
                    }
                }
                div {
                    class: "text-sm text-gray-400 width-full",
                    span {
                        class: "text-black",
                        "Crossing Number: "
                    }
                    span {
                        class: "italic",
                        crossing_number
                    }
                }
            }
        }
        div {
            class: "width-full",
            div {
                class: "relative mx-auto",
                width: "{viewbox_width}px",
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
    let (drawing_sender, drawing_receiver) = futures_channel::mpsc::unbounded::<(Option<usize>, Vec<Node>)>();

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let mut prev_model: Option<String> = None;
                while let Some(model) = model_receiver.next().await {
                    if Some(&model) != prev_model.as_ref() {
                        let nodes = if model.trim().is_empty() {
                            Ok(Ok((None, vec![])))
                        } else {
                            catch_unwind(|| {
                                draw(model.clone())
                            })
                        };
                        let model = model.clone();
                        match nodes {
                            Ok(Ok((crossing_number, nodes))) => {
                                prev_model = Some(model);
                                drawing_sender.unbounded_send((crossing_number, nodes)).unwrap();
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
        |c| c.with_window(|c| {
            let mut menu_bar = tao::menu::MenuBar::new();
            let mut app_menu = tao::menu::MenuBar::new();
            let mut edit_menu = tao::menu::MenuBar::new();
            
            edit_menu.add_native_item(tao::menu::MenuItem::Undo);
            edit_menu.add_native_item(tao::menu::MenuItem::Redo);
            edit_menu.add_native_item(tao::menu::MenuItem::Separator);
            edit_menu.add_native_item(tao::menu::MenuItem::Cut);
            edit_menu.add_native_item(tao::menu::MenuItem::Copy);
            edit_menu.add_native_item(tao::menu::MenuItem::Paste);
            edit_menu.add_native_item(tao::menu::MenuItem::Separator);
            edit_menu.add_native_item(tao::menu::MenuItem::SelectAll);

            app_menu.add_native_item(tao::menu::MenuItem::CloseWindow);
            app_menu.add_native_item(tao::menu::MenuItem::Quit);
            menu_bar.add_submenu("STAMPEDE", true, app_menu);
            menu_bar.add_submenu("Edit", true, edit_menu);

            c
                .with_inner_size(LogicalSize::new(1200.0f64, 700.0f64))
                .with_menu(menu_bar)
        }));

    // let mut vdom = VirtualDom::new(app);
    // let _ = vdom.rebuild();
    // let text = render_vdom(&vdom);
    // eprintln!("{text}");

    Ok(())
}
