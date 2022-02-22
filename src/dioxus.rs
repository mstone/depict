use std::cell::Cell;
use std::io;
use std::panic::catch_unwind;

use dioxus::core::exports::futures_channel;
use dioxus::prelude::*;

use diagrams::parser::{parse, Ident};
use diagrams::graph_drawing::*;
use diagrams::render::filter_fact;

use dioxus_desktop::tao::dpi::{LogicalSize};
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender};
use indoc::indoc;

use futures::StreamExt;

#[derive(Clone, PartialEq, PartialOrd)]
pub enum Node {
  Div { key: String, label: String, hpos: f64, vpos: f64, width: f64 },
  Svg { key: String, path: String },
}

fn draw(data: String) -> Result<Vec<Node>, ()> {
    let v = parse(&data[..]).map_err(|_| ())?.1;
    let mut ds = filter_fact(v.iter(), &Ident("draw"));
    let draw = ds.next().ok_or(())?;

    let Vcg{vert, v_nodes, h_name} = calculate_vcg(&v, draw);

    // eprintln!("VERT: {:?}", Dot::new(&vert));

    let Cvcg{condensed, condensed_vxmap: _} = condense(&vert);

    let roots = roots(&condensed);

    let paths_by_rank = rank(&condensed, &roots);

    let Placement{locs_by_level, hops_by_level, hops_by_edge, loc_to_node, node_to_loc} = calculate_locs_and_hops(&condensed, &paths_by_rank);

    let solved_locs = minimize_edge_crossing(&locs_by_level, &hops_by_level);

    let layout_problem = calculate_sols(&solved_locs, &loc_to_node, &hops_by_level, &hops_by_edge);

    let (ls, rs, ss) = position_sols(&vert, &v_nodes, &hops_by_edge, &node_to_loc, &solved_locs, &layout_problem);

    let LayoutProblem{sol_by_loc, sol_by_hop, ..} = layout_problem;

    // let width_scale = 0.9;
    let height_scale = 80.0;

    let mut texts = vec![];

    for (loc, node) in loc_to_node.iter() {
        let (ovr, ohr) = loc;
        let n = sol_by_loc[&(*ovr, *ohr)];

        let lpos = ls[n];
        let rpos = rs[n];

        let vpos = 80.0 * (*ovr as f64) + 100.0;
        // let hpos = 1024.0 * ((lpos + rpos) / 2.0);
        let hpos = 1024.0 * lpos + 0.1 * 1024.0 * (rpos - lpos);
        let width = 1024.0 * 0.9 * (rpos - lpos);

        if let Loc::Node(vl) = node {
            let key = String::from(*vl);
            let label = h_name[*vl].clone();
            texts.push(Node::Div{key, label, hpos, vpos, width});
        }
    }

    let mut arrows = vec![];

    for cer in condensed.edge_references() {
        for (vl, wl, ew) in cer.weight().iter() {
            let hops = &hops_by_edge[&(*vl, *wl)];
            // eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

            let mut path = vec![];

            for (n, hop) in hops.iter().enumerate() {
                let (lvl, (_mhr, nhr)) = hop;
                let hn = sol_by_hop[&(lvl+1, *nhr, *vl, *wl)];
                let spos = ss[hn];
                let hpos = 1024.0 * spos;
                let vpos = (*lvl as f64) * height_scale + 100.0;
                let vpos2 = ((*lvl + 1) as f64) * height_scale + 100.0;

                if n == 0 {
                    path.push(format!("M {hpos} {vpos}"));
                }

                path.push(format!("L {hpos} {vpos2}"));

            }

            let key = format!("{vl}_{wl}_{ew}");
            let path = path.join(" ");
            arrows.push(Node::Svg{key, path});
        }
    }

    let nodes = texts
        .into_iter()
        .chain(arrows.into_iter())
        .collect::<Vec<_>>();

    Ok(nodes)
}

pub fn render<P>(_cx: Scope<P>, mut nodes: Vec<Node>) -> LazyNodes {
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
                            view_box: "0 0 1024 768",
                            width: "1024px",
                            height: "768px",
                            path {
                                d: "{path}",
                            }
                        }
                    }
                });
            },
        }
    }
    rsx!(children)
}

const PLACEHOLDER: &str = indoc!("
driver: name: Driver controls: car actuates: wheel brakes accel screen senses: wheel brakes accel screen
car: name: Car hosts: wheel brakes accel thermostat
computer: name: Computer hosts: screen actuates: thermostat senses: thermostat
wheel: name: Wheel action: Turn $name percept: $name Angle
brakes: name: Brakes action: Brake percept: $action Position
accel: name: Accelerator action: Accelerate percept: Acceleration
screen: name: Touchscreen action: Click percept: Read UI
thermostat: name: Thermostat action: Set $name percept: Car Temperature
draw: drawing1: compact: driver car computer parallel: wheel brakes accel screen thermostat
");

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

    let nodes = render(cx, drawing.to_owned());
    let model_sender = cx.props.model_sender.clone().unwrap();
    model_sender.unbounded_send(model.clone()).unwrap();

    cx.render(rsx!{
        link { href:"https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css", rel:"stylesheet" }
        div {
            class: "width-full z-20 p-4",
            div {
                class: "max-w-3xl mx-auto flex flex-col",
                div {
                    class: "fs-24",
                    "Model"
                }
                div {
                    textarea {
                        class: "border",
                        rows: "10",
                        cols: "80",
                        // placeholder: "",
                        value: "{model}",
                        oninput: move |e| { 
                            set_model(e.value.clone()); 
                            model_sender.unbounded_send(e.value.clone()).unwrap(); 
                        }
                    }
                }
            }
        }
        div {
            class: "relative",
            nodes
        }
    })   
}

pub fn main() -> io::Result<()> {

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
                        if let Ok(Ok(nodes)) = nodes {
                            prev_model = Some(model);
                            drawing_sender.unbounded_send(nodes).unwrap();
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
            c.with_inner_size(LogicalSize::new(1200.0, 900.0))));

    // let mut vdom = VirtualDom::new(app);
    // let _ = vdom.rebuild();
    // let text = render_vdom(&vdom);
    // eprintln!("{text}");

    Ok(())
}
