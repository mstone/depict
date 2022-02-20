use std::env::args;
use std::fs::read_to_string;
use std::io;

use dioxus::prelude::*;

use diagrams::parser::{parse, Ident};
use diagrams::graph_drawing::*;
use diagrams::render::filter_fact;
use petgraph::dot::Dot;

#[derive(Clone)]
pub enum Node {
  Div(String, f64, f64, f64),
  Svg(String),
}

fn draw() -> Vec<Node> {
    let path = args().skip(1).next().unwrap();
    let data = read_to_string(path).unwrap();

    let v = parse(&data[..]).unwrap().1;
    let mut ds = filter_fact(v.iter(), &Ident("draw"));
    let draw = ds.next().unwrap();

    let (vert, v_nodes, h_name) = calculate_vcg(&v, draw);

    eprintln!("VERT: {:?}", Dot::new(&vert));

    let (condensed, _) = condense(&vert);

    let roots = roots(&condensed);

    let paths_by_rank = rank(&condensed, &roots);

    let (locs_by_level, hops_by_level, hops_by_edge, loc_to_node, node_to_loc) = calculate_locs_and_hops(&condensed, &paths_by_rank);

    let solved_locs = minimize_edge_crossing(&locs_by_level, &hops_by_level);

    let (all_locs, all_hops0, all_hops, sol_by_loc, sol_by_hop) = calculate_sols(&solved_locs, &loc_to_node, &hops_by_level, &hops_by_edge);

    let (ls, rs, ss) = position_sols(&vert, &v_nodes, &hops_by_edge, &node_to_loc, &solved_locs, &all_locs, &all_hops0, &all_hops, &sol_by_loc, &sol_by_hop);

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
        let hpos = 1024.0 * lpos;
        let width = 1024.0 * (rpos - lpos);

        match node {
            Loc::Node(vl) => {
                let label = h_name[*vl].clone();
                texts.push(Node::Div(label, hpos, vpos, width));
            },
            _ => {},
        }
    }

    let mut arrows = vec![];

    for cer in condensed.edge_references() {
        for (vl, wl, _ew) in cer.weight().iter() {
            let hops = &hops_by_edge[&(*vl, *wl)];
            eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

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

            let svg_path = path.join(" ");
            arrows.push(Node::Svg(svg_path));
        }
    }

    let nodes = texts
        .into_iter()
        .chain(arrows.into_iter())
        .collect::<Vec<_>>();

    nodes
}

pub fn render(cx: Scope, nodes: Vec<Node>) -> Element {
    let mut children = vec![];

    for node in nodes {
        match node {
            Node::Div(label, hpos, vpos, width) => {
                children.push(rsx! {
                    div {
                        class: "absolute border border-blue text-center z-10 bg-white",
                        top: "{vpos}px",
                        left: "{hpos}px",
                        width: "{width}px",
                        span {
                            class: "border border-red",
                            "{label}"
                        }
                    }
                });
            },
            Node::Svg(svg_path) => {
                children.push(rsx!{
                    svg {
                        class: "absolute",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        stroke_width: "1",
                        view_box: "0 0 1024 768",
                        width: "1024px",
                        height: "768px",
                        path {
                            d: "{svg_path}",
                        }
                    }
                });
            },
        }
    }
    cx.render(rsx!(
        link { href:"https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css", rel:"stylesheet" }
        div {
            div {
                class: "relative border border-red",
                children
            }
        }
    ))
}

pub fn app(cx: Scope) -> Element {
    let nodes = use_future(&cx, || { async move {
        draw()
    }}).value();

    // let desktop = cx.consume_context::<dioxus_desktop::desktop_context::DesktopContext>().unwrap();
    // desktop.devtool();

    nodes.and_then(|nodes| render(cx, nodes.to_owned()))
}

pub fn main() -> io::Result<()> {
    dioxus::desktop::launch(app);

    // let mut vdom = VirtualDom::new(app);
    // let _ = vdom.rebuild();
    // let text = render_vdom(&vdom);
    // eprintln!("{text}");

    Ok(())
}
