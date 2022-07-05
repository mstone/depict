#![feature(c_variadic)]

use std::{default::Default, panic::catch_unwind, io::BufWriter, borrow::Cow};

use depict::{graph_drawing::{
    error::{Kind, Error, OrErrExt},
    layout::{Loc}, 
    index::{LocSol, HopSol, VerticalRank, OriginalHorizontalRank}},
};

use dioxus::{prelude::*, core::to_owned};

use futures::StreamExt;
use indoc::indoc;

use tracing::{event, Level};

#[no_mangle]
unsafe extern "C" fn malloc(size: ::std::os::raw::c_ulong) -> *mut ::std::os::raw::c_void {
    use std::alloc::{alloc, Layout};
    let layout = Layout::from_size_align(size as usize + std::mem::size_of::<Layout>(), 16).unwrap();
    let ptr = alloc(layout);
    *(ptr as *mut Layout) = layout;
    (ptr as *mut Layout).offset(1) as *mut ::std::os::raw::c_void
}

#[no_mangle]
unsafe extern "C" fn calloc(count: ::std::os::raw::c_ulong, size: ::std::os::raw::c_ulong) -> *mut ::std::os::raw::c_void {
    use std::alloc::{alloc_zeroed, Layout};
    let layout = Layout::from_size_align((count * size) as usize + std::mem::size_of::<Layout>(), 16).unwrap();
    let ptr = alloc_zeroed(layout);
    *(ptr as *mut Layout) = layout;
    (ptr as *mut Layout).offset(1) as *mut ::std::os::raw::c_void
}

#[no_mangle]
unsafe extern "C" fn realloc(ptr: *mut ::std::os::raw::c_void, size: ::std::os::raw::c_ulong) -> *mut ::std::os::raw::c_void {
    use std::alloc::{realloc, Layout};
    let ptr = (ptr as *mut Layout).offset(-1);
    let layout = *ptr;
    let ptr = realloc(ptr as *mut u8, layout, size as usize + std::mem::size_of::<Layout>());
    *(ptr as *mut Layout) = Layout::from_size_align(size as usize + std::mem::size_of::<Layout>(), 16).unwrap();
    (ptr as *mut Layout).offset(1) as *mut ::std::os::raw::c_void
}

#[no_mangle]
unsafe extern "C" fn free(ptr: *mut ::std::os::raw::c_void) {
    use std::alloc::{dealloc, Layout};
    let ptr = (ptr as *mut Layout).offset(-1);
    let layout = *ptr;
    dealloc(ptr as *mut u8, layout);
}

#[no_mangle]
unsafe extern "C" fn printf(format: *const ::std::os::raw::c_char, mut args: ...) -> ::std::os::raw::c_int {
    // use std::ffi::CStr;
    // let c_str = unsafe { CStr::from_ptr(format_string) };
    // let c_str = c_str.to_string_lossy();
    // return c_str.len().try_into().unwrap();
    let mut s = String::new();
    #[cfg(target_family="wasm")]
    let format = format as *const u8;
    #[cfg(not(target_family="wasm"))]
    let format = format as *const i8;
    let bytes_written = printf_compat::format(
        format, 
        args.as_va_list(), 
        printf_compat::output::fmt_write(&mut s)
    );
    log::info!("{s}");
    bytes_written
}

fn now() -> i64 {
    let window = web_sys::window().expect("should have a window in this context");
    let performance = window
        .performance()
        .expect("performance should be available");
    performance.now() as i64
}

#[no_mangle]
unsafe extern "C" fn mach_absolute_time() -> ::std::os::raw::c_longlong {
    now()
}

use osqp_rust_sys::src::src::util::{mach_timebase_info_t, kern_return_t};

#[no_mangle]
unsafe extern "C" fn mach_timebase_info(info: mach_timebase_info_t) -> kern_return_t {
    let info = &mut *info;
    info.numer = 1; // wrong, but may work?
    info.denom = 1;
    0 // KERN_SUCCESS
}

#[no_mangle]
unsafe extern "C" fn dlopen(__path: *const ::std::os::raw::c_char, __mode: ::std::os::raw::c_int) -> *mut ::std::os::raw::c_void {
    todo!()
}

#[no_mangle]
unsafe extern "C" fn dlclose(__handle: *mut ::std::os::raw::c_void) -> ::std::os::raw::c_int {
    todo!()
}

#[no_mangle]
unsafe extern "C" fn dlerror() -> *mut ::std::os::raw::c_char {
    todo!()
}

#[no_mangle]
unsafe extern "C"     fn dlsym(
    __handle: *mut ::std::os::raw::c_void,
    __symbol: *const ::std::os::raw::c_char,
) -> *mut ::std::os::raw::c_void {
    todo!()
}

#[no_mangle]
unsafe extern "C" fn sqrt(x: ::std::os::raw::c_double) -> ::std::os::raw::c_double {
    x.sqrt()
}

use osqp_rust_sys::src::lin_sys::lib_handler::__darwin_ct_rune_t;

#[no_mangle]
unsafe extern "C" fn __tolower(_: __darwin_ct_rune_t) -> __darwin_ct_rune_t {
    todo!()
}

#[no_mangle]
unsafe extern "C" fn __toupper(_: __darwin_ct_rune_t) -> __darwin_ct_rune_t {
    todo!()
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

fn draw(data: String) -> Result<Drawing, Error> {
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
    let width_by_loc = &depiction.geometry_problem.width_by_loc;
    let width_by_hop = &depiction.geometry_problem.width_by_hop;
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
            let label = vert_node_labels
                .get(vl)
                .or_err(Kind::KeyNotFoundError{key: vl.to_string()})?
                .clone();
            // if !label.is_screaming_snake_case() {
            //     label = label.to_title_case();
            // }
            let estimated_width = width_by_loc[&(*ovr, *ohr)];
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
                    estimated_width0 = Some(width_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())]);
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

                if n < hops.len() - 1 {
                    vpos2 += 26.0;
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

use svg::{Document, node::{element::{Group, Marker, Path, Rectangle, Text as TextElt}, Node as _, Text}};

pub fn as_data_svg(drawing: Drawing) -> String {
    let viewbox_width = drawing.viewbox_width;
    let mut nodes = drawing.nodes;
    let viewbox_height = 768f64;

    let mut svg = Document::new()
        .set("viewBox", (0f64, 0f64, viewbox_width, viewbox_height))
        .set("text-rendering", "optimizeLegibility");

    svg.append(Marker::new()
        .set("id", "arrowhead")
        .set("markerWidth", 7)
        .set("markerHeight", 10)
        .set("refX", 0)
        .set("refY", 5)
        .set("orient", "auto")
        .set("viewBox", "0 0 10 10")
        .add(Path::new()
            .set("d", "M 0 0 L 10 5 L 0 10 z")
            .set("fill", "black")
        )
    );
    svg.append(Marker::new()
        .set("id", "arrowheadrev")
        .set("markerWidth", 7)
        .set("markerHeight", 10)
        .set("refX", 0)
        .set("refY", 5)
        .set("orient", "auto-start-reverse")
        .set("viewBox", "0 0 10 10")
        .add(Path::new()
            .set("d", "M 0 0 L 10 5 L 0 10 z")
            .set("fill", "black")
        )
    );

    nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for node in nodes {
        match node {
            Node::Div{label, hpos, vpos, width, ..} => {
                    svg.append(Group::new()
                        .set("transform", format!("translate({hpos}, {vpos})"))
                        .add(Rectangle::new()
                            .set("width", width)
                            .set("height", 26f64)
                            .set("stroke", "black")
                            .set("fill", "none"))
                        .add(TextElt::new()
                            .set("text-anchor", "middle")
                            .set("transform", format!("translate({}, {})", width / 2., 16.))
                            .set("font-family", "serif") // 'Times New Roman', Times, serif
                            .set("fill", "black")
                            .set("stroke", "none")
                            .add(Text::new(label)))
                    );
            },
            Node::Svg{path, rel, label, ..} => {
                
                let mut path_elt = Path::new()
                    .set("d", path)
                    .set("stroke", "black");
                
                match rel.as_str() {
                    "actuates" => path_elt = path_elt.set("marker-end", "url(%23arrowhead)"),
                    "senses" => path_elt = path_elt.set("marker-start", "url(%23arrowheadrev)"),
                    _ => {},
                };

                if let Some(Label{text, hpos, width: _, vpos, ..}) = label {
                    for (lineno, line) in text.lines().enumerate() {
                        let translate = if rel == "actuates" { 
                            format!("translate({}, {})", hpos-12., vpos + 56. + (20. * lineno as f64))
                        } else { 
                            format!("translate({}, {})", hpos+12., vpos + 56. + (20. * lineno as f64))
                        };
                        let anchor = if rel == "actuates" {
                            "end"
                        } else {
                            "start"
                        };
                        svg.append(Group::new()
                            .add(TextElt::new()
                                .set("font-family", "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",\"Courier New\",monospace")
                                .set("fill", "black")
                                .set("stroke", "none")
                                .set("transform", translate)
                                .set("text-anchor", anchor)
                                .add(Text::new(line))
                            )
                        );
                    }
                }

                svg.append(Group::new()
                    .add(path_elt)
                );
            },
        }
    }

    let mut buf = BufWriter::new(Vec::new());
    svg::write(&mut buf, &svg).unwrap();
    let bytes = buf.into_inner().unwrap();
    let svg_str = String::from_utf8(bytes).unwrap();
    format!("data:image/svg+xml;utf8,{svg_str}")
}

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
                        style: "position: absolute; padding-top: 3px; padding-bottom: 3px; box-sizing: border-box; border: 1px solid black; text-align: center; z-index: 10; background-color: #fff;", // bg-opacity-50
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
                let marker_id = if rel == "actuates" { "arrowhead" } else { "arrowheadrev" };
                let marker_orient = if rel == "actuates" { "auto" } else { "auto-start-reverse" };
                let stroke_dasharray = if rel == "fake" { "5 5" } else { "none" };
                let stroke_color = if rel == "fake" { "hsl(0, 0%, 50%)" } else { "currentColor" };
                children.push(cx.render(rsx!{
                    div {
                        key: "{key}",
                        style: "position: absolute;",
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
                                id: "{marker_id}",
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
                                            "marker-start": "url(#arrowheadrev)",
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
                                // let border = match rel.as_str() { 
                                //     // "actuates" => "border border-red-300",
                                //     // "senses" => "border border-blue-300",
                                //     _ => "",
                                // };
                                rsx!(div {
                                    style: "position: absolute;",
                                    left: "{hpos}px",
                                    // width: "{width}px",
                                    top: "calc({vpos}px + 40px)",
                                    div {
                                        style: "white-space: pre; z-index: 50; background-color: #fff; box-sizing: border-box; font-size: .875rem; line-height: 1.25rem; font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",\"Courier New\",monospace;",
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
    let drawing = use_state(&cx, || draw(PLACEHOLDER.into()).unwrap());
    
    let drawing_client = use_coroutine(&cx, |mut rx: UnboundedReceiver<String>| {
        to_owned![drawing];
        async move {
            while let Some(model) = rx.next().await {
                let nodes = if model.trim().is_empty() {
                    Ok(Ok(Drawing::default()))
                } else {
                    catch_unwind(|| {
                        draw(model.clone())
                    })
                };
                match nodes {
                    Ok(Ok(drawing_nodes)) => {
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

    let data_svg = as_data_svg(drawing.get().clone());

    cx.render(rsx!{
        div {
            // key: "editor",
            style: "width: 100%; z-index: 20; padding: 1rem;",
            div {
                style: "max-width: 36rem; margin-left: auto; margin-right: auto; flex-direction: column;",
                div {
                    // key: "editor_label",
                    "Model"
                }
                div {
                    // key: "editor_editor",
                    textarea {
                        style: "border-width: 1px; border-color: #000;",
                        rows: "6",
                        cols: "80",
                        autocomplete: "off",
                        // autocorrect: "off",
                        "autocapitalize": "off",
                        autofocus: "true",
                        spellcheck: "false",
                        // placeholder: "",
                        oninput: move |e| { 
                            event!(Level::TRACE, "INPUT");
                            drawing_client.send(e.value.clone());
                        },
                        "{model}"
                    }
                }
                div { 
                    style: "display: flex; flex-direction: row; justify-content: space-between;",
                    div {
                        style: "font-size: 0.875rem; line-height: 1.25rem; --tw-text-opacity: 1; color: rgba(156, 163, 175, var(--tw-text-opacity)); width: calc(100% - 8rem);",
                        div {
                            span {
                                style: "color: #000;",
                                "Syntax: "
                            }
                            span {
                                style: "font-style: italic;",
                                "node ... : action ... / feedback ... : action ... / feedback ..."
                            }
                        }
                        div {
                            span {
                                style: "color: #000;",
                                "Example: "
                            }
                            span {
                                style: "font-style: italic;",
                                "person microwave food: open, start, stop / beep : heat"
                            },
                        }
                    }
                    div {
                        style: "display: flex; flex-direction: column; align-items: end;",
                        a {
                            href: "{data_svg}",
                            download: "depict.svg",
                            "Export SVG"
                        }
                        span {
                            style: "font-style: italic; font-size: 0.875rem; line-height: 1.25rem; --tw-text-opacity: 1; color: rgba(156, 163, 175, var(--tw-text-opacity));",
                            "('Copy Link' to use)"
                        }
                    }
                }
                div {
                    details {
                        summary {
                            style: "font-size: 0.875rem; line-height: 1.25rem; --tw-text-opacity: 1; color: rgba(156, 163, 175, var(--tw-text-opacity)); text-align: right;",
                            "Licenses",
                        },
                        div {
                            (depict::licenses::LICENSES.dirs().map(|dir| {
                                let path = dir.path().display();
                                cx.render(rsx!{
                                    div {
                                        key: "{path}",
                                        span {
                                            style: "font-style: italic; text-decoration: underline;",
                                            "{path}"
                                        },
                                        ul {
                                            dir.files().map(|f| {
                                                let file_path = f.path();
                                                let file_contents = f.contents_utf8().unwrap();
                                                cx.render(rsx!{
                                                    details {
                                                        key: "{file_path:?}",
                                                        style: "white-space: pre;",
                                                        summary {
                                                            "{file_path:?}"
                                                        }
                                                        "{file_contents}"
                                                    }
                                                })
                                            })
                                        }
                                    }
                                })
                            }))
                        }
                    }
                }
                // div {
                //     style: "font-size: 0.875rem; line-height: 1.25rem; --tw-text-opacity: 1; color: rgba(156, 163, 175, var(--tw-text-opacity)); width: 100%;",
                //     span {
                //         style: "color: #000;",
                //         "Crossing Number: "
                //     }
                //     span {
                //         style: "font-style: italic;",
                //         crossing_number
                //     }
                // }
            }
        }
        div {
            style: "width: 100%;",
            div {
                style: "position: relative; margin-left: auto; margin-right: auto; border-width: 1px; border-color: #000;",
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


