#![feature(c_variadic)]

use std::{default::Default, panic::catch_unwind, io::BufWriter, borrow::Cow};

use depict::{graph_drawing::{
    error::{Kind, Error, OrErrExt},
    layout::{Loc}, 
    index::{LocSol, HopSol, VerticalRank, OriginalHorizontalRank},
    frontend::dom::{draw, Drawing, Label, Node},
}};

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
            Node::Div{label, hpos, vpos, width, height, ..} => {
                    svg.append(Group::new()
                        .set("transform", format!("translate({hpos}, {vpos})"))
                        .add(Rectangle::new()
                            .set("width", width)
                            .set("height", height)
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
            Node::Div{key, label, hpos, vpos, width, height, z_index, ..} => {
                children.push(cx.render(rsx! {
                    div {
                        key: "{key}",
                        style: "position: absolute; padding-top: 3px; padding-bottom: 3px; box-sizing: border-box; border: 1px solid black; text-align: center; z-index: 10; background-color: #fff;", // bg-opacity-50
                        top: "{vpos}px",
                        left: "{hpos}px",
                        width: "{width}px",
                        height: "{height}px",
                        z_index: "{z_index}",
                        span {
                            "{label}"
                        }
                    }
                }));
            },
            Node::Svg{key, path, z_index, rel, label, ..} => {
                let marker_id = if rel == "actuates" || rel == "forward" { "arrowhead" } else { "arrowheadrev" };
                let marker_orient = if rel == "actuates" || rel == "forward" { "auto" } else { "auto-start-reverse" };
                let stroke_dasharray = if rel == "fake" { "5 5" } else { "none" };
                let stroke_color = if rel == "fake" { "hsl(0, 0%, 50%)" } else { "currentColor" };
                children.push(cx.render(rsx!{
                    div {
                        key: "{key}",
                        style: "position: absolute;",
                        z_index: "{z_index}",
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
                                    "actuates" | "forward" => {
                                        rsx!(path {
                                            d: "{path}",
                                            marker_end: "url(#arrowhead)",
                                        })
                                    },
                                    "senses" | "reverse" => {
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
                                let translate = match &rel[..] {
                                    "actuates" | "forward" => "translate(calc(-100% - 1.5ex))",
                                    "senses" | "reverse" => "translate(1.5ex)",
                                    _ => "translate(0px, 0px)",
                                };
                                let offset = match &rel[..] {
                                    "actuates" | "senses" => "40px",
                                    "forward" => "-24px",
                                    "reverse" => "4px",
                                    _ => "0px",
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
                                    top: "calc({vpos}px + {offset})",
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


