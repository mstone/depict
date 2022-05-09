use std::default::Default;

use depict::rest::*;

use dioxus::{prelude::*, core::to_owned};

use futures::StreamExt;
use indoc::indoc;

use reqwasm::http::{Request, Response};

use tracing::{event, Level};

async fn click(s: String) -> Response {
    let draw_req = serde_json::to_string(&Draw{text: s}).unwrap();
    Request::post("/api/draw/v1")
        .header("Content-Type", "application/json")
        .body(draw_req)
        .send()
        .await
        .unwrap()
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

    let drawing = use_state(&cx, || serde_json::from_str::<DrawResp>(PLACEHOLDER_DRAWING).unwrap().drawing);

    let drawing_client = use_coroutine(&cx, |mut rx: UnboundedReceiver<String>| {
        to_owned![drawing];
        async move {
            while let Some(model) = rx.next().await {
                let res: Result<DrawResp, _> = click(model).await.json().await;
                if let Ok(drawing_resp) = res {
                    drawing.set(drawing_resp.drawing);
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
