use std::borrow::Cow;
use std::cell::Cell;
use std::io::{self};
use std::panic::catch_unwind;

use depict::graph_drawing::error::{Error, OrErrExt, Kind};
use depict::graph_drawing::index::{VerticalRank, OriginalHorizontalRank, LocSol, HopSol};
use depict::graph_drawing::frontend::dom::{draw, Drawing, Label, Node, Log};
use depict::graph_drawing::frontend::dioxus::{render, as_data_svg};
use dioxus::core::exports::futures_channel;
use dioxus::prelude::*;

// use dioxus_desktop::tao::dpi::{LogicalSize};
// use dioxus_desktop::use_window;
use tao::dpi::LogicalSize;

use color_spantrace::colorize;

use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender};
use futures::StreamExt;

use indoc::indoc;

use tracing::{instrument, event, Level};
use tracing_error::{ExtractSpanTrace};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const PLACEHOLDER: &str = indoc!("
k [ - s b ]
- c s
");
// c: a b [ z ]
// c: a b, [ z ]
// c: a b [ ]
// c: a b, [ ]
// c: a, b [ z ]

/*
a [ b t: / qqqq, qqqq, qqqq, qqqq ; - t u: foo, foo, foo, foo ]
*/
/*
a [ b [ - c e ] ]
z c
*/
/*
a [ b c ] 
*/
/*
a [  - b c: z ] 
d b
*/
/*
a [ b c ]
a d
*/
/*
p a > : foo / bar
pn: p
q pn
q a
*/
/* 
p q
r q >: sdf / zlk
r t
*/
/*
a [ b ]
a c
");
*/
/*
a [ b [ c ] ]
a d
p q r
");
*/

// person microwave food: open, start, stop / beep : heat
// person food: stir
// LEFT test person: aaaaaaaaaa / bbbbbbb
// ");
// driver wheel car: turn / wheel angle
// driver accel car: accelerate / pedal position
// driver brakes car: brake /  pedal position
// driver screen computer: press screen / read display
// computer thermostat car: set temperature / measure temperature
// ");

pub struct AppProps {
    model_sender: Option<UnboundedSender<String>>,
    #[allow(clippy::type_complexity)]
    drawing_receiver: Cell<Option<UnboundedReceiver<Drawing>>>,
}

pub fn render_logs<P>(cx: Scope<P>, drawing: Drawing) -> Option<VNode> {
    let logs = drawing.logs;
    cx.render(rsx!{
        logs.iter().map(|m| match m {
            Log::String(s) => rsx!{div { "{s}" }},
        })
    })
}

pub fn app(cx: Scope<AppProps>) -> Element {
    let model = use_state(&cx, || String::from(PLACEHOLDER));
    let drawing = use_state(&cx, Drawing::default);

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

    let nodes = render(cx, drawing.get().clone());
    let logs = render_logs(cx, drawing.get().clone());

    let mut show_logs = use_state(&cx, || false);

    let model_sender = cx.props.model_sender.clone().unwrap();
    model_sender.unbounded_send(model.get().clone()).unwrap();

    let viewbox_width = drawing.get().viewbox_width;
    let _crossing_number = cx.render(rsx!(match drawing.get().crossing_number {
        Some(cn) => rsx!(span { "{cn}" }),
        None => rsx!(div{}),
    }));

    let data_svg = as_data_svg(drawing.get().clone());
    let keyword = "font-weight: bold; color: rgb(207, 34, 46);";
    let example = "font-size: 0.625rem; font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",\"Courier New\",monospace;";

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
                        // autocapitalize: "off",
                        "autocapitalize": "off",
                        autofocus: "true",
                        spellcheck: "false",
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
                    style: "display: flex; flex-direction: row; justify-content: space-between;",
                    div {
                        style: "font-size: 0.875rem; line-height: 1.25rem; width: calc(100% - 8rem);",
                        div {
                            details {
                                // open: "true",
                                summary {
                                    span {
                                        style: "color: #000;",
                                        "Syntax + Examples"
                                    }
                                }
                                div {
                                    p {
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "nodes"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            ":"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "actions"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            "/"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "feedback"
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            style: "{example}",
                                            "person microwave food: open start stop / beep : heat"
                                        },
                                    }
                                    p {
                                        span {
                                            style: "{keyword}",
                                            "-"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "nodes"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            ":"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "flows"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            "/"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "responses"
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            style: "{example}",
                                            "- left right: input / reply"
                                        },
                                    }
                                    p {
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "container"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            "["
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "components"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            "]"
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            style: "{example}",
                                            "plane [ pilot navigator ]"
                                        },
                                    }
                                    p {
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "abbreviation"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            ":"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "..."
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            style: "{example}",
                                            "c: controller; p: process; c p: setpoint / feedback"
                                        },
                                    }
                                    p {
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "nodes"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            ":"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "long-label"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            ","
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "labels"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            "/"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "long-label"
                                        }
                                        " "
                                        span {
                                            style: "{keyword}",
                                            ","
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "labels"
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            style: "{example}",
                                            "controller process: a long action, / a long feedback, another feedback"
                                        },
                                    }
                                }
                            }
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
                div {
                    button {
                        onclick: move |_| show_logs.modify(|v| !v),
                        "Show debug logs"
                    }
                    show_logs.then(|| rsx!{
                        logs
                    })
                }
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

pub fn main() -> io::Result<()> {    
    tracing_subscriber::Registry::default()
        .with(tracing_error::ErrorLayer::default())
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let (model_sender, mut model_receiver) = futures_channel::mpsc::unbounded::<String>();
    let (drawing_sender, drawing_receiver) = futures_channel::mpsc::unbounded::<Drawing>();

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let mut prev_model: Option<String> = None;
                while let Some(model) = model_receiver.next().await {
                    if Some(&model) != prev_model.as_ref() {
                        let model_str: &str = &model;
                        let nodes = if model_str.trim().is_empty() {
                            Ok(Ok(Drawing::default()))
                        } else {
                            catch_unwind(|| {
                                draw(model.clone())
                            })
                        };
                        let model = model.clone();
                        match nodes {
                            Ok(Ok(drawing)) => {
                                prev_model = Some(model);
                                drawing_sender.unbounded_send(drawing).unwrap();
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
            menu_bar.add_submenu("Depict", true, app_menu);
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
