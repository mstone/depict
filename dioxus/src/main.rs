use std::borrow::Cow;
use std::cell::Cell;
use std::io::{self};
use std::panic::catch_unwind;

use depict::graph_drawing::error::{Error, OrErrExt, Kind};
use depict::graph_drawing::eval::{Val, Body};
use depict::graph_drawing::frontend::log::Record;
use depict::graph_drawing::index::{VerticalRank, OriginalHorizontalRank, LocSol, HopSol};
use depict::graph_drawing::frontend::dom::{draw, Drawing, Label, Node};
use depict::graph_drawing::frontend::dioxus::{render, as_data_svg};

use anyhow;

use dioxus::prelude::*;
use dioxus_desktop::{self, Config, WindowBuilder};

use futures::StreamExt;

// use dioxus_desktop::tao::dpi::{LogicalSize};
// use dioxus_desktop::use_window;
use tao::dpi::LogicalSize;

use color_spantrace::colorize;

use indoc::indoc;

use tracing::{instrument, event, Level};
use tracing_error::{ExtractSpanTrace};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const PLACEHOLDER: &str = indoc!("
k [ - s b ]
- c s
");

pub struct AppProps {
}

pub fn render_one<P>(cx: Scope<P>, record: Record) -> Result<VNode, anyhow::Error> {
    match record {
        Record::String{name, ty, names, val} => {
            let classes = names.iter().map(|n| format!("highlight_{n}")).collect::<Vec<_>>().join(" ");
            cx.render(rsx!{
                div {
                    key: "debug_{name}",
                    class: "{classes}",
                    div {
                        style: "font-weight: 700;",
                        "{name}"
                    }
                    div {
                        style: "white-space: pre; margin-left: 10px;",
                        "{val}"
                    }
                }
            })
        },
        _ => cx.render(rsx!{""}),
    }
}

fn render_many<P>(cx: Scope<P>, record: Record) -> Result<VNode, anyhow::Error> {
    match record {
        Record::String{..} => render_one(cx, record),
        Record::Group{name, ty, names, val} => {
            let classes = names.iter().map(|n| format!("highlight_{n}")).collect::<Vec<_>>().join(" ");
            eprintln!("LOG: record: {name} {ty:?} {names:#?}");
            cx.render(rsx!{
                div {
                    key: "debug_{name}",
                    class: "{classes}",
                    details {
                        style: "padding-left: 4px;",
                        open: "true",
                        summary {
                            style: "font-weight: 700;",
                            "{name}"
                        }
                        div {
                            style: "white-space: pre; margin-left: 10px; border-left: 1px gray solid;",
                            val.into_iter().map(|r| render_many(cx, r))
                        }
                    }
                }
            })
        },
    }
}

pub fn render_logs<P>(cx: Scope<P>, drawing: Drawing) -> Result<VNode, anyhow::Error> {
    let logs = drawing.logs;
    cx.render(rsx!{
        logs.into_iter().map(|r| render_many(cx, r))
    })
}

pub fn parse_highlights<'s>(data: &'s Cow<'s, str>) -> Result<Val<Cow<'s, str>>, Error> {
    use depict::parser::{Parser, Token};
    use depict::graph_drawing::eval::{eval, index, resolve};
    use logos::Logos;
    use std::collections::HashMap;
    use tracing_error::InstrumentResult;

    if data.trim().is_empty() {
        return Ok(Val::default())
    }
    let data = &data;

    let mut p = Parser::new();
    {
        let lex = Token::lexer(data);
        let tks = lex.collect::<Vec<_>>();
        event!(Level::TRACE, ?tks, "HIGHLIGHT LEX");
    }
    let mut lex = Token::lexer(data);
    while let Some(tk) = lex.next() {
        p.parse(tk)
            .map_err(|_| {
                Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
            })
            .in_current_span()?
    }

    let items = p.end_of_input()
        .map_err(|_| {
            Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
        })?;

    event!(Level::TRACE, ?items, "HIGHLIGHT PARSE");
    eprintln!("HIGHLIGHT PARSE {items:#?}");

    let mut val = eval(&items[..]);

    event!(Level::TRACE, ?val, "HIGHLIGHT EVAL");
    eprintln!("HIGHLIGHT EVAL {val:#?}");

    let mut scopes = HashMap::new();
    let val2 = val.clone();
    index(&val2, &mut vec![], &mut scopes);
    resolve(&mut val, &mut vec![], &scopes);

    eprintln!("HIGHLIGHT SCOPES: {scopes:#?}");
    eprintln!("HIGHLIGHT RESOLVE: {val:#?}");
    Ok(val)
}

pub fn render_highlight_styles<'s, 't, P>(cx: Scope<'t, P>, highlight_val: Val<Cow<'s, str>>) -> Result<VNode<'t>, anyhow::Error> {
    if let Val::Process { name, label, body: Some(Body::All(bs)) } = highlight_val {
        cx.render(rsx!{(
            bs.iter().map(|b| {
                match b {
                    Val::Process { name: pname, .. } => {
                        let style = ".box.highlight_{pname} { color: red; }";
                        rsx!{
                            style {
                                "{style}"
                            }
                        }
                    },
                    Val::Chain{ name: cname, .. } => {
                        let style = ".arrow.highlight_{cname} { color: red; }";
                        rsx!{
                            style {
                                "{style}"
                            }
                        }
                    }
                }
            })
        )})
    } else {
        cx.render(rsx!{()})
    }
}

pub fn app(cx: Scope<AppProps>) -> Element {
    let model = use_state(&cx, || String::from(PLACEHOLDER));
    let drawing = use_state(&cx, Drawing::default);
    let highlight = use_state(&cx, || String::new());

    let drawing_sender = use_coroutine(cx, |mut rx| { 
        let drawing = drawing.clone();
        async move {
            while let Some(msg) = rx.next().await {
                drawing.set(msg);
            }
        }
    });

    let model_sender = use_coroutine(cx, |mut rx| {
        let drawing_sender = drawing_sender.clone();
        async move {
            let mut prev_model: Option<String> = None;
            while let Some(model) = rx.next().await {
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
                            drawing_sender.send(drawing);
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
        }
    });

    // let desktop = cx.consume_context::<dioxus_desktop::desktop_context::DesktopContext>().unwrap();
    // desktop.devtool();
    // let window = use_window(&cx);
    // window.devtool();

    let nodes = render(cx, drawing.get().clone());
    let logs = render_logs(cx, drawing.get().clone());

    let mut show_logs = use_state(&cx, || false);

    model_sender.send(model.get().clone());

    let viewbox_width = drawing.get().viewbox_width;
    let _crossing_number = cx.render(rsx!(match drawing.get().crossing_number {
        Some(cn) => rsx!(span { "{cn}" }),
        None => rsx!(div{}),
    }));

    let data_svg = as_data_svg(drawing.get().clone());
    let keyword = "font-weight: bold; color: rgb(207, 34, 46);";
    let example = "font-size: 0.625rem; font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",\"Courier New\",monospace;";
    let box_highlight_s = ".box.highlight_s { background-color: blue; color: white; }";
    let highlight_s = ".highlight_s { color: blue; }";
    let highlight_0h = ".highlight_0h { color: red; }";

    // parse and eval the highlight string to get a sub-model to highlight
    let highlight_data = Cow::from(highlight.get());
    let highlight_val = parse_highlights(&highlight_data)?;

    let highlight_styles = render_highlight_styles(cx, highlight_val)?;

    cx.render(rsx!{
        highlight_styles
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
                            model_sender.send(e.value.clone());
                        },
                        "{model}"
                    }
                }
                div {
                    "Sub-model to Highlight"
                }
                div {
                    textarea {
                        style: "border-width 1px; border-color: #000;",
                        rows: "1",
                        cols: "80",
                        autocomplete: "off",
                        "autocapitalize": "off",
                        spellcheck: "false",
                        oninput: move |e| {
                            event!(Level::TRACE, "HIGHLIGHT INPUT");
                            highlight.set(e.value.clone());
                        }
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
                }
            }
        }
        div {
            style: "width: 100%;",
            div {
                style: "display: flex; gap: 20px;",
                div {
                    style: "position: relative; margin-left: auto; margin-right: auto; border-width: 1px; border-color: #000;",
                    width: "{viewbox_width}px",
                    nodes
                }
                div {
                    style: "display: flex; gap: 20px; overflow-x: auto;",
                    show_logs.then(|| rsx!{
                        logs
                    })
                }
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

    dioxus_desktop::launch_with_props(app,
        AppProps {},
        Config::new().with_window(
            WindowBuilder::new()
                .with_inner_size(LogicalSize::new(1200.0f64, 700.0f64))
                .with_menu(menu_bar)
        ));

    // let mut vdom = VirtualDom::new(app);
    // let _ = vdom.rebuild();
    // let text = render_vdom(&vdom);
    // eprintln!("{text}");

    Ok(())
}
