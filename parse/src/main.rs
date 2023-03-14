use std::{borrow::Cow, path::Path, fs::File, io::{Write, stdout}};

use clap::Parser;
use depict::graph_drawing::{error::Error, frontend::{dioxus::{as_data_svg, render, default_css}, dom::Drawing}};

use dioxus::prelude::{*};
use thiserror::__private::PathAsDisplay;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(short = 'f')]
    paths: Vec<String>,

    #[clap(short = 'e')]
    exprs: Vec<String>,

    #[clap(short = 'o', default_value = "-")]
    output: String,
}

fn do_one_path<P: AsRef<Path> + Clone>(output: &mut Option<Box<dyn Write>>, path: P) -> Result<(), Error> {
    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let data = std::fs::read_to_string(entry.path())
                .expect("read error");

            output.as_mut().map(|output| {
                output.write(r#"<div class="file">"#.as_bytes());
                output.write(r#"<div class=path">"#.as_bytes());
                output.write(entry.path().to_string_lossy().as_bytes());
                output.write(r#"/<div>"#.as_bytes());
                output.write(r#"<div class="data">"#.as_bytes());
                output.write(data.as_bytes());
                output.write("</div>".as_bytes());
            });
            output.as_mut().map(|output| output.write(r#"<div class="drawing">"#.as_bytes()));
            do_one_expr(output, entry.path(), data)?;
            output.as_mut().map(|output| output.write("</div>".as_bytes()));
            output.as_mut().map(|output| output.write("</div>".as_bytes()));
            output.as_mut().map(|output| output.write("</div>".as_bytes()));
            output.as_mut().map(|output| output.write("</div>".as_bytes()));
        }
    }
    Ok(())
}

struct Props {
    drawing: Drawing,
}

fn do_one_expr<P: AsRef<Path> + Clone>(output: &mut Option<Box<dyn Write>>, _path: P, data: String) -> Result<(), Error> {
    let drawing = depict::graph_drawing::frontend::dom::draw(data);
    output.as_mut().map(|output| {
        if let Ok(drawing) = drawing {
            fn app(cx: Scope<Props>) -> Element {
                let drawing = render(cx, cx.props.drawing.clone());
                cx.render(rsx!{
                    div {
                        style: "position: relative;",
                        drawing
                    }
                })
            }
            let mut vdom = VirtualDom::new_with_props(app, Props{drawing});
            let _ = vdom.rebuild();
            output.write(dioxus_ssr::render(&vdom).as_bytes());
        } else {
            output.write(format!("<p>Error: {drawing:?}</p>").as_bytes());
        }
    });
    Ok(())
}

fn main() -> Result<(), Error> {
    miette::set_hook(Box::new(|_| {
        Box::new(
            miette::MietteHandlerOpts::new()
                // .terminal_links(true)
                // .unicode(false)
                .context_lines(2)
                // .tab_width(4)
                .build(),
        )
    })).unwrap();

    let args = Args::parse();


    let mut output = if args.output.is_empty() {
        None
    } else if &args.output == "-" {
        Some(Box::new(stdout()) as Box<dyn Write>)
    } else {
        Some(Box::new(File::create(args.output).unwrap()) as Box<dyn Write>)
    };

    output.as_mut().map(|output| {
        output.write(r#"
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <style>
        body {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(54pc, 1fr)); grid-gap: 2pc; justify-content: space-between;
            padding-left: 2pc;
            padding-right: 2pc;
            width: 100%;
        }
        div.file { max-width: calc(60pc - 40px); height: 40pc; border: 1px dashed black; margin-bottom: 20px; padding: 20px; }
        div.data { white-space: pre; overflow-x: scroll; max-width: 100%; }
        /* div.drawing { overflow: scroll; } */
        div.drawing { margin-top: 1em; }
        @media (prefers-color-scheme: dark) {
            div.file { border: 1px dashed #aaa; }
        }
        "#.as_bytes());
        output.write(default_css.as_bytes());
        output.write(r#"
        </style>
        </head>
        <body>
        "#.as_bytes())
    });
    for path in args.paths {
        do_one_path(&mut output, path)?;
    }

    for expr in args.exprs {
        do_one_expr(&mut output, String::new(), expr)?;
    }
    output.as_mut().map(|output| output.write("</body></html>".as_bytes()));
    Ok(())
}
