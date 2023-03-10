use std::{borrow::Cow, path::Path, fs::File, io::{Write, stdout}};

use clap::Parser;
use depict::graph_drawing::{error::Error, frontend::dioxus::as_data_svg};

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
                output.write(r#"<div class="example">"#.as_bytes());
                output.write(r#"<div class=path">"#.as_bytes());
                output.write(entry.path().to_string_lossy().as_bytes());
                output.write(r#"/<div>"#.as_bytes());
                output.write(r#"<div class="data">"#.as_bytes());
                output.write(data.as_bytes());
                output.write("</div>".as_bytes());
            });
            do_one_expr(output, entry.path(), data)?;
            output.as_mut().map(|output| output.write("</div>".as_bytes()));
            output.as_mut().map(|output| output.write("</div>".as_bytes()));
            output.as_mut().map(|output| output.write("</div>".as_bytes()));
        }
    }
    Ok(())
}

fn do_one_expr<P: AsRef<Path> + Clone>(output: &mut Option<Box<dyn Write>>, _path: P, data: String) -> Result<(), Error> {
    let drawing = depict::graph_drawing::frontend::dom::draw(data)?;
    output.as_mut().map(|output| {
        let svg = as_data_svg(drawing);
        let svg = svg.splitn(2, ',').last().unwrap();
        output.write(svg.as_bytes());
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

    output.as_mut().map(|output| output.write(r#"
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <style>
        div.example { width: 100px; border: 1px dashed black; margin-left: auto; margin-right: auto; margin-bottom: 20px; }
        div.data { whitespace: pre; }
        </style>
        </head>
        "#.as_bytes()));
    for path in args.paths {
        do_one_path(&mut output, path)?;
    }

    for expr in args.exprs {
        do_one_expr(&mut output, String::new(), expr)?;
    }
    output.as_mut().map(|output| output.write("</html>".as_bytes()));
    Ok(())
}
