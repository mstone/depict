use std::borrow::Cow;

use clap::Parser;
use depict::graph_drawing::{error::Error};

// use tracing::{instrument, event, Level};
// use tracing_error::{InstrumentResult, ExtractSpanTrace, TracedError};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(short = 'f')]
    paths: Vec<String>,

    #[clap(short = 'e')]
    exprs: Vec<String>,
}


fn do_one_path(path: String) -> Result<(), Error> {
    let data = std::fs::read_to_string(&path)
        .expect("read error");

    println!("\npath: {path}");
    do_one_expr(path, data)
}

fn do_one_expr(_path: String, data: String) -> Result<(), Error> {
    depict::graph_drawing::frontend::render(Cow::Owned(data))
        .map(|_| ())
}

fn main() -> Result<(), Error> {
    tracing_subscriber::Registry::default()
        .with(tracing_error::ErrorLayer::default())
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

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

    for path in args.paths {
        do_one_path(path)?;
    }

    for expr in args.exprs {
        do_one_expr(String::new(), expr)?;
    }
    Ok(())
}
