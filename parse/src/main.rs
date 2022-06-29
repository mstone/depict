use clap::Parser;
use depict::{parser::*, graph_drawing::layout::eval::eval};

use logos::Logos;

use miette::{Diagnostic, NamedSource, Result};

// use tracing::{instrument, event, Level};
// use tracing_error::{InstrumentResult, ExtractSpanTrace, TracedError};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Diagnostic, thiserror::Error)]
#[error("parse error")]
#[diagnostic(code(depict::parse_error))]
pub struct Error {
    #[source_code]
    pub src: NamedSource,

    #[label = "Unexpected token"]
    pub span: std::ops::Range<usize>,

    pub text: String,
}

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(short = 'f')]
    paths: Vec<String>,

    #[clap(short = 'e')]
    exprs: Vec<String>,
}


fn do_one_path(path: String) -> Result<()> {
    let data = std::fs::read_to_string(&path)
        .expect("read error");

    println!("\npath: {path}");
    do_one_expr(path, data)
}

fn do_one_expr(path: String, data: String) -> Result<()> {
    println!("data:\n\n{data}\n");
    { 
        let lex = Token::lexer(&data);
        let tks = lex.collect::<Vec<_>>();
        println!("lex: {tks:?}");
    }

    let mut lex = Token::lexer(&data);
    let mut p = depict::parser::Parser::new();

    while let Some(tk) = lex.next() {
        println!("token: {:?}", tk);
        p.parse(tk)
            .map_err(|_| { 
                Error{
                    src: NamedSource::new(path.clone(), data.clone()), 
                    span: lex.span(), 
                    text: lex.slice().into()
                }
            })?
    }
    let v = p.end_of_input()
        .map_err(|_| { 
            Error{
                src: NamedSource::new(path.clone(), data.clone()), 
                span: lex.span(), 
                text: lex.slice().into()
            }
        })?;
        
    println!("PARSE: {v:#?}");

    let ev = eval(&v[..]);
    println!("EVAL: {ev:#?}");

    Ok(())
}

fn main() -> Result<()> {
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
    }))?;

    let args = Args::parse();

    for path in args.paths {
        do_one_path(path)?;
    }

    for expr in args.exprs {
        do_one_expr(String::new(), expr)?;
    }
    Ok(())
}
