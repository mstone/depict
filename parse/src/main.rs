use std::env::args;

use diagrams::parser::*;

use logos::Logos;

use miette::{Diagnostic, NamedSource, Result};

// use tracing::{instrument, event, Level};
// use tracing_error::{InstrumentResult, ExtractSpanTrace, TracedError};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Diagnostic, thiserror::Error)]
#[error("parse error")]
#[diagnostic(code(diadym::parse_error))]
pub struct Error {
    #[source_code]
    pub src: NamedSource,

    #[label = "Unexpected token"]
    pub span: std::ops::Range<usize>,

    pub text: String,
}

fn do_one(path: String) -> Result<()> {
    let data = std::fs::read_to_string(&path)
        .expect("read error");

    println!("\npath: {path}");
    println!("data:\n\n{data}\n");

    { 
        let lex = Token::lexer(&data);
        let tks = lex.collect::<Vec<_>>();
        println!("lex: {tks:?}");
    }

    let mut lex = Token::lexer(&data);
    let mut p = Parser::new();

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
        
    println!("{v:#?}");
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

    for path in args().skip(1) {
        do_one(path)?;
    }
    Ok(())
}
