use diagrams::parser::*;
use std::fmt::Debug;
use std::fs::read_to_string;
use std::env::args;
use std::io;
use std::process::exit;
use nom::error::convert_error;

type Syn<'a> = diagrams::parser::Syn::<&'a str>;
type Ident<'a> = diagrams::parser::Ident<&'a str>;
type Directive<'a> = diagrams::parser::Directive<&'a str>;
   
pub fn filter_directives(v: Vec<Syn>) -> Vec<Directive> {
    v
        .iter()
        .filter_map(|ref e| if let Syn::Directive(d) = e { Some(d) } else { None })
        .cloned()
        .collect()
}

pub fn render(v: Vec<Syn>) {
    println!("ok\n\n");

    let ds = filter_directives(v);
    println!("directives:\n{:#?}\n\n", ds);
    // use directives to identify objects
    // use facts to figure out directions + labels?
    // print out dot repr?
    //   header
    //   render nodes
    //   render edges
    //   footer
    for d in ds {
        if d.0 == "compact" {
            println!("COMPACT\n{:#?}", d.1)
        }
    }

}

pub fn main() -> io::Result<()> {
    for path in args().skip(1) {
        let contents = read_to_string(path)?;
        println!("{}\n\n", &contents);
        let v = parse(&contents[..]);
        match v {
            Err(nom::Err::Error(v2)) => {
                println!("{}", convert_error(&contents[..], v2));
                exit(1);
            },
            Ok(("", v2)) => {
                render(v2);
            }
            _ => {
                println!("{:#?}", v);
                exit(2);
            }
        }
    }
    Ok(())
}