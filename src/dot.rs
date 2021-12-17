use diagrams::parser::*;
use std::fs::read_to_string;
use std::env::args;
use std::io;
use std::process::exit;
use nom::error::convert_error;

type Syn<'a> = diagrams::parser::Syn::<&'a str>;
type Ident<'a> = diagrams::parser::Ident<&'a str>;
type Directive<'a> = diagrams::parser::Directive<Ident<'a>>;
type Fact<'a> = diagrams::parser::Fact<Ident<'a>>;
   
pub fn filter_directives<'a>(v: &'a Vec<Syn>) -> Vec<&'a Directive<'a>> {
    v
        .iter()
        .filter_map(|ref e| if let Syn::Directive(ref d) = e { Some(d) } else { None })
        .collect()
}

pub fn filter_draw<'a>(v: &'a Vec<Syn>) -> Vec<&'a Fact<'a>> {
    v
        .iter()
        .filter_map(|ref e| if let Syn::Fact(ref f) = e { Some(f) } else { None })
        .collect()
}

pub struct Process<I> {
    name: I,
    controls: Vec<Path<I>>,
    senses: Vec<Path<I>>,
}

pub struct Path<I> {
    name: I,
    action: I,
    percept: I,
}

pub struct Draw<I> {
    name: I,
}

pub struct Drawing<I> {
    names: Vec<I>,
}

pub enum Item<I> {
    Process(Process<I>),
    Path(Path<I>),
    Draw(Draw<I>),
    Drawing(Drawing<I>),
}

// pub fn resolve<'a>(v: &'a Vec<Syn>) -> ??? {

// }

pub fn render(v: Vec<Syn>) {
    println!("ok\n\n");

    let ds = filter_directives(&v);
    println!("directives:\n{:#?}\n\n", ds);
    // use directives to identify objects
    // use facts to figure out directions + labels?
    // print out dot repr?
    //   header
    //   render nodes
    //   render edges
    //   footer
    let mut compact: &Vec<Ident> = &ds.iter().find(|d| d.0 == Ident("compact")).unwrap().1;
    println!("COMPACT\n{:#?}", compact)

    // for id in compact {
    //     match resolve(&v, id) {

    //     }
    // }
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