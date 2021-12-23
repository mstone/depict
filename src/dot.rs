use diagrams::parser::*;
use std::collections::HashMap;
use std::fs::read_to_string;
use std::env::args;
use std::io;
use std::process::exit;
use nom::error::convert_error;
use auto_enums::auto_enum;

type Syn<'a> = diagrams::parser::Syn::<&'a str>;
type Ident<'a> = diagrams::parser::Ident<&'a str>;
type Directive<'a> = diagrams::parser::Directive<Ident<'a>>;
type Fact<'a> = diagrams::parser::Fact<Ident<'a>>;

// pub fn filter_directives<'a, I: Iterator<Item = Syn<'a>>>(v: I) -> Vec<&'a Directive<'a>> {
//     v
//         .filter_map(|e| if let Syn::Directive(d) = e { Some(d) } else { None })
//         .collect()
// }

// pub fn filter_fact<'a>(v: &'a Vec<Syn>, i: &'a Ident) -> impl Iterator<Item = &'a Fact<'a>> {
// pub fn filter_fact<'a, I: Iterator<Item = &'a Syn<'a>>>(v: I, i: &'a Ident) -> impl Iterator<Item = &'a Fact<'a>> {
pub fn filter_fact<'a, I: Iterator<Item = Item>, Item: TryInto<&'a Fact<'a>, Error=E>, E>(v: I, q: &'a Ident) -> impl Iterator<Item = &'a Fact<'a>> {
    v
        .filter_map(move |e| match e.try_into() { Ok(Fact::Fact(ref i, f)) if q == i => Some(f), _ => None, })
        .flatten()
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

// pub fn resolve<'a>(v: &'a Vec<Syn>, r: &'a Fact<'a>) -> Vec<&'a Fact<'a>> {
#[auto_enum(Iterator)]
pub fn resolve<'a, I: Iterator<Item = Item>, Item: TryInto<&'a Fact<'a>, Error=E>, E>(v: I, r: &'a Fact<'a>) -> impl Iterator<Item = &'a Fact<'a>> {
    match r {
        Fact::Atom(i) => {
            return filter_fact(v, i);
        },
        Fact::Fact(_i, fs) => {
            return fs.iter();
        },
    }
}

pub fn unwrap_atom<'a>(a: &'a Fact<'a>) -> Option<&'a str> {
    match a {
        Fact::Atom(Ident(i)) => Some(*i),
        _ => None,
    }
}

pub fn to_ident<'a>(a: &'a Fact<'a>) -> &'a Ident<'a> {
    match a {
        Fact::Atom(i) => i,
        Fact::Fact(i, _fs) => i,
    }
}

pub fn first_ident<'a, I: Iterator<Item = &'a Fact<'a>>>(mut v: I) -> Option<&'a str> {
    v
        .find_map(unwrap_atom)
}

pub fn find_parent<'a, I: Iterator<Item = Item>, Item: PartialEq + TryInto<&'a Fact<'a>, Error=E>, E>(v: I, q1: &'a Ident, q2: &'a Ident) -> impl Iterator<Item = &'a Ident<'a>> {
    v
        .filter_map(move |item|
            match item.try_into() {
                Ok(Fact::Fact(i, fs)) => {
                    let mut candidates = filter_fact(fs.iter(), q1);
                    candidates.find(|c| match c {
                        Fact::Atom(a) if a == q2 => true,
                        _ => false,
                    }).map(|_| i)
                },
                _ => None,
            }
        )
}

pub fn render(v: Vec<Syn>) {
    // println!("ok\n\n");

    let ds = filter_fact(v.iter(), &Ident("draw"));
    // let ds2 = ds.collect::<Vec<&Fact>>();
    // println!("draw:\n{:#?}\n\n", ds2);

    for draw in ds {
        // println!("draw:\n{:#?}\n\n", draw);

        let res = resolve(v.iter(), draw);

        // println!("resolution: {:?}\n", res);

        println!("{}", "digraph {");
        println!("{}", "graph [splines=ortho, nodesep=1];");

        let action_query = Fact::Atom(Ident("action"));
        let percept_query = Fact::Atom(Ident("percept"));
        let name_query = Fact::Atom(Ident("name"));

        for hint in res {
            match hint {
                Fact::Fact(Ident("compact"), items) => {
                    let mut h_acts = HashMap::new();
                    let mut h_sens = HashMap::new();

                    for item in items {
                        let resolved_item = resolve(v.iter(), item).collect::<Vec<&Fact>>();
                        let resolved_name = resolve(resolved_item.iter().map(|v| *v), &name_query);
                        let name = first_ident(resolved_name).unwrap_or(
                            unwrap_atom(item).unwrap()
                        );
                        println!("{} [label=\"{}\",shape=rect];", unwrap_atom(item).unwrap(), name);

                        let resolved_actuates = find_parent(v.iter(), &Ident("actuates"), to_ident(item)).next();
                        let resolved_senses = find_parent(v.iter(), &Ident("senses"), to_ident(item)).next();
                        let resolved_hosts = find_parent(v.iter(), &Ident("hosts"), to_ident(item)).next();
                        let resolved_action = resolve(resolved_item.iter().map(|v| *v), &action_query);
                        let action = first_ident(resolved_action).unwrap_or(
                            unwrap_atom(item).unwrap()
                        );
                        let resolved_percept = resolve(resolved_item.iter().map(|v| *v), &percept_query);
                        let percept = first_ident(resolved_percept).unwrap_or(
                            unwrap_atom(item).unwrap()
                        );

                        h_acts.entry((resolved_actuates, resolved_hosts))
                            .or_insert(vec![])
                            .push((unwrap_atom(item).unwrap(), action));

                        h_sens.entry((resolved_senses, resolved_hosts))
                            .or_insert(vec![])
                            .push((unwrap_atom(item).unwrap(), percept));
                    }

                    for ((src, tgt), mediators) in h_acts.iter() {
                        if let (Some(s), Some(t)) = (src, tgt) {
                            for (med, action) in mediators {
                                println!("{} -> {} [label=\"{}\"];", s, med, action);
                                println!("{} -> {};", med, t);
                            }
                        }
                    }
                    for ((src, tgt), mediators) in h_sens.iter() {
                        if let (Some(s), Some(t)) = (src, tgt) {
                            for (med, percept) in mediators {
                                println!("{} -> {} [dir=back,label=\"{}\"];", s, med, percept);
                                println!("{} -> {} [dir=back];", med, t);
                            }
                        }
                    }
                },
                Fact::Fact(Ident("coalesce"), items) => {
                    let mut h_acts = HashMap::new();
                    let mut h_sens = HashMap::new();

                    for item in items {
                        let resolved_item = resolve(v.iter(), item).collect::<Vec<&Fact>>();

                        // let src_query = Fact::Atom(Ident(""))  // "x where q \in x.actuates"
                            // handle via unification? via a parser on facts??? via customs search routines?
                            // via some kind of relational hackery?
                        let resolved_actuates = find_parent(v.iter(), &Ident("actuates"), to_ident(item)).next();
                        let resolved_senses = find_parent(v.iter(), &Ident("senses"), to_ident(item)).next();
                        let resolved_hosts = find_parent(v.iter(), &Ident("hosts"), to_ident(item)).next();
                        // println!("{:?}", resolved_src);

                        let resolved_action = resolve(resolved_item.iter().map(|v| *v), &action_query);
                        let action = first_ident(resolved_action).unwrap_or(
                            unwrap_atom(item).unwrap()
                        );
                        let resolved_percept = resolve(resolved_item.iter().map(|v| *v), &percept_query);
                        let percept = first_ident(resolved_percept).unwrap_or(
                            unwrap_atom(item).unwrap()
                        );

                        h_acts.entry((resolved_actuates, resolved_hosts))
                            .or_insert(vec![])
                            .push(action);

                        h_sens.entry((resolved_senses, resolved_hosts))
                            .or_insert(vec![])
                            .push(percept);
                    }

                    for ((src, tgt), actions) in h_acts.iter() {
                        if let (Some(s), Some(t)) = (src, tgt) {
                            let rendered_actions = actions.as_slice().join("\\n");
                            println!("{} -> {} [label=\"{}\"];", s, t, rendered_actions);
                        }
                    }

                    for ((src, tgt), percepts) in h_sens.iter() {
                        if let (Some(s), Some(t)) = (src, tgt) {
                            let rendered_percepts = percepts.as_slice().join("\\n");
                            println!("{} -> {} [dir=back,label=\"{}\"];", t, s, rendered_percepts);
                        }
                    }
                },
                _ => {},
            }
        }

        println!("{}", "}");
    }
    // use top-level "draw" fact to identify inline or top-level drawings to draw
    // resolve top-level drawings + use inline drawings to identify objects to draw to make particular drawings
    // use object facts to figure out directions + labels?
    // print out dot repr?
    //   header
    //   render nodes
    //   render edges
    //   footer
    // let mut compact: &Vec<Ident> = &ds.find(|d| d == Ident("compact")).unwrap().1;
    // println!("COMPACT\n{:#?}", compact)

    // for id in compact {
    //     match resolve(&v, id) {

    //     }
    // }
}

pub fn main() -> io::Result<()> {
    for path in args().skip(1) {
        let contents = read_to_string(path)?;
        // println!("{}\n\n", &contents);
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