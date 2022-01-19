use diagrams::parser::{Ident, parse};
use diagrams::render::{Fact, Syn, filter_fact, resolve, unwrap_atom, find_parent, to_ident, as_string};
use std::collections::HashMap;
use std::collections::hash_map::{Entry};
use std::fs::read_to_string;
use std::env::args;
use std::io;
use std::process::exit;
use nom::error::convert_error;
use indoc::indoc;
use petgraph::graph::{Graph, DefaultIx, NodeIndex};
use petgraph::dot::{Dot};

pub fn tikz_escape(s: &str) -> String {
    s
        .replace("$", "\\$")
        .replace("\\n", "\\\\")
}

pub fn or_insert<'a, 'g, 'h>(g: &'g mut Graph<&'a str, &'a str>, h: &'h mut HashMap<&'a str, NodeIndex<DefaultIx>>, v: &'a str) -> NodeIndex<DefaultIx> {
    let e = h.entry(v);
    let ix = match e {
        Entry::Vacant(ve) => ve.insert(g.add_node(v.clone())),
        Entry::Occupied(ref oe) => oe.get(),
    };
    println!("OR_INSERT {} -> {:?}", v, ix);
    return *ix;
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

        println!("{}", indoc!(r#"
            \documentclass[tikz,border=5mm]{standalone}
            \usetikzlibrary{graphs,graphdrawing,quotes,arrows.meta,calc,backgrounds,decorations.markings}
            \usegdlibrary{layered}
            \begin{document}

            \tikzset{process/.style = { 
                minimum width = 2cm,
                draw
            }}
            
            \def\la{0.2} % centered on 0.25, lhs 0.2 evenly divides 5, 10% of 5
            \def\lb{0.7} % centered on 0.75, rhs 0.8 evenly divides 5
            \def\lc{0.4} % centered on 0.5
            \def\ld{0.4}
            \def\le{0.7}
            \def\ra{0.3}
            \def\rb{0.8}
            \def\rc{0.6} % centered on 0.5, 20% of 2.5
            \def\rd{0.6}
            \def\re{0.8}
            
            \tikz[align=left, decoration={markings, mark=at position 0.5 with {\fill[red] (0, 0) circle (1pt);}}] {"#));
        // println!("{}", "graph [splines=ortho, nodesep=1];");

        let action_query = Ident("action");
        let percept_query = Ident("percept");
        let name_query = Ident("name");

        let mut vert = Graph::<&str, &str>::new();
        let mut horz = Graph::<&str, &str>::new();

        let mut v_nodes = HashMap::<&str, NodeIndex<DefaultIx>>::new();
        let mut h_nodes = HashMap::<&str, NodeIndex<DefaultIx>>::new();

        for hint in res {
            match hint {
                Fact::Fact(Ident("compact"), compacts) => {
                    let mut h_acts = HashMap::new();
                    let mut h_sens = HashMap::new();

                    for item in compacts {
                        let item_ident = unwrap_atom(item).unwrap();
                        let v_ix = or_insert(&mut vert, &mut v_nodes, item_ident);
                        // let _h_ix = or_insert(&mut horz, &h_nodes, item_ident);
                        
                        let resolved_item = resolve(v.iter(), item).collect::<Vec<&Fact>>();
                        let name = as_string(&resolved_item, &name_query, item_ident.into());
                        println!(r#"{}/{};"#, unwrap_atom(item).unwrap(), tikz_escape(&name));

                        // TODO: need to loop here to resolve all the actuates/senses/hosts pairs, not just the first
                        let resolved_actuates = find_parent(v.iter(), &Ident("actuates"), to_ident(item)).next();
                        let resolved_senses = find_parent(v.iter(), &Ident("senses"), to_ident(item)).next();
                        let resolved_hosts = find_parent(v.iter(), &Ident("hosts"), to_ident(item)).next();
                        let action = as_string(&resolved_item, &action_query, unwrap_atom(item).unwrap().into());
                        let percept = as_string(&resolved_item, &percept_query, unwrap_atom(item).unwrap().into());

                        h_acts.entry((resolved_actuates, resolved_hosts))
                            .or_insert(vec![])
                            .push((item_ident, action));

                        h_sens.entry((resolved_senses, resolved_hosts))
                            .or_insert(vec![])
                            .push((item_ident, percept));
                        
                        if let Some(actuator) = resolved_actuates {
                            let actuator_vix = or_insert(&mut vert, &mut v_nodes, actuator.0);
                            vert.add_edge(v_ix, actuator_vix, "actuates");
                        }

                        if let Some(sensor) = resolved_senses {
                            let sensor_vix = or_insert(&mut vert, &mut v_nodes, sensor.0);
                            vert.add_edge(v_ix, sensor_vix, "senses");
                        }

                        if let Some(client) = resolved_hosts {
                            let client_vix = or_insert(&mut vert, &mut v_nodes, client.0);
                            vert.add_edge(client_vix, v_ix, "rides");
                        }
                    }

                    for ((src, tgt), mediators) in h_acts.iter() {
                        if let (Some(s), Some(t)) = (src, tgt) {
                            for (med, action) in mediators {
                                println!(r#"({}) ->["{}"] ({});"#, s, tikz_escape(action), med);
                                println!(r#"({}) -> ({});"#, med, t);
                            }
                        }
                    }
                    for ((src, tgt), mediators) in h_sens.iter() {
                        if let (Some(s), Some(t)) = (src, tgt) {
                            for (med, percept) in mediators {
                                println!(r#"({}) <-["{}"] ({});"#, s, tikz_escape(percept), med);
                                println!(r#"({}) <- ({});"#, med, t);
                            }
                        }
                    }
                },
                Fact::Fact(Ident("coalesce"), coalesces) => {
                    let mut h_acts = HashMap::new();
                    let mut h_sens = HashMap::new();

                    for item in coalesces {
                        let item_ident = unwrap_atom(item).unwrap();
                        let v_ix = or_insert(&mut vert, &mut v_nodes, item_ident);
                        // let _h_ix = or_insert(&mut horz, &h_nodes, item_ident);
                        
                        let resolved_item = resolve(v.iter(), item).collect::<Vec<&Fact>>();

                        // let src_query = Fact::Atom(Ident(""))  // "x where q \in x.actuates"
                            // handle via unification? via a parser on facts??? via customs search routines?
                            // via some kind of relational hackery?
                        let resolved_actuates = find_parent(v.iter(), &Ident("actuates"), to_ident(item)).next();
                        let resolved_senses = find_parent(v.iter(), &Ident("senses"), to_ident(item)).next();
                        let resolved_hosts = find_parent(v.iter(), &Ident("hosts"), to_ident(item)).next();
                        // println!("{:?}", resolved_src);

                        let action = as_string(&resolved_item, &action_query, item_ident.into());
                        let percept = as_string(&resolved_item, &percept_query, item_ident.into());

                        h_acts.entry((resolved_actuates, resolved_hosts))
                            .or_insert(vec![])
                            .push(action);

                        h_sens.entry((resolved_senses, resolved_hosts))
                            .or_insert(vec![])
                            .push(percept);

                        if let Some(actuator) = resolved_actuates {
                            let actuator_vix = or_insert(&mut vert, &mut v_nodes, actuator.0);
                            vert.add_edge(actuator_vix, v_ix, "actuates");
                        }

                        if let Some(sensor) = resolved_senses {
                            let sensor_vix = or_insert(&mut vert, &mut v_nodes, sensor.0);
                            vert.add_edge(sensor_vix, v_ix, "senses");
                        }

                        if let Some(client) = resolved_hosts {
                            let client_vix = or_insert(&mut vert, &mut v_nodes, client.0);
                            vert.add_edge(v_ix, client_vix, "rides");
                        }
                    }

                    for ((src, tgt), actions) in h_acts.iter() {
                        if let (Some(s), Some(t)) = (src, tgt) {
                            let rendered_actions = actions.as_slice().join("\\n");
                            println!(r#"({}) ->["{}"] ({});"#, s, tikz_escape(&rendered_actions), t);
                        }
                    }

                    for ((src, tgt), percepts) in h_sens.iter() {
                        if let (Some(s), Some(t)) = (src, tgt) {
                            let rendered_percepts = percepts.as_slice().join("\\n");
                            println!(r#"({}) <-["{}"'] ({});"#, s, tikz_escape(&rendered_percepts), t);
                        }
                    }
                },
                _ => {},
            }
        }

        println!("{}", indoc!(r#"
            \scope[on background layer]
            \draw[help lines,very thin,step=1] (-3.2,-4.2) grid (3.2,1.2);
            \foreach \x in {-3,...,3} {
            \foreach \y in {-4,...,1} {
                \draw [fill, black] (\x, \y) circle (0.5pt); 
                \node[scale=0.5, anchor=south east] at (\x, \y) {\x,\y};
            }
            }
            \endscope

            }
            \end{document}
        "#));

        println!("{}", "\n\n\n");
        println!("{:?}", Dot::new(&vert));
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