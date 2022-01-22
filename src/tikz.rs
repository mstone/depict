use diagrams::parser::{Ident, parse};
use diagrams::render::{Fact, Syn, filter_fact, resolve, unwrap_atom, find_parent, to_ident, as_string};
use petgraph::EdgeDirection::{Incoming, Outgoing};
use std::collections::{HashMap, BTreeMap};
use std::collections::hash_map::{Entry};
use std::fs::read_to_string;
use std::env::args;
use std::io;
use std::process::exit;
use nom::error::convert_error;
use indoc::indoc;
use petgraph::graph::{Graph, DefaultIx, NodeIndex, EdgeReference};
use petgraph::dot::{Dot};
use petgraph::visit::{NodeFiltered};
use petgraph::algo::floyd_warshall::floyd_warshall;

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
    // println!("OR_INSERT {} -> {:?}", v, ix);
    return *ix;
}

// pub fn find_roots<'a, 'g>(g: &'g mut Graph<&'a str, &'a str>) -> Vec<NodeIndex<DefaultIx>> {
//     let nf = NodeFiltered::from_fn(g, |ix| {
//         g.edges_directed(ix, Incoming).next().is_none()
//     });
// }

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

        let mut h_name = HashMap::new();
        let mut h_styl = HashMap::new();
        let mut h_acts = HashMap::new();
        let mut h_sens = HashMap::new();

        for hint in res {
            match hint {
                Fact::Fact(Ident(style), items) => {
                    for item in items {
                        let item_ident = unwrap_atom(item).unwrap();
                        let v_ix = or_insert(&mut vert, &mut v_nodes, item_ident);
                        // let _h_ix = or_insert(&mut horz, &h_nodes, item_ident);
                        
                        let resolved_item = resolve(v.iter(), item).collect::<Vec<&Fact>>();
                        let name = as_string(&resolved_item, &name_query, item_ident.into());
                        // println!(r#"{}/{};"#, unwrap_atom(item).unwrap(), tikz_escape(&name));

                        // TODO: need to loop here to resolve all the actuates/senses/hosts pairs, not just the first
                        let resolved_actuates = find_parent(v.iter(), &Ident("actuates"), to_ident(item)).next();
                        let resolved_senses = find_parent(v.iter(), &Ident("senses"), to_ident(item)).next();
                        let resolved_hosts = find_parent(v.iter(), &Ident("hosts"), to_ident(item)).next();
                        let action = as_string(&resolved_item, &action_query, unwrap_atom(item).unwrap().into());
                        let percept = as_string(&resolved_item, &percept_query, unwrap_atom(item).unwrap().into());

                        h_styl.insert(item_ident, style);
                        h_name.insert(item_ident, name);

                        h_acts.entry((resolved_actuates, resolved_hosts))
                            .or_insert(vec![])
                            .push((item_ident, action));

                        h_sens.entry((resolved_senses, resolved_hosts))
                            .or_insert(vec![])
                            .push((item_ident, percept));
                        
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
                },
                _ => {},
            }
        }

        // find graph roots
        // in a digraph, the roots are nodes with in-degree 0
        let roots = vert.externals(Incoming).collect::<Vec<NodeIndex<DefaultIx>>>();
        // println!("ROOTS {:?}", roots);

        // let rvert = vert.filter_map(|_vx, vw| {Some(vw)}, |_ex, ew| {
        //     match *ew {
        //         "actuates" | "rides" => Some(ew),
        //         _ => None,
        //     }
        // });

        // println!("RVERT {:?}", rvert);

        // println!("{}", "\n\n\n");
        // println!("{:?}", Dot::new(&vert));
        // println!("{}", "\n\n\n");

        let paths = floyd_warshall(&vert, |_ex| { -1 }).unwrap();
        let mut paths_from_roots = paths
            .iter()
            .filter_map(|((vx, wx), wgt)| {
                if *wgt <= 0 && roots.contains(vx) {
                    Some((*vx, *wx, -(*wgt)))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut paths_by_rank = BTreeMap::new();
        for (vx, wx, wgt) in &paths_from_roots {
            paths_by_rank
                .entry(*wgt)
                .or_insert(vec![])
                .push((*vx, *wx));
        }

        let mut h_rank = HashMap::new();
        // paths_from_roots
        //     .sort_by_key(|(vx, wx, wgt)| { *wgt });
        // println!("PATHS");
        for (rank, paths) in paths_by_rank.iter() {
            for (n, (vx, wx)) in paths.iter().enumerate() {
                let vl = vert.node_weight(*vx).unwrap();
                let wl = vert.node_weight(*wx).unwrap();
                let name = h_name.get(wl).unwrap();
                let vpos = -1.5 * (*rank as f64);
                let hpos = 3.5 * (n as f64);

                h_rank.insert(wl, n);
                println!("% GRR  {:?} {} -> {} {:?}: {}", vx, vl, wl, wx, rank);
                println!(indoc!(r#"
                    \node[minimum width = 2.5cm, fill=white, fill opacity=0.9, draw, text opacity=1.0]({}) at ({}, {}) {{{}}};"#), 
                    wl, hpos, vpos, name);
            }
        }

        // eprintln!("{:?}", Dot::new(&vert));

        for vx in vert.node_indices() {
            let src = vert.node_weight(vx).unwrap();
            // let style = h_styl.get(src).unwrap();
            // if let Some(&&"coalesced") = h_styl.get(src) {
                let mut sorted_nbrs = BTreeMap::new();
                let mut nbrs = vert.neighbors_directed(vx, Outgoing).detach();
                while let Some((ex, wx)) = nbrs.next(&vert) {
                    let dst = vert.node_weight(wx).unwrap();
                    let edge = vert.edge_weight(ex).unwrap();
                    let dst_hrank = h_rank.get(dst).unwrap();
                    println!("% ARGH {:?} {} {:?} {:?} {} {} {}", vx, src, ex, wx, dst, edge, dst_hrank);
                    sorted_nbrs
                        .entry(dst_hrank)
                        .or_insert(vec![])
                        .push((dst, edge));
                }

                let num_bundles = sorted_nbrs.len() as f64;
                let arrow_position = |bundle_rank: usize, adj: f64| {
                    (((bundle_rank as f64) + 1.0) / ((num_bundles as f64) + 1.0)) + adj
                };
                for (n, (_hrank, edges)) in sorted_nbrs.iter().enumerate() {
                    for (dst, edge) in edges {
                        match **edge {
                            "actuates" => {
                                println!("% {} {} -> {}, {}", n, src, dst, edge);
                                println!(indoc!(r#"
                                    \draw [-{{Latex[]}},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)        to[]  node[scale=0.8, anchor=north east, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} ($({}.north west)!\lc!({}.north east)$);"#),
                                    src, arrow_position(n, -0.05), src, edge, dst, dst    
                                );
                            },
                            "senses" => {
                                println!(indoc!(r#"
                                    \draw [{{Latex[]-}},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)        to[]  node[scale=0.8, anchor=south west, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} ($({}.north west)!\rc!({}.north east)$);"#),
                                    src, arrow_position(n, 0.05), src, edge, dst, dst    
                                );
                            },
                            _ => {
                                println!(indoc!(r#"
                                    \draw [-{{Stealth[]}},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)        to[]  node[scale=0.8, anchor=north east, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} ($({}.north west)!0.5!({}.north east)$);"#),
                                    src, arrow_position(n, 0.0), src, edge, dst, dst    
                                );
                            },
                        };
                    }
                }
            // }
        }

        // let l = -3;
        // let r = 12;
        // let t = 7;
        // let b = -1;
        // println!(indoc!(r#"
        //     \scope[on background layer]
        //     \draw[help lines,very thin,step=1] ({}.2,{}.2) grid ({}.2,{}.2);
        //     \foreach \x in {{{},...,{}}} {{
        //     \foreach \y in {{{},...,{}}} {{
        //         \draw [fill, black] (\x, \y) circle (0.5pt); 
        //         \node[scale=0.5, anchor=south east] at (\x, \y) {{\x,\y}};
        //     }}
        //     }}
        //     \endscope
        // "#), l, b, r, t, l, r, b, t);
        println!(indoc!(r#"
            }}
            \end{{document}}
        "#));

        // println!("{}", "\n\n\n");
        // println!("{:?}", Dot::new(&vert));
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