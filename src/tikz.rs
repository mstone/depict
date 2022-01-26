use diagrams::parser::{Ident, parse};
use diagrams::render::{Fact, Syn, filter_fact, resolve, unwrap_atom, find_parent, to_ident, as_string};
use petgraph::EdgeDirection::{Incoming, Outgoing};
use std::collections::{HashMap, BTreeMap};
use std::collections::hash_map::{Entry};
use std::fs::read_to_string;
use std::env::args;
use std::io::{self, Write};
use std::process::{exit, Command, Stdio};
use ndarray::{Array, Array3};
use nom::error::convert_error;
use indoc::indoc;
use petgraph::graph::{Graph, DefaultIx, NodeIndex, EdgeReference, EdgeIndex};
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

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum NodeHRank {
    Real(NodeIndex),
    Fake(EdgeIndex),
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
        // let host_query = Ident("host");
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
                        let resolved_item = resolve(v.iter(), item).collect::<Vec<&Fact>>();
                        let name = as_string(&resolved_item, &name_query, item_ident.into());
                        // println!(r#"{}/{};"#, unwrap_atom(item).unwrap(), tikz_escape(&name));

                        // TODO: need to loop here to resolve all the actuates/senses/hosts pairs, not just the first
                        let resolved_actuates = find_parent(v.iter(), &Ident("actuates"), to_ident(item)).next();
                        let resolved_senses = find_parent(v.iter(), &Ident("senses"), to_ident(item)).next();
                        let resolved_hosts = find_parent(v.iter(), &Ident("hosts"), to_ident(item)).next();
                        let action = as_string(&resolved_item, &action_query, item_ident.into());
                        let percept = as_string(&resolved_item, &percept_query, item_ident.into());
                        // let host = as_string(&resolved_item, &host_query, item_ident.into());

                        h_styl.insert(item_ident, style);
                        h_name.insert(item_ident, name);

                        h_acts.entry((resolved_actuates, resolved_hosts))
                            .or_insert(vec![])
                            .push((item_ident, action));

                        h_sens.entry((resolved_senses, resolved_hosts))
                            .or_insert(vec![])
                            .push((item_ident, percept));

                        match *style {
                            "compact" => {
                                let _v_ix = or_insert(&mut vert, &mut v_nodes, item_ident);
                            },
                            "coalesce" => {
                                if let (Some(actuator), Some(process)) = (resolved_actuates, resolved_hosts) {
                                    let controller_ix = or_insert(&mut vert, &mut v_nodes, actuator.0);
                                    let process_ix = or_insert(&mut vert, &mut v_nodes, process.0);
                                    vert.add_edge(controller_ix, process_ix, "actuates"); // controls?
                                }
                                if let (Some(sensor), Some(process)) = (resolved_senses, resolved_hosts) {
                                    let controller_ix = or_insert(&mut vert, &mut v_nodes, sensor.0);
                                    let process_ix = or_insert(&mut vert, &mut v_nodes, process.0);
                                    vert.add_edge(controller_ix, process_ix, "senses"); // reads?
                                }
                            },
                            "embed" => {
                                let v_ix = or_insert(&mut vert, &mut v_nodes, item_ident);
                            },
                            "parallel" => {
                                let v_ix = or_insert(&mut vert, &mut v_nodes, item_ident);
                                // let _h_ix = or_insert(&mut horz, &h_nodes, item_ident);
                            
                                if let Some(actuator) = resolved_actuates {
                                    let controller_ix = or_insert(&mut vert, &mut v_nodes, actuator.0);
                                    vert.add_edge(controller_ix, v_ix, "actuates");
                                }
        
                                if let Some(sensor) = resolved_senses {
                                    let controller_ix = or_insert(&mut vert, &mut v_nodes, sensor.0);
                                    vert.add_edge(controller_ix, v_ix, "senses");
                                }
        
                                if let Some(platform) = resolved_hosts {
                                    let platform_ix = or_insert(&mut vert, &mut v_nodes, platform.0);
                                    vert.add_edge(v_ix, platform_ix, "rides");
                                }
                            },
                            _ => {
                                unimplemented!("{}", style);
                            }
                        }
                        

                        eprintln!("READ {} {} {:?} {:?} {:?}", item_ident, style, resolved_actuates, resolved_senses, resolved_hosts)
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

        let paths = floyd_warshall(&vert, |_ex| { -1 as i32 }).unwrap();
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

        let mut vx_rank = HashMap::new();
        let mut hx_rank = HashMap::new();
        for (rank, paths) in paths_by_rank.iter() {
            for (n, (_vx, wx)) in paths.iter().enumerate() {
                vx_rank.insert(*wx, *rank);
                hx_rank.insert(*wx, n);
            }
        }

        let mut locs_by_level = BTreeMap::new();
        for (rank, paths) in paths_by_rank.iter() {
            let l = rank;
            let n = paths.len();
            for a in 0..(n-1) {
                locs_by_level.entry(*l).or_insert(vec![]).push(a);
            }
        }

        let mut hops_by_level = BTreeMap::new();
        for ex in vert.edge_indices() {
            let el = vert.edge_weight(ex).unwrap();
            match *el {
                "actuates" | "senses" | "rides" => {
                    let (vx, wx) = vert.edge_endpoints(ex).unwrap();
                    let vvr = *vx_rank.get(&vx).unwrap();
                    let wvr = *vx_rank.get(&wx).unwrap();
                    let vhr = *hx_rank.get(&vx).unwrap();
                    let whr = *hx_rank.get(&wx).unwrap();
                    
                    let mut mhrs = vec![vhr];
                    for mid_level in (vvr+1)..(wvr) {
                        let mhr = locs_by_level.get(&mid_level).map_or(0, |v| v.len());
                        locs_by_level.entry(mid_level).or_insert(vec![]).push(mhr);
                        mhrs.push(mhr);
                    }
                    mhrs.push(whr);
                    
                    for lvl in vvr..(wvr as i32) {
                        let mx = (lvl as i32 - vvr as i32) as usize;
                        let nx = (lvl as i32 + 1 - vvr as i32) as usize;
                        let mhr = mhrs[mx];
                        let nhr = mhrs[nx];
                        hops_by_level
                            .entry(lvl)
                            .or_insert(vec![])
                            .push((lvl, mhr, nhr, ex));
                    }
                },
                _ => {},
            }
        }

        let max_level = hops_by_level.keys().max().unwrap();
        let max_width = hops_by_level.values().map(|paths| paths.len()).max().unwrap();

        let mut csp = String::new();
        csp.push_str("MINION 3\n");
        csp.push_str("**VARIABLES**\n");
        csp.push_str(&format!("BOOL x[{},{},{}]\n", max_level, max_width, max_width));
        csp.push_str(&format!("BOOL c[{},{},{},{},{}]\n", max_level, max_width, max_width, max_width, max_width));
        csp.push_str("BOUND csum {0..1000}\n");
        csp.push_str("\n**SEARCH**\n");
        csp.push_str("MINIMISING csum\n");
        // csp.push_str("PRINT ALL");
        csp.push_str("PRINT [[csum]]\n");
        csp.push_str("PRINT [[x]]\n");
        csp.push_str("\n**CONSTRAINTS**\n");
        for (rank, paths) in paths_by_rank.iter() {
            let l = rank;
            let n = paths.len();
            for a in 0..(n-1) {
                for b in 0..(n-1) {
                    if a != b {
                        csp.push_str(&format!("sumleq([x[{l},{a},{b}], x[{l},{b},{a}]],1)\n", l=l, a=a, b=b));
                        csp.push_str(&format!("sumgeq([x[{l},{a},{b}], x[{l},{b},{a}]],1)\n", l=l, a=a, b=b));
                        for c in 0..(n-1) {
                            if b != c && a != c {
                                csp.push_str(&format!("sumleq([x[{l},{c},{b}], x[{l},{b},{a}], -1],x[{l},{c},{a}])\n", l=l, a=a, b=b, c=c));
                            }
                        }
                    }
                }
            }
        }
        for (k, hops) in hops_by_level.iter() {
            if *k < max_level-1 {
                for (_lvl, a, c, _ex) in hops {
                    for (_lvl2, b, d, _fx)  in hops {
                        if (a,c) != (b,d) {
                            csp.push_str(&format!("sumgeq([c[{k},{a},{b},{c},{d}],x[{k},{c},{a}],x[{j},{b},{d}]],1)\n", a=a, b=b, c=c, d=d, k=k, j=k+1));
                            csp.push_str(&format!("sumgeq([c[{k},{a},{b},{c},{d}],x[{k},{a},{c}],x[{j},{d},{b}]],1)\n", a=a, b=b, c=c, d=d, k=k, j=k+1));
                        }
                    }
                }
            }
        }
        csp.push_str("\nsumleq([c[_,_,_,_,_]],csum)\n");
        csp.push_str("sumgeq([c[_,_,_,_,_]],csum)\n");
        csp.push_str("\n\n**EOF**");

        eprintln!("{}", csp);


        // std::process::exit(0);

        let mut minion = Command::new("/Users/mstone/src/minion/result/bin/minion");
        minion
            .arg("-printsolsonly")
            .arg("-printonlyoptimal")
            .arg("-timelimit")
            .arg("1")
            .arg("--")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        let mut child = minion.spawn()
            .expect("failed to execute minion");
        let mut stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(csp.as_bytes()).expect("failed to write csp");
        drop(stdin);

        let output = child
            .wait_with_output()
            .expect("failed to wait on child");

        let outs = std::str::from_utf8(&output.stdout[..]).unwrap();

        // eprintln!("{}", outs);

        let lines = outs.split("\n").collect::<Vec<_>>();
        let crossing_number = lines[lines.len()-3]
            .trim()
            .parse::<i32>()
            .expect("unable to parse crossing number");
        let soln = lines[lines.len()-2]
            .split(" ")
            .filter_map(|s| s.parse::<i32>().ok())
            .collect::<Vec<_>>();
        
        eprintln!("{:?}, {:?}\n{:?}", crossing_number, soln, lines);

        let mut perm = Array3::<i32>::zeros((*max_level as usize, max_width, max_width));
        for (n, ix) in perm.iter_mut().enumerate() {
            *ix = soln[n];
        }

        eprintln!("{:?}", perm);

        // turn the data into hranks.
        std::process::exit(0);

        let mut v_rank = HashMap::new();
        let mut h_rank = HashMap::new();
        // paths_from_roots
        //     .sort_by_key(|(vx, wx, wgt)| { *wgt });
        // println!("PATHS");
        for (rank, paths) in paths_by_rank.iter() {
            for (vx, wx) in paths.iter() {
                let vl = vert.node_weight(*vx).unwrap();
                let wl = vert.node_weight(*wx).unwrap();
                let name = h_name.get(wl).unwrap();
                let vpos = -1.5 * (*rank as f64);
                let n = 0;
                let hpos = 3.5 * (n as f64);

                v_rank.insert(wl, *rank);
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
            let mut sorted_out_nbrs = BTreeMap::new();
            let mut sorted_in_nbrs = BTreeMap::new();
            let mut out_nbrs = vert.neighbors_directed(vx, Outgoing).detach();
            while let Some((ex, wx)) = out_nbrs.next(&vert) {
                let dst = vert.node_weight(wx).unwrap();
                let edge = vert.edge_weight(ex).unwrap();
                let dst_hrank = h_rank.get(dst).unwrap();
                println!("% OUT {:?} {} {:?} {:?} {} {} {}", vx, src, ex, wx, dst, edge, dst_hrank);
                sorted_out_nbrs
                    .entry(dst_hrank)
                    .or_insert(vec![])
                    .push((wx, dst, edge));
            }
            

            let num_out_bundles = sorted_out_nbrs.len() as f64;
            let arrow_position = |bundle_rank: usize, adj: f64| {
                let br = bundle_rank as f64;
                let nob = num_out_bundles as f64;
                ((br + 1.0) / (nob + 1.0)) + (adj / nob)
            };
            for (n, (_hrank, edges)) in sorted_out_nbrs.iter().enumerate() {
                for (wx, dst, edge) in edges {
                    let mut in_nbrs = vert.neighbors_directed(*wx, Incoming).detach();
                    while let Some((ex, ux)) = in_nbrs.next(&vert) {
                        let src2 = vert.node_weight(ux).unwrap();
                        let edge2 = vert.edge_weight(ex).unwrap();
                        let src2_vrank = v_rank.get(src2).unwrap();
                        let src2_hrank = h_rank.get(src2).unwrap();
                        println!("% IN2 {:?} {} {:?} {:?} {} {} {}", ux, src2, ex, wx, dst, edge2, src2_hrank);
                        sorted_in_nbrs
                            .entry((src2_hrank, src2_vrank))
                            .or_insert(vec![])
                            .push((ux, src2, edge2));
                    }
                    let num_in_bundles = sorted_in_nbrs.len() as f64;
                    let arrow_position2 = |bundle_rank: usize, adj: f64| {
                        let br = bundle_rank as f64;
                        let nib = num_in_bundles as f64;
                        ((br + 1.0) / (nib + 1.0)) + (adj / nib)
                    };
                    let m = sorted_in_nbrs
                        .iter()
                        .enumerate()
                        .find(|(m, (_rank2, edges2))| 
                            edges2
                                .iter()
                                .find(|(ux, _, _)| *ux == vx)
                                .is_some()).unwrap().0;
                    println!("% ARR {} {} -> {}, {}, {}", n, src, dst, edge, m);
                    match **edge {
                        "actuates" => {
                            println!(indoc!(r#"
                                \draw [-{{Latex[]}},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)        to[]  node[scale=0.8, anchor=north east, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} ($({}.north west)!{}!({}.north east)$);"#),
                                src, arrow_position(n, -0.05), src, edge, dst, arrow_position2(m, -0.05), dst    
                            );
                        },
                        "senses" => {
                            println!(indoc!(r#"
                                \draw [{{Latex[]-}},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)        to[]  node[scale=0.8, anchor=south west, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} ($({}.north west)!{}!({}.north east)$);"#),
                                src, arrow_position(n, 0.05), src, edge, dst, arrow_position2(m, 0.05), dst    
                            );
                        },
                        _ => {
                            println!(indoc!(r#"
                                \draw [-{{Stealth[]}},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)        to[]  node[scale=0.8, anchor=north east, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} ($({}.north west)!{}!({}.north east)$);"#),
                                src, arrow_position(n, 0.0), src, edge, dst, arrow_position2(m, 0.0), dst    
                            );
                        },
                    };
                }
            }
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