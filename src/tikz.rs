use diagrams::parser::{Ident, parse};
use diagrams::render::{Fact, Syn, filter_fact, resolve, unwrap_atom, find_parent, to_ident, as_string};
use petgraph::EdgeDirection::{Incoming, Outgoing};
use petgraph::algo::bellman_ford;
use sorted_vec::SortedVec;
use std::collections::{HashMap, BTreeMap};
use std::collections::hash_map::{Entry};
use std::fs::read_to_string;
use std::env::args;
use std::hash::Hash;
use std::io::{self, Write};
use std::process::{exit, Command, Stdio};
use ndarray::{Array2};
use nom::error::convert_error;
use indoc::indoc;
use petgraph::graph::{Graph, DefaultIx, NodeIndex, EdgeReference, EdgeIndex};
use petgraph::dot::{Dot};
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use petgraph::algo::floyd_warshall::floyd_warshall;

pub fn tikz_escape(s: &str) -> String {
    s
        .replace("$", "\\$")
        .replace("\\n", "\\\\")
}

pub fn or_insert<V, E>(g: &mut Graph<V, E>, h: &mut HashMap<V, NodeIndex>, v: V) -> NodeIndex where V: Eq + Hash + Clone {
    let e = h.entry(v.clone());
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

        let mut v_nodes = HashMap::<&str, NodeIndex>::new();
        let mut h_nodes = HashMap::<&str, NodeIndex>::new();

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

        eprintln!("VERT: {:?}", Dot::new(&vert));

        let mut condensed = Graph::<&str, SortedVec<(&str, &str, &str)>>::new();
        let mut condensed_vxmap = HashMap::new();
        for (vx, vl) in vert.node_references() {
            let mut dsts = HashMap::new();
            for er in vert.edges_directed(vx, Outgoing) {
                let wx = er.target();
                let wl = vert.node_weight(wx).unwrap();
                dsts.entry(wl).or_insert(SortedVec::new()).insert((*vl, *wl, *er.weight()));
            }
            
            let cvx = or_insert(&mut condensed, &mut condensed_vxmap, vl);
            for (wl, exs) in dsts {
                let cwx = or_insert(&mut condensed, &mut condensed_vxmap, wl);
                condensed.add_edge(cvx, cwx, exs);
            }
        }

        eprintln!("CONDENSED: {:?}", Dot::new(&condensed));

        // find graph roots
        // in a digraph, the roots are nodes with in-degree 0
        let roots = SortedVec::from_unsorted(
            condensed
            .externals(Incoming)
            .map(|vx| *vert.node_weight(vx).unwrap())
            .collect::<Vec<_>>());
        eprintln!("ROOTS {:?}\n", roots);

        // let paths = floyd_warshall(&condensed, |_ex| { -1 as i32 }).unwrap();

        // eprintln!("FLOYD-WARSHALL: {:#?}\n", SortedVec::from_unsorted(
        //     paths
        //         .iter()
        //         .map(|((vx, wx), wgt)| {
        //             let vl = *vert.node_weight(*vx).unwrap();
        //             let wl = *vert.node_weight(*wx).unwrap();
        //             (wgt, vl, wl)
        //         }).collect::<Vec<_>>()
        //     )
        // );
        let root_vx = condensed_vxmap[roots[0]];
        let paths = bellman_ford(&condensed.map(|vx, vl| vl, |ex, el| -1.0), root_vx).unwrap();

        // let mut paths_from_roots = SortedVec::from_unsorted(
        //     paths
        //     .iter()
        //     .filter_map(|((vx, wx), wgt)| {
        //         let vl = *vert.node_weight(*vx).unwrap();
        //         let wl = *vert.node_weight(*wx).unwrap();
        //         if *wgt <= 0 && roots.contains(&vl) {
        //             Some((-(*wgt) as usize, vl, wl))
        //         } else {
        //             None
        //         }
        //     })
        //     .collect::<Vec<_>>());
        let mut paths_from_roots = SortedVec::from_unsorted(
            paths
                .distances
                .iter()
                .enumerate()
                .map(|(n, wgt)| {
                    (-wgt as usize, roots[0], *condensed.node_weight(NodeIndex::new(n)).unwrap())
                }).collect::<Vec<_>>()
        );

        eprintln!("PATHS_FROM_ROOTS: {:#?}", paths_from_roots);

        let mut paths_by_rank = BTreeMap::new();
        for (wgt, vx, wx) in paths_from_roots.iter() {
            paths_by_rank
                .entry(*wgt)
                .or_insert(SortedVec::new())
                .insert((*vx, *wx));
        }

        eprintln!("PATHS_BY_RANK: {:#?}", paths_by_rank);

        let mut vx_rank = HashMap::new();
        let mut hx_rank = HashMap::new();
        for (rank, paths) in paths_by_rank.iter() {
            for (n, (_vx, wx)) in paths.iter().enumerate() {
                vx_rank.insert(*wx, *rank);
                hx_rank.insert(*wx, n);
            }
        }


        #[derive(Eq, PartialEq, Ord, PartialOrd, Hash)]
        enum Loc<V,E> {
            Node(V),
            Hop(E, E),
        }
        let mut loc_to_node = HashMap::new();
        let mut node_to_loc = HashMap::new();

        let mut locs_by_level = BTreeMap::new();
        for (rank, paths) in paths_by_rank.iter() {
            let l = *rank;
            for (a, (cvl, cwl)) in paths.iter().enumerate() {
                locs_by_level.entry(l).or_insert(vec![]).push(a);
                loc_to_node.insert((l, a), Loc::Node(cwl));
                node_to_loc.insert(Loc::Node(cwl), (l, a));
            }
        }

        eprintln!("LOCS_BY_LEVEL V1: {:#?}", locs_by_level);

        let sorted_condensed_edges = SortedVec::from_unsorted(
            condensed
                .edge_references()
                .map(|er| {
                    let (vx, wx) = (er.source(), er.target());
                    let vl = condensed.node_weight(vx).unwrap();
                    let wl = condensed.node_weight(wx).unwrap();
                    (*vl, *wl, er.weight())
                })
                .collect::<Vec<_>>()
        );

        eprintln!("CONDENSED GRAPH: {:#?}", sorted_condensed_edges);

        let mut hops_by_edge = BTreeMap::new();
        let mut hops_by_level = BTreeMap::new();
        for (vl, wl, _) in sorted_condensed_edges.iter() {
            let vvr = *vx_rank.get(*vl).unwrap();
            let wvr = *vx_rank.get(*wl).unwrap();
            let vhr = *hx_rank.get(*vl).unwrap();
            let whr = *hx_rank.get(*wl).unwrap();
            
            let mut mhrs = vec![vhr];
            for mid_level in (vvr+1)..(wvr) {
                let mhr = locs_by_level.get(&mid_level).map_or(0, |v| v.len());
                locs_by_level.entry(mid_level).or_insert(vec![]).push(mhr);
                loc_to_node.insert((mid_level, mhr), Loc::Hop(vl, wl));
                node_to_loc.insert(Loc::Hop(vl, wl), (mid_level, mhr));
                mhrs.push(mhr);
            }
            mhrs.push(whr);

            eprintln!("HOP {} {} {} {} {} {} {:?}", vl, wl, vvr, wvr, vhr, whr, mhrs);
            
            for lvl in vvr..wvr {
                let mx = (lvl as i32 - vvr as i32) as usize;
                let nx = (lvl as i32 + 1 - vvr as i32) as usize;
                let mhr = mhrs[mx];
                let nhr = mhrs[nx];
                hops_by_level
                    .entry(lvl)
                    .or_insert(SortedVec::new())
                    .insert((mhr, nhr, vl, wl, lvl));
                hops_by_edge
                    .entry((*vl, *wl))
                    .or_insert(BTreeMap::new())
                    .insert(lvl, (mhr, nhr));
            }
        }
        eprintln!("LOCS_BY_LEVEL V2: {:#?}", locs_by_level);

        eprintln!("HOPS_BY_LEVEL: {:#?}", hops_by_level);

        let mut g_hops = Graph::<(usize, usize), (usize, &str, &str)>::new();
        let mut g_hops_vx = HashMap::new();
        for (rank, hops) in hops_by_level.iter() {
            for (mhr, nhr, vl, wl, lvl) in hops.iter() {
                let gvx = or_insert(&mut g_hops, &mut g_hops_vx, (*lvl, *mhr));
                let gwx = or_insert(&mut g_hops, &mut g_hops_vx, (lvl+1, *nhr));
                g_hops.add_edge(gvx, gwx, (*lvl, vl, wl));
            }
        }
        eprintln!("HOPS GRAPH: {:?}\n", Dot::new(&g_hops));

        // std::process::exit(0);

        let max_level = *hops_by_level.keys().max().unwrap();
        let max_width = hops_by_level.values().map(|paths| paths.len()).max().unwrap();

        eprintln!("max_level: {}, max_width: {}", max_level, max_width);

        let mut csp = String::new();
        csp.push_str("MINION 3\n");
        csp.push_str("**VARIABLES**\n");
        csp.push_str("BOUND csum {0..1000}\n");
        for (rank, locs) in locs_by_level.iter() {
            csp.push_str(&format!("BOOL x{}[{},{}]\n", rank, locs.len(), locs.len()));
        }
        for rank in 0..locs_by_level.len() - 2 {
            let w1 = locs_by_level[&rank].len();
            let w2 = locs_by_level[&(rank+1)].len();
            csp.push_str(&format!("BOOL c{}[{},{},{},{}]\n", rank, w1, w2, w1, w2));
        }
        csp.push_str("\n**SEARCH**\n");
        csp.push_str("MINIMISING csum\n");
        // csp.push_str("PRINT ALL\n");
        csp.push_str("PRINT [[csum]]\n");
        for (rank, _) in locs_by_level.iter() {
            csp.push_str(&format!("PRINT [[x{}]]\n", rank));
        }
        // for rank in 0..max_level {
        //     csp.push_str(&format!("PRINT [[c{}]]\n", rank));
        // }
        csp.push_str("\n**CONSTRAINTS**\n");
        for (rank, locs) in locs_by_level.iter() {
            let l = rank;
            let n = locs.len();
            // let n = max_width;
            for a in 0..n {
                csp.push_str(&format!("sumleq(x{l}[{a},{a}],0)\n", l=l, a=a));
                for b in 0..n {
                    if a != b {
                        csp.push_str(&format!("sumleq([x{l}[{a},{b}], x{l}[{b},{a}]],1)\n", l=l, a=a, b=b));
                        csp.push_str(&format!("sumgeq([x{l}[{a},{b}], x{l}[{b},{a}]],1)\n", l=l, a=a, b=b));
                        for c in 0..(n) {
                            if b != c && a != c {
                                csp.push_str(&format!("sumleq([x{l}[{c},{b}], x{l}[{b},{a}], -1],x{l}[{c},{a}])\n", l=l, a=a, b=b, c=c));
                            }
                        }
                    }
                }
            }
        }
        for (k, hops) in hops_by_level.iter() {
            if *k < max_level {
                for (u1, v1, ..) in hops.iter() {
                    for (u2, v2, ..)  in hops.iter() {
                        // if (u1,v1) != (u2,v2) {
                        if u1 != u2 && v1 != v2 {
                            csp.push_str(&format!("sumgeq([c{k}[{u1},{v1},{u2},{v2}],x{k}[{u2},{u1}],x{j}[{v1},{v2}]],1)\n", u1=u1, u2=u2, v1=v1, v2=v2, k=k, j=k+1));
                            csp.push_str(&format!("sumgeq([c{k}[{u1},{v1},{u2},{v2}],x{k}[{u1},{u2}],x{j}[{v2},{v1}]],1)\n", u1=u1, u2=u2, v1=v1, v2=v2, k=k, j=k+1));
                            // csp.push_str(&format!("sumleq(c{k}[{a},{c},{b},{d}],c{k}[{b},{d},{a},{c}])\n", a=a, b=b, c=c, d=d, k=k));
                            // csp.push_str(&format!("sumgeq(c{k}[{a},{c},{b},{d}],c{k}[{b},{d},{a},{c}])\n", a=a, b=b, c=c, d=d, k=k));
                        }
                    }
                }
            }
        }
        csp.push_str("\nsumleq([");
        for rank in 0..max_level {
            if rank > 0 {
                csp.push_str(",");
            }
            csp.push_str(&format!("c{}[_,_,_,_]", rank));
        }
        csp.push_str("],csum)\n");
        csp.push_str("sumgeq([");
        for rank in 0..max_level {
            if rank > 0 {
                csp.push_str(",");
            }
            csp.push_str(&format!("c{}[_,_,_,_]", rank));
        }
        csp.push_str("],csum)\n");
        csp.push_str("\n\n**EOF**");

        eprintln!("{}", csp);


        // std::process::exit(0);

        let mut minion = Command::new("/Users/mstone/src/minion/result/bin/minion");
        minion
            .arg("-printsolsonly")
            .arg("-printonlyoptimal")
            // .arg("-timelimit")
            // .arg("30")
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

        eprintln!("{}", outs);

        // std::process::exit(0);

        let lines = outs.split("\n").collect::<Vec<_>>();
        eprintln!("cn line: {}", lines[2]);
        

        let crossing_number = lines[2]
            .trim()
            .parse::<i32>()
            .expect("unable to parse crossing number");

        // std::process::exit(0);
        
        let solns = &lines[3..lines.len()];
        
        eprintln!("{:?}\n{:?}\n{:?}", lines, solns, crossing_number);

        let mut perm = Vec::<Array2<i32>>::new();
        for (rank, locs) in locs_by_level.iter() {
            let mut arr = Array2::<i32>::zeros((locs.len(), locs.len()));
            let mut parsed_solns = solns[*rank]
                .split(" ")
                .filter_map(|s| {
                    s
                        .trim()
                        .parse::<i32>()
                        .ok()
                })
                .collect::<Vec<_>>();
            for (n, ix) in arr.iter_mut().enumerate() {
                *ix = parsed_solns[n];
            }
            perm.push(arr);
        }

        for (n, p) in perm.iter().enumerate() {
            eprintln!("{}:\n{:?}\n", n, p);
        };

        let mut solved_locs = BTreeMap::new();
        for (n, p) in perm.iter().enumerate() {
            let mut sums = p.rows().into_iter().enumerate().map(|(i, r)| (i, r.sum() as usize)).collect::<Vec<_>>();
            sums.sort_by_key(|(i,s)| *s);
            eprintln!("row sums: {:?}", sums);
            for (nhr, (i,s)) in sums.into_iter().enumerate() {
                solved_locs.entry(n).or_insert(BTreeMap::new()).insert(i, nhr);
            }
        }
        eprintln!("SOLVED_LOCS: {:#?}", solved_locs);

        // turn the data into hranks.
        // std::process::exit(0);

        for (loc, node) in loc_to_node.iter() {   
            let (ovr, ohr) = loc;
            let (svr, shr) = (ovr, solved_locs[&ovr][&ohr]);
            let vpos = -1.5 * (*ovr as f64);
            let hpos = 3.5 * (shr as f64);
            match node {
                Loc::Node(vl) => {
                    println!(indoc!(r#"
                        \node[minimum width = 2.5cm, fill=white, fill opacity=0.9, draw, text opacity=1.0]({}) at ({}, {}) {{{}}};"#), 
                        vl, hpos, vpos, h_name[*vl]);
                },
                Loc::Hop(_, _) => {
                    println!(indoc!(r#"
                    \draw [fill, black] ({}, {}) circle (0.5pt);
                    \node[](aux_{}_{}) at ({}, {}) {{}};"#), 
                    hpos, vpos, ovr, ohr, hpos, vpos);
                },
            }
        }

        for cer in condensed.edge_references() {
            for (vl, wl, ew) in cer.weight().iter() {
                let hops = &hops_by_edge[&(*vl, *wl)];
                eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);
                match hops.len() {
                    0 => { unreachable!(); }
                    1 => {
                        println!(indoc!(r#"
                            \draw [-{{Stealth[]}},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)        to[]  node[scale=0.8, anchor=north east, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} ($({}.north west)!{}!({}.north east)$);"#),
                            vl, 0.5, vl, ew, wl, 0.5, wl    
                        );
                    },
                    2 => {
                        let (lvl, (mhr, nhr)) = hops.iter().next().unwrap();
                        let (ovr, ohr) = (lvl+1, nhr);
                        println!(indoc!(r#"
                            \draw [-{{Stealth[]}},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$) -- node[scale=0.8, anchor=north east, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} at ({},{}) -- ($({}.north west)!{}!({}.north east)$);"#),
                            vl, 0.5, vl, ew, ovr, ohr, wl, 0.5, wl    
                        );
                    },
                    max_levels => {
                        print!(indoc!(r#"\draw [-{{Stealth[]}},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)"#), vl, 0.5, vl);
                        let mid = max_levels / 2;
                        for (n, hop) in hops.iter().enumerate() {
                            if n < max_levels-1 {
                                let (lvl, (mhr, nhr)) = hop;
                                let (ovr, ohr) = (lvl+1, nhr);
                                // let (ovr, ohr) = (lvl, mhr);

                                // println!("% HOP {} {:?}", n, hop);
                                if n == mid {
                                    // print!(indoc!(r#" -- node[scale=0.8, anchor=north east, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} at (aux_{}_{})"#), ew, ovr, ohr);
                                    print!(r#" -- (aux_{}_{})"#, ovr, ohr);
                                } else {
                                    print!(r#" -- (aux_{}_{})"#, ovr, ohr);
                                }
                            }
                        }
                        println!(indoc!(r#" -- ($({}.north west)!{}!({}.north east)$);"#), wl, 0.5, wl);
                    },
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