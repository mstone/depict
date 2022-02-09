use diagrams::parser::{Ident, parse};
use diagrams::render::{Fact, Syn, filter_fact, resolve, unwrap_atom, find_parent, to_ident, as_string};
use petgraph::EdgeDirection::{Incoming, Outgoing};
use petgraph::algo::{floyd_warshall};
use pyo3::types::{PyModule, IntoPyDict, PyList};
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
use numpy::{PyArray1, PyReadonlyArray1};
use indoc::indoc;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::dot::{Dot};
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use pyo3::{prelude::*, prepare_freethreaded_python};
use pyo3::class::basic::CompareOp;

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

pub fn mul<'py>(a: &'py PyAny, b: &'py PyAny) -> &'py PyAny {
    a.call_method1("__mul__", (b,)).unwrap()
}

pub fn sub<'py>(a: &'py PyAny, b: &'py PyAny) -> &'py PyAny {
    a.call_method1("__sub__", (b,)).unwrap()
}

pub fn add<'py>(a: &'py PyAny, b: &'py PyAny) -> &'py PyAny {
    a.call_method1("__add__", (b,)).unwrap()
}

pub fn get<'py>(a: &'py PyAny, b: usize) -> &'py PyAny {
    a.get_item(b.into_py(a.py())).unwrap()
}

pub fn geq<'py>(a: &'py PyAny, b: &'py PyAny) -> &'py PyAny {
    a.rich_compare(b, CompareOp::Ge).unwrap()
}

pub fn leq<'py>(a: &'py PyAny, b: &'py PyAny) -> &'py PyAny {
    a.rich_compare(b, CompareOp::Le).unwrap()
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum Loc<V,E> {
    Node(V),
    Hop(usize, E, E),
}

pub fn get_value<'py>(a: &'py PyAny) -> PyResult<PyReadonlyArray1<'py, f64>> {
    Ok(a
        .getattr("value")?
        .extract::<&PyArray1<f64>>()?
        .readonly())
}

pub fn calculate_vcg<'s>(v: &'s Vec<Syn>, draw: &'s Fact) -> (Graph<&'s str, &'s str>, HashMap<&'s str, NodeIndex>, HashMap<&'s str, String>) {
    // println!("draw:\n{:#?}\n\n", draw);

    let res = resolve(v.iter(), draw);

    // println!("resolution: {:?}\n", res);

    let name_query = Ident("name");

    let mut vert = Graph::<&str, &str>::new();

    let mut v_nodes = HashMap::<&str, NodeIndex>::new();

    let mut h_name = HashMap::new();

    for hint in res {
        match hint {
            Fact::Fact(Ident(style), items) => {
                for item in items {
                    let item_ident = unwrap_atom(item).unwrap();
                    let resolved_item = resolve(v.iter(), item).collect::<Vec<&Fact>>();
                    let name = as_string(&resolved_item, &name_query, item_ident.into());

                    // TODO: need to loop here to resolve all the actuates/senses/hosts pairs, not just the first
                    let resolved_actuates = find_parent(v.iter(), &Ident("actuates"), to_ident(item)).next();
                    let resolved_senses = find_parent(v.iter(), &Ident("senses"), to_ident(item)).next();
                    let resolved_hosts = find_parent(v.iter(), &Ident("hosts"), to_ident(item)).next();

                    h_name.insert(item_ident, name);

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
                            let _v_ix = or_insert(&mut vert, &mut v_nodes, item_ident);
                        },
                        "parallel" => {
                            let v_ix = or_insert(&mut vert, &mut v_nodes, item_ident);
                        
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

    (vert, v_nodes, h_name)
}

pub fn condense<'s>(vert: &'s Graph<&'s str, &'s str>) -> (Graph<&'s str, SortedVec<(&'s str, &'s str, &'s str)>>, HashMap::<&'s str, NodeIndex>) {
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
    (condensed, condensed_vxmap)
}

pub fn render(v: Vec<Syn>) {
    let ds = filter_fact(v.iter(), &Ident("draw"));
    // let ds2 = ds.collect::<Vec<&Fact>>();
    // println!("draw:\n{:#?}\n\n", ds2);

    for draw in ds {
        
        let (vert, v_nodes, h_name) = calculate_vcg(&v, draw);

        let (condensed, _) = condense(&vert);

        eprintln!("CONDENSED: {:?}", Dot::new(&condensed));

        // find graph roots
        // in a digraph, the roots are nodes with in-degree 0
        let roots = SortedVec::from_unsorted(
            condensed
            .externals(Incoming)
            .map(|vx| *condensed.node_weight(vx).unwrap())
            .collect::<Vec<_>>());
        eprintln!("ROOTS {:?}\n", roots);

        let paths_fw = floyd_warshall(&condensed, |_ex| { -1 as i32 }).unwrap();

        eprintln!("FLOYD-WARSHALL: {:#?}\n", SortedVec::from_unsorted(
            paths_fw
                .iter()
                .map(|((vx, wx), wgt)| {
                    let vl = *condensed.node_weight(*vx).unwrap();
                    let wl = *condensed.node_weight(*wx).unwrap();
                    (wgt, vl, wl)
                }).collect::<Vec<_>>()
            )
        );

        let paths_from_roots = SortedVec::from_unsorted(
            paths_fw
                .iter()
                .filter_map(|((vx, wx), wgt)| {
                    let vl = *condensed.node_weight(*vx).unwrap();
                    let wl = *condensed.node_weight(*wx).unwrap();
                    if *wgt <= 0 && roots.contains(&vl) {
                        Some((-(*wgt) as usize, vl, wl))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
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


        let mut loc_to_node = HashMap::new();
        let mut node_to_loc = HashMap::new();

        let mut locs_by_level = BTreeMap::new();
        for (rank, paths) in paths_by_rank.iter() {
            let l = *rank;
            for (a, (_cvl, cwl)) in paths.iter().enumerate() {
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
                loc_to_node.insert((mid_level, mhr), Loc::Hop(mid_level, vl, wl));
                node_to_loc.insert(Loc::Hop(mid_level, vl, wl), (mid_level, mhr)); // BUG: what about the endpoints?
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
        for (_rank, hops) in hops_by_level.iter() {
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
                        // if (u1,v1) != (u2,v2) { // BUG!
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
        let stdin = child.stdin.as_mut().unwrap();
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
            let parsed_solns = solns[*rank]
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
            sums.sort_by_key(|(_i,s)| *s);
            eprintln!("row sums: {:?}", sums);
            for (nhr, (i,_s)) in sums.into_iter().enumerate() {
                solved_locs.entry(n).or_insert(BTreeMap::new()).insert(i, nhr);
            }
        }
        eprintln!("SOLVED_LOCS: {:#?}", solved_locs);

        let all_locs = solved_locs
            .iter()
            .flat_map(|(ovr, nodes)| nodes
                .iter()
                .map(|(ohr, shr)| (*ovr, *ohr, *shr, &loc_to_node[&(*ovr,*ohr)])))
            .enumerate()
            .map(|(n, (ovr, ohr, shr, loc))| (ovr, ohr, shr, loc, n))
            .collect::<Vec<_>>();
        let num_locs = all_locs.len();

        let mut sol_by_loc = HashMap::new();
        for (ovr, ohr, _shr, _loc, n) in all_locs.iter() {
            sol_by_loc.insert((*ovr, *ohr), *n);
        }

        let all_hops0 = hops_by_level
            .iter()
            .flat_map(|h| 
                h.1.iter().map(|(mhr, nhr, vl, wl, lvl)| {
                    (*mhr, *nhr, **vl, **wl, *lvl)
                })
            ).enumerate()
            .map(|(n, (mhr, nhr, vl, wl, lvl))| {
                (lvl, mhr, nhr, vl, wl, n)
            })
            .collect::<Vec<_>>();
        let all_hops = hops_by_level
            .iter()
            .flat_map(|h| 
                h.1.iter().map(|(mhr, nhr, vl, wl, lvl)| {
                    (*mhr, *nhr, **vl, **wl, *lvl)
                })
            )
            .chain(
                hops_by_edge.iter().map(|((vl, wl), hops)| {
                        let (lvl, (_mhr, nhr)) = hops.iter().rev().next().unwrap();
                        (*nhr, std::usize::MAX, *vl, *wl, lvl+1)
                }) 
            )
            .enumerate()
            .map(|(n, (mhr, nhr, vl, wl, lvl))| {
                (lvl, mhr, nhr, vl, wl, n)
            })
            .collect::<Vec<_>>();
        
        let mut sol_by_hop = HashMap::new();
        for (lvl, mhr, _nhr, vl, wl, n) in all_hops.iter() {
            sol_by_hop.insert((*lvl, *mhr, *vl, *wl), *n);
        }
        let num_hops = all_hops.len();

        prepare_freethreaded_python();
        let res: PyResult<(Vec<_>, Vec<_>, Vec<_>)> = Python::with_gil(|py| {
            let cp = PyModule::import(py, "cvxpy")?;
            let var = cp.getattr("Variable")?;
            let sq = cp.getattr("square")?;
            let constant = cp.getattr("Constant")?;
            let problem = cp.getattr("Problem")?;
            let minimize = cp.getattr("Minimize")?;
            let square = |a: &PyAny| {sq.call1((a,)).unwrap()};
            let as_constant = |a: i32| { constant.call1((a.into_py(py),)).unwrap() };
            let as_constantf = |a: f64| { constant.call1((a.into_py(py),)).unwrap() };
            let hundred: &PyAny = as_constant(100);
            let thousand: &PyAny = as_constant(1000);
            let _ten: &PyAny = as_constant(10);
            let one: &PyAny = as_constant(1);
            let zero: &PyAny = as_constant(0);
            let l = var.call((num_locs,), Some([("pos", true)].into_py_dict(py)))?;
            let r = var.call((num_locs,), Some([("pos", true)].into_py_dict(py)))?;
            let s = var.call((num_hops,), Some([("pos", true)].into_py_dict(py)))?;
            let eps = var.call((), Some([("pos", true)].into_py_dict(py)))?;
            let sep = as_constantf(0.05);
            let mut cvec: Vec<&PyAny> = vec![];
            let mut obj = zero;

            for (ovr, ohr, _shr, _loc, _n) in all_locs.iter() {
                let ovr = *ovr; 
                let ohr = *ohr;
                let locs = &solved_locs[&ovr];
                let shr = locs[&ohr];
                let n = sol_by_loc[&(ovr, ohr)];
                // if loc.is_node() { 
                    eprint!("C0: ");
                    eprint!("r{} >= l{} + S, ", n, n);
                    cvec.push(geq(get(r,n), add(get(l,n), &sep)));
                    if shr > 0 {
                        let ohrp = locs.iter().position(|(_, shrp)| *shrp == shr-1).unwrap();
                        let np = sol_by_loc[&(ovr, ohrp)];
                        let shrp = locs[&ohrp];
                        cvec.push(geq(get(l, n), get(r, np)));
                        eprint!("l{:?} >= r{:?}, ", (ovr, ohr, shr, n), (ovr, ohrp, shrp, np));
                    }
                    if shr < locs.len()-1 {
                        let ohrn = locs.iter().position(|(_, shrp)| *shrp == shr+1).unwrap();
                        let nn = sol_by_loc[&(ovr,ohrn)];
                        let shrn = locs[&(ohr+1)];
                        cvec.push(leq(get(r,n), get(l,nn)));
                        eprint!("r{:?} <= l{:?}, ", (ovr, ohr, shr, n), (ovr, ohrn, shrn, nn));
                    }
                    if shr == locs.len()-1 {
                        cvec.push(leq(get(r,n), one));
                        eprint!("r{:?} <= 1", (ovr, ohr, shr, n));
                    }
                    eprintln!();  
                // }
            }
            for (lvl, mhr, nhr, vl, wl, _n) in all_hops0.iter() {
                let v_ers = vert.edges_directed(v_nodes[vl], Outgoing).into_iter().collect::<Vec<_>>();
                let w_ers = vert.edges_directed(v_nodes[wl], Incoming).into_iter().collect::<Vec<_>>();

                let mut v_dsts = v_ers.iter().map(|er| { *vert.node_weight(er.target()).unwrap() }).collect::<Vec<_>>();
                let mut w_srcs = w_ers.iter().map(|er| { *vert.node_weight(er.source()).unwrap() }).collect::<Vec<_>>();

                v_dsts.sort(); v_dsts.dedup();
                v_dsts.sort_by_key(|dst| {
                    let (ovr, ohr) = node_to_loc[&Loc::Node(dst)];
                    let (svr, shr) = (ovr, solved_locs[&ovr][&ohr]);
                    (shr, -(svr as i32))
                });
                let v_outs = v_dsts
                    .iter()
                    .map(|dst| { (*vl, *dst) })
                    .collect::<Vec<_>>();

                w_srcs.sort(); w_srcs.dedup();
                w_srcs.sort_by_key(|src| {
                    let (ovr, ohr) = node_to_loc[&Loc::Node(src)];
                    let (svr, shr) = (ovr, solved_locs[&ovr][&ohr]);
                    (shr, -(svr as i32))
                });
                let w_ins = w_srcs
                    .iter()
                    .map(|src| { (*src, *wl) })
                    .collect::<Vec<_>>();

                let bundle_src_pos = v_outs.iter().position(|e| { *e == (vl, wl) }).unwrap();
                let bundle_dst_pos = w_ins.iter().position(|e| { *e == (vl, wl) }).unwrap();
                let hops = &hops_by_edge[&(*vl, *wl)];
                eprintln!("lvl: {}, vl: {}, wl: {}, hops: {:?}", lvl, vl, wl, hops);
                
                let ln = sol_by_loc[&(*lvl, *mhr)];
                let rn = ln;
                let n = sol_by_hop[&(*lvl, *mhr, *vl, *wl)];
                let nd = sol_by_hop[&(lvl+1, *nhr, *vl, *wl)];
                let mut n = n;
                let mut nd = nd;

                // ORDER, SYMMETRY
                if bundle_src_pos == 0 {
                    eprint!("C1: ");
                    cvec.push(geq(get(s, n), add(get(l, ln), &eps)));
                    eprintln!("s{} >= l{}+ε", n, ln);
                    obj = add(obj, square(sub(get(s, n), get(l, ln))));
                }
                if bundle_src_pos > 0 {
                    eprint!("C2: ");
                    let (vll, wll) = v_outs[bundle_src_pos-1];
                    let hopsl = &hops_by_edge[&(vll, wll)];
                    let (lvll, (mhrl, _nhrl)) = hopsl.iter().next().unwrap();
                    let nl = sol_by_hop[&(*lvll, *mhrl, vll, wll)];
                    cvec.push(geq(get(s, n), add(get(s, nl), &eps)));
                    eprintln!("s{} >= s{}+ε", n, nl);
                    obj = add(obj, square(sub(get(s, nl), get(s, n))));
                }
                if bundle_src_pos < v_outs.len()-1 {
                    eprint!("C3: ");
                    let (vlr, wlr) = v_outs[bundle_src_pos+1];
                    let hopsr = &hops_by_edge[&(vlr, wlr)];
                    let (lvlr, (mhrr, _nhrr)) = hopsr.iter().next().unwrap();
                    let nr = sol_by_hop[&(*lvlr, *mhrr, vlr, wlr)];
                    cvec.push(leq(get(s, n), sub(get(s, nr), &eps)));
                    eprintln!("s{} <= s{}-ε", n, nr);
                }
                if bundle_src_pos == v_outs.len()-1 {
                    eprint!("C4: ");
                    cvec.push(leq(get(s, n), sub(get(r, rn), &eps)));
                    eprintln!("s{} <= r{}-ε", n, rn);
                    obj = add(obj, square(sub(get(s, n), get(r, rn))));
                }

                // AGREEMENT
                for (i, (lvl, (mhr, nhr))) in hops.iter().enumerate() {
                    if i < hops.len() {
                        // s, t are really indexed by hops.
                        let n2 = sol_by_hop[&(*lvl, *mhr, *vl, *wl)];
                        let nd2 = sol_by_hop[&(lvl+1, *nhr, *vl, *wl)];
                        n = n2;
                        nd = nd2;
                        obj = add(obj, mul(thousand, square(sub(get(s,n), get(s,nd)))));
                    }
                }

                let last_hop = hops.iter().rev().next().unwrap();
                let (llvl, (_lmhr, lnhr)) = last_hop;
                let lnd = sol_by_loc[&(*llvl+1, *lnhr)];
                let rnd = lnd;
                
                // ORDER, SYMMETRY
                if bundle_dst_pos == 0 {
                    eprint!("C5: ");
                    cvec.push(geq(get(s, nd), add(get(l, lnd), &eps)));
                    eprintln!("s{} >= l{}+ε", nd, lnd);
                    obj = add(obj, square(sub(get(s, nd), get(l, lnd))));
                }
                if bundle_dst_pos > 0 {
                    eprint!("C6: ");
                    let (vll, wll) = w_ins[bundle_dst_pos-1];
                    let hopsl = &hops_by_edge[&(vll, wll)];
                    let (lvll, (_mhrl, nhrl)) = hopsl.iter().rev().next().unwrap();
                    let ndl = sol_by_hop[&(lvll+1, *nhrl, vll, wll)];
                    cvec.push(geq(get(s, nd), add(get(s, ndl), &eps)));
                    eprintln!("s{} >= s{}+ε", nd, ndl);
                    obj = add(obj, square(sub(get(s, ndl), get(s, nd))));
                }
                if bundle_dst_pos < w_ins.len()-1 {
                    eprint!("C7: ");
                    let (vlr, wlr) = w_ins[bundle_dst_pos+1];
                    let hopsr = &hops_by_edge[&(vlr, wlr)];
                    let (lvlr, (_mhrr, nhrr)) = hopsr.iter().rev().next().unwrap();
                    let ndr = sol_by_hop[&(lvlr+1, *nhrr, vlr, wlr)];
                    cvec.push(leq(get(s, nd), sub(get(s, ndr), &eps)));
                    eprintln!("s{} <= s{}-ε", nd, ndr);
                }
                if bundle_dst_pos == w_ins.len()-1 {
                    eprint!("C8: ");
                    cvec.push(leq(get(s, nd), sub(get(r, rnd), &eps)));
                    eprintln!("s{} <= r{}-ε", nd, rnd);
                    obj = add(obj, square(sub(get(s, nd), get(r, rnd))));
                }
            }

            // SPACE
            obj = sub(obj, mul(hundred, eps));
            obj = minimize.call1((obj,))?;
            
            let constr = PyList::new(py, cvec);

            eprintln!("OBJECTIVE: {}\n", obj.str()?);

            let prb = problem.call1((obj, constr))?;
            let is_dcp = prb.call_method1("is_dcp", ())?;
            eprintln!("IS_DCP: {}\n", is_dcp.str()?);

            prb.call_method1("solve", ())?;

            let lv = get_value(l)?;
            let lv = lv.as_slice().unwrap();

            let rv = get_value(r)?;
            let rv = rv.as_slice().unwrap();

            let sv = get_value(s)?;
            let sv = sv.as_slice().unwrap();

            eprintln!("L: {:.2?}\n", lv);
            eprintln!("R: {:.2?}\n", rv);
            eprintln!("S: {:.2?}\n", sv);

            Ok((lv.to_vec(), rv.to_vec(), sv.to_vec()))
        });
        eprintln!("PY: {:#?}", res);
        let (ls, rs, ss) = res.unwrap();

        // std::process::exit(0);

        let width_scale = 0.9;
        println!("{}", indoc!(r#"
        \documentclass[tikz,border=5mm]{standalone}
        \usetikzlibrary{graphs,graphdrawing,quotes,arrows.meta,calc,backgrounds,decorations.markings}
        \usegdlibrary{layered}
        \begin{document}
        
        \tikz[align=left, decoration={markings, mark=at position 0.5 with {\fill[red] (0, 0) circle (1pt);}}] {"#));

        for (loc, node) in loc_to_node.iter() {   
            let (ovr, ohr) = loc;
            let n = sol_by_loc[&(*ovr, *ohr)];

            let lpos = ls[n];
            let rpos = rs[n];

            let vpos = -1.5 * (*ovr as f64);
            let hpos = 10.0 * ((lpos + rpos) / 2.0);
            let width = 10.0 * width_scale * (rpos - lpos);

            match node {
                Loc::Node(vl) => {
                    println!(indoc!(r#"
                        \node[minimum width = {}cm, fill=white, fill opacity=0.9, draw, text opacity=1.0]({}) at ({}, {}) {{{}}};"#), 
                        width, vl, hpos, vpos, h_name[*vl]);
                },
                Loc::Hop(_, vl, wl) => {
                    let hn = sol_by_hop[&(*ovr, *ohr, **vl, **wl)];
                    let spos = ss[hn];
                    let hpos = 10.0 * spos;
                    println!(indoc!(r#"
                        \draw [fill, black] ({}, {}) circle (0.5pt);
                        \node[](aux_{}_{}) at ({}, {}) {{}};"#), 
                        hpos, vpos, ovr, ohr, hpos, vpos);
                },
            }
        }

        for cer in condensed.edge_references() {
            for (vl, wl, ew) in cer.weight().iter() {
                let (ovr, ohr) = node_to_loc[&Loc::Node(vl)];
                let (ovrd, ohrd) = node_to_loc[&Loc::Node(wl)];

                let snv = sol_by_hop[&(ovr, ohr, *vl, *wl)];
                let snw = sol_by_hop[&(ovrd, ohrd, *vl, *wl)];

                let sposv = ss[snv];
                let sposw = ss[snw];

                let nnv = sol_by_loc[&(ovr, ohr)];
                let nnw = sol_by_loc[&(ovrd, ohrd)];

                let lposv = ls[nnv];
                let lposw = ls[nnw];

                let rposv = rs[nnv];
                let rposw = rs[nnw];

                let src_width = rposv - lposv;
                let dst_width = rposw - lposw;

                let bundle_src_frac = ((((sposv - lposv) / src_width) - 0.5) / width_scale) + 0.5;
                let bundle_dst_frac = ((((sposw - lposw) / dst_width) - 0.5) / width_scale) + 0.5;

                let arr_src_frac = match *ew {
                    "actuates" => (bundle_src_frac) - (0.025 / src_width),
                    "senses" => (bundle_src_frac) + (0.025 / src_width),
                    _ => (bundle_src_frac),
                };
                let arr_dst_frac = match *ew {
                    "actuates" => bundle_dst_frac - (0.025 / dst_width),
                    "senses" => bundle_dst_frac + (0.025 / dst_width),
                    _ => bundle_dst_frac,
                };

                let hops = &hops_by_edge[&(*vl, *wl)];
                eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

                let dir = match *ew {
                    "actuates" | "rides" => "-{Stealth[]}",
                    "senses" => "{Stealth[]}-",
                    _ => "--",
                };

                let anchor = match *ew {
                    "actuates" => "north east",
                    "senses" => "south west",
                    _ => "south east",
                };

                match hops.len() {
                    0 => { unreachable!(); }
                    1 => {
                        println!(indoc!(r#"
                            \draw [{}, postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)        to[]  node[scale=0.8, anchor={}, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} ($({}.north west)!{}!({}.north east)$);"#),
                            dir, vl, arr_src_frac, vl, anchor, ew, wl, arr_dst_frac, wl    
                        );
                    },
                    2 => {
                        let (lvl, (_mhr, nhr)) = hops.iter().next().unwrap();
                        let (ovr, ohr) = (lvl+1, nhr);
                        println!(indoc!(r#"
                            \draw [rounded corners, {},postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$) -- node[scale=0.8, anchor={}, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] {{{}}} at ({},{}) -- ($({}.north west)!{}!({}.north east)$);"#),
                            dir, vl, arr_src_frac, vl, anchor, ew, ovr, ohr, wl, arr_dst_frac, wl    
                        );
                    },
                    max_levels => {
                        print!(indoc!(r#"\draw [rounded corners, {}, postaction={{decorate}}] ($({}.south west)!{}!({}.south east)$)"#), 
                            dir, vl, arr_src_frac, vl);
                        let mid = max_levels / 2;
                        let mut mid_ovr = 0;
                        let mut mid_ohr = 0;
                        let mut mid_ovrd = 0;
                        let mut mid_ohrd = 0;
                        for (n, hop) in hops.iter().enumerate() {
                            if n < max_levels-1 {
                                let (lvl, (mhr, nhr)) = hop;
                                let (ovr, ohr) = (lvl+1, nhr);
                                // let (ovr, ohr) = (lvl, mhr);
                                println!("% HOP {} {:?}", n, hop);
                                print!(r#" -- (aux_{}_{}.center)"#, ovr, ohr);
                                if n == mid {
                                    mid_ovr = *lvl;
                                    mid_ohr = *mhr;
                                    mid_ovrd = lvl+1;
                                    mid_ohrd = *nhr;
                                }
                            }
                        }
                        println!(indoc!(r#" -- ($({}.north west)!{}!({}.north east)$);"#), wl, arr_dst_frac, wl);
                        println!(indoc!(r#"\node[scale=0.8, anchor={}, fill=white, fill opacity = 0.8, text opacity = 1.0, draw, ultra thin] (mid_{}_{}_{}) at ($(aux_{}_{})!0.5!(aux_{}_{})$) {{{}}};"#), 
                            anchor, vl, wl, ew, mid_ovr, mid_ohr, mid_ovrd, mid_ohrd, ew);
                    },
                }
            }
        }

        use rand::Rng;
        let mut rng = rand::thread_rng();

        for (ovr, ohr, _shr, loc, _n) in all_locs.iter() {
            let n = sol_by_loc[&(*ovr, *ohr)];

            let lpos = ls[n];
            let rpos = rs[n];

            let vpos = -1.5 * (*ovr as f64) + 0.5 - rng.gen_range(0.25..0.75);
            let hpos = 10.0 * ((lpos + rpos) / 2.0);
            let hposl = 10.0 * lpos;
            let hposr = 10.0 * rpos;

            let loc_color = match loc {
                Loc::Node(_) => "red",
                Loc::Hop(_, _, _) => "blue",
            };
            let loc_str = match loc {
                Loc::Node(vl) => format!("{}", vl),
                Loc::Hop(_, vl, wl) => format!("{}{}", vl.chars().nth(0).unwrap(), wl.chars().nth(0).unwrap()),
            };

            println!(indoc!(r#"%\draw [{}] ({}, {}) circle (1pt);"#), loc_color, hpos, vpos);
            println!(indoc!(r#"%\draw [fill,violet] ({}, {}) circle (0.5pt);"#) , hposl, vpos);
            println!(indoc!(r#"%\draw [fill,orange] ({}, {}) circle (0.5pt);"#), hposr, vpos);
            println!(indoc!(r#"%\draw [--] ({},{}) -- ({}, {});"#), hposl, vpos, hposr, vpos);
            println!(indoc!(r#"%\node[scale=0.5, anchor=south west] at ({}, {}) {{{}}};"#), hpos, vpos, loc_str);
        }

        for ((lvl, _mhr, vl, wl), n) in sol_by_hop.iter() {
            let spos = ss[*n];

            // let vpos = -1.5 * (*lvl as f64) - 0.5 + rng.gen_range(0.5..1.0);
            let vpos = -1.5 * (*lvl as f64);
            let hpos = 10.0 * (spos);// + rng.gen_range(0.0..0.25);
            
            println!(indoc!(r#"%\draw [fill, black] ({}, {}) circle (1pt);"#), hpos, vpos);
            println!(indoc!(r#"%\node[scale=0.5, anchor=south east] at ({}, {}) {{{}{}}};"#), hpos, vpos, vl.chars().nth(0).unwrap(), wl.chars().nth(0).unwrap());
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