use petgraph::graph::{Graph, NodeIndex};
use petgraph::dot::{Dot};
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use pyo3::{prelude::*, prepare_freethreaded_python};
use pyo3::class::basic::CompareOp;
use petgraph::EdgeDirection::{Incoming, Outgoing};
use petgraph::algo::{floyd_warshall};
use pyo3::types::{PyModule, IntoPyDict, PyList};
use sorted_vec::SortedVec;
use std::collections::{HashMap, BTreeMap};
use std::collections::hash_map::{Entry};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::io::{Write};
use std::process::{Command, Stdio};
use ndarray::{Array2};
use numpy::{PyArray1, PyReadonlyArray1};
use tracing::{event, Level};
use crate::parser::Ident;
use crate::render::{Fact, Syn, resolve, unwrap_atom, find_parent, to_ident, as_string};

pub fn or_insert<V, E>(g: &mut Graph<V, E>, h: &mut HashMap<V, NodeIndex>, v: V) -> NodeIndex where V: Eq + Hash + Clone {
    let e = h.entry(v.clone());
    let ix = match e {
        Entry::Vacant(ve) => ve.insert(g.add_node(v)),
        Entry::Occupied(ref oe) => oe.get(),
    };
    // println!("OR_INSERT {} -> {:?}", v, ix);
    *ix
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

pub fn get(a: &PyAny, b: usize) -> &PyAny {
    a.get_item(b.into_py(a.py())).unwrap()
}

pub fn geq<'py>(a: &'py PyAny, b: &'py PyAny) -> &'py PyAny {
    a.rich_compare(b, CompareOp::Ge).unwrap()
}

pub fn leq<'py>(a: &'py PyAny, b: &'py PyAny) -> &'py PyAny {
    a.rich_compare(b, CompareOp::Le).unwrap()
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone)]
pub enum Loc<V,E> {
    Node(V),
    Hop(usize, E, E),
}

pub fn get_value(a: &PyAny) -> PyResult<PyReadonlyArray1<f64>> {
    Ok(a
        .getattr("value")?
        .extract::<&PyArray1<f64>>()?
        .readonly())
}

pub struct Vcg<'s> {
    /// vert is a vertical constraint graph. 
    /// Edges (v, w) in vert indicate that v needs to be placed above w. 
    /// Node weights must be unique.
    pub vert: Graph<&'s str, &'s str>,

    /// v_nodes maps node weights in vert to node-indices.
    pub v_nodes: HashMap<&'s str, NodeIndex>,

    /// h_name maps node weights to display names/labels.
    pub h_name: HashMap<&'s str, String>,
}

pub fn calculate_vcg<'s>(v: &'s [Syn], draw: &'s Fact) -> Vcg<'s> {
    // println!("draw:\n{:#?}\n\n", draw);

    let res = resolve(v.iter(), draw);

    // println!("resolution: {:?}\n", res);

    let mut vert = Graph::<&str, &str>::new();

    let mut v_nodes = HashMap::<&str, NodeIndex>::new();

    let mut h_name = HashMap::new();

    for hint in res {
        if let Fact::Fact(Ident(style), items) = hint {
            for item in items {
                let item_ident = unwrap_atom(item).unwrap();
                let resolved_item = resolve(v.iter(), item).collect::<Vec<&Fact>>();
                let name = as_string(&resolved_item, &Ident("name"), item_ident.into());

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
                    // hide? elide? skip?
                    // autodraw?
                    _ => {
                        unimplemented!("{}", style);
                    }
                }
                
                event!(Level::DEBUG, %item_ident, %style, ?resolved_actuates, ?resolved_senses, ?resolved_hosts, "READ");
            }
        }
    }

    Vcg{vert, v_nodes, h_name}
}

pub struct Cvcg<V: Clone + Debug + Ord + Hash, E: Clone + Debug + Ord> {
    pub condensed: Graph<V, SortedVec<(V, V, E)>>,
    pub condensed_vxmap: HashMap::<V, NodeIndex>
}

pub fn condense<V: Clone + Debug + Ord + Hash, E: Clone + Debug + Ord>(vert: &Graph<V, E>) -> Cvcg<V,E> {
    let mut condensed = Graph::<V, SortedVec<(V, V, E)>>::new();
    let mut condensed_vxmap = HashMap::new();
    for (vx, vl) in vert.node_references() {
        let mut dsts = HashMap::new();
        for er in vert.edges_directed(vx, Outgoing) {
            let wx = er.target();
            let wl = vert.node_weight(wx).unwrap();
            dsts.entry(wl).or_insert_with(SortedVec::new).insert((vl.clone(), wl.clone(), (*er.weight()).clone()));
        }
        
        let cvx = or_insert(&mut condensed, &mut condensed_vxmap, vl.clone());
        for (wl, exs) in dsts {
            let cwx = or_insert(&mut condensed, &mut condensed_vxmap, wl.clone());
            condensed.add_edge(cvx, cwx, exs);
        }
    }
    let dot = Dot::new(&condensed);
    event!(Level::DEBUG, ?dot, "CONDENSED");
    Cvcg{condensed, condensed_vxmap}
}

pub fn roots<V: Clone + Debug + Ord, E>(dag: &Graph<V, E>) -> SortedVec<V> {
    let roots = SortedVec::from_unsorted(
        dag
            .externals(Incoming)
            .map(|vx| dag.node_weight(vx).unwrap().clone())
            .collect::<Vec<_>>()
    );
    event!(Level::DEBUG, ?roots, "ROOTS");
    roots
}

pub fn rank<'s, V: Clone + Debug + Ord, E>(dag: &'s Graph<V, E>, roots: &'s SortedVec<V>) -> BTreeMap<usize, SortedVec<(V, V)>> {
    let paths_fw = floyd_warshall(&dag, |_ex| { -1 }).unwrap();

    let paths_fw2 = SortedVec::from_unsorted(
        paths_fw
            .iter()
            .map(|((vx, wx), wgt)| {
                let vl = (*dag.node_weight(*vx).unwrap()).clone();
                let wl = (*dag.node_weight(*wx).unwrap()).clone();
                (*wgt, vl, wl)
            }).collect::<Vec<_>>()
    );
    event!(Level::DEBUG, ?paths_fw2, "FLOYD-WARSHALL");

    let paths_from_roots = SortedVec::from_unsorted(
        paths_fw2
            .iter()
            .filter_map(|(wgt, vl, wl)| {
                if *wgt <= 0 && roots.contains(vl) {
                    Some((-(*wgt) as usize, vl.clone(), wl.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    );
    event!(Level::DEBUG, ?paths_from_roots, "PATHS_FROM_ROOTS");

    let mut paths_by_rank = BTreeMap::new();
    for (wgt, vx, wx) in paths_from_roots.iter() {
        paths_by_rank
            .entry(*wgt)
            .or_insert_with(SortedVec::new)
            .insert((vx.clone(), wx.clone()));
    }
    event!(Level::DEBUG, ?paths_by_rank, "PATHS_BY_RANK");

    paths_by_rank
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Hop<V: Clone + Debug + Display + Ord + Hash> {
    pub mhr: usize,
    pub nhr: usize,
    pub vl: V,
    pub wl: V,
    pub lvl: usize,
}
pub struct Placement<V: Clone + Debug + Display + Ord + Hash> {
    pub locs_by_level: BTreeMap<usize, Vec<usize>>, 
    pub hops_by_level: BTreeMap<usize, SortedVec<Hop<V>>>,
    pub hops_by_edge: BTreeMap<(V, V), BTreeMap<usize, (usize, usize)>>,
    pub loc_to_node: HashMap<(usize, usize), Loc<V, V>>,
    pub node_to_loc: HashMap<Loc<V, V>, (usize, usize)>
}

pub fn calculate_locs_and_hops<'s, V, E>(
    dag: &'s Graph<V, E>, 
    paths_by_rank: &'s BTreeMap<usize, SortedVec<(V, V)>>
) -> Placement<V>
        where 
    V: Clone + Debug + Display + Ord + Hash, 
    E: Clone + Debug + Ord 
{
    let mut vx_rank = HashMap::new();
    let mut hx_rank = HashMap::new();
    for (rank, paths) in paths_by_rank.iter() {
        for (n, (_vx, wx)) in paths.iter().enumerate() {
            vx_rank.insert(wx.clone(), *rank);
            hx_rank.insert(wx.clone(), n);
        }
    }

    let mut loc_to_node = HashMap::new();
    let mut node_to_loc = HashMap::new();

    let mut locs_by_level = BTreeMap::new();
    for (rank, paths) in paths_by_rank.iter() {
        let l = *rank;
        for (a, (_cvl, cwl)) in paths.iter().enumerate() {
            locs_by_level.entry(l).or_insert_with(Vec::new).push(a);
            loc_to_node.insert((l, a), Loc::Node(cwl.clone()));
            node_to_loc.insert(Loc::Node(cwl.clone()), (l, a));
        }
    }

    event!(Level::DEBUG, ?locs_by_level, "LOCS_BY_LEVEL V1");

    let sorted_condensed_edges = SortedVec::from_unsorted(
        dag
            .edge_references()
            .map(|er| {
                let (vx, wx) = (er.source(), er.target());
                let vl = dag.node_weight(vx).unwrap();
                let wl = dag.node_weight(wx).unwrap();
                (vl.clone(), wl.clone(), er.weight())
            })
            .collect::<Vec<_>>()
    );

    event!(Level::DEBUG, ?sorted_condensed_edges, "CONDENSED GRAPH");

    let mut hops_by_edge = BTreeMap::new();
    let mut hops_by_level = BTreeMap::new();
    for (vl, wl, _) in sorted_condensed_edges.iter() {
        let vvr = *vx_rank.get(vl).unwrap();
        let wvr = *vx_rank.get(wl).unwrap();
        let vhr = *hx_rank.get(vl).unwrap();
        let whr = *hx_rank.get(wl).unwrap();
        
        let mut mhrs = vec![vhr];
        for mid_level in (vvr+1)..(wvr) {
            let mhr = locs_by_level.get(&mid_level).map_or(0, |v| v.len());
            locs_by_level.entry(mid_level).or_insert_with(Vec::new).push(mhr);
            loc_to_node.insert((mid_level, mhr), Loc::Hop(mid_level, vl.clone(), wl.clone()));
            node_to_loc.insert(Loc::Hop(mid_level, vl.clone(), wl.clone()), (mid_level, mhr)); // BUG: what about the endpoints?
            mhrs.push(mhr);
        }
        mhrs.push(whr);

        event!(Level::DEBUG, %vl, %wl, %vvr, %wvr, %vhr, %whr, ?mhrs, "HOP");
        
        for lvl in vvr..wvr {
            let mx = (lvl as i32 - vvr as i32) as usize;
            let nx = (lvl as i32 + 1 - vvr as i32) as usize;
            let mhr = mhrs[mx];
            let nhr = mhrs[nx];
            hops_by_level
                .entry(lvl)
                .or_insert_with(SortedVec::new)
                .insert(Hop{mhr, nhr, vl: vl.clone(), wl: wl.clone(), lvl});
            hops_by_edge
                .entry((vl.clone(), wl.clone()))
                .or_insert_with(BTreeMap::new)
                .insert(lvl, (mhr, nhr));
        }
    }
    event!(Level::DEBUG, ?locs_by_level, "LOCS_BY_LEVEL V2");

    event!(Level::DEBUG, ?hops_by_level, "HOPS_BY_LEVEL");

    let mut g_hops = Graph::<(usize, usize), (usize, V, V)>::new();
    let mut g_hops_vx = HashMap::new();
    for (_rank, hops) in hops_by_level.iter() {
        for Hop{mhr, nhr, vl, wl, lvl} in hops.iter() {
            let gvx = or_insert(&mut g_hops, &mut g_hops_vx, (*lvl, *mhr));
            let gwx = or_insert(&mut g_hops, &mut g_hops_vx, (lvl+1, *nhr));
            g_hops.add_edge(gvx, gwx, (*lvl, vl.clone(), wl.clone()));
        }
    }
    let g_hops_dot = Dot::new(&g_hops);
    event!(Level::DEBUG, ?g_hops_dot, "HOPS GRAPH");

    Placement{locs_by_level, hops_by_level, hops_by_edge, loc_to_node, node_to_loc}
}

pub fn minimize_edge_crossing<'s, V>(
    locs_by_level: &'s BTreeMap<usize, Vec<usize>>,
    hops_by_level: &'s BTreeMap<usize, SortedVec<Hop<V>>>
) -> BTreeMap<usize, BTreeMap<usize, usize>> where
    V: Clone + Debug + Display + Ord + Hash
{
    let max_level = *hops_by_level.keys().max().unwrap();
    let max_width = hops_by_level.values().map(|paths| paths.len()).max().unwrap();

    event!(Level::DEBUG, %max_level, %max_width, "max_level, max_width");

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
            for Hop{mhr: u1, nhr: v1, ..} in hops.iter() {
                for Hop{mhr: u2, nhr: v2, ..}  in hops.iter() {
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
            csp.push(',');
        }
        csp.push_str(&format!("c{}[_,_,_,_]", rank));
    }
    csp.push_str("],csum)\n");
    csp.push_str("sumgeq([");
    for rank in 0..max_level {
        if rank > 0 {
            csp.push(',');
        }
        csp.push_str(&format!("c{}[_,_,_,_]", rank));
    }
    csp.push_str("],csum)\n");
    csp.push_str("\n\n**EOF**");

    event!(Level::DEBUG, %csp, "CSP");


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

    let output = child
        .wait_with_output()
        .expect("failed to wait on child");

    let outs = std::str::from_utf8(&output.stdout[..]).unwrap();

    event!(Level::DEBUG, %outs, "CSP OUT");

    // std::process::exit(0);

    let lines = outs.split('\n').collect::<Vec<_>>();
    let cn_line = lines[2];
    event!(Level::DEBUG, %cn_line, "cn line");
    
    let crossing_number = cn_line
        .trim()
        .parse::<i32>()
        .expect("unable to parse crossing number");

    // std::process::exit(0);
    
    let solns = &lines[3..lines.len()];
    
    event!(Level::DEBUG, ?lines, ?solns, ?crossing_number, "LINES, SOLNS, CN");

    let mut perm = Vec::<Array2<i32>>::new();
    for (rank, locs) in locs_by_level.iter() {
        let mut arr = Array2::<i32>::zeros((locs.len(), locs.len()));
        let parsed_solns = solns[*rank]
            .split(' ')
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

    // for (n, p) in perm.iter().enumerate() {
    //     eprintln!("{}:\n{:?}\n", n, p);
    // };

    let mut solved_locs = BTreeMap::new();
    for (n, p) in perm.iter().enumerate() {
        let mut sums = p.rows().into_iter().enumerate().map(|(i, r)| (i, r.sum() as usize)).collect::<Vec<_>>();
        sums.sort_by_key(|(_i,s)| *s);
        // eprintln!("row sums: {:?}", sums);
        for (nhr, (i,_s)) in sums.into_iter().enumerate() {
            solved_locs.entry(n).or_insert_with(BTreeMap::new).insert(i, nhr);
        }
    }
    event!(Level::DEBUG, ?solved_locs, "SOLVED_LOCS");

    solved_locs
}

pub struct LocRow<V: Clone + Debug + Display + Ord + Hash> {
    pub ovr: usize,
    pub ohr: usize,
    pub shr: usize,
    pub loc: Loc<V, V>,
    pub n: usize,
}

pub struct HopRow<V: Clone + Debug + Display + Ord + Hash> {
    pub lvl: usize,
    pub mhr: usize,
    pub nhr: usize,
    pub vl: V,
    pub wl: V,
    pub n: usize,
}

pub struct LayoutProblem<V: Clone + Debug + Display + Ord + Hash> {
    pub all_locs: Vec<LocRow<V>>,
    pub all_hops0: Vec<HopRow<V>>,
    pub all_hops: Vec<HopRow<V>>,
    pub sol_by_loc: HashMap<(usize, usize), usize>,
    pub sol_by_hop: HashMap<(usize, usize, V, V), usize>,
}

/// ovr, ohr
pub type LocIx = (usize, usize);

/// ovr, ohr -> loc
pub type LocNodeMap<V> = HashMap<LocIx, Loc<V, V>>;

/// lvl -> (mhr, nhr)
pub type HopMap = BTreeMap<usize, (usize, usize)>;

pub fn calculate_sols<'s, V>(
    solved_locs: &'s BTreeMap<usize, BTreeMap<usize, usize>>,
    loc_to_node: &'s HashMap<LocIx, Loc<V, V>>,
    hops_by_level: &'s BTreeMap<usize, SortedVec<Hop<V>>>,
    hops_by_edge: &'s BTreeMap<(V, V), HopMap>,
) -> LayoutProblem<V> where
    V: Clone + Debug + Display + Ord + Hash
{
    let all_locs = solved_locs
        .iter()
        .flat_map(|(ovr, nodes)| nodes
            .iter()
            .map(|(ohr, shr)| (*ovr, *ohr, *shr, &loc_to_node[&(*ovr,*ohr)])))
        .enumerate()
        .map(|(n, (ovr, ohr, shr, loc))| LocRow{ovr, ohr, shr, loc: loc.clone(), n})
        .collect::<Vec<_>>();

    let mut sol_by_loc = HashMap::new();
    for LocRow{ovr, ohr, shr: _, loc: _, n} in all_locs.iter() {
        sol_by_loc.insert((*ovr, *ohr), *n);
    }

    let all_hops0 = hops_by_level
        .iter()
        .flat_map(|h| 
            h.1.iter().map(|Hop{mhr, nhr, vl, wl, lvl}| {
                (*mhr, *nhr, vl.clone(), wl.clone(), *lvl)
            })
        ).enumerate()
        .map(|(n, (mhr, nhr, vl, wl, lvl))| {
            HopRow{lvl, mhr, nhr, vl, wl, n}
        })
        .collect::<Vec<_>>();
    let all_hops = hops_by_level
        .iter()
        .flat_map(|h| 
            h.1.iter().map(|Hop{mhr, nhr, vl, wl, lvl}| {
                (*mhr, *nhr, vl.clone(), wl.clone(), *lvl)
            })
        )
        .chain(
            hops_by_edge.iter().map(|((vl, wl), hops)| {
                    let (lvl, (_mhr, nhr)) = hops.iter().rev().next().unwrap();
                    (*nhr, std::usize::MAX, vl.clone(), wl.clone(), lvl+1)
            }) 
        )
        .enumerate()
        .map(|(n, (mhr, nhr, vl, wl, lvl))| {
            HopRow{lvl, mhr, nhr, vl, wl, n}
        })
        .collect::<Vec<_>>();
    
    let mut sol_by_hop = HashMap::new();
    for HopRow{lvl, mhr, nhr: _, vl, wl, n} in all_hops.iter() {
        sol_by_hop.insert((*lvl, *mhr, vl.clone(), wl.clone()), *n);
    }

    LayoutProblem{all_locs, all_hops0, all_hops, sol_by_loc, sol_by_hop}
}

pub fn position_sols<'s, V, E>(
    dag: &'s Graph<V, E>,
    dag_map: &'s HashMap::<V, NodeIndex>,
    hops_by_edge: &'s BTreeMap<(V, V), HopMap>,
    node_to_loc: &'s HashMap::<Loc<V, V>, (usize, usize)>,
    solved_locs: &'s BTreeMap<usize, BTreeMap<usize, usize>>,
    layout_problem: &'s LayoutProblem<V>,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>
) where 
    V: Clone + Debug + Display + Ord + Hash,
    E: Clone + Debug
{
    let LayoutProblem{all_locs, all_hops0, all_hops, sol_by_loc, sol_by_hop} = layout_problem;
    
    let num_locs = all_locs.len();
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

        for LocRow{ovr, ohr, ..} in all_locs.iter() {
            let ovr = *ovr; 
            let ohr = *ohr;
            let locs = &solved_locs[&ovr];
            let shr = locs[&ohr];
            let n = sol_by_loc[&(ovr, ohr)];
            // if loc.is_node() { 
                // eprint!("C0: ");
                // eprint!("r{} >= l{} + S, ", n, n);
                cvec.push(geq(get(r,n), add(get(l,n), sep)));
                if shr > 0 {
                    let ohrp = locs.iter().position(|(_, shrp)| *shrp == shr-1).unwrap();
                    let np = sol_by_loc[&(ovr, ohrp)];
                    // let _shrp = locs[&ohrp];
                    cvec.push(geq(get(l, n), get(r, np)));
                    // eprint!("l{:?} >= r{:?}, ", (ovr, ohr, shr, n), (ovr, ohrp, shrp, np));
                }
                if shr < locs.len()-1 {
                    let ohrn = locs.iter().position(|(_, shrp)| *shrp == shr+1).unwrap();
                    let nn = sol_by_loc[&(ovr,ohrn)];
                    // let _shrn = locs[&(ohr+1)];
                    cvec.push(leq(get(r,n), get(l,nn)));
                    // eprint!("r{:?} <= l{:?}, ", (ovr, ohr, shr, n), (ovr, ohrn, shrn, nn));
                }
                if shr == locs.len()-1 {
                    cvec.push(leq(get(r,n), one));
                    // eprint!("r{:?} <= 1", (ovr, ohr, shr, n));
                }
                // eprintln!();  
            // }
        }
        for HopRow{lvl, mhr, nhr, vl, wl, ..} in all_hops0.iter() {
            let v_ers = dag.edges_directed(dag_map[vl], Outgoing).into_iter().collect::<Vec<_>>();
            let w_ers = dag.edges_directed(dag_map[wl], Incoming).into_iter().collect::<Vec<_>>();

            let mut v_dsts = v_ers.iter().map(|er| { (*dag.node_weight(er.target()).unwrap()).clone() }).collect::<Vec<_>>();
            let mut w_srcs = w_ers.iter().map(|er| { (*dag.node_weight(er.source()).unwrap()).clone() }).collect::<Vec<_>>();

            v_dsts.sort(); v_dsts.dedup();
            v_dsts.sort_by_key(|dst| {
                let (ovr, ohr) = node_to_loc[&Loc::Node(dst.clone())];
                let (svr, shr) = (ovr, solved_locs[&ovr][&ohr]);
                (shr, -(svr as i32))
            });
            let v_outs = v_dsts
                .iter()
                .map(|dst| { (vl.clone(), dst.clone()) })
                .collect::<Vec<_>>();

            w_srcs.sort(); w_srcs.dedup();
            w_srcs.sort_by_key(|src| {
                let (ovr, ohr) = node_to_loc[&Loc::Node(src.clone())];
                let (svr, shr) = (ovr, solved_locs[&ovr][&ohr]);
                (shr, -(svr as i32))
            });
            let w_ins = w_srcs
                .iter()
                .map(|src| { (src.clone(), wl.clone()) })
                .collect::<Vec<_>>();

            let bundle_src_pos = v_outs.iter().position(|e| { *e == (vl.clone(), wl.clone()) }).unwrap();
            let bundle_dst_pos = w_ins.iter().position(|e| { *e == (vl.clone(), wl.clone()) }).unwrap();
            let hops = &hops_by_edge[&(vl.clone(), wl.clone())];
            // eprintln!("lvl: {}, vl: {}, wl: {}, hops: {:?}", lvl, vl, wl, hops);
            
            let ln = sol_by_loc[&(*lvl, *mhr)];
            let rn = ln;
            let n = sol_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())];
            let nd = sol_by_hop[&(lvl+1, *nhr, vl.clone(), wl.clone())];
            let mut n = n;
            let mut nd = nd;

            // ORDER, SYMMETRY
            if bundle_src_pos == 0 {
                // eprint!("C1: ");
                cvec.push(geq(get(s, n), add(get(l, ln), eps)));
                // eprintln!("s{} >= l{}+ε", n, ln);
                obj = add(obj, square(sub(get(s, n), get(l, ln))));
            }
            if bundle_src_pos > 0 {
                // eprint!("C2: ");
                let (vll, wll) = &v_outs[bundle_src_pos-1];
                let hopsl = &hops_by_edge[&(vll.clone(), wll.clone())];
                let (lvll, (mhrl, _nhrl)) = hopsl.iter().next().unwrap();
                let nl = sol_by_hop[&(*lvll, *mhrl, vll.clone(), wll.clone())];
                cvec.push(geq(get(s, n), add(get(s, nl), eps)));
                // eprintln!("s{} >= s{}+ε", n, nl);
                obj = add(obj, square(sub(get(s, nl), get(s, n))));
            }
            if bundle_src_pos < v_outs.len()-1 {
                // eprint!("C3: ");
                let (vlr, wlr) = &v_outs[bundle_src_pos+1];
                let hopsr = &hops_by_edge[&(vlr.clone(), wlr.clone())];
                let (lvlr, (mhrr, _nhrr)) = hopsr.iter().next().unwrap();
                let nr = sol_by_hop[&(*lvlr, *mhrr, vlr.clone(), wlr.clone())];
                cvec.push(leq(get(s, n), sub(get(s, nr), eps)));
                // eprintln!("s{} <= s{}-ε", n, nr);
            }
            if bundle_src_pos == v_outs.len()-1 {
                // eprint!("C4: ");
                cvec.push(leq(get(s, n), sub(get(r, rn), eps)));
                // eprintln!("s{} <= r{}-ε", n, rn);
                obj = add(obj, square(sub(get(s, n), get(r, rn))));
            }

            // AGREEMENT
            for (i, (lvl, (mhr, nhr))) in hops.iter().enumerate() {
                if i < hops.len() {
                    // s, t are really indexed by hops.
                    let n2 = sol_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())];
                    let nd2 = sol_by_hop[&(lvl+1, *nhr, vl.clone(), wl.clone())];
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
                // eprint!("C5: ");
                cvec.push(geq(get(s, nd), add(get(l, lnd), eps)));
                // eprintln!("s{} >= l{}+ε", nd, lnd);
                obj = add(obj, square(sub(get(s, nd), get(l, lnd))));
            }
            if bundle_dst_pos > 0 {
                // eprint!("C6: ");
                let (vll, wll) = &w_ins[bundle_dst_pos-1];
                let hopsl = &hops_by_edge[&(vll.clone(), wll.clone())];
                let (lvll, (_mhrl, nhrl)) = hopsl.iter().rev().next().unwrap();
                let ndl = sol_by_hop[&(lvll+1, *nhrl, vll.clone(), wll.clone())];
                cvec.push(geq(get(s, nd), add(get(s, ndl), eps)));
                // eprintln!("s{} >= s{}+ε", nd, ndl);
                obj = add(obj, square(sub(get(s, ndl), get(s, nd))));
            }
            if bundle_dst_pos < w_ins.len()-1 {
                // eprint!("C7: ");
                let (vlr, wlr) = &w_ins[bundle_dst_pos+1];
                let hopsr = &hops_by_edge[&(vlr.clone(), wlr.clone())];
                let (lvlr, (_mhrr, nhrr)) = hopsr.iter().rev().next().unwrap();
                let ndr = sol_by_hop[&(lvlr+1, *nhrr, vlr.clone(), wlr.clone())];
                cvec.push(leq(get(s, nd), sub(get(s, ndr), eps)));
                // eprintln!("s{} <= s{}-ε", nd, ndr);
            }
            if bundle_dst_pos == w_ins.len()-1 {
                // eprint!("C8: ");
                cvec.push(leq(get(s, nd), sub(get(r, rnd), eps)));
                // eprintln!("s{} <= r{}-ε", nd, rnd);
                obj = add(obj, square(sub(get(s, nd), get(r, rnd))));
            }
        }

        // SPACE
        obj = sub(obj, mul(hundred, eps));
        obj = minimize.call1((obj,))?;
        
        let constr = PyList::new(py, cvec);

        event!(Level::DEBUG, ?obj, "OBJECTIVE");

        let prb = problem.call1((obj, constr))?;
        let is_dcp = prb.call_method1("is_dcp", ())?;
        let is_dcp_str = is_dcp.str()?;
        event!(Level::DEBUG, ?is_dcp_str, "IS_DCP");

        prb.call_method1("solve", ())?;

        let lv = get_value(l)?;
        let lv = lv.as_slice().unwrap();

        let rv = get_value(r)?;
        let rv = rv.as_slice().unwrap();

        let sv = get_value(s)?;
        let sv = sv.as_slice().unwrap();

        // eprintln!("L: {:.2?}\n", lv);
        // eprintln!("R: {:.2?}\n", rv);
        // eprintln!("S: {:.2?}\n", sv);

        Ok((lv.to_vec(), rv.to_vec(), sv.to_vec()))
    });
    event!(Level::DEBUG, ?res, "PY");
    res.unwrap()
}