#![deny(clippy::unwrap_used)]

pub mod error {
    use std::ops::Range;

    use petgraph::algo::NegativeCycle;
    use tracing_error::{TracedError, ExtractSpanTrace, SpanTrace, InstrumentError};


    #[non_exhaustive]
    #[derive(Debug, thiserror::Error)]
    pub enum Kind {
        #[error("indexing error")]
        IndexingError {},
        #[error("key not found error")]
        KeyNotFoundError {key: String},
        #[error("missing drawing error")]
        MissingDrawingError {},
        #[error("missing fact error")]
        MissingFactError {ident: String},
        #[error("unimplemented drawing style error")]
        UnimplementedDrawingStyleError { style: String },
        #[error("pomelo error")]
        PomeloError { span: Range<usize>, text: String },
    }
    
    #[non_exhaustive]
    #[derive(Debug, thiserror::Error)]
    pub enum RankingError {
        #[error("negative cycle error")]
        NegativeCycleError{cycle: NegativeCycle},
        #[error("io error")]
        IoError{#[from] source: std::io::Error},
        #[error("utf8 error")]
        Utf8Error{#[from] source: std::str::Utf8Error},
    }
    
    #[non_exhaustive]
    #[derive(Debug, thiserror::Error)]
    pub enum LayoutError {
        #[error("cvxpy error")]
        CvxpyError{#[from] source: pyo3::PyErr},
    }
    
    #[non_exhaustive]
    #[derive(Debug, thiserror::Error)]
    pub enum TypeError {
        #[error("deep name error")]
        DeepNameError{name: String},
        #[error("unknown mode")]
        UnknownModeError{mode: String},
    }
    
    #[non_exhaustive]
    #[derive(Debug, thiserror::Error)]
    pub enum Error {
        #[error(transparent)]
        ParsingError{
            #[from] source: TracedError<nom::Err<String>>,
        },
        #[error(transparent)]
        TypeError{
            #[from] source: TracedError<TypeError>,
        },
        #[error(transparent)]
        GraphDrawingError{
            #[from] source: TracedError<Kind>,
        },
        #[error(transparent)]
        RankingError{
            #[from] source: TracedError<RankingError>,
        },
        #[error(transparent)]
        LayoutError{
            #[from] source: TracedError<LayoutError>,
        }
    }
    
    impl From<Kind> for Error {
        fn from(source: Kind) -> Self {
            Self::GraphDrawingError {
                source: source.into(),
            }
        }
    }
    
    impl ExtractSpanTrace for Error {
        fn span_trace(&self) -> Option<&SpanTrace> {
            use std::error::Error as _;
            match self {
                Error::ParsingError{source} => source.source().and_then(ExtractSpanTrace::span_trace),
                Error::TypeError{source} => source.source().and_then(ExtractSpanTrace::span_trace),
                Error::GraphDrawingError{source} => source.source().and_then(ExtractSpanTrace::span_trace),
                Error::RankingError{source} => source.source().and_then(ExtractSpanTrace::span_trace),
                Error::LayoutError{source} => source.source().and_then(ExtractSpanTrace::span_trace),
            }
        }
    }

    pub trait OrErrExt<E> {
        type Item;
        fn or_err(self, error: E) -> Result<Self::Item, Error>;
    }

    impl<V, E> OrErrExt<E> for Option<V> where tracing_error::TracedError<E>: From<E>, Error: From<tracing_error::TracedError<E>> {
        type Item = V;
        fn or_err(self, error: E) -> Result<V, Error> {
            self.ok_or_else(|| Error::from(error.in_current_span()))
        }
    }

    pub trait OrErrMutExt {
        type Item;
        fn or_err_mut(&mut self) -> Result<&mut Self::Item, Error>;
    }

    impl<V> OrErrMutExt for Option<V> {
        type Item = V;
        fn or_err_mut(&mut self) -> Result<&mut V, Error> {
            match self {
                Some(ref mut inner) => {
                    Ok(inner)
                },
                None => {
                    Err(Error::from(Kind::IndexingError{}.in_current_span()))
                },
            }
        }
    }
}

pub mod graph {
    use std::fmt::Debug;

    use petgraph::{EdgeDirection::Incoming, Graph};
    use sorted_vec::SortedVec;
    use tracing::{event, Level};

    use super::error::{Error, Kind, OrErrExt};

    pub fn roots<V: Clone + Debug + Ord, E>(dag: &Graph<V, E>) -> Result<SortedVec<V>, Error> {
        let roots = dag
            .externals(Incoming)
            .map(|vx| dag.node_weight(vx).or_err(Kind::IndexingError{}).map(Clone::clone))
            .into_iter()
            .collect::<Result<Vec<_>, Error>>()?;
        let roots = SortedVec::from_unsorted(roots);
        event!(Level::DEBUG, ?roots, "ROOTS");
        Ok(roots)
    }
}

pub mod layout {
    use std::borrow::Cow;
    use std::collections::BTreeMap;
    use std::collections::{HashMap, hash_map::Entry};
    use std::fmt::{Debug, Display};
    use std::hash::Hash;
    
    use petgraph::EdgeDirection::Outgoing;
    use petgraph::algo::floyd_warshall;
    use petgraph::dot::Dot;
    use petgraph::graph::{Graph, NodeIndex};
    use petgraph::visit::{EdgeRef, IntoNodeReferences};
    use sorted_vec::SortedVec;
    use tracing::{event, Level, instrument};
    use tracing_error::InstrumentError;
    use typed_index_collections::TiVec;

    use crate::graph_drawing::error::{Error, Kind, OrErrExt, RankingError};
    use crate::graph_drawing::graph::roots;
    use crate::parser::{Fact, Item, Labels};

    #[derive(Clone, Debug)]
    pub struct Vcg<V, E> {
        /// vert is a vertical constraint graph. 
        /// Edges (v, w) in vert indicate that v needs to be placed above w. 
        /// Node weights must be unique.
        pub vert: Graph<V, E>,
    
        /// vert_vxmap maps node weights in vert to node-indices.
        pub vert_vxmap: HashMap<V, NodeIndex>,
    
        /// vert_node_labels maps node weights in vert to display names/labels.
        pub vert_node_labels: HashMap<V, String>,
    
        /// vert_edge_labels maps (v,w,rel) node weight pairs to display edge labels.
        pub vert_edge_labels: HashMap<V, HashMap<V, HashMap<V, Vec<E>>>>,
    }

    pub fn or_insert<V, E>(g: &mut Graph<V, E>, h: &mut HashMap<V, NodeIndex>, v: V) -> NodeIndex where V: Eq + Hash + Clone {
        let e = h.entry(v.clone());
        let ix = match e {
            Entry::Vacant(ve) => ve.insert(g.add_node(v)),
            Entry::Occupied(ref oe) => oe.get(),
        };
        // println!("OR_INSERT {} -> {:?}", v, ix);
        *ix
    }

    pub trait Trim {
        fn trim(self) -> Self;
    }

    impl Trim for &str {
        fn trim(self) -> Self {
            str::trim(self)
        }
    }

    impl Trim for String {
        fn trim(self) -> Self {
            str::trim(&self).into()
        }
    }

    impl<'s> Trim for Cow<'s, str> {
        fn trim(self) -> Self {
            match self {
                Cow::Borrowed(b) => {
                    Cow::Borrowed(b.trim())
                },
                Cow::Owned(o) => {
                    Cow::Owned(o.trim())
                }
            }
        }
    }

    pub trait IsEmpty {
        fn is_empty(&self) -> bool;
    }

    impl IsEmpty for &str {
        fn is_empty(&self) -> bool {
            str::is_empty(self)
        }
    }

    impl IsEmpty for String {
        fn is_empty(&self) -> bool {
            String::is_empty(self)
        }
    }

    impl<'s> IsEmpty for Cow<'s, str> {
        fn is_empty(&self) -> bool {
            match self {
                Cow::Borrowed(b) => b.is_empty(),
                Cow::Owned(o) => o.is_empty(),
            }
        }
    }

    pub trait Len {
        fn len(&self) -> usize;
    }

    impl Len for &str {
        fn len(&self) -> usize {
            str::len(self)
        }
    }

    impl Len for String {
        fn len(&self) -> usize {
            str::len(&self)
        }
    }

    impl<'s> Len for Cow<'s, str> {
        fn len(&self) -> usize {
            str::len(&self)
        }
    }


    // fn levels_colon_helper<'s>(lvls: &mut Vec<(Labels<&'s str>, Labels<&'s str>)>, b: &Body<'s>) {
    //     if let Body::Slash(up, down) = &b {
    //         let up = up.0.iter().map(|s| Some(*s)).collect::<Vec<_>>();
    //         let down = down.0.iter().map(|s| Some(*s)).collect::<Vec<_>>();
    //         lvls.push((up, down));
    //     }
    //     if let Body::Nest(a, b) = b {
    //         levels_colon_helper(lvls, a);
    //         levels_helper(lvls, b);
    //     }
    // }

    // fn levels_helper<'s>(lvls: &mut Vec<(Labels<&'s str>, Labels<&'s str>)>, body: &Body<'s>) {
    //     match body {
    //         Body::Colon(Item(i, Some(b))) => {
    //             match b.as_ref() {
    //                 Body::Colon(_) => {
    //                     levels_helper(lvls, b);
    //                 },
    //                 Body::Sq(_) | Body::Br(_) => {
    //                     let up = i.iter().map(|s| Some(*s)).collect::<Vec<_>>();
    //                     lvls.push((up, vec![]));
    //                 },
    //                 Body::Slash(_, _) | Body::Nest(_, _) => {
    //                     levels_colon_helper(lvls, b); 
    //                 },
    //             }
    //         },
    //         Body::Colon(Item(i, None)) => {
    //             let up = i.iter().map(|s| Some(*s)).collect::<Vec<_>>();
    //             lvls.push((up, vec![]));
    //         },
    //         Body::Slash(_, _) => {
    //             levels_colon_helper(lvls, body);
    //         },
    //         _ => {}
    //     }
    // }

    #[instrument()]
    fn helper_path<'s>(l: &'s [Item<'s>]) -> Vec<Cow<'s, str>>{
        l.iter().filter_map(|i| {
            match i {
                Item::Text(s) => Some(s.clone()),
                _ => None
            }
        }).collect::<Vec<_>>()
    }

    #[instrument()]
    fn helper_labels<'s>(labels: &mut Vec<(Labels<Cow<'s, str>>, Labels<Cow<'s, str>>)>, r: &'s [Item<'s>]) {
        match r.first() {
            Some(_f @ Item::Colon(rl, rr)) => {
                let mut lvl = (vec![], vec![]);
                helper_lvl(&mut lvl, rl);
                event!(Level::TRACE, ?lvl, "HELPER_LABELS");
                labels.push(lvl);
                helper_labels(labels, rr);
            }
            Some(Item::Slash(rl, rr)) => {
                let mut lvl = (vec![], vec![]);
                helper_slash(&mut lvl.0, rl);
                helper_slash(&mut lvl.1, rr);
                labels.push(lvl);
            },
            Some(Item::Text(_)) => {
                let lvl = (r.iter().map(|i| 
                    if let Item::Text(s) = i { 
                        Some(s.clone()) 
                    } else { 
                        None
                    })
                    .collect::<Vec<_>>(), vec![]);
                labels.push(lvl);
            },
            Some(Item::Seq(r)) => {
                let lvl = (r.iter().map(|i| 
                    if let Item::Text(s) = i { 
                        Some(s.clone()) 
                    } else if let Item::Seq(s) = i { 
                        eprintln!("TODO seq support");
                        None
                    } else { None })
                    .collect::<Vec<_>>(), vec![]);
                labels.push(lvl);
            }
            _ => (),
        }
    }

    #[instrument()]
    fn helper_lvl<'s>(lvl: &mut (Vec<Option<Cow<'s, str>>>, Vec<Option<Cow<'s, str>>>), rl: &'s [Item<'s>]) {
        if let [Item::Slash(l, r)] = rl {
            helper_slash(&mut lvl.0, l);
            helper_slash(&mut lvl.1, r);
        }
        event!(Level::TRACE, ?lvl, "HELPER_LVL");
    }

    #[instrument()]
    fn helper_slash<'s>(side: &mut Vec<Option<Cow<'s, str>>>, items: &'s [Item<'s>]) {
        for i in items {
            if let Item::Text(s) = i {
                side.push(Some(s.clone()))
            } else if let Item::Comma(cs) = i {
                helper_slash(side, cs);
            }
        }
        event!(Level::TRACE, ?side, "HELPER_SLASH");
    }

    #[instrument()]
    fn helper<'s>(vs: &mut Vec<Fact<Cow<'s, str>>>, item: &'s Item<'s>) {
        let mut labels = vec![];
        match item {
            Item::Colon(l, r) => {
                if l.len() == 1 {
                    helper(vs, &l[0]);
                } else {
                    helper_labels(&mut labels, r);
                    let lvl = Fact{
                        path: helper_path(l),
                        labels_by_level: labels,
                    };
                    vs.push(lvl);
                }
            },
            Item::Seq(ls) => {
                vs.push(Fact{path: helper_path(ls), labels_by_level: vec![]})
            },
            Item::Text(s) => {
                vs.push(Fact{path: vec![s.clone()], labels_by_level: vec![]})
            }
            _ => {},
        }
        event!(Level::TRACE, ?vs, "HELPER_MAIN");
    }

    pub fn calculate_vcg2<'s>(v: &'s [Item<'s>]) -> Result<Vcg<Cow<'s, str>, Cow<'s, str>>, Error> where 
    {
        let mut vs = Vec::new();
        for i in v {
            helper(&mut vs, i);
        }
        event!(Level::TRACE, ?vs, "CALCULATE_VCG2");
        calculate_vcg(&vs)
    }

    #[instrument()]
    pub fn calculate_vcg<'s, V>(v: &[Fact<V>]) -> Result<Vcg<V, V>, Error> where 
        V: 's + Clone + Debug + Eq + Hash + Ord + AsRef<str> + From<&'s str> + Trim + IsEmpty,
        String: From<V>
    {
        event!(Level::TRACE, "CALCULATE_VCG");
        let vert = Graph::<V, V>::new();
        let vert_vxmap = HashMap::<V, NodeIndex>::new();
        let vert_node_labels = HashMap::new();
        let vert_edge_labels = HashMap::new();
        let mut vcg = Vcg{vert, vert_vxmap, vert_node_labels, vert_edge_labels};

        let _ = v;

        for Fact{path, labels_by_level} in v {
            for n in 0..path.len()-1 {
                let src = &path[n];
                let dst = &path[n+1];
                let src_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, src.clone());
                let dst_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, dst.clone());

                // TODO: record associated action/percept texts.
                let empty = (vec![], vec![]);
                let (actions, percepts) = labels_by_level.get(n).unwrap_or(&empty);
                let rels = vcg.vert_edge_labels.entry(src.clone()).or_default().entry(dst.clone()).or_default();
                for action in actions {
                    let action = action.clone().map(|a| a.trim());
                    if let Some(action) = action {
                        if !action.is_empty() {
                            vcg.vert.add_edge(src_ix, dst_ix, "actuates".into());
                            rels.entry("actuates".into()).or_default().push(action);
                        }
                    }
                }
                for percept in percepts {
                    let percept = percept.clone().map(|p| p.trim());
                    if let Some(percept) = percept {
                        if !percept.is_empty() {
                            vcg.vert.add_edge(src_ix, dst_ix, "senses".into());
                            rels.entry("senses".into()).or_default().push(percept);
                        }
                    }
                }
            }
            for node in path {
                or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, node.clone());
                vcg.vert_node_labels.insert(node.clone(), node.clone().into());
            }
        }

        for (src, dsts) in vcg.vert_edge_labels.iter_mut() {
            for (dst, rels) in dsts.iter_mut() {
                if rels.is_empty() {
                    let src_ix = vcg.vert_vxmap[src];
                    let dst_ix = vcg.vert_vxmap[dst];
                    vcg.vert.add_edge(src_ix, dst_ix, "fake".into());
                    rels.entry("fake".into()).or_default().push("?".into());
                }
            }
        }

        let roots = roots(&vcg.vert)?;
        let root_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, "root".into());
        vcg.vert_node_labels.insert("root".into(), "".to_string());
        for node in roots.iter() {
            let node_ix = vcg.vert_vxmap[node];
            vcg.vert.add_edge(root_ix, node_ix, "fake".into());
        }

        event!(Level::TRACE, ?vcg, "VCG");

        Ok(vcg)
    }

    pub struct Cvcg<V: Clone + Debug + Ord + Hash, E: Clone + Debug + Ord> {
        pub condensed: Graph<V, SortedVec<(V, V, E)>>,
        pub condensed_vxmap: HashMap::<V, NodeIndex>
    }
    
    pub fn condense<V: Clone + Debug + Ord + Hash, E: Clone + Debug + Ord>(vert: &Graph<V, E>) -> Result<Cvcg<V,E>, Error> {
        let mut condensed = Graph::<V, SortedVec<(V, V, E)>>::new();
        let mut condensed_vxmap = HashMap::new();
        for (vx, vl) in vert.node_references() {
            let mut dsts = HashMap::new();
            for er in vert.edges_directed(vx, Outgoing) {
                let wx = er.target();
                let wl = vert.node_weight(wx).or_err(Kind::IndexingError{})?;
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
        Ok(Cvcg{condensed, condensed_vxmap})
    }

    pub fn rank<'s, V: Clone + Debug + Ord, E>(dag: &'s Graph<V, E>, roots: &'s SortedVec<V>) -> Result<BTreeMap<VerticalRank, SortedVec<(V, V)>>, Error> {
        let paths_fw = floyd_warshall(&dag, |_ex| { -1 })
            .map_err(|cycle| 
                Error::from(RankingError::NegativeCycleError{cycle}.in_current_span())
            )?;

        let paths_fw2 = SortedVec::from_unsorted(
            paths_fw
                .iter()
                .map(|((vx, wx), wgt)| {
                    let vl = dag.node_weight(*vx).or_err(Kind::IndexingError{})?.clone();
                    let wl = dag.node_weight(*wx).or_err(Kind::IndexingError{})?.clone();
                    Ok((*wgt, vl, wl))
                })
                .into_iter()
                .collect::<Result<Vec<_>, Error>>()?
        );
        event!(Level::DEBUG, ?paths_fw2, "FLOYD-WARSHALL");

        let paths_from_roots = SortedVec::from_unsorted(
            paths_fw2
                .iter()
                .filter_map(|(wgt, vl, wl)| {
                    if *wgt <= 0 && roots.contains(vl) {
                        Some((VerticalRank(-(*wgt) as usize), vl.clone(), wl.clone()))
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

        Ok(paths_by_rank)
    }

    use crate::graph_drawing::index::{OriginalHorizontalRank, VerticalRank};

    #[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone)]
    pub enum Loc<V,E> {
        Node(V),
        Hop(VerticalRank, E, E),
    }

    #[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
    pub struct Hop<V: Clone + Debug + Display + Ord + Hash> {
        pub mhr: OriginalHorizontalRank,
        pub nhr: OriginalHorizontalRank,
        pub vl: V,
        pub wl: V,
        pub lvl: VerticalRank,
    }
    
    #[derive(Clone, Debug)]
    pub struct Placement<V: Clone + Debug + Display + Ord + Hash> {
        pub locs_by_level: BTreeMap<VerticalRank, TiVec<OriginalHorizontalRank, OriginalHorizontalRank>>, 
        pub hops_by_level: BTreeMap<VerticalRank, SortedVec<Hop<V>>>,
        pub hops_by_edge: BTreeMap<(V, V), BTreeMap<VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank)>>,
        pub loc_to_node: HashMap<(VerticalRank, OriginalHorizontalRank), Loc<V, V>>,
        pub node_to_loc: HashMap<Loc<V, V>, (VerticalRank, OriginalHorizontalRank)>
    }

    pub type RankedPaths<V> = BTreeMap<VerticalRank, SortedVec<(V, V)>>;

    pub fn calculate_locs_and_hops<'s, V, E>(
        dag: &'s Graph<V, E>, 
        paths_by_rank: &'s RankedPaths<V>
    ) -> Result<Placement<V>, Error>
            where 
        V: Clone + Debug + Display + Ord + Hash, 
        E: Clone + Debug + Ord 
    {
        let mut vx_rank = HashMap::new();
        let mut hx_rank = HashMap::new();
        for (rank, paths) in paths_by_rank.iter() {
            for (n, (_vx, wx)) in paths.iter().enumerate() {
                let n = OriginalHorizontalRank(n);
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
                let a = OriginalHorizontalRank(a);
                locs_by_level.entry(l).or_insert_with(TiVec::new).push(a);
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
                    let vl = dag.node_weight(vx).or_err(Kind::IndexingError{})?;
                    let wl = dag.node_weight(wx).or_err(Kind::IndexingError{})?;
                    Ok((vl.clone(), wl.clone(), er.weight()))
                })
                .into_iter()
                .collect::<Result<Vec<_>, Error>>()?
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
            for mid_level in (vvr+1).0..(wvr.0) {
                let mid_level = VerticalRank(mid_level); // pending https://github.com/rust-lang/rust/issues/42168
                let mhr = locs_by_level.get(&mid_level).map_or(OriginalHorizontalRank(0), |v| OriginalHorizontalRank(v.len()));
                locs_by_level.entry(mid_level).or_insert_with(TiVec::<OriginalHorizontalRank, OriginalHorizontalRank>::new).push(mhr);
                loc_to_node.insert((mid_level, mhr), Loc::Hop(mid_level, vl.clone(), wl.clone()));
                node_to_loc.insert(Loc::Hop(mid_level, vl.clone(), wl.clone()), (mid_level, mhr)); // BUG: what about the endpoints?
                mhrs.push(mhr);
            }
            mhrs.push(whr);

            event!(Level::DEBUG, %vl, %wl, %vvr, %wvr, %vhr, %whr, ?mhrs, "HOP");
            
            for lvl in vvr.0..wvr.0 {
                let lvl = VerticalRank(lvl); // pending https://github.com/rust-lang/rust/issues/42168
                let mx = (lvl.0 as i32 - vvr.0 as i32) as usize;
                let nx = (lvl.0 as i32 + 1 - vvr.0 as i32) as usize;
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

        let mut g_hops = Graph::<(VerticalRank, OriginalHorizontalRank), (VerticalRank, V, V)>::new();
        let mut g_hops_vx = HashMap::new();
        for (_rank, hops) in hops_by_level.iter() {
            for Hop{mhr, nhr, vl, wl, lvl} in hops.iter() {
                let gvx = or_insert(&mut g_hops, &mut g_hops_vx, (*lvl, *mhr));
                let gwx = or_insert(&mut g_hops, &mut g_hops_vx, (*lvl+1, *nhr));
                g_hops.add_edge(gvx, gwx, (*lvl, vl.clone(), wl.clone()));
            }
        }
        let g_hops_dot = Dot::new(&g_hops);
        event!(Level::DEBUG, ?g_hops_dot, "HOPS GRAPH");

        Ok(Placement{locs_by_level, hops_by_level, hops_by_edge, loc_to_node, node_to_loc})
    }

    pub mod minion {
        use std::collections::{BTreeMap, HashMap};
        use std::fmt::{Debug, Display};
        use std::hash::Hash;
        use std::io::Write;
        use std::process::{Command, Stdio};

        use ndarray::Array2;
        use petgraph::Graph;
        use petgraph::dot::Dot;
        use tracing::{event, Level};
        use tracing_error::InstrumentError;

        use crate::graph_drawing::error::{Error, RankingError, OrErrMutExt};
        use crate::graph_drawing::index::{VerticalRank, OriginalHorizontalRank, SolvedHorizontalRank};
        use crate::graph_drawing::layout::{Hop, Loc, or_insert};

        use super::Placement;

        /// minimize_edge_crossing returns the obtained crossing number and a map of (ovr -> (ohr -> shr))
        #[allow(clippy::type_complexity)]
        pub fn minimize_edge_crossing<V>(
            placement: &Placement<V>
        ) -> Result<(usize, BTreeMap<VerticalRank, BTreeMap<OriginalHorizontalRank, SolvedHorizontalRank>>), Error> where
            V: Clone + Debug + Display + Ord + Hash
        {
            let Placement{locs_by_level, hops_by_level, hops_by_edge, node_to_loc, ..} = placement;
            
            if hops_by_level.is_empty() {
                return Ok((0, BTreeMap::new()));
            }
            #[allow(clippy::unwrap_used)]
            let max_level = *hops_by_level.keys().max().unwrap();
            #[allow(clippy::unwrap_used)]
            let max_width = hops_by_level.values().map(|paths| paths.len()).max().unwrap();

            event!(Level::DEBUG, %max_level, %max_width, "max_level, max_width");

            let mut csp = String::new();

            csp.push_str("MINION 3\n");
            csp.push_str("**VARIABLES**\n");
            csp.push_str("BOUND csum {0..1000}\n");
            for (rank, locs) in locs_by_level.iter() {
                csp.push_str(&format!("BOOL x{}[{},{}]\n", rank, locs.len(), locs.len()));
            }
            for rank in 0..locs_by_level.len() - 1 {
                let rank = VerticalRank(rank);
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
                if *k <= max_level {
                    for Hop{mhr: u1, nhr: v1, ..} in hops.iter() {
                        for Hop{mhr: u2, nhr: v2, ..}  in hops.iter() {
                            // if (u1,v1) != (u2,v2) { // BUG!
                            if u1 != u2 && v1 != v2 {
                                csp.push_str(&format!("sumgeq([c{k}[{u1},{v1},{u2},{v2}],x{k}[{u2},{u1}],x{j}[{v1},{v2}]],1)\n", u1=u1, u2=u2, v1=v1, v2=v2, k=k, j=*k+1));
                                csp.push_str(&format!("sumgeq([c{k}[{u1},{v1},{u2},{v2}],x{k}[{u1},{u2}],x{j}[{v2},{v1}]],1)\n", u1=u1, u2=u2, v1=v1, v2=v2, k=k, j=*k+1));
                                // csp.push_str(&format!("sumleq(c{k}[{a},{c},{b},{d}],c{k}[{b},{d},{a},{c}])\n", a=a, b=b, c=c, d=d, k=k));
                                // csp.push_str(&format!("sumgeq(c{k}[{a},{c},{b},{d}],c{k}[{b},{d},{a},{c}])\n", a=a, b=b, c=c, d=d, k=k));
                            }
                        }
                    }
                }
            }
            csp.push_str("\nsumleq([");
            for rank in 0..=max_level.0 {
                if rank > 0 {
                    csp.push(',');
                }
                csp.push_str(&format!("c{}[_,_,_,_]", rank));
            }
            csp.push_str("],csum)\n");
            csp.push_str("sumgeq([");
            for rank in 0..max_level.0 {
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
            let mut child = minion.spawn().map_err(|e| Error::from(RankingError::from(e).in_current_span()))?;
            let stdin = child.stdin.or_err_mut()?;
            stdin.write_all(csp.as_bytes()).map_err(|e| Error::from(RankingError::from(e).in_current_span()))?;

            let output = child
                .wait_with_output()
                .map_err(|e| Error::from(RankingError::from(e).in_current_span()))?;

            let outs = std::str::from_utf8(&output.stdout[..]).map_err(|e| Error::from(RankingError::from(e).in_current_span()))?;

            event!(Level::DEBUG, %outs, "CSP OUT");

            // std::process::exit(0);

            let lines = outs.split('\n').collect::<Vec<_>>();
            let cn_line = lines[2];
            event!(Level::DEBUG, %cn_line, "cn line");
            
            let crossing_number = cn_line
                .trim()
                .parse::<usize>()
                .expect("unable to parse crossing number");

            // std::process::exit(0);
            
            let solns = &lines[3..lines.len()];
            
            event!(Level::DEBUG, ?lines, ?solns, ?crossing_number, "LINES, SOLNS, CN");

            let mut perm = Vec::<Array2<i32>>::new();
            for (rank, locs) in locs_by_level.iter() {
                let mut arr = Array2::<i32>::zeros((locs.len(), locs.len()));
                let parsed_solns = solns[rank.0]
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
            // let perm = perm.into_iter().map(|p| p.permuted_axes([1, 0])).collect::<Vec<_>>();
            event!(Level::TRACE, ?perm, "PERM");
            // for (n, p) in perm.iter().enumerate() {
            //     event!(Level::TRACE, %n, ?p, "PERM2");
            // };

            let mut solved_locs = BTreeMap::new();
            for (n, p) in perm.iter().enumerate() {
                let n = VerticalRank(n);
                let mut sums = p.rows().into_iter().enumerate().map(|(i, r)| (OriginalHorizontalRank(i), r.sum() as usize)).collect::<Vec<_>>();
                sums.sort_by_key(|(_i,s)| *s);
                // eprintln!("row sums: {:?}", sums);
                event!(Level::TRACE, %n, ?p, ?sums, "PERM2");
                for (shr, (i,_s)) in sums.into_iter().enumerate() {
                    let shr = SolvedHorizontalRank(shr);
                    solved_locs.entry(n).or_insert_with(BTreeMap::new).insert(i, shr);
                }
            }
            event!(Level::DEBUG, ?solved_locs, "SOLVED_LOCS");

            let mut layout_debug = Graph::<String, String>::new();
            let mut layout_debug_vxmap = HashMap::new();
            for ((vl, wl), hops) in hops_by_edge.iter() {
                // if *vl == "root" { continue; }
                let vn = node_to_loc[&Loc::Node(vl.clone())];
                let wn = node_to_loc[&Loc::Node(wl.clone())];
                let vshr = solved_locs[&vn.0][&vn.1];
                let wshr = solved_locs[&wn.0][&wn.1];

                let vx = or_insert(&mut layout_debug, &mut layout_debug_vxmap, format!("{vl} {vshr}"));
                let wx = or_insert(&mut layout_debug, &mut layout_debug_vxmap, format!("{wl} {wshr}"));

                for (n, (lvl, (mhr, nhr))) in hops.iter().enumerate() {
                    let shr = solved_locs[lvl][mhr];
                    let shrd = solved_locs[&(*lvl+1)][nhr];
                    let lvl1 = *lvl+1;
                    let vxh = or_insert(&mut layout_debug, &mut layout_debug_vxmap, format!("{vl} {wl} {lvl},{shr}"));
                    let wxh = or_insert(&mut layout_debug, &mut layout_debug_vxmap, format!("{vl} {wl} {lvl1},{shrd}"));
                    layout_debug.add_edge(vxh, wxh, format!("{lvl},{shr}->{lvl1},{shrd}"));
                    if n == 0 {
                        layout_debug.add_edge(vx, vxh, format!("{lvl1},{shrd}"));
                    }
                    if n == hops.len()-1 {
                        layout_debug.add_edge(wxh, wx, format!("{lvl1},{shrd}"));
                    }
                }
            }
            let layout_debug_dot = Dot::new(&layout_debug);
            event!(Level::TRACE, %layout_debug_dot, "LAYOUT GRAPH");

            Ok((crossing_number, solved_locs))
        }
    }

    pub use minion::minimize_edge_crossing;
}

pub mod py {
    use std::marker::PhantomData;
    use numpy::{PyReadonlyArray1, PyArray1};
    use pyo3::{prelude::*, basic::CompareOp};

    pub fn mul<'py>(a: &'py PyAny, b: &'py PyAny) -> PyResult<&'py PyAny> {
        a.call_method1("__mul__", (b,))
    }
    
    pub fn sub<'py>(a: &'py PyAny, b: &'py PyAny) -> PyResult<&'py PyAny> {
        a.call_method1("__sub__", (b,))
    }
    
    pub fn add<'py>(a: &'py PyAny, b: &'py PyAny) -> PyResult<&'py PyAny> {
        a.call_method1("__add__", (b,))
    }
    
    pub struct TiPyAny<'py, K>(pub &'py PyAny, PhantomData<fn(K) -> K>)
        where usize: From<K>;

    impl<'py, K> TiPyAny<'py, K> where usize: From<K> {
        pub fn new(py: &'py PyAny) -> TiPyAny<'py, K> {
            Self(py, PhantomData)
        }
    }
    
    impl<'py, K> TiPyAny<'py, K> where usize: From<K> {
        pub fn get(&self, b: K) -> PyResult<&PyAny> {
            let k: usize = b.into();
            self.0.get_item(k.into_py(self.0.py()))
        }
    }
    
    pub fn get(a: &PyAny, b: usize) -> PyResult<&PyAny> {
        a.get_item(b.into_py(a.py()))
    }
    
    pub fn geq<'py>(a: &'py PyAny, b: &'py PyAny) -> PyResult<&'py PyAny> {
        a.rich_compare(b, CompareOp::Ge)
    }
    
    pub fn leq<'py>(a: &'py PyAny, b: &'py PyAny) -> PyResult<&'py PyAny> {
        a.rich_compare(b, CompareOp::Le)
    }
    
    pub fn eq<'py>(a: &'py PyAny, b: &'py PyAny) -> PyResult<&'py PyAny> {
        a.rich_compare(b, CompareOp::Eq)
    }

    pub fn get_value(a: &PyAny) -> PyResult<PyReadonlyArray1<f64>> {
        Ok(a
            .getattr("value")?
            .extract::<&PyArray1<f64>>()?
            .readonly())
    }
}

pub mod index {
    use std::{ops::{Add, Sub}, fmt::Display};

    use derive_more::{From, Into};

    #[derive(Clone, Copy, Debug, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct VerticalRank(pub usize);

    #[derive(Clone, Copy, Debug, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct OriginalHorizontalRank(pub usize);

    #[derive(Clone, Copy, Debug, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct SolvedHorizontalRank(pub usize);

    #[derive(Clone, Copy, Debug, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct LocSol(pub usize);

    #[derive(Clone, Copy, Debug, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct HopSol(pub usize);

    impl Add<usize> for VerticalRank {
        type Output = Self;

        fn add(self, rhs: usize) -> Self::Output {
            Self(self.0 + rhs)
        }
    }

    impl Add<usize> for OriginalHorizontalRank {
        type Output = Self;

        fn add(self, rhs: usize) -> Self::Output {
            Self(self.0 + rhs)
        }
    }

    impl Add<usize> for SolvedHorizontalRank {
        type Output = Self;

        fn add(self, rhs: usize) -> Self::Output {
            Self(self.0 + rhs)
        }
    }

    impl Add<usize> for LocSol {
        type Output = Self;

        fn add(self, rhs: usize) -> Self::Output {
            Self(self.0 + rhs)
        }
    }

    impl Add<usize> for HopSol {
        type Output = Self;

        fn add(self, rhs: usize) -> Self::Output {
            Self(self.0 + rhs)
        }
    }

    impl Sub<usize> for VerticalRank {
        type Output = Self;

        fn sub(self, rhs: usize) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    impl Sub<usize> for OriginalHorizontalRank {
        type Output = Self;

        fn sub(self, rhs: usize) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    impl Sub<usize> for SolvedHorizontalRank {
        type Output = Self;

        fn sub(self, rhs: usize) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    impl Sub<usize> for LocSol {
        type Output = Self;

        fn sub(self, rhs: usize) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    impl Sub<usize> for HopSol {
        type Output = Self;

        fn sub(self, rhs: usize) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    impl Display for VerticalRank {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.0))
        }
    }

    impl Display for OriginalHorizontalRank {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.0))
        }
    }

    impl Display for SolvedHorizontalRank {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.0))
        }
    }

    impl Display for LocSol {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.0))
        }
    }

    impl Display for HopSol {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.0))
        }
    }
}

pub mod geometry {
    use petgraph::EdgeDirection::{Outgoing, Incoming};
    use petgraph::visit::EdgeRef;
    use pyo3::types::{PyModule, IntoPyDict, PyList};
    use pyo3::{prepare_freethreaded_python, PyResult, Python, PyAny, IntoPy, PyErr};
    use sorted_vec::SortedVec;
    use tracing::{event, Level};
    use tracing_error::InstrumentError;
    use typed_index_collections::TiVec;

    use crate::graph_drawing::error::LayoutError;
    use crate::graph_drawing::py::{TiPyAny, add, leq, geq, sub, mul, get_value};

    use super::error::Error;
    use super::index::{VerticalRank, OriginalHorizontalRank, SolvedHorizontalRank, LocSol, HopSol};
    use super::layout::{Loc, Hop, Vcg, Placement};

    use std::cmp::max;
    use std::collections::{HashMap, BTreeMap, HashSet};
    use std::fmt::{Debug, Display};
    use std::hash::Hash;

    #[derive(Clone, Debug)]
    pub struct LocRow<V: Clone + Debug + Display + Ord + Hash> {
        pub ovr: VerticalRank,
        pub ohr: OriginalHorizontalRank,
        pub shr: SolvedHorizontalRank,
        pub loc: Loc<V, V>,
        pub n: LocSol,
    }
    
    #[derive(Clone, Debug)]
    pub struct HopRow<V: Clone + Debug + Display + Ord + Hash> {
        pub lvl: VerticalRank,
        pub mhr: OriginalHorizontalRank,
        pub nhr: OriginalHorizontalRank,
        pub vl: V,
        pub wl: V,
        pub n: HopSol,
    }
    
    #[derive(Clone, Debug, Default)]
    pub struct LayoutProblem<V: Clone + Debug + Display + Ord + Hash> {
        pub all_locs: Vec<LocRow<V>>,
        pub all_hops0: Vec<HopRow<V>>,
        pub all_hops: Vec<HopRow<V>>,
        pub sol_by_loc: HashMap<(VerticalRank, OriginalHorizontalRank), LocSol>,
        pub sol_by_hop: HashMap<(VerticalRank, OriginalHorizontalRank, V, V), HopSol>,
        pub width_by_loc: HashMap<LocIx, f64>,
        pub width_by_hop: HashMap<(VerticalRank, OriginalHorizontalRank, V, V), (f64, f64)>,
    }
    
    /// ovr, ohr
    pub type LocIx = (VerticalRank, OriginalHorizontalRank);
    
    /// ovr, ohr -> loc
    pub type LocNodeMap<V> = HashMap<LocIx, Loc<V, V>>;
    
    /// lvl -> (mhr, nhr)
    pub type HopMap = BTreeMap<VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank)>;
    
    pub fn calculate_sols<'s, V>(
        solved_locs: &'s BTreeMap<VerticalRank, BTreeMap<OriginalHorizontalRank, SolvedHorizontalRank>>,
        loc_to_node: &'s HashMap<LocIx, Loc<V, V>>,
        hops_by_level: &'s BTreeMap<VerticalRank, SortedVec<Hop<V>>>,
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
            .map(|(n, (ovr, ohr, shr, loc))| LocRow{ovr, ohr, shr, loc: loc.clone(), n: LocSol(n)})
            .collect::<Vec<_>>();
    
        let mut sol_by_loc = HashMap::new();
        let mut sol_by_loc2 = HashMap::<LocIx, Vec<_>>::new();
        for LocRow{ovr, ohr, shr, loc, n} in all_locs.iter() {
            sol_by_loc.insert((*ovr, *ohr), *n);
            sol_by_loc2.entry((*ovr, *ohr)).or_default().push((shr, loc, *n))
        }
        for (loc, duplications) in sol_by_loc2.iter() {
            if duplications.len() > 1 {
                event!(Level::ERROR, ?loc, ?duplications, "all_locs DUPLICATION");
            }
        }
    
        let all_hops0 = hops_by_level
            .iter()
            .flat_map(|h| 
                h.1.iter().map(|Hop{mhr, nhr, vl, wl, lvl}| {
                    (*mhr, *nhr, vl.clone(), wl.clone(), *lvl)
                })
            ).enumerate()
            .map(|(n, (mhr, nhr, vl, wl, lvl))| {
                HopRow{lvl, mhr, nhr, vl, wl, n: HopSol(n)}
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
                        #[allow(clippy::unwrap_used)] // an edge with no hops really should panic
                        let (lvl, (mhr, nhr)) = hops.iter().rev().next().unwrap();
                        (*nhr, OriginalHorizontalRank(std::usize::MAX - mhr.0), vl.clone(), wl.clone(), *lvl+1)
                }) 
            )
            .enumerate()
            .map(|(n, (mhr, nhr, vl, wl, lvl))| {
                HopRow{lvl, mhr, nhr, vl, wl, n: HopSol(n)}
            })
            .collect::<Vec<_>>();
        
        let mut sol_by_hop = HashMap::new();
        let mut sol_by_hop2 = HashMap::<(VerticalRank, OriginalHorizontalRank, OriginalHorizontalRank), Vec<_>>::new();
        let mut sol_by_hop3 = HashMap::<(VerticalRank, OriginalHorizontalRank, OriginalHorizontalRank), Vec<_>>::new();
        for HopRow{lvl, mhr, nhr, vl, wl, n} in all_hops.iter() {
            sol_by_hop.insert((*lvl, *mhr, vl.clone(), wl.clone()), *n);
            sol_by_hop2.entry((*lvl, *mhr, *nhr)).or_default().push((vl.clone(), wl.clone(), *nhr, *n));
        }
        for HopRow{lvl, mhr, nhr, vl, wl, n} in all_hops0.iter() {
            sol_by_hop3.entry((*lvl, *mhr, *nhr)).or_default().push((vl.clone(), wl.clone(), *nhr, *n));
        }
        for (loc, duplications) in sol_by_hop2.iter() {
            if duplications.len() > 1 {
                event!(Level::ERROR, ?loc, ?duplications, "all_hops DUPLICATION");
            }
        }
        for (loc, duplications) in sol_by_hop3.iter() {
            if duplications.len() > 1 {
                event!(Level::ERROR, ?loc, ?duplications, "all_hops0 DUPLICATION");
            }
        }
    
        let width_by_loc = HashMap::new();
        let width_by_hop = HashMap::new();
    
        LayoutProblem{all_locs, all_hops0, all_hops, sol_by_loc, sol_by_hop, width_by_loc, width_by_hop}
    }
    
    #[derive(Debug)]
    pub struct LayoutSolution {
        pub ls: TiVec<LocSol, f64>,
        pub rs: TiVec<LocSol, f64>,
        pub ss: TiVec<HopSol, f64>,
        pub ts: TiVec<VerticalRank, f64>,
    }
    
    pub fn position_sols<'s, V, E>(
        vcg: &'s Vcg<V, E>,
        placement: &'s Placement<V>,
        solved_locs: &'s BTreeMap<VerticalRank, BTreeMap<OriginalHorizontalRank, SolvedHorizontalRank>>,
        layout_problem: &'s LayoutProblem<V>,
    ) -> Result<LayoutSolution, Error> where 
        V: Clone + Debug + Display + Hash + Ord + PartialEq,
        E: Clone + Debug
    {
        let Vcg{vert: dag, vert_vxmap: dag_map, vert_node_labels: _, vert_edge_labels: dag_edge_labels, ..} = vcg;
        let Placement{hops_by_edge, node_to_loc, ..} = placement;
        let LayoutProblem{all_locs, all_hops, sol_by_loc, sol_by_hop, width_by_loc, width_by_hop, ..} = layout_problem;
    
        let mut edge_label_heights = BTreeMap::<VerticalRank, usize>::new();
        for (node, loc) in node_to_loc.iter() {
            let (ovr, _ohr) = loc;
            if let Loc::Node(vl) = node {
                let height_max = edge_label_heights.entry(*ovr).or_default();
                for (vl2, dsts) in dag_edge_labels.iter() {
                    if vl == vl2 {
                        let edge_labels = dsts.iter().flat_map(|(_, rels)| rels.iter().map(|(_, labels)| labels.len())).max().unwrap_or(1);
                        *height_max = max(*height_max, max(0, (edge_labels as i32) - 1) as usize);
                    } 
                }
            }
        }
        let mut row_height_offsets = BTreeMap::<VerticalRank, f64>::new();
        let mut cumulative_offset = 0.0;
        for (lvl, max_height) in edge_label_heights {
            row_height_offsets.insert(lvl, cumulative_offset);
            cumulative_offset += max_height as f64;
        }
        event!(Level::TRACE, ?row_height_offsets, "ROW HEIGHT OFFSETS");
        event!(Level::TRACE, ?width_by_hop, "WIDTH BY HOP");
        
        
        let num_locs = all_locs.len();
        let num_hops = all_hops.len();
        prepare_freethreaded_python();
        let res: PyResult<LayoutSolution> = Python::with_gil(|py| {
            let cp = PyModule::import(py, "cvxpy")?;
            let var = cp.getattr("Variable")?;
            let sq = cp.getattr("norm")?;
            let abs = cp.getattr("abs")?;
            let constant = cp.getattr("Constant")?;
            let problem = cp.getattr("Problem")?;
            let minimize = cp.getattr("Minimize")?;
            let square = |a: &PyAny| {sq.call1((a,))};
            let _absv = |a: &PyAny| {abs.call1((a,))};
            let as_constant = |a: i32| { constant.call1((a.into_py(py),)) };
            let as_constantf = |a: f64| { constant.call1((a.into_py(py),)) };
            let hundred: &PyAny = as_constant(100)?;
            let _thousand: &PyAny = as_constant(1000)?;
            let _ten: &PyAny = as_constant(10)?;
            let _one: &PyAny = as_constant(1)?;
            let _two: &PyAny = as_constant(2)?;
            let zero: &PyAny = as_constant(0)?;
            let symmetry_cost: &PyAny = as_constantf(1.0)?;
            let l = var.call((num_locs,), Some([("pos", true)].into_py_dict(py)))?;
            let l = TiPyAny::<LocSol>::new(l);
            let r = var.call((num_locs,), Some([("pos", true)].into_py_dict(py)))?;
            let r = TiPyAny::<LocSol>::new(r);
            let s = var.call((num_hops,), Some([("pos", true)].into_py_dict(py)))?;
            let s = TiPyAny::<HopSol>::new(s);
            let sep = as_constantf(20.0)?;
            let dsep = as_constantf(40.0)?;
            let _tenf = as_constantf(10.0)?;
            let mut cvec: Vec<&PyAny> = vec![];
            let mut obj = zero;
    
            let root_n = sol_by_loc[&(VerticalRank(0), OriginalHorizontalRank(0))];
            // cvec.push(eq(l.get(root_n)?, zero)?);
            // cvec.push(eq(r.get(root_n)?, one)?);
            obj = add(obj, r.get(root_n)?)?;
    
            #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
            enum Loc2<V> {
                Node{vl: V, loc: LocIx, shr: SolvedHorizontalRank, sol: LocSol},
                Hop{vl: V, wl: V, loc: LocIx, shr: SolvedHorizontalRank, sol: HopSol},
            }
    
            let mut level_to_object = BTreeMap::<VerticalRank, BTreeMap<SolvedHorizontalRank, HashSet<_>>>::new();
            for ((vl, wl), hops) in hops_by_edge.iter() {
                let vn = node_to_loc[&Loc::Node(vl.clone())];
                let wn = node_to_loc[&Loc::Node(wl.clone())];
                let vshr = solved_locs[&vn.0][&vn.1];
                let wshr = solved_locs[&wn.0][&wn.1];
                let vsol = sol_by_loc[&vn];
                let wsol = sol_by_loc[&wn];
    
                level_to_object.entry(vn.0).or_default().entry(vshr).or_default().insert(Loc2::Node{vl, loc: vn, shr: vshr, sol: vsol});
                level_to_object.entry(wn.0).or_default().entry(wshr).or_default().insert(Loc2::Node{vl: wl, loc: wn, shr: wshr, sol: wsol});
    
                for (_n, (lvl, (mhr, nhr))) in hops.iter().enumerate() {
                    let shr = solved_locs[lvl][mhr];
                    let shrd = solved_locs[&(*lvl+1)][nhr];
                    let lvl1 = *lvl+1;
                    let src_sol = sol_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())];
                    let dst_sol = sol_by_hop[&(lvl1, *nhr, vl.clone(), wl.clone())];
                    level_to_object.entry(*lvl).or_default().entry(shr).or_default().insert(Loc2::Hop{vl, wl, loc: (*lvl, *mhr), shr, sol: src_sol});
                    level_to_object.entry(lvl1).or_default().entry(shrd).or_default().insert(Loc2::Hop{vl, wl, loc: (lvl1, *nhr), shr: shrd, sol: dst_sol});
                    // let vxh = format!("{vl} {wl} {lvl},{shr}"));
                    // let wxh = format!("{vl} {wl} {lvl1},{shrd}");
                    // layout_debug.add_edge(vxh, wxh, format!("{lvl},{shr}->{lvl1},{shrd}"));
                    // if n == 0 {
                    //     layout_debug.add_edge(vx, vxh, format!("{lvl1},{shrd}"));
                    // }
                    // if n == hops.len()-1 {
                    //     layout_debug.add_edge(wxh, wx, format!("{lvl1},{shrd}"));
                    // }
                }
            }
            event!(Level::TRACE, ?level_to_object, "LEVEL TO OBJECT");
            // eprintln!("LEVEL TO OBJECT: {level_to_object:#?}");
    
            for LocRow{ovr, ohr, loc, ..} in all_locs.iter() {
                let ovr = *ovr; 
                let ohr = *ohr;
                let locs = &solved_locs[&ovr];
                let shr = locs[&ohr];
                let n = sol_by_loc[&(ovr, ohr)];
                let min_width = *width_by_loc.get(&(ovr, ohr))
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("missing node width: {ovr}, {ohr}")))?;
                let mut min_width = min_width.round() as usize;
    
                if let Loc::Node(vl) = loc {
                    let v_ers = dag.edges_directed(dag_map[vl], Outgoing).into_iter().collect::<Vec<_>>();
                    let w_ers = dag.edges_directed(dag_map[vl], Incoming).into_iter().collect::<Vec<_>>();
                    let mut v_dsts = v_ers
                        .iter()
                        .map(|er| { 
                            dag
                                .node_weight(er.target())
                                .map(Clone::clone)
                                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("missing node weight"))
                        })
                        .into_iter()
                        .collect::<Result<Vec<_>, PyErr>>()?;
                    let mut w_srcs = w_ers
                        .iter()
                        .map(|er| { 
                            dag
                                .node_weight(er.source())
                                .map(Clone::clone)
                                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("missing node weight"))
                        })
                        .into_iter()
                        .collect::<Result<Vec<_>, PyErr>>()?;
    
                    v_dsts.sort(); v_dsts.dedup();
                    v_dsts.sort_by_key(|dst| {
                        let (ovr, ohr) = node_to_loc[&Loc::Node(dst.clone())];
                        let (svr, shr) = (ovr, solved_locs[&ovr][&ohr]);
                        (shr, -(svr.0 as i32))
                    });
                    let v_outs = v_dsts
                        .iter()
                        .map(|dst| { (vl.clone(), dst.clone()) })
                        .collect::<Vec<_>>();
    
                    w_srcs.sort(); w_srcs.dedup();
                    w_srcs.sort_by_key(|src| {
                        let (ovr, ohr) = node_to_loc[&Loc::Node(src.clone())];
                        let (svr, shr) = (ovr, solved_locs[&ovr][&ohr]);
                        (shr, -(svr.0 as i32))
                    });
                    let w_ins = w_srcs
                        .iter()
                        .map(|src| { (src.clone(), vl.clone()) })
                        .collect::<Vec<_>>();
    
                    let v_out_first_hops = v_outs
                        .iter()
                        .map(|(vl, wl)| {
                            #[allow(clippy::unwrap_used)]
                            let (lvl, (mhr, _nhr)) = hops_by_edge[&(vl.clone(), wl.clone())].iter().next().unwrap();
                            (*lvl, *mhr, vl.clone(), wl.clone())
                        })
                        .collect::<Vec<_>>();
                    let w_in_last_hops = w_ins
                        .iter()
                        .map(|(vl, wl)| {
                            #[allow(clippy::unwrap_used)]
                            let (lvl, (mhr, _nhr)) = hops_by_edge[&(vl.clone(), wl.clone())].iter().rev().next().unwrap();
                            (*lvl, *mhr, vl.clone(), wl.clone())
                        })
                        .collect::<Vec<_>>();
                    
                    let out_width: f64 = v_out_first_hops
                        .iter()
                        .map(|idx| {
                            let widths = width_by_hop[idx];
                            widths.0 + widths.1
                        })
                        .sum();
                    let in_width: f64 = w_in_last_hops
                        .iter()
                        .map(|idx| {
                            let widths = width_by_hop[idx];
                            widths.0 + widths.1
                        })
                        .sum();
    
                    let in_width = in_width.round() as usize;
                    let out_width = out_width.round() as usize;
                    let orig_width = min_width;
                    // min_width += max_by(out_width, in_width, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
                    min_width = max(orig_width, max(in_width, out_width));
                    event!(Level::TRACE, %vl, %min_width, %orig_width, %in_width, %out_width, "MIN WIDTH");
                    // eprintln!("lvl: {}, vl: {}, wl: {}, hops: {:?}", lvl, vl, wl, hops);
                }
    
                if let Loc::Hop(_lvl, vl, wl) = loc {
                    let ns = sol_by_hop[&(ovr, ohr, vl.clone(), wl.clone())];
                    cvec.push(leq(l.get(n)?, s.get(ns)?)?);
                    cvec.push(leq(s.get(ns)?, r.get(n)?)?);
                    event!(Level::TRACE, ?loc, %n, %min_width, "X3: l{n} <= s{ns} <= r{n}");
                }
            
                cvec.push(geq(l.get(n)?, l.get(root_n)?)?);
                cvec.push(leq(r.get(n)?, r.get(root_n)?)?);
    
                event!(Level::TRACE, ?loc, %n, %min_width, "X0: r{n} >= l{n} + {min_width:.0?}");
                cvec.push(geq(r.get(n)?, add(l.get(n)?, as_constantf(min_width as f64)?)?)?);
    
                // WIDTH
                // BUG! WHY DOES THIS MATTER???????
                // obj = add(obj, sub(r.get(n)?, l.get(n)?)?)?;
                if shr > SolvedHorizontalRank(0) {
                    #[allow(clippy::unwrap_used)]
                    let ohrp = OriginalHorizontalRank(locs.iter().position(|(_, shrp)| *shrp == shr-1).unwrap());
                    let np = sol_by_loc[&(ovr, ohrp)];
                    let shrp = locs[&ohrp];
                    cvec.push(geq(l.get(n)?, add(r.get(np)?, sep)?)?);
                    event!(Level::TRACE, ?loc, %ovr, %ohr, %shr, %n, %ovr, %ohrp, %shrp, %np, "X1: l{n} >= r{np} + ε")
                }
                if shr < SolvedHorizontalRank(locs.len()-1) {
                    #[allow(clippy::unwrap_used)]
                    let ohrn = OriginalHorizontalRank(locs.iter().position(|(_, shrp)| *shrp == shr+1).unwrap());
                    let nn = sol_by_loc[&(ovr,ohrn)];
                    let shrn = locs[&(ohrn)];
                    cvec.push(leq(r.get(n)?, sub(l.get(nn)?, sep)?)?);
                    event!(Level::TRACE, ?loc, %ovr, %ohr, %shr, %n, %ovr, %ohrn, %shrn, %nn, "X2: r{n} <= l{nn} - ε")
                }
            }
            for hop_row in all_hops.iter() {
                let HopRow{lvl, mhr, nhr, vl, wl, ..} = &hop_row;
    
                let shr = &solved_locs[lvl][mhr];
                let n = sol_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())];
    
                // the hop that we're positioning is either freefloating or attached.
                // we'll have separate hoprows for the top and the bottom of each hop.
                // terminal hoprows have nhr >> |graph|.
                // if we're freefloating, then our horizontal adjacencies are null, 
                // boxes, or (possibly-labeled) hops.
                // otherwise, if we're attached, we have a node and so we need to 
                // figure out if we're on that node; hence whether we are terminal 
                // or we have a downward successor.
                // finally, if we are attached, we need to position ourselves w.r.t. the
                // boundaries of the node we're attached to and to any other parallel hops.
                let all_objects = &level_to_object[lvl][shr];
                let node = all_objects.iter().find(|loc| matches!(loc, Loc2::Node{..}));
                let num_objects: usize = level_to_object.iter().flat_map(|row| row.1.iter().map(|cell| cell.1.len())).sum();
                let terminal = nhr.0 > num_objects;
                let default_hop_width = (20.0, 20.0);
                let (action_width, percept_width) = {
                    width_by_hop.get(&(*lvl, *mhr, vl.clone(), wl.clone())).unwrap_or(&default_hop_width)
                };
                let action_width = *action_width;
                let percept_width = *percept_width;
    
                cvec.push(geq(s.get(n)?, add(l.get(root_n)?, as_constantf(action_width)?)?)?);
                cvec.push(leq(s.get(n)?, add(r.get(root_n)?, as_constantf(percept_width)?)?)?);
    
                if !terminal {
                    let nd = sol_by_hop[&((*lvl+1), *nhr, (*vl).clone(), (*wl).clone())];
                    obj = add(obj, mul(hundred, square(sub(s.get(n)?, s.get(nd)?)?)?)?)?;
                }
    
                event!(Level::TRACE, ?hop_row, ?node, ?all_objects, "POS HOP START");
                if let Some(Loc2::Node{sol: nd, ..}) = node {
                    cvec.push(geq(s.get(n)?, add(l.get(*nd)?, add(sep, as_constantf(action_width)?)?)?)?);
                    cvec.push(leq(s.get(n)?, sub(r.get(*nd)?, add(sep, as_constantf(percept_width)?)?)?)?);
    
                    if terminal {
                        let mut terminal_hops = all_objects
                            .iter()
                            .filter(|obj| { matches!(obj, Loc2::Hop{loc: (_, onhr), ..} if onhr.0 > num_objects) })
                            .collect::<Vec<_>>();
                            #[allow(clippy::unit_return_expecting_ord)]
    
                        terminal_hops.sort_by_key(|hop| {
                            if let Loc2::Hop{shr: tshr, ..} = hop {
                                tshr
                            } else {
                                unreachable!();
                            }
                        });
    
                        for (ox, hop) in terminal_hops.iter().enumerate() {
                            if let Loc2::Hop{vl: ovl, wl: owl, loc: (oovr, oohr), sol: on, ..} = hop {
                                let owidth = width_by_hop.get(&(*oovr, *oohr, (*ovl).clone(), (*owl).clone())).unwrap_or(&default_hop_width);
                                if let Some(Loc2::Hop{vl: ovll, wl: owll, loc: (oovrl, oohrl), sol: onl, ..}) = ox.checked_sub(1).and_then(|oxl| terminal_hops.get(oxl)) {
                                    let owidth_l = width_by_hop.get(&(*oovrl, *oohrl, (*ovll).clone(), (*owll).clone())).unwrap_or(&default_hop_width);
                                    cvec.push(leq(s.get(*onl)?, sub(s.get(*on)?, add(sep, as_constantf(owidth_l.1 + owidth.0)?)?)?)?);
                                    obj = add(obj, mul(symmetry_cost, square(sub(s.get(*on)?, s.get(*onl)?)?)?)?)?;
                                }
                                else {
                                    obj = add(obj, square(sub(l.get(*nd)?, s.get(*on)?)?)?)?;
                                }
                                if let Some(Loc2::Hop{vl: ovlr, wl: owlr, loc: (ovrr, oohrr), sol: onr, ..}) = terminal_hops.get(ox+1) {
                                    let owidth_r = width_by_hop.get(&(*ovrr, *oohrr, (*ovlr).clone(), (*owlr).clone())).unwrap_or(&default_hop_width);
                                    cvec.push(leq(s.get(*on)?, sub(s.get(*onr)?, add(sep, as_constantf(owidth_r.0 + owidth.1)?)?)?)?);
                                    obj = add(obj, mul(symmetry_cost, square(sub(s.get(*onr)?, s.get(*on)?)?)?)?)?;
                                }
                                else {
                                    obj = add(obj, square(sub(r.get(*nd)?, s.get(*on)?)?)?)?;
                                }
                            }
                        }
                    } else {
                        let mut initial_hops = all_objects
                            .iter()
                            .filter(|obj| { matches!(obj, Loc2::Hop{loc: (_, onhr), ..} if onhr.0 <= num_objects) })
                            .collect::<Vec<_>>();
    
                        #[allow(clippy::unit_return_expecting_ord)]
                        initial_hops.sort_by_key(|hop| {
                            if let Loc2::Hop{vl: hvl, wl: hwl, ..} = hop {
                                #[allow(clippy::unwrap_used)]
                                let (hvr, (_hmhr, hnhr)) = hops_by_edge[&((*hvl).clone(), (*hwl).clone())].iter().next().unwrap();
                                solved_locs[&(*hvr+1)][hnhr]
                            } else {
                                unreachable!();
                            }
                        });
    
                        for (ox, hop) in initial_hops.iter().enumerate() {
                            if let Loc2::Hop{vl: ovl, wl: owl, loc: (oovr, oohr), sol: on, ..} = hop {
                                let owidth = width_by_hop.get(&(*oovr, *oohr, (*ovl).clone(), (*owl).clone())).unwrap_or(&default_hop_width);
                                if let Some(Loc2::Hop{vl: ovll, wl: owll, loc: (oovrl, oohrl), sol: onl, ..}) = ox.checked_sub(1).and_then(|oxl| initial_hops.get(oxl)) {
                                    let owidth_l = width_by_hop.get(&(*oovrl, *oohrl, (*ovll).clone(), (*owll).clone())).unwrap_or(&default_hop_width);
                                    cvec.push(leq(s.get(*onl)?, sub(s.get(*on)?, add(sep, as_constantf(owidth_l.1 + owidth.0)?)?)?)?);
                                    obj = add(obj, mul(symmetry_cost, square(sub(s.get(*on)?, s.get(*onl)?)?)?)?)?;
                                } else {
                                    obj = add(obj, square(sub(l.get(*nd)?, s.get(*on)?)?)?)?;
                                }
                                if let Some(Loc2::Hop{vl: ovlr, wl: owlr, loc: (ovrr, oohrr), sol: onr, ..}) = initial_hops.get(ox+1) {
                                    let owidth_r = width_by_hop.get(&(*ovrr, *oohrr, (*ovlr).clone(), (*owlr).clone())).unwrap_or(&default_hop_width);
                                    cvec.push(leq(s.get(*on)?, sub(s.get(*onr)?, add(sep, as_constantf(owidth_r.0 + owidth.1)?)?)?)?);
                                    obj = add(obj, mul(symmetry_cost, square(sub(s.get(*onr)?, s.get(*on)?)?)?)?)?;
                                }
                                else {
                                    obj = add(obj, square(sub(r.get(*nd)?, s.get(*on)?)?)?)?;
                                }
                            }
                        }
                    }
                }
            
                let same_height_objects = &level_to_object[lvl];
    
                if let Some(left_objects) = shr.0.checked_sub(1).and_then(|shrl| same_height_objects.get(&SolvedHorizontalRank(shrl))) {
                    event!(Level::TRACE, ?left_objects, "POS LEFT OBJECTS");
                    for lo in left_objects.iter() {
                        event!(Level::TRACE, ?lo, "POS LEFT OBJECT");
                        match lo {
                            Loc2::Node{sol: ln, ..} => {
                                cvec.push(geq(s.get(n)?, add(l.get(*ln)?, add(sep, as_constantf(action_width)?)?)?)?);
                            },
                            Loc2::Hop{vl: lvl, wl: lwl, loc: (lvr, lhr), sol: ln, ..} => {
                                let (_action_width_l, percept_width_l) = width_by_hop.get(&(*lvr, *lhr, (*lvl).clone(), (*lwl).clone())).unwrap_or(&default_hop_width);
                                cvec.push(geq(s.get(n)?, add(s.get(*ln)?, add(dsep, as_constantf(percept_width_l + action_width)?)?)?)?);
                            },
                        }
                    }
                }
    
                if let Some(right_objects) = same_height_objects.get(&(*shr+1)) {
                    event!(Level::TRACE, ?right_objects, "POS RIGHT OBJECTS");
                    for ro in right_objects.iter() {
                        event!(Level::TRACE, ?ro, "POS RIGHT OBJECT");
                        match ro {
                            Loc2::Node{sol: rn, ..} => {
                                cvec.push(leq(s.get(n)?, add(l.get(*rn)?, add(sep, as_constantf(action_width)?)?)?)?);
                            },
                            Loc2::Hop{vl: rvl, wl: rwl, loc: (rvr, rhr), sol: rn, ..} => {
                                let (action_width_r, _percept_width_r) = width_by_hop.get(&(*rvr, *rhr, (*rvl).clone(), (*rwl).clone())).unwrap_or(&default_hop_width);
                                cvec.push(leq(s.get(n)?, sub(s.get(*rn)?, add(dsep, as_constantf(action_width_r + percept_width)?)?)?)?);
                            },
                        }
                    }
                }
                
                event!(Level::TRACE, ?hop_row, ?node, ?all_objects, "POS HOP END");
            }
    
            // SPACE
            obj = minimize.call1((obj,))?;
            
            let constr = PyList::new(py, cvec);
    
            event!(Level::DEBUG, ?obj, "OBJECTIVE");
    
            let prb = problem.call1((obj, constr))?;
            let is_dcp = prb.call_method1("is_dcp", ())?;
            let is_dcp_str = is_dcp.str()?;
            event!(Level::DEBUG, ?is_dcp_str, "IS_DCP");
    
            // prb.call_method1("solve", ("ECOS",))?;
            prb.call_method1("solve", ())?;
            let status = prb.getattr("status")?;
            let status_str = status.str()?;
            event!(Level::DEBUG, ?status_str, "PROBLEM STATUS");
    
            let lv = get_value(l.0)?;
            let lv = lv.as_slice()?;
    
            let rv = get_value(r.0)?;
            let rv = rv.as_slice()?;
    
            let sv = get_value(s.0)?;
            let sv = sv.as_slice()?;
    
            // eprintln!("L: {:.2?}\n", lv);
            // eprintln!("R: {:.2?}\n", rv);
            // eprintln!("S: {:.2?}\n", sv);
            let ts = row_height_offsets.values().copied().collect::<TiVec<VerticalRank, _>>();
            let ls = lv.iter().copied().collect::<TiVec<LocSol, _>>();
            let rs = rv.iter().copied().collect::<TiVec<LocSol, _>>(); 
            let ss = sv.iter().copied().collect::<TiVec<HopSol, _>>();
    
            Ok(LayoutSolution{ls, rs, ss, ts})
        });
        event!(Level::DEBUG, ?res, "PY");
        res
            .map_err(|e| Error::from(LayoutError::from(e).in_current_span()))
    }
}

#[cfg(test)]
mod tests {
    use super::{error::Error};
    use crate::{parser::parse, graph_drawing::{layout::{*}, graph::roots, index::VerticalRank, geometry::calculate_sols}};
    use tracing_error::InstrumentResult;

    #[test]
    #[allow(clippy::unwrap_used)]    
    pub fn no_swaps() -> Result<(), Error> {
        let data = "A c q: y / z. d e af: w / x";
        let v = parse(data)
            .map_err(|e| match e {
                nom::Err::Error(e) => { nom::Err::Error(nom::error::convert_error(data, e)) },
                nom::Err::Failure(e) => { nom::Err::Failure(nom::error::convert_error(data, e)) },
                nom::Err::Incomplete(n) => { nom::Err::Incomplete(n) },
            })
            .in_current_span()?
            .1;

        let Vcg{vert, vert_vxmap, ..} = calculate_vcg(&v)?;
        let vx = vert_vxmap["e"];
        let wx = vert_vxmap["af"];
        assert_eq!(vert.node_weight(vx), Some(&"e"));
        assert_eq!(vert.node_weight(wx), Some(&"af"));

        let Cvcg{condensed, condensed_vxmap} = condense(&vert)?;
        let cvx = condensed_vxmap["e"];
        let cwx = condensed_vxmap["af"];
        assert_eq!(condensed.node_weight(cvx), Some(&"e"));
        assert_eq!(condensed.node_weight(cwx), Some(&"af"));

        let roots = roots(&condensed)?;

        let paths_by_rank = rank(&condensed, &roots)?;
        assert_eq!(paths_by_rank[&VerticalRank(3)][0], ("root", "af"));

        let placement = calculate_locs_and_hops(&condensed, &paths_by_rank)?;
        let Placement{hops_by_level, hops_by_edge, loc_to_node, node_to_loc, ..} = &placement;
        let nv: Loc<&str, &str> = Loc::Node("e");
        let nw: Loc<&str, &str> = Loc::Node("af");
        let np: Loc<&str, &str> = Loc::Node("c");
        let nq: Loc<&str, &str> = Loc::Node("q");
        let lv = node_to_loc[&nv];
        let lw = node_to_loc[&nw];
        let lp = node_to_loc[&np];
        let lq = node_to_loc[&nq];
        // assert_eq!(lv, (2, 1));
        // assert_eq!(lw, (3, 0));
        // assert_eq!(lp, (2, 0));
        // assert_eq!(lq, (3, 1));
        assert_eq!(lv.1.0 + lw.1.0, 1); // lv.1 != lw.1
        assert_eq!(lp.1.0 + lq.1.0, 1); // lp.1 != lq.1

        let nv2 = &loc_to_node[&lv];
        let nw2 = &loc_to_node[&lw];
        let np2 = &loc_to_node[&lp];
        let nq2 = &loc_to_node[&lq];
        assert_eq!(nv2, &nv);
        assert_eq!(nw2, &nw);
        assert_eq!(np2, &np);
        assert_eq!(nq2, &nq);
        
        let (crossing_number, solved_locs) = minimize_edge_crossing(&placement)?;
        assert_eq!(crossing_number, 0);
        // let sv = solved_locs[&2][&1];
        // let sw = solved_locs[&3][&0];
        // let sp = solved_locs[&2][&0];
        // let sq = solved_locs[&3][&1];
        let sv = solved_locs[&lv.0][&lv.1];
        let sw = solved_locs[&lw.0][&lw.1];
        let sp = solved_locs[&lp.0][&lp.1];
        let sq = solved_locs[&lq.0][&lq.1];
        // assert_eq!(sv, 1);
        // assert_eq!(sw, 1);
        // assert_eq!(sp, 0);
        // assert_eq!(sq, 0);
        assert_eq!(sv, sw); // uncrossing happened
        assert_eq!(sp, sq);

        let layout_problem = calculate_sols(&solved_locs, loc_to_node, hops_by_level, hops_by_edge);
        let all_locs = &layout_problem.all_locs;
        let lrv = all_locs.iter().find(|lr| lr.loc == nv).unwrap();
        let lrw = all_locs.iter().find(|lr| lr.loc == nw).unwrap();
        let lrp = all_locs.iter().find(|lr| lr.loc == np).unwrap();
        let lrq = all_locs.iter().find(|lr| lr.loc == nq).unwrap();
        assert_eq!(lrv.ovr, lv.0);
        assert_eq!(lrw.ovr, lw.0);
        assert_eq!(lrp.ovr, lp.0);
        assert_eq!(lrq.ovr, lq.0);
        assert_eq!(lrv.ohr, lv.1);
        assert_eq!(lrw.ohr, lw.1);
        assert_eq!(lrp.ohr, lp.1);
        assert_eq!(lrq.ohr, lq.1);
        assert_eq!(lrv.shr, sv);
        assert_eq!(lrw.shr, sw);
        assert_eq!(lrp.shr, sp);
        assert_eq!(lrq.shr, sq);

        Ok(())
    }
}
