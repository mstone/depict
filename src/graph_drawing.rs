//! The depict compiler backend
//! 
//! # Summary
//! 
//! Like most compilers, depict has a front-end [parser](crate::parser), 
//! a backend, and a intermediate representation (IR) to connect them.
//! 
//! The depict backend is responsible for gradually transforming the IR 
//! into lower-level constructs: ultimately, into geometric representations of 
//! visual objects that, when drawn, will beautifully and correctly portray
//! the modelling relationships recorded in the IR being compiled.

#![deny(clippy::unwrap_used)]

pub mod error {
    //! Error types for graph_drawing
    //! 
    //! # Summary
    //! 
    //! depict's backend exposes errors with fine-grained types and 
    //! makes extensive use of [SpanTrace]s for error reporting.
    //! 
    //! [OrErrExt] and [OrErrMutExt] provide methods to help attach
    //! current span information to [Option]s. 
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
        #[error("osqp solver error")]
        OsqpError{error: String},
        #[error("osqp setup error")]
        OsqpSetupError{#[from] source: osqp::SetupError},
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
                Error::TypeError{source} => source.source().and_then(ExtractSpanTrace::span_trace),
                Error::GraphDrawingError{source} => source.source().and_then(ExtractSpanTrace::span_trace),
                Error::RankingError{source} => source.source().and_then(ExtractSpanTrace::span_trace),
                Error::LayoutError{source} => source.source().and_then(ExtractSpanTrace::span_trace),
            }
        }
    }

    /// A trait to use to annotate [Option] values with rich error information.
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

    /// A trait to use to annotate `&mut` [Option] references values with rich error information.
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
    //! Graph-theoretic helpers
    use std::fmt::Debug;

    use petgraph::{EdgeDirection::Incoming, Graph};
    use sorted_vec::SortedVec;
    use tracing::{event, Level};

    use super::error::{Error, Kind, OrErrExt};

    /// Find the roots of the input graph `dag`
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

pub mod index {
    //! Index types for graph_drawing
    //! 
    //! # Summary
    //! 
    //! A key challenge the depict backend faces is to transform collections 
    //! of relations into collections of constraints and collections of 
    //! constraints into collections of variables representing geometry.
    //! 
    //! Typed collections ease this task.
    //! 
    //! This module collects the types of indices that may be used to index
    //! these typed collections.
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


pub mod layout {
    //! Choose geometric relations to use to express model relationships
    //! 
    //! # Summary
    //! 
    //! The purpose of the [layout](self) module is to convert descriptions of 
    //! model relationships to be drawn into geometric relationships between 
    //! visual "objects". For example, suppose we would like to express that a 
    //! "person" controls a "microwave" via actions named "open", "start", or 
    //! "stop" and via feedback named "beep". One conceptual picture that we 
    //! might wish to draw of this scenario looks roughly like:
    //! 
    //! ```text
    //!         person
    //! open     | ^     beep
    //! start    | |  
    //! stop     v |
    //!       microwave
    //! ```
    //! 
    //! In this situation, we need to convert the modeling relations like 
    //!   "`person` controls `microwave`" 
    //! and details like
    //!   "`person` can `open` `microwave'`"
    //! into visual choices like "we're going to represent `person` 
    //! and `microwave` as boxes, person should be vertically positioned 
    //! above `microwave`, and in the space between the `person` box and 
    //! the `microwave` box, we are going to place a downward-directed 
    //! arrow, an upward-directed arrow, one label to the left of the 
    //! downward arrrow with the text 'open, start, stop', and one label 
    //! to the right of the upward arrow with the text 'beep'."
    //! 
    //! # Guide-level explanation
    //! 
    //! The heart of the [layout](self) algorithm is to 
    //! 
    //! 1. form a [Vcg], witnessing a partial order on items as a "vertical constraint graph"
    //! 2. [`condense()`] the [Vcg] to a [Cvcg], which is a simple graph, by merging parallel edges into single edges labeled with lists of Vcg edge labels.
    //! 3. [`rank()`] the condensed VCG by finding longest-paths
    //! 4. Calculate a [Placement] of the ranked Cvcg by 
    //! 
    //! use the integer program 
    //! described in <cite>[Optimal Sankey Diagrams Via Integer Programming]</cite> ([author's copy])
    //! implemented via the [minion](https://github.com/minion/minion) constraint
    //! solver to minimize edge-crossing 
    //! 
    //! [Optimal Sankey Diagrams Via Integer Programming]: https://doi.org/10.1109/PacificVis.2018.00025
    //! [author's copy]: https://ialab.it.monash.edu/~dwyer/papers/optimal-sankey-diagrams.pdf
    
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

    pub trait Len: IsEmpty {
        fn len(&self) -> usize;
    }

    impl Len for &str {
        fn len(&self) -> usize {
            str::len(self)
        }
    }

    impl Len for String {
        fn len(&self) -> usize {
            str::len(self)
        }
    }

    impl<'s> Len for Cow<'s, str> {
        fn len(&self) -> usize {
            str::len(self)
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
        eprint!("HELPER_LABELS, r: {r:#?} ");
        match r.first() {
            Some(_f @ Item::Colon(rl, rr)) => {
                helper_labels(labels, rl);
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
            Some(Item::Seq(r)) | Some(Item::Comma(r)) => {
                let lvl = (r.iter().map(|i| 
                    if let Item::Text(s) = i { 
                        Some(s.clone()) 
                    } else if let s@Item::Seq(_) = i {
                        Some(Cow::from(crate::printer::print1(s)))
                    } else { None })
                    .collect::<Vec<_>>(), vec![]);
                labels.push(lvl);
            }
            _ => (),
        }
        eprintln!("-> labels: {labels:#?}");
        event!(Level::TRACE, ?labels, "HELPER_LABELS");
    }

    #[instrument()]
    fn helper_slash<'s>(side: &mut Vec<Option<Cow<'s, str>>>, items: &'s [Item<'s>]) {
        for i in items {
            if let Item::Text(s) = i {
                side.push(Some(s.clone()))
            } else if let Item::Comma(cs) = i {
                for i in cs {
                    side.push(Some(Cow::from(crate::printer::print1(i))));
                }
            }
        }
        event!(Level::TRACE, ?side, "HELPER_SLASH");
    }

    #[instrument()]
    fn helper<'s>(vs: &mut Vec<Fact<Cow<'s, str>>>, item: &'s Item<'s>) {
        let mut labels = vec![];
        match item {
            Item::Colon(l, r) => {
                if l.len() == 1 && matches!(l[0], Item::Text(..)){
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

    /// A graphical object to be positioned relative to other objects
    #[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone)]
    pub enum Loc<V,E> {
        /// A "box"
        Node(V),
        /// One hop of an "arrow"
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

    /// Set up a [Placement] problem
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

    /// A placement solver based on the [minion](https://github.com/minion/minion) constraint solver
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

            let mut minion = Command::new("minion");
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

    /// Solve for horizontal ranks that minimize edge crossing
    pub use minion::minimize_edge_crossing;
}

pub mod geometry {
    //! Generate beautiful geometry consistent with given relations.
    //! 
    //! # Summary 
    //! 
    //! The purpose of the `geometry` module is to generate beautiful geometry 
    //! (positions, widths, paths) consistent with given geometric relations between 
    //! graphical objects to be positioned (ex: "object A should be placed to the left 
    //! of object b").
    //! 
    //! # Guide-level explanation
    //! 
    //! Convex optimization is a powerful framework for finding sets of numbers 
    //! (representing the coordinates of points and guides) that "solve" a given
    //! layout problem in an optimal way consistent with given costs and constraints.
    //! 
    //! Conveniently, it allows us to express both geometric constraints like 
    //! "the horizontal position of the right-hand border of box A must be less 
    //! the horizontal position of the left-hand side of the bounding box of hop C1"
    //! as well as more flexible desiderata like "subject to these constraints, 
    //! minimize the square of the difference of the positions of hops C1.1 and C2.2".
    //! 
    //! # Reference-level explanation
    //! 
    //! `geometry` is implemented in terms of the [OSQP](https://osqp.org) convex 
    //! quadratic program solver and its Rust bindings in the [osqp] crate.
    //! 
    //! Abstractly, the data and the steps required to convert geometric relations into geometry are:
    //! 
    //! 1. the input data need to be organized (here, via [`calculate_sols()`]) so that constraints can be generated and so that the optimization objective can be formed.
    //! 2. then, once constraints and the objective are generated, they need to be formatted as an [osqp::CscMatrix] and associated `&[f64]` slices, passed to [osqp::Problem], and solved.
    //! 3. then, the resulting [osqp::Solution] needs to be destructured so that the resulting solution values can be returned to [`position_sols()`]'s caller as a [LayoutSolution].

    use osqp::{self, CscMatrix};
    use petgraph::EdgeDirection::{Outgoing, Incoming};
    use petgraph::visit::EdgeRef;
    use sorted_vec::SortedVec;
    use tracing::{event, Level};
    use tracing_error::InstrumentError;
    use typed_index_collections::TiVec;

    use crate::graph_drawing::error::{LayoutError, OrErrExt};

    use super::error::Error;
    use super::index::{VerticalRank, OriginalHorizontalRank, SolvedHorizontalRank, LocSol, HopSol};
    use super::layout::{Loc, Hop, Vcg, Placement};

    use std::borrow::{Cow};
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

    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum AnySol {
        L(LocSol),
        R(LocSol),
        S(HopSol),
        T(usize)
    }

    #[derive(Debug)]
    pub struct Vars {
        vars: HashMap<AnySol, Var>
    }

    impl Display for Vars {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let vs = self.vars.iter().map(|(a, b)| (b, a)).collect::<BTreeMap<_, _>>();
            write!(f, "Vars {{")?;
            for (var, _sol) in vs.iter() {
                write!(f, "{var}, ")?;
            }
            write!(f, "}}")
        }
    }

    impl Display for Var {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "v{}({})", self.index, self.sol)
        }
    }

    impl Display for Monomial {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self.coeff {
                x if x == -1. => write!(f, "-{}", self.var),
                x if x == 1. => write!(f, "{}", self.var),
                _ => write!(f, "{}{}", self.coeff, self.var)
            }
        }
    }

    impl Display for AnySol {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                AnySol::L(loc) => write!(f, "l{}", loc.0),
                AnySol::R(loc) => write!(f, "r{}", loc.0),
                AnySol::S(hop) => write!(f, "s{}", hop.0),
                AnySol::T(idx) => write!(f, "t{}", idx),
            }
        }
    }

    impl Display for Constraints {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "Constraints {{")?;
            for (l, comb, u) in self.constrs.iter() {
                write!(f, "    {l} <= ")?;
                for term in comb.iter() {
                    write!(f, "{term} ")?;
                }
                if *u != f64::INFINITY {
                    writeln!(f, "<= {u},")?;
                } else {
                    writeln!(f, ",")?;
                }
            }
            write!(f, "}}")
        }
    }

    impl Vars {
        fn new() -> Self {
            Self { vars: Default::default() }
        }

        fn get(&mut self, index: AnySol) -> Monomial {
            let len = self.vars.len();
            let var = self.vars
                .entry(index)
                .or_insert(Var{index: len, sol: index});
            From::from(&*var)
        }
    }

    pub struct Constraints {
        constrs: Vec<(f64, Vec<Monomial>, f64)>,
    }

    impl Constraints {
        pub fn new() -> Self {
            Self { constrs: Default::default() }
        }

        // L <= Ax <= U 
        /// l < r => r - l > 0 => A = A += [1(r) ... -1(l) ...], L += 0, U += (FMAX/infty)
        pub fn leq(&mut self, v: &mut Vars, lhs: AnySol, rhs: AnySol) {
            if lhs == rhs {
                return
            }
            self.constrs.push((0., vec![-v.get(lhs), v.get(rhs)], f64::INFINITY));
        }

        /// l + c < r => c < r - l => A = A += [1(r) ... -1(l) ...], L += 0, U += (FMAX/infty)
        pub fn leqc(&mut self, v: &mut Vars, lhs: AnySol, rhs: AnySol, c: f64) {
            self.constrs.push((c, vec![-v.get(lhs), v.get(rhs)], f64::INFINITY));
        }

        /// l > r => l - r > 0 => A += [-1(r) ... 1(s) ...], L += 0, U += (FMAX/infty)
        pub fn geq(&mut self, v: &mut Vars, lhs: AnySol, rhs: AnySol) {
            if lhs == rhs {
                return
            }
            self.constrs.push((0., vec![v.get(lhs), -v.get(rhs)], f64::INFINITY));
        }

        /// l > r + c => l - r > c => A += [1(r) ... -1(s) ...], L += c, U += (FMAX/infty)
        pub fn geqc(&mut self, v: &mut Vars, lhs: AnySol, rhs: AnySol, c: f64) {
            self.constrs.push((c, vec![v.get(lhs), -v.get(rhs)], f64::INFINITY));
        }

        pub fn eq(&mut self, lc: &[Monomial]) {
            self.constrs.push((0., Vec::from(lc), 0.));
        }

        pub fn sym(&mut self, v: &mut Vars, pd: &mut Vec<Monomial>, lhs: AnySol, rhs: AnySol) {
            // P[i, j] = 100 => obj += 100 * x_i * x_j
            // we want 100 * (x_i-x_j)^2 => we need a new variable for x_i - x_j?
            // x_k = x_i - x_j => x_k - x_i + x_j = 0
            // then P[k,k] = 100, and [l,A,u] += [0],[1(k), -1(i), 1(j)],[0]
            // 0 <= k-i+j && k-i+j <= 0    =>    i <= k+j && k+j <= i       => i-j <= k && k <= i-j => k == i-j
            // obj = add(obj, mul(hundred, square(sub(s.get(n)?, s.get(nd)?)?)?)?)?;
            // obj.push(...)
            let t = v.get(AnySol::T(v.vars.len()));
            let symmetry_cost = 100.0;
            pd.push(symmetry_cost * t);
            self.eq(&[t, -v.get(lhs), v.get(rhs)]);
        }
    }

    impl Default for Constraints {
        fn default() -> Self {
            Self::new()
        }
    }

    fn as_csc_matrix<'s>(nrows: Option<usize>, ncols: Option<usize>, rows: &[&[Monomial]]) -> CscMatrix<'s> {
        let mut cols: BTreeMap<usize, BTreeMap<usize, f64>> = BTreeMap::new();
        let mut indptr = vec![];
        let mut indices = vec![];
        let mut data = vec![];
        let mut cur_col = 0;
        for (row, comb) in rows.iter().enumerate() {
            for term in comb.iter() {
                if term.coeff != 0. {
                    let old = cols
                        .entry(term.var.index)
                        .or_default()
                        .insert(row, term.coeff);
                    assert!(old.is_none());
                }
            }
        }
        let nrows = nrows.unwrap_or(rows.len());
        let ncols = ncols.unwrap_or(cols.len());
        // indptr.push(data.len());
        for (col, rows) in cols.iter() {
            for _ in cur_col..=*col {
                indptr.push(data.len());
            }
            for (row, coeff) in rows {
                indices.push(*row);
                data.push(*coeff);
            }
            // indptr.push(data.len());
            cur_col = *col+1;
        }
        for _ in cur_col..=ncols {
            indptr.push(data.len());
        }
        CscMatrix{
            nrows,
            ncols,
            indptr: Cow::Owned(indptr),
            indices: Cow::Owned(indices),
            data: Cow::Owned(data),
        }
    }

    fn as_diag_csc_matrix<'s>(nrows: Option<usize>, ncols: Option<usize>, rows: &[Monomial]) -> CscMatrix<'s> {
        let mut cols: BTreeMap<usize, BTreeMap<usize, f64>> = BTreeMap::new();
        let mut indptr = vec![];
        let mut indices = vec![];
        let mut data = vec![];
        let mut cur_col = 0;
        for (_, term) in rows.iter().enumerate() {
            if term.coeff != 0. {
                let old = cols
                    .entry(term.var.index)
                    .or_default()
                    .insert(term.var.index, term.coeff);
                assert!(old.is_none());
            }
        }
        let nrows = nrows.unwrap_or(rows.len());
        let ncols = ncols.unwrap_or(cols.len());
        // indptr.push(data.len());
        for (col, rows) in cols.iter() {
            for _ in cur_col..=*col {
                indptr.push(data.len());
            }
            for (row, coeff) in rows {
                indices.push(*row);
                data.push(*coeff);
            }
            // indptr.push(data.len());
            cur_col = *col+1;
        }
        for _ in cur_col..=ncols {
            indptr.push(data.len());
        }
        CscMatrix{
            nrows,
            ncols,
            indptr: Cow::Owned(indptr),
            indices: Cow::Owned(indices),
            data: Cow::Owned(data),
        }
    }

    impl<'s> From<Constraints> for CscMatrix<'s> {
        fn from(c: Constraints) -> Self {
            let a = &c.constrs
                .iter()
                .map(|(_, comb, _)| &comb[..])
                .collect::<Vec<_>>();
            as_csc_matrix(None, None, a)
        }
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
    pub struct Var {
        index: usize,
        sol: AnySol,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Monomial {
        var: Var,
        coeff: f64,
    }

    impl std::ops::Neg for Monomial {
        type Output = Self;
        fn neg(mut self) -> Self::Output {
            self.coeff = -self.coeff;
            self
        }
    }

    impl From<&Var> for Monomial {
        fn from(var: &Var) -> Self {
            Monomial{ var: *var, coeff: 1. }
        }
    }

    impl std::ops::Mul<Monomial> for f64 {
        type Output = Monomial;
        fn mul(self, mut rhs: Monomial) -> Self::Output {
            rhs.coeff *= self;
            rhs
        }
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
        
    
        let sep = 20.0;

        let mut V: Vars = Vars::new();
        let mut C: Constraints = Constraints::new();
        let mut Pd: Vec<Monomial> = vec![];
        let mut Q: Vec<Monomial> = vec![];

        let L = AnySol::L;
        let R = AnySol::R;
        let S = AnySol::S;
        
        let root_n = sol_by_loc[&(VerticalRank(0), OriginalHorizontalRank(0))];
        // q[r[root_n]] = 1  <-- obj += r[root_n]
        // obj = add(obj, r.get(root_n)?)?;
        Q.push(V.get(R(root_n)));

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
                .ok_or_else::<Error,_>(|| LayoutError::OsqpError{error: format!("missing node width: {ovr}, {ohr}")}.in_current_span().into())?;
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
                            .ok_or_else::<Error, _>(|| LayoutError::OsqpError{error: "missing node weight".into()}.in_current_span().into())
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?;
                let mut w_srcs = w_ers
                    .iter()
                    .map(|er| { 
                        dag
                            .node_weight(er.source())
                            .map(Clone::clone)
                            .ok_or_else::<Error, _>(|| LayoutError::OsqpError{error: "missing node weight".into()}.in_current_span().into())
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?;

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
                
                C.leq(&mut V, L(n), S(ns));
                C.leq(&mut V, S(ns), R(n));
                event!(Level::TRACE, ?loc, %n, %min_width, "X3: l{n} <= s{ns} <= r{n}");
            }
        
            C.leq(&mut V, L(root_n), L(n));
            C.leq(&mut V, R(n), R(root_n));

            event!(Level::TRACE, ?loc, %n, %min_width, "X0: r{n} >= l{n} + {min_width:.0?}");
            C.leqc(&mut V, L(n), R(n), min_width as f64);

            // WIDTH
            // BUG! WHY DOES THIS MATTER???????
            // obj = add(obj, sub(r.get(n)?, l.get(n)?)?)?;
            if shr > SolvedHorizontalRank(0) {
                #[allow(clippy::unwrap_used)]
                let ohrp = OriginalHorizontalRank(locs.iter().position(|(_, shrp)| *shrp == shr-1).unwrap());
                let np = sol_by_loc[&(ovr, ohrp)];
                let shrp = locs[&ohrp];
                C.leqc(&mut V, R(np), L(n), sep);
                event!(Level::TRACE, ?loc, %ovr, %ohr, %shr, %n, %ovr, %ohrp, %shrp, %np, "X1: l{n} >= r{np} + ε")
            }
            if shr < SolvedHorizontalRank(locs.len()-1) {
                #[allow(clippy::unwrap_used)]
                let ohrn = OriginalHorizontalRank(locs.iter().position(|(_, shrp)| *shrp == shr+1).unwrap());
                let nn = sol_by_loc[&(ovr,ohrn)];
                let shrn = locs[&(ohrn)];
                C.leqc(&mut V, R(n), L(nn), sep);
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

            C.leqc(&mut V, L(root_n), S(n), action_width);
            C.leqc(&mut V, S(n), R(root_n), percept_width);

            if !terminal {
                let nd = sol_by_hop[&((*lvl+1), *nhr, (*vl).clone(), (*wl).clone())];
                C.sym(&mut V, &mut Pd, S(n), S(nd));
            }

            event!(Level::TRACE, ?hop_row, ?node, ?all_objects, "POS HOP START");
            if let Some(Loc2::Node{sol: nd, ..}) = node {
                // Sn >= Lnd + sep + aw
                // Lnd + sep + aw <= Sn
                // Lnd <= Sn-sep-aw
                C.geqc(&mut V, S(n), L(*nd), sep + action_width);
                C.leqc(&mut V, S(n), R(*nd), sep + percept_width);

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
                                C.leqc(&mut V, S(*onl), S(*on), sep + owidth_l.1 + owidth.0);
                                C.sym(&mut V, &mut Pd, S(*on), S(*onl));
                            }
                            else {
                                C.sym(&mut V, &mut Pd, L(*nd), S(*on));
                            }
                            if let Some(Loc2::Hop{vl: ovlr, wl: owlr, loc: (ovrr, oohrr), sol: onr, ..}) = terminal_hops.get(ox+1) {
                                let owidth_r = width_by_hop.get(&(*ovrr, *oohrr, (*ovlr).clone(), (*owlr).clone())).unwrap_or(&default_hop_width);
                                C.leqc(&mut V, S(*on), S(*onr), sep + owidth_r.0 + owidth.1);
                                C.sym(&mut V, &mut Pd, S(*onr), S(*on));
                            }
                            else {
                                C.sym(&mut V, &mut Pd, R(*nd), S(*on));
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
                                C.leqc(&mut V, S(*onl), S(*on), sep + owidth_l.1 + owidth.0);
                                C.sym(&mut V, &mut Pd, S(*on), S(*onl));
                            } else {
                                C.sym(&mut V, &mut Pd, L(*nd), S(*on));
                            }
                            if let Some(Loc2::Hop{vl: ovlr, wl: owlr, loc: (ovrr, oohrr), sol: onr, ..}) = initial_hops.get(ox+1) {
                                let owidth_r = width_by_hop.get(&(*ovrr, *oohrr, (*ovlr).clone(), (*owlr).clone())).unwrap_or(&default_hop_width);
                                C.leqc(&mut V, S(*on), S(*onr), sep + owidth_r.0 + owidth.1);
                                C.sym(&mut V, &mut Pd, S(*onr), S(*on));
                            }
                            else {
                                C.sym(&mut V, &mut Pd, R(*nd), S(*on));
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
                            C.geqc(&mut V, S(n), L(*ln), sep + action_width);
                        },
                        Loc2::Hop{vl: lvl, wl: lwl, loc: (lvr, lhr), sol: ln, ..} => {
                            let (_action_width_l, percept_width_l) = width_by_hop.get(&(*lvr, *lhr, (*lvl).clone(), (*lwl).clone())).unwrap_or(&default_hop_width);
                            C.geqc(&mut V, S(n), S(*ln), (2.*sep) + percept_width_l + action_width);
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
                            C.leqc(&mut V, S(n), L(*rn), sep + action_width);
                        },
                        Loc2::Hop{vl: rvl, wl: rwl, loc: (rvr, rhr), sol: rn, ..} => {
                            let (action_width_r, _percept_width_r) = width_by_hop.get(&(*rvr, *rhr, (*rvl).clone(), (*rwl).clone())).unwrap_or(&default_hop_width);
                            C.leqc(&mut V, S(n), S(*rn), (2.*sep) + action_width_r + percept_width);
                        },
                    }
                }
            }
            
            event!(Level::TRACE, ?hop_row, ?node, ?all_objects, "POS HOP END");
        }

        // add non-negativity constraints for all vars
        for (sol, var) in V.vars.iter() {
            if matches!(sol, AnySol::L(_) | AnySol::R(_) | AnySol::S(_)) {
                C.constrs.push((0., vec![var.into()], f64::INFINITY));
            }
        }

        use osqp::{Problem};

        let n = V.vars.len();
        // eprintln!("VARS: {V:#?}");
        // let nnz = Pd.iter().filter(|v| v.coeff != 0.).count();
        // P, q, A, l, u.
        // conceptually, we walk over the columns, then the rows, 
        // recording each non-zero value + its row index, and 
        // as we finish each column, the current data length.
        // let P = CscMatrix::from(&[[4., 1.], [1., 0.]]).into_upper_tri();

        let sparsePd = &Pd[..];
        eprintln!("sparsePd: {sparsePd:?}");
        let P2 = as_diag_csc_matrix(Some(n), Some(n), sparsePd);
        print_tuples("P2", &P2);

        let mut Q2 = Vec::with_capacity(n);
        Q2.resize(n, 0.);
        for q in Q.iter() {
            Q2[q.var.index] += q.coeff; 
        }
        

        let mut L2 = vec![];
        let mut U2 = vec![];
        for (l, _, u) in C.constrs.iter() {
            L2.push(*l);
            U2.push(*u);
        }
        eprintln!("V[{}]: {V}", V.vars.len());
        eprintln!("C[{}]: {C}", &C.constrs.len());

        let A2: CscMatrix = C.into();

        eprintln!("P2[{},{}]: {P2:?}", P2.nrows, P2.ncols);
        eprintln!("Q2[{}]: {Q2:?}", Q2.len());
        eprintln!("L2[{}]: {L2:?}", L2.len());
        eprintln!("U2[{}]: {U2:?}", U2.len());
        eprintln!("A2[{},{}]: {A2:?}", A2.nrows, A2.ncols);
        
        // let q = &[1., 1.];
        // let A = &[
        //     [1., 1.],
        //     [1., 0.],
        //     [0., 1.],
        // ];
        // let l = &[0., 0., 0.];
        // let u = &[1., 1., 1.];

        let settings = osqp::Settings::default()
            .verbose(false);

        // let mut prob = Problem::new(P, q, A, l, u, &settings)
        let mut prob = Problem::new(P2, &Q2[..], A2, &L2[..], &U2[..], &settings)
            .map_err(|e| Error::from(LayoutError::from(e).in_current_span()))?;
        
        let result = prob.solve();
        eprintln!("STATUS {:?}", result);
        let x = result.x().or_err(LayoutError::OsqpError{error: "failed to solve problem".into()})?;
        // eprintln!("{:?}", x);
        let mut solutions = V.vars.iter().map(|(_sol, var)| (*var, x[var.index])).collect::<Vec<_>>();
        solutions.sort_by_key(|(a, _)| *a);
        for (var, val) in solutions {
            if !matches!(var.sol, AnySol::T(_)) {
                eprintln!("{} = {}", var.sol, val);
            }
        }

        // eprintln!("L: {:.2?}\n", lv);
        // eprintln!("R: {:.2?}\n", rv);
        // eprintln!("S: {:.2?}\n", sv);
        let ts = row_height_offsets.values().copied().collect::<TiVec<VerticalRank, _>>();
        let mut ls = V.vars.iter()
            .filter_map(|(sol, var)| {
                if let AnySol::L(l) = sol { 
                    Some((l, x[var.index]))
                } else { 
                    None 
                }
            })
            .collect::<Vec<_>>();
        ls.sort_by_key(|(l, _)| **l);
        eprintln!("ls: {ls:?}");
        let ls = ls.iter().map(|(_, v)| *v).collect::<TiVec<LocSol, _>>();

        let mut rs = V.vars.iter()
            .filter_map(|(sol, var)| {
                if let AnySol::R(r) = sol { 
                    Some((r, x[var.index]))
                } else { 
                    None 
                }
            })
            .collect::<Vec<_>>();
        rs.sort_by_key(|(r, _)| **r);
        eprintln!("rs: {rs:?}");
        let rs = rs.iter().map(|(_, v)| *v).collect::<TiVec<LocSol, _>>();

        let mut ss = V.vars.iter()
            .filter_map(|(sol, var)| {
                if let AnySol::S(s) = sol { 
                    Some((s, x[var.index]))
                } else { 
                    None 
                }
            })
            .collect::<Vec<_>>();
        ss.sort_by_key(|(s, _)| **s);
        eprintln!("ss: {ss:?}");
        let ss = ss.iter().map(|(_, v)| *v).collect::<TiVec<HopSol, _>>();

        let res = LayoutSolution{ls, rs, ss, ts};
        event!(Level::DEBUG, ?res, "LAYOUT");
        Ok(res)
    }

    fn print_tuples(name: &str, m: &CscMatrix) {
        // conceptually, we walk over the columns, then the rows,
        // recording each non-zero value + its row index, and
        // as we finish each column, the current data length.
        // let P = CscMatrix::from(&[[4., 1.], [1., 0.]]).into_upper_tri();
        eprintln!("{name}: {:?}", m);
        let mut n = 0;
        let mut col = 0;
        let indptr = &m.indptr[..];
        let indices = &m.indices[..];
        let data = &m.data[..];
        while n < data.len() {
            while indptr[col+1] <= n {
                col += 1;
            }
            let row = indices[n];
            let val = data[n];
            n += 1;
            eprintln!("{name}[{},{}] = {}", row, col, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{error::Error};
    use crate::{parser::{Parser, Token, Item}, graph_drawing::{layout::{*}, graph::roots, index::VerticalRank, geometry::calculate_sols, error::Kind}};
    use tracing_error::InstrumentResult;
    use logos::Logos;

    #[test]
    #[allow(clippy::unwrap_used)]    
    pub fn no_swaps() -> Result<(), Error> {
        let data = "A c q: y / z\nd e af: w / x";
        let mut p = Parser::new();
        let mut lex = Token::lexer(data);
        while let Some(tk) = lex.next() {
            p.parse(tk)
                .map_err(|_| {
                    Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
                })
                .in_current_span()?
        }

        let v: Vec<Item> = p.end_of_input()
            .map_err(|_| {
                Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
            })
            .in_current_span()?;

        let vcg = calculate_vcg2(&v)?;
        let Vcg{vert, vert_vxmap, ..} = vcg;
        let vx = vert_vxmap["e"];
        let wx = vert_vxmap["af"];
        assert_eq!(vert.node_weight(vx), Some(&Cow::from("e")));
        assert_eq!(vert.node_weight(wx), Some(&Cow::from("af")));

        let Cvcg{condensed, condensed_vxmap} = condense(&vert)?;
        let cvx = condensed_vxmap["e"];
        let cwx = condensed_vxmap["af"];
        assert_eq!(condensed.node_weight(cvx), Some(&Cow::from("e")));
        assert_eq!(condensed.node_weight(cwx), Some(&Cow::from("af")));

        let roots = roots(&condensed)?;

        let paths_by_rank = rank(&condensed, &roots)?;
        assert_eq!(paths_by_rank[&VerticalRank(3)][0], (Cow::from("root"), Cow::from("af")));

        let placement = calculate_locs_and_hops(&condensed, &paths_by_rank)?;
        let Placement{hops_by_level, hops_by_edge, loc_to_node, node_to_loc, ..} = &placement;
        let nv: Loc<Cow<'_, str>, Cow<'_, str>> = Loc::Node(Cow::from("e"));
        let nw: Loc<Cow<'_, str>, Cow<'_, str>> = Loc::Node(Cow::from("af"));
        let np: Loc<Cow<'_, str>, Cow<'_, str>> = Loc::Node(Cow::from("c"));
        let nq: Loc<Cow<'_, str>, Cow<'_, str>> = Loc::Node(Cow::from("q"));
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
