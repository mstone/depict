//! The depict compiler backend
//! 
//! # Summary
//! 
//! Like most compilers, depict has a front-end [parser](crate::parser), 
//! a multi-stage backend, and various intermediate representations (IRs) 
//! to connect them.
//! 
//! The depict backend is responsible for gradually transforming the IR 
//! into lower-level constructs: ultimately, into geometric representations of 
//! visual objects that, when drawn, beautifully and correctly portray
//! the modelling relationships recorded in the IR being compiled.
//! 
//! # Guide-level Explanation
//! 
//! At a high level, the depict backend consists of [layout] and [geometry] modules.
//! 
//! `layout` is responsible for producing a coarse-grained map, called a 
//! [LayoutProblem](layout::LayoutProblem), that "positions" visual components relative 
//! to one another literally by calculating ordering relations between them.
//! 
//! `geometry` is then responsible for calculating positions and dimensions for these
//! components -- essentially, for "meshing" them -- in order to produce instructions
//! that can be passed to a *depict* front-end for drawing using a conventional 
//! drawing backend like [Dioxus](https://dioxuslabs.com) or [TikZ](https://ctan.org/pkg/pgf).
//! 
//! # Reference-level Explanation
//! 
//! ## Approach
//! 
//! Both layout and geometry calculation follow the same general approach which is:
//! 
//! 1. elaborate the input data into a collection of lower-level entities
//! along with additional collections that map indices for the input data into 
//! indices for the corresponding refined objects that enable navigation.
//! 
//! 2. map the refinement and the associated collections of indices into the input 
//! format for a solver, such as a generate-and-test solver for layout
//! or [OSQP](https://osqp.org) for geometry.
//! 
//! 3. solve the relevant problem, and then map the resulting solution back to
//! data about the refinement.
//! 
//! ## Vocabulary
//! 
//! The refinement for `layout` is a collection of "locs". Each "loc" represents a visual
//! cell that is ordered vertically into ranks and horizontally within ranks, into which
//! geometry can conceptually be placed.
//! 
//! Due to requirements of the edge crossing minimization algorithm, it is necessary to
//! refine edges that span more than one rank into collections of "hops" that such that
//! each hop spans only a single rank.
//! 
//! As a result, there are two kinds of "locs", "node locs" and "hop locs".
//! 
//! Locs are indexed by pairs of (VerticalRank, OriginalHorizontalRank), called LocIx.
//! 
//! Node locs have just these coordinates. Hop locs, by contrast, span ranks and so 
//! have coordinates for both their upper end-point -- (ovr, mhr), for "Original 
//! Vertical Rank, m-Horizontal Rank", and their lower end-point (ovr+1, nhr). 
//! Lexically, ohr+1 is also often abbreviated ohr**d** for "down".
//! 
//! The result of solving the layout problem is a map from 
//! 
//!   VerticalRank -> OriginalHorizontalRank -> SolvedHorizontalRank
//! 
//! describing a permutation of the indices of the locs being ordered.
//! 
//! As with original horizontal ranks, solved horizontal ranks get abbreviated as
//! "shr" (for the solved horizontal rank of the upper endpoint of a hop), "shrd"
//! for the solved horizontal rank of the lower endpoint of a hop, and with further
//! abbreviations like "shrl", "shrr" for the solved horizontal ranks of left and 
//! right neighbors.
//! 
//! The layout problem itself has additional further vocabulary: for every distinct
//! pair of locs with a given rank, there is an ordering variable x_ij that will be
//! set to 1 if loc i should be left of loc j and for every distinct pair of hops 
//! with the same starting rank there are crossing number variables c[r][u1, v1, u2, v2] 
//! that will be summed to count whether or not hop (u1, v1) crosses hop (u2, v2)
//! given the relative ordering of locs (r, u1), (r, u2), (r+1, v1), and (r+1, v2).
//! 
//! Finally, hops have an additional subtlety which is that the collections of hops
//! used by geometry are not the same as those solved for by layout because the 
//! refinement used in geometry refines collections of hops into collections of control 
//! points to use to represent the edge bundle to be meshed by adding a fake/sentinel
//! hop with an unreasonably large "nhr" value for each previous final hop so that 
//! procedures that iterate over sequences of hops can simulate iterating over control 
//! points.
//! 
//! ## Other Details
//! 
//! To make the layout refinement from the input data, we vertically rank the nodes 
//! by using Floyd-Warshall with negative weights to find all-pairs longest paths, 
//! and then filter down to paths that start at a root.
//! 
//! These paths are then sorted/grouped by length to produce `paths_by_rank`, which
//! tells us the possible ranks of each destination node, of which we then pick the
//! largest.
//! 
//! The next step is to create a [LayoutProblem](layout::LayoutProblem), 
//! via [`calculate_locs_and_hops()`](layout::calculate_locs_and_hops).
//! 
//! The data of this refinement (and its associated maps of indices) is captured 
//! in multiple collections, including
//! 
//! * `hops_by_level` (which tells us all the hops starting on a given level)
//! * `hops_by_edge` (which tells us all the hops for a given edge, sorted by start level)
//! * `locs_by_level` (which collects the indices of all node or intermediate hop locs)
//! * `loc_to_node` (which records what kind of loc is present at a given index)
//! * `node_to_loc` (which records the indices of each kind of (node | intermediate hop))
//! * `mhrs` (which, at a given level, records the original horizontal ranks of the objects and is used for mhr assignment)
//! 
//! Then [`minimize_edge_crossing()`](layout::minimize_edge_crossing) consumes the given LayoutProblem 
//! (LayoutProblem) and produces a LayoutSolution (a crossing number + (lvl, mhr) -> (shr) map) 
//! that permutes (nodes and intermediate hops), within levels, to minimize edge crossing, by further 
//! embedding the layout IR into variables and constraints that model the possible ordering relations between
//! layout problem objects and their crossing numbers.

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

    use miette::Diagnostic;

    #[cfg(all(feature="osqp", not(feature="osqp-rust")))]
    use osqp;
    #[cfg(all(not(feature="osqp"), feature="osqp-rust"))]
    use osqp_rust as osqp;

    #[non_exhaustive]
    #[derive(Debug, Diagnostic, thiserror::Error)]
    #[diagnostic(code(depict::graph_drawing::error))]
    pub enum Kind {
        #[error("indexing error")]
        IndexingError {},
        #[error("key not found error")]
        KeyNotFoundError {key: String},
        #[error("missing drawing error")]
        MissingDrawingError {},
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
        #[error("heaps solver error")]
        HeapsError{error: String},
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
    #[derive(Debug, Diagnostic, thiserror::Error)]
    #[diagnostic(code(depict::graph_drawing::error))]
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

    /// A trait to use to annotate `&mut` [Option] references values with rich error information.
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
    use std::{ops::{Add, Sub}, fmt::{Debug, Display}};

    use derive_more::{From, Into};

    #[derive(Clone, Copy, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct VerticalRank(pub usize);

    #[derive(Clone, Copy, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct OriginalHorizontalRank(pub usize);

    #[derive(Clone, Copy, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct SolvedHorizontalRank(pub usize);

    #[derive(Clone, Copy, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct LocSol(pub usize);

    #[derive(Clone, Copy, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
    pub struct HopSol(pub usize);

    impl Debug for VerticalRank {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}_vr", self.0)
        }
    }

    impl Debug for OriginalHorizontalRank {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}_ohr", self.0)
        }
    }

    impl Debug for SolvedHorizontalRank {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}_shr", self.0)
        }
    }

    impl Debug for LocSol {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}_ls", self.0)
        }
    }

    impl Debug for HopSol {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}_hs", self.0)
        }
    }

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

pub mod eval {
    //! The main "value" type of depiction parts.

    use std::{collections::{BTreeMap, HashMap}, borrow::{Cow}, vec::IntoIter, ops::Deref, slice::Iter};

    use crate::{parser::Item};
    
    /// What kind of relationship between processes does 
    /// the containing [Val::Chain] describe?
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum Rel {
        Vertical,
        Horizontal,
    }

    #[non_exhaustive]
    #[derive(Clone, Debug)]
    pub enum Position {
        /// north, west, outside
        NWO,
        /// north, east, outside
        NEO,
        /// north, west, inside
        NWI,
        /// south, inside
        SI,
    }

    #[derive(Clone, Debug)]
    pub struct Note<V> {
        /// Content of the note
        pub label: V,
        /// Location of the note
        pub pos: Position,
    }

    /// What are the labels for the forward and reverse 
    /// directions for a given link of the containing 
    /// [Val::Chain]?
    #[derive(Clone, Debug)]
    pub struct Level<V> {
        /// What are the labels for the "forward" direction 
        /// of this link of the chain?
        pub forward: Option<Vec<V>>,
        /// What are the labels for the "reverse" direction
        /// of this link of the chain?
        pub reverse: Option<Vec<V>>,
    }

    /// How are these parts related to the whole they make up?
    #[derive(Clone, Debug)]
    pub enum Body<V> {
        /// States or modes the process could be in
        Any(Vec<Val<V>>),
        /// Parts or constituents making up the process
        All(Vec<Val<V>>),
    }

    impl<V> IntoIterator for Body<V> {
        type Item = Val<V>;

        type IntoIter = IntoIter<Val<V>>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                Body::Any(b) => b.into_iter(),
                Body::All(b) => b.into_iter(),
            }
        }
    }

    impl<'a, V> IntoIterator for &'a Body<V> {
        type Item = &'a Val<V>;

        type IntoIter = Iter<'a, Val<V>>;

        fn into_iter(self) -> Iter<'a, Val<V>> {
            self.iter()
        }
    }

    impl<V> From<Body<V>> for Vec<Val<V>> {
        fn from(b: Body<V>) -> Self {
            match b {
                Body::Any(b) => b,
                Body::All(b) => b,
            }
        }
    }

    impl<V> Default for Body<V> {
        fn default() -> Self {
            Body::All(Default::default())
        }
    }

    impl<V> Deref for Body<V> {
        type Target = [Val<V>];

        fn deref(&self) -> &Self::Target {
            match self {
                Body::Any(b) => &b[..],
                Body::All(b) => &b[..],
            }
        }
    }

    impl<V> Body<V> {
        fn push(&mut self, item: Val<V>) {
            match self {
                Body::Any(b) => b.push(item),
                Body::All(b) => b.push(item),
            }
        }

        fn append(&mut self, rhs: &mut Vec<Val<V>>) {
            match self {
                Body::Any(b) => b.append(rhs),
                Body::All(b) => b.append(rhs),
            }
        }
    }

    /// What processes and labeled relationships between them
    /// are we depicting?
    #[derive(Clone, Debug)]
    pub enum Val<V> {
        Ref {
            name: V,
        },
        Process {
            /// Maybe this process is named?
            name: Option<V>,
            /// Maybe this process has a label?
            label: Option<V>,
            /// Maybe this process has nested parts?
            body: Option<Body<V>>,
            /// Maybe there are nested parts accessible by name?
            name_to_part: Option<BTreeMap<V, usize>>,
            /// Maybe there are annotations?
            notes: Option<Vec<Note<V>>>,
        },
        Chain {
            /// Maybe this chain is named?
            name: Option<V>,
            /// How are the given processes related?
            rel: Rel,
            /// What processes are in the chain?
            path: Vec<Self>,
            /// What labels separate and connect the processes in the chain?
            labels: Vec<Level<V>>
        },
    }

    impl<V> Default for Val<V> {
        fn default() -> Self {
            Self::Process{
                name: None,
                label: None,
                body: None,
                name_to_part: None,
                notes: None,
            }
        }
    }

    impl<V> Val<V> {
        pub fn name(&self) -> Option<&V> {
            match self {
                Val::Ref { name } => Some(name),
                Val::Process { name, .. } => name.as_ref(),
                Val::Chain { name, .. } => name.as_ref(),
            }
        }

        pub fn set_name(&mut self, name: V) -> &mut Self {
            match self {
                Val::Ref { name: n } => { *n = name; },
                Val::Process { name: n, .. } => { *n = Some(name); },
                Val::Chain { name: n, .. } => { *n = Some(name); },
            }
            self
        }

        pub fn label(&self) -> Option<&V> {
            match self {
                Val::Ref { .. } => None, // really?
                Val::Process { label, .. } => label.as_ref(),
                Val::Chain { .. } => None,
            }
        }

        pub fn set_label(&mut self, label: Option<V>) -> &mut Self {
            match self {
                Val::Ref { .. } => unreachable!(),
                Val::Process { label: l, .. } => { *l = label; },
                Val::Chain { .. } => unreachable!(),
            }
            self
        }
    }

    fn eval_path<'s, 't>(path: &'t [Item<'s>]) -> Vec<Val<Cow<'s, str>>> {
        path.iter().filter_map(|i| {
            match i {
                Item::Text(s) if s == "LEFT" || s == "<" || s == ">" || s == "-" => None,
                Item::Text(s) => Some(s.clone()),
                Item::Seq(s) => Some(itertools::join(s, " ").into()),
                _ => None
            }
        }).map(|label| {
            Val::Process{
                name: None,
                label: Some(label),
                body: None,
                name_to_part: None,
                notes: None,
            }
        }).collect::<Vec<_>>()
    }

    fn eval_slash<'s, 't>(forward: Option<&'t [Item<'s>]>, reverse: Option<&'t [Item<'s>]>) -> Option<Vec<Cow<'s, str>>> {
        let mut res: Option<Vec<Cow<'s, str>>> = None;
        if let Some(side) = forward.or(reverse) {
            for item in side {
                match item {
                    Item::Text(s) => {
                        res.get_or_insert_with(Default::default).push(s.clone())
                    },
                    Item::Seq(s) | Item::Comma(s) => {
                        res.get_or_insert_with(Default::default).append(&mut s
                            .iter()
                            .map(|c| Cow::from(crate::printer::print1(c)))
                            .collect::<Vec<_>>()
                        );
                    },
                    _ => {},
                }
            }
        }
        res
    }

    fn eval_labels<'s, 't>(labels: Option<Vec<Level<Cow<'s, str>>>>, r: &'t [Item<'s>]) -> Vec<Level<Cow<'s, str>>> {
        let mut labels = labels.unwrap_or_default();
        match r.first() {
            Some(_f @ Item::Colon(rl, rr)) => {
                labels = eval_labels(Some(labels), rl);
                labels = eval_labels(Some(labels), rr);
            }
            Some(Item::Slash(rl, rr)) => {
                labels.push(Level{
                    forward: eval_slash(Some(&rl[..]), None), 
                    reverse: eval_slash(None, Some(&rr[..])),
                });
            },
            Some(Item::Text(_)) | Some(Item::Seq(_)) | Some(Item::Comma(_)) => {
                labels.push(Level{
                    forward: eval_slash(Some(r), None),
                    reverse: None,
                });
            },
            _ => {},
        }
        labels
    }

    fn eval_rel<'s, 't>(path: &'t [Item<'s>]) -> Rel {
        if path.len() > 0 {
            if let Item::Text(t) = &path[0] {
                let t = t.as_ref();
                match t {
                    "LEFT" | "<" | ">" | "-" => { return Rel::Horizontal; }
                    _ => {}
                }
            }
            if let Item::Text(t) = &path[path.len()-1] {
                let t = t.as_ref();
                match t {
                    "LEFT" | "<" | ">" | "-" => { return Rel::Horizontal; }
                    _ => {}
                }
            }
        } 
        Rel::Vertical
    }

    fn eval_seq<'s, 't>(ls: &'t [Item<'s>]) -> Option<Body<Cow<'s, str>>> {
        let mut body: Option<Body<_>> = None;

        if ls.iter().all(|l| matches!(l, Item::Text(_) | Item::Seq(_))) {
            body.get_or_insert_with(Default::default).push(Val::Chain{
                name: None,
                rel: eval_rel(&ls[..]),
                path: eval_path(&ls[..]),
                labels: vec![],
            });
            return body;
        }
        if ls.len() == 2 {
            if let Item::Text(label) = &ls[0] {
                match &ls[1] {
                    Item::Sq(nest) | Item::Br(nest) => {
                        if let Val::Process{body: Some(nest_val), ..} = eval(&nest[..]) {
                            let nest_val = if matches!(ls[1], Item::Sq(_)) { 
                                Body::All(nest_val.into()) 
                            } else { 
                                Body::Any(nest_val.into()) 
                            };
                            body.get_or_insert_with(Default::default).push(Val::Process{
                                name: None,
                                label: Some(label.clone()),
                                body: Some(nest_val),
                                name_to_part: None,
                                notes: None,
                            });
                            return body;
                        }
                    },
                    _ => {},
                }
            }
        }
        for expr in ls {
            match expr {
                Item::Sq(nest) | Item::Br(nest) => {
                    if let Val::Process{body: Some(nest_val), ..} = eval(&nest[..]) {
                        let nest_val = if matches!(expr, Item::Sq(_)) { 
                            Body::All(nest_val.into()) 
                        } else { 
                            Body::Any(nest_val.into()) 
                        };
                        body.get_or_insert_with(Default::default).push(Val::Process{
                            name: None,
                            label: None,
                            body: Some(nest_val),
                            name_to_part: None,
                            notes: None,
                        });
                    }
                },
                Item::Text(s) => {
                    body.get_or_insert_with(Default::default).push(Val::Process{
                        name: None, 
                        label: Some(s.clone()), 
                        body: None, 
                        name_to_part: None, 
                        notes: None
                    })
                },
                _ => {},
            }
        }

        body
    }

    pub fn index<'s, 't, 'u>(
        val: &'t Val<Cow<'s, str>>,
        current_scope: &'u mut Vec<Cow<'s, str>>,
        scopes: &'u mut HashMap<Vec<Cow<'s, str>>, &'t Val<Cow<'s, str>>>,
    ) {
        match val {
            Val::Ref { name: _ } => {},
            Val::Process { name, body, .. } => {
                if let Some(name) = name {
                    current_scope.push(name.clone());
                    scopes.insert(current_scope.clone(), &val);
                }
                if let Some(body) = body {
                    for val in body.iter() {
                        index(val, current_scope, scopes);
                    }
                }
                if let Some(_name) = name {
                    current_scope.pop();
                }
            },
            Val::Chain { name, path, .. } => {
                if let Some(name) = name {
                    current_scope.push(name.clone());
                    scopes.insert(current_scope.clone(), &val);
                    for val in path.iter() {
                        index(val, current_scope, scopes);
                    }
                    current_scope.pop();
                }
            },
        }
    }

    pub fn resolve<'s, 't, 'u>(
        val: &'t mut Val<Cow<'s, str>>,
        current_path: &'u mut Vec<Cow<'s, str>>,
        scopes: &'u HashMap<Vec<Cow<'s, str>>, &'t Val<Cow<'s, str>>>,
    ) {
        eprintln!("RESOLVE {current_path:?}");
        let mut resolution = None;
        match val {
            Val::Ref { name: _ } => {
                todo!()
            },
            Val::Process { name, body, label, .. } => {
                if let Some(name) = name {
                    current_path.push(name.clone());
                }
                if name.is_none() && body.is_none() {
                    if let Some(label) = label {
                        eprintln!("RESOLVE {current_path:?} found reference: {label}");
                        let label = label.to_string();
                        let mut base_path = current_path.clone();
                        let path = label.split(".").map(|s| s.to_string()).map(Cow::Owned).collect::<Vec<Cow<'s, str>>>();
                        while !base_path.is_empty() {
                            let mut test_path = base_path.clone();
                            test_path.append(&mut path.clone());
                            eprintln!("RESOLVE test path: {test_path:?}");
                            if let Some(val) = scopes.get(&test_path).copied() {
                                resolution = Some(val.clone());
                                break
                            }
                            base_path.pop();
                        }
                        if resolution.is_none() {
                            if let Some(val) = scopes.get(&path).copied() {
                                resolution = Some(val.clone())
                            }
                        }
                    }
                }
                if let Some(body) = body {
                    let bs = match body {
                        Body::Any(bs) => bs,
                        Body::All(bs) => bs,
                    };
                    for val in bs.iter_mut() {
                        resolve(val, current_path, scopes);
                    }
                }
                if let Some(_name) = name {
                    current_path.pop();
                }
            },
            Val::Chain { name, path, .. } => {
                if let Some(name) = name {
                    current_path.push(name.clone());
                }
                for val in path.iter_mut() {
                    resolve(val, current_path, scopes);
                }
                if let Some(_name) = name {
                    current_path.pop();
                }
            },
        }
        eprintln!("RESOLVE resolution: {resolution:?}");
        if let Some(resolution) = resolution {
            *val = resolution;   
        }
    }

    /// What depiction do the given depict-expressions denote?
    pub fn eval<'s, 't>(exprs: &'t [Item<'s>]) -> Val<Cow<'s, str>> {
        // let mut body: Option<Vec<Val<Cow<'s, str>>>> = None;
        let mut body: Option<Body<_>> = None;

        for expr in exprs {
            let _: &'t Item<'s> = expr;
            match expr {
                Item::Colon(l, r) => {
                    if l.len() == 1 && matches!(l[0], Item::Text(..)){
                        let mut rbody = eval(&r[..]);
                        if let Item::Text(name) = &l[0] {
                            let sublabel = match &rbody {
                                Val::Ref { .. } => None,
                                Val::Process { label, body, .. } => {
                                    if label.is_some() { 
                                        label.as_ref().cloned()
                                    } else {
                                        if let Some(body) = body {
                                            match &body[..] {
                                                [Val::Process{label, ..}, ..] => { label.as_ref().cloned() },
                                                _ => {None},
                                            }
                                        } else {
                                            None
                                        }
                                    }
                                },
                                Val::Chain { .. } => None,
                            };
                            let label = sublabel.or_else(|| Some(name.clone()));
                            rbody = match rbody {
                                Val::Process { body: Some(Body::All(bs)), .. } if bs.len() == 1 && matches!(bs[0], Val::Process{..}) => {
                                    bs[0].clone()
                                },
                                _ => rbody,
                            };
                            rbody.set_name(name.clone());
                            rbody.set_label(label.clone());
                            body.get_or_insert_with(Default::default).push(rbody);
                        }
                    } else {
                        body.get_or_insert_with(Default::default).push(Val::Chain{
                            name: None,
                            rel: eval_rel(&l[..]),
                            path: eval_path(&l[..]),
                            labels: eval_labels(None, &r[..]),
                        });
                    }
                },
                Item::Seq(ls) | Item::Comma(ls) => {
                    if let Some(seq_body) = eval_seq(ls) {
                        body.get_or_insert_with(Default::default).append(&mut seq_body.into());
                    }
                }
                Item::Sq(nest) | Item::Br(nest) => {
                    if let Val::Process{body: Some(nest_val), ..} = eval(&nest[..]) {
                        let nest_val = if matches!(expr, Item::Sq(_)) { 
                            Body::All(nest_val.into()) 
                        } else { 
                            Body::Any(nest_val.into()) 
                        };
                        body.get_or_insert_with(Default::default).push(Val::Process{
                            name: None,
                            label: None,
                            body: Some(nest_val),
                            name_to_part: None,
                            notes: None,
                        });
                    }
                }
                Item::Text(s) => {
                    body.get_or_insert_with(Default::default).push(Val::Process{
                        name: None, 
                        label: Some(s.clone()), 
                        body: None, 
                        name_to_part: None, 
                        notes: None
                    })
                }
                _ => {},
            }
        }

        Val::Process{ 
            name: None, 
            label: None, 
            body, 
            name_to_part: None,
            notes: None,
        }
    }
}

pub mod osqp {
    //! Types for optimization problems and conversions to osqp types
    //! 
    //! # Summary
    //! 
    //! This module helps pose problems to minimize an objective defined 
    //! in terms of constrained variables.
    use std::{borrow::Cow, collections::{HashMap, BTreeMap, HashSet}, fmt::{Debug, Display}, hash::Hash, ops::{Mul, Neg}};

    #[cfg(all(feature="osqp", not(feature="osqp-rust")))]
    use osqp;
    #[cfg(all(not(feature="osqp"), feature="osqp-rust"))]
    use osqp_rust as osqp;

    use super::layout::Graphic;

    /// A map from Sols to Vars. `get()`ing an not-yet-seen sol 
    /// allocates a new `Var` for that sol and returns a monomial
    /// containing that sol with a 1.0 coefficient.
    #[derive(Clone, Debug)]
    pub struct Vars<S: Sol> {
        pub vars: HashMap<S, Var<S>>
    }

    impl<S: Sol> Display for Vars<S> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let vs = self.vars.iter().map(|(a, b)| (b, a)).collect::<BTreeMap<_, _>>();
            write!(f, "Vars {{")?;
            for (var, _sol) in vs.iter() {
                write!(f, "{var}, ")?;
            }
            write!(f, "}}")
        }
    }

    impl<S: Sol> Display for Var<S> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "v{}({})", self.index, self.sol)
        }
    }

    impl<S: Sol, C: Coeff> Display for Monomial<S, C> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self.coeff {
                x if x == -1. => write!(f, "-{}", self.var),
                x if x == 1. => write!(f, "{}", self.var),
                _ => write!(f, "{}{}", self.coeff, self.var)
            }
        }
    }

    impl<S: Sol, C: Coeff> Display for Constraints<S, C> {
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

    impl<S: Sol> Vars<S> {
        pub fn new() -> Self {
            Self { vars: Default::default() }
        }

        pub fn len(&self) -> usize {
            self.vars.len()
        }

        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        pub fn get<C: Coeff>(&mut self, index: S) -> Monomial<S, C> {
            let len = self.vars.len();
            let var = self.vars
                .entry(index)
                .or_insert(Var{index: len, sol: index});
            From::from(&*var)
        }

        pub fn iter(&self) -> impl Iterator<Item=(&S, &Var<S>)> {
            self.vars.iter()
        }
    }

    impl<S: Sol> Default for Vars<S> {
        fn default() -> Self {
            Self::new()
        }
    }

    pub trait Coeff : Copy + Clone + Debug + Display + Eq + From<f64> + Hash + Into<f64> + Mul<Output=Self> + Neg<Output=Self> + Ord + PartialEq + PartialEq<f64> + PartialOrd {}
    impl<C: Copy + Clone + Debug + Display + Eq + From<f64> + Hash + Into<f64> + Mul<Output=C> + Neg<Output=C> + Ord + PartialEq + PartialEq<f64> + PartialOrd> Coeff for C {}

    /// A collection of affine constraints: L <= Ax <= U.
    #[derive(Debug, Clone)]
    pub struct Constraints<S: Sol, C: Coeff> {
        pub constrs: HashSet<(C, Vec<Monomial<S, C>>, C)>,
    }

    impl<S: Sol, C: Coeff> Constraints<S, C> {
        pub fn new() -> Self {
            Self { constrs: Default::default() }
        }

        pub fn len(&self) -> usize {
            self.constrs.len()
        }

        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        pub fn push(&mut self, value: (C, Vec<Monomial<S, C>>, C)) {
            self.constrs.insert(value);
        }

        pub fn iter(&self) -> impl Iterator<Item=&(C, Vec<Monomial<S, C>>, C)> {
            self.constrs.iter()
        }

        /// Constrain `lhs` to be less than `rhs`.
        /// l < r => r - l > 0 => A = A += [1(r) ... -1(l) ...], L += 0, U += (FMAX/infty)
        pub fn leq(&mut self, v: &mut Vars<S>, lhs: S, rhs: S) {
            if lhs == rhs {
                return
            }
            self.constrs.insert(((0.).into(), vec![-v.get(lhs), v.get(rhs)], f64::INFINITY.into()));
        }

        /// Constrain `lhs + c` to be less than `rhs`.
        /// l + c < r => c < r - l => A = A += [1(r) ... -1(l) ...], L += 0, U += (FMAX/infty)
        pub fn leqc<C2: Into<C>>(&mut self, v: &mut Vars<S>, lhs: S, rhs: S, c: C2) {
            self.constrs.insert((c.into(), vec![-v.get(lhs), v.get(rhs)], f64::INFINITY.into()));
        }

        /// Constrain `lhs` to be greater than `rhs`.
        /// l > r => l - r > 0 => A += [-1(r) ... 1(s) ...], L += 0, U += (FMAX/infty)
        pub fn geq(&mut self, v: &mut Vars<S>, lhs: S, rhs: S) {
            if lhs == rhs {
                return
            }
            self.constrs.insert(((0.).into(), vec![v.get(lhs), -v.get(rhs)], f64::INFINITY.into()));
        }

        /// Constrain `lhs` to be greater than `rhs + c`.
        /// l > r + c => l - r > c => A += [1(r) ... -1(s) ...], L += c, U += (FMAX/infty)
        pub fn geqc<C2: Into<C>>(&mut self, v: &mut Vars<S>, lhs: S, rhs: S, c: C2) {
            self.constrs.insert((c.into(), vec![v.get(lhs), -v.get(rhs)], f64::INFINITY.into()));
        }

        /// Constrain the linear combination `lc` to be equal to 0.
        pub fn eq(&mut self, lc: &[Monomial<S, C>]) {
            self.constrs.insert(((0.).into(), Vec::from(lc), (0.).into()));
        }

        /// Constrain the linear combination `lc` to be equal to `c`.
        pub fn eqc(&mut self, lc: &[Monomial<S, C>], c: C) {
            self.constrs.insert((c, Vec::from(lc), c));
        }

        /// Constrain `lhs` to be similar to `rhs` by introducing a fresh variable, 
        /// `t`, constraining `t` to be equal to `lhs - rhs`, and adding `t` to a 
        /// collection representing the diagonal of the quadratic form P of the 
        /// objective `1/2 x'Px + Qx`.
        pub fn sym<C2: Into<C>>(&mut self, v: &mut Vars<S>, pd: &mut Vec<Monomial<S, C>>, lhs: S, rhs: S, coeff: C2) {
            // P[i, j] = 100 => obj += 100 * x_i * x_j
            // we want 100 * (x_i-x_j)^2 => we need a new variable for x_i - x_j?
            // x_k = x_i - x_j => x_k - x_i + x_j = 0
            // then P[k,k] = 100, and [l,A,u] += [0],[1(k), -1(i), 1(j)],[0]
            // 0 <= k-i+j && k-i+j <= 0    =>    i <= k+j && k+j <= i       => i-j <= k && k <= i-j => k == i-j
            // obj = add(obj, mul(hundred, square(sub(s.get(n)?, s.get(nd)?)?)?)?)?;
            // obj.push(...)
            let mut t = v.get(S::fresh(v.vars.len()));
            self.eq(&[t.clone(), -v.get(lhs), v.get(rhs)]);
            t.coeff = t.coeff * coeff.into();
            pd.push(t);
            eprintln!("SYM {t} {lhs} {rhs}");
        }
    }

    impl<S: Sol, C: Coeff> Default for Constraints<S, C> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Used to allocate fresh variables
    pub trait Fresh {
        /// Construct a fresh sol
        fn fresh(index: usize) -> Self;
    }

    /// Used to define indexed families of variables
    pub trait Sol: Clone + Copy + Debug + Display + Eq + Fresh + Hash + Ord + PartialEq + PartialOrd {}

    impl<S: Clone + Copy + Debug + Display + Eq + Fresh + Hash + Ord + PartialEq + PartialOrd> Sol for S {}

    /// Convert `rows` into an `osqp::CscMatrix` in "compressed sparse column" format.
    fn as_csc_matrix<'s, S: Sol, C: Coeff>(nrows: Option<usize>, ncols: Option<usize>, rows: &[&[Monomial<S, C>]]) -> osqp::CscMatrix<'s> {
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
                        .insert(row, term.coeff.into());
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
        osqp::CscMatrix{
            nrows,
            ncols,
            indptr: Cow::Owned(indptr),
            indices: Cow::Owned(indices),
            data: Cow::Owned(data),
        }
    }

    /// Convert `rows` into a *diagonal* `osqp::CscMatrix` in "compressed sparse column" format.
    pub fn as_diag_csc_matrix<'s, S: Sol, C: Coeff>(nrows: Option<usize>, ncols: Option<usize>, rows: &[Monomial<S, C>]) -> osqp::CscMatrix<'s> {
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
                    .insert(term.var.index, term.coeff.into());
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
        osqp::CscMatrix{
            nrows,
            ncols,
            indptr: Cow::Owned(indptr),
            indices: Cow::Owned(indices),
            data: Cow::Owned(data),
        }
    }

    pub fn as_scipy(name: impl Display, m: &osqp::CscMatrix) {
        let osqp::CscMatrix{
            nrows,
            ncols,
            indptr,
            indices,
            data,
        } = m;
        eprintln!("{name} = sp.csc_array(({data:?},{indices:?},{indptr:?}), shape=({nrows},{ncols}))");
    }

    pub fn as_numpy(name: impl Display, v: &Vec<f64>) {
        eprintln!("{name} = np.array({v:?})");
    }

    impl<'s, S: Sol, C: Coeff> From<Constraints<S, C>> for osqp::CscMatrix<'s> {
        fn from(c: Constraints<S, C>) -> Self {
            let a = &c.constrs
                .iter()
                .map(|(_, comb, _)| &comb[..])
                .collect::<Vec<_>>();
            as_csc_matrix(None, None, a)
        }
    }

    /// An optimization variable
    /// 
    /// Vars map indices of business-domain quantities of interest
    /// represented by `sol` to densely packed optimization-domain 
    /// indices `index` that can be used as row or column indices 
    /// in matrix or array formulations of the optimization problem
    /// to be solved.
    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
    pub struct Var<S: Sol> {
        pub index: usize,
        pub sol: S,
    }

    /// A weighted optimization variable
    /// 
    /// Most solvers optimize an objective function that is typically 
    /// defined as, e.g., a weighted linear or quadratic function of
    /// given optimization variables.
    /// 
    /// We use "monomials" to specify the data (weights and variables) 
    /// of individual terms of these linear or quadratic forms to be 
    /// optimized via convenient syntax implemented via operator overloading.
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct Monomial<S: Sol, C: Coeff> {
        pub var: Var<S>,
        pub coeff: C,
    }

    impl<S: Sol, C: Coeff> std::ops::Neg for Monomial<S, C> {
        type Output = Self;
        fn neg(mut self) -> Self::Output {
            self.coeff = -self.coeff;
            self
        }
    }

    impl<S: Sol, C: Coeff> From<&Var<S>> for Monomial<S, C> {
        fn from(var: &Var<S>) -> Self {
            Monomial{ var: *var, coeff: (1.).into() }
        }
    }

    impl<S: Sol, C: Coeff> std::ops::Mul<Monomial<S, C>> for f64 {
        type Output = Monomial<S, C>;
        fn mul(self, mut rhs: Monomial<S, C>) -> Self::Output {
            rhs.coeff = rhs.coeff * self.into();
            rhs
        }
    }

    /// A pretty-printer for linear combinations of monomials.
    pub struct Printer<'s, S: Sol, C: Coeff>(pub &'s Vec<Monomial<S, C>>);

    impl<'s, S: Sol, C: Coeff> Display for Printer<'s, S, C> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "[")?;
            for (n, m) in self.0.iter().enumerate() {
                write!(f, "{m}")?;
                if n < self.0.len()-1 {
                    write!(f, " ")?;
                }
            }
            write!(f, "]")
        }
    }

    /// A debug print helper for dumping `osqp::CscMatrix`s
    pub fn print_tuples(name: &str, m: &osqp::CscMatrix) {
        // conceptually, we walk over the columns, then the rows,
        // recording each non-zero value + its row index, and
        // as we finish each column, the current data length.
        // let P = osqp::CscMatrix::from(&[[4., 1.], [1., 0.]]).into_upper_tri();
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

pub mod layout {
    //! Choose geometric relations to use to express model relationships
    //! 
    //! # Summary
    //! 
    //! The purpose of the [layout](self) module is to convert descriptions of 
    //! model relationships to be drawn into order relationships between 
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
    //! 4. [`calculate_locs_and_hops()`] from the ranked paths of the CVCG to form a [LayoutProblem] by refining edge bundles (i.e., condensed edges) into hops
    //! 5. [`minimize_edge_crossing()`] by direct enumeration, inspired by the integer program described in <cite>[Optimal Sankey Diagrams Via Integer Programming]</cite> ([author's copy])
    //! resulting in a `ovr -> ohr -> shr` map.
    //! 
    //! [Optimal Sankey Diagrams Via Integer Programming]: https://doi.org/10.1109/PacificVis.2018.00025
    //! [author's copy]: https://ialab.it.monash.edu/~dwyer/papers/optimal-sankey-diagrams.pdf
    //! 
    //! # Reference-level explanation
    //! 
    //! Many general solvers are challenging to run in all the situations we care about,
    //! notably on wasm32-unknown-unknown. Consequently, after experimenting with a 
    //! variety of LP-relaxation-based solvers, we have found that
    //! 
    //! * [heaps], a direct "generate-and-test" solver, based on a version of 
    //!   [Heap's algorithm](https://en.wikipedia.org/wiki/Heap%27s_algorithm) 
    //!   for enumerating permutations that has been modified to enumerate 
    //!   all vertical-rank-preserving permutations of our layouts.
    //! 
    //! seems to serve us best.
    
    use std::borrow::{Cow};
    use std::collections::{BTreeMap, HashSet};
    use std::collections::{HashMap, hash_map::Entry};
    use std::fmt::{Debug, Display};
    use std::hash::Hash;
    
    use petgraph::EdgeDirection::Outgoing;
    use petgraph::algo::floyd_warshall;
    use petgraph::dot::Dot;
    use petgraph::graph::{Graph, NodeIndex, EdgeReference};
    use petgraph::visit::{EdgeRef, IntoNodeReferences};
    use sorted_vec::SortedVec;
    use tracing::{event, Level};
    use tracing_error::InstrumentError;

    use crate::graph_drawing::error::{Error, Kind, OrErrExt, RankingError};
    use crate::graph_drawing::eval::{Val, self, Body};
    use crate::graph_drawing::graph::roots;

    #[derive(Clone, Debug, Default)]
    pub struct Hcg<V: Graphic> {
        pub constraints: HashSet<HorizontalConstraint<V>>,
        pub labels: HashMap<(V, V), eval::Level<V>>,
    }

    /// Require a to be left of b
    #[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
    pub struct HorizontalConstraint<V: Graphic> {
        pub a: V,
        pub b: V,
    }

    impl<V: Graphic> Hcg<V> {
        fn iter(&self) -> impl Iterator<Item=&HorizontalConstraint<V>> {
            self.constraints.iter()
        }
    }

    fn calculate_hcg_chain<'s, 't>(
        hcg: &'t mut Hcg<Cow<'s, str>>,
        chain: &'t Val<Cow<'s, str>>,
    ) -> Result<(), Error> {
        if let Val::Chain{rel, path, labels, ..} = chain {
            if *rel == Rel::Horizontal {
                for n in 0..path.len()-1 {
                    if let Val::Process{label: Some(al), ..} = &path[n] {
                        if let Val::Process{label: Some(bl), ..} = &path[n+1] {
                            hcg.constraints.insert(
                                HorizontalConstraint{
                                    a: al.clone(), 
                                    b: bl.clone()
                                }
                            );
                            if let Some(level) = labels.get(n) {
                                let eval::Level{forward, reverse} = level.clone();
                                let hlvl = hcg.labels.entry((al.clone(), bl.clone()))
                                    .or_insert(eval::Level{forward: None, reverse: None});
                                // if hf.is_none() then forward else hf.map()
                                match (&mut hlvl.forward, forward) {
                                    (Some(f1), Some(mut f2)) => { f1.append(&mut f2); }
                                    (Some(_), None) => {},
                                    (None, Some(f2)) => { hlvl.forward.replace(f2); },
                                    _ => {},
                                };
                                match (&mut hlvl.reverse, reverse) {
                                    (Some(r1), Some(mut r2)) => { r1.append(&mut r2); }
                                    (Some(_), None) => {},
                                    (None, Some(r2)) => { hlvl.reverse.replace(r2); },
                                    _ => {},
                                };
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn calculate_hcg_helper<'s, 't>(
        hcg: &'t mut Hcg<Cow<'s, str>>,
        process: &'t Val<Cow<'s, str>>,
    ) -> Result<(), Error> {
        match process { 
            Val::Chain{..} => {
                calculate_hcg_chain(hcg, process)?;
            },
            Val::Process{body, ..} => {
                for part in body.iter().flatten() {
                    match part {
                        Val::Chain{..} => {
                            calculate_hcg_chain(hcg, part)?;
                        }
                        Val::Process{body, ..} => {
                            for part in body.iter().flatten() {
                                calculate_hcg_helper(hcg, part)?;
                            }
                        },
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn calculate_hcg<'s, 't>(process: &'t Val<Cow<'s, str>>) -> Result<Hcg<Cow<'s, str>>, Error> {
        let mut hcg = Hcg::default();
        calculate_hcg_helper(&mut hcg, process)?;
        Ok(hcg)
    }
    
    pub fn fixup_hcg_rank<'s, 't>(hcg: &'t Hcg<Cow<'s, str>>, paths_by_rank: &'t mut BTreeMap<VerticalRank, SortedVec<(Cow<'s, str>, Cow<'s, str>)>>) {
        let mut preliminary_rank = HashMap::new();
        for (rank, paths) in paths_by_rank.iter() {
            for (_, wl) in paths.iter() {
                preliminary_rank.insert(wl.clone(), *rank);
            }
        }
        let mut modified_rank = HashMap::new();
        for (node, rank) in preliminary_rank.iter() {
            let rank = *rank;
            for HorizontalConstraint{a, b} in hcg.constraints.iter() {
                let ar = modified_rank.get(a).copied().unwrap_or(preliminary_rank[a]);
                let br = modified_rank.get(b).copied().unwrap_or(preliminary_rank[b]);
                if node == a || node == b {
                    let (o, or) = if node == a { (b, br) } else { (a, ar) };
                    if ar != br {
                        if rank < or {
                            modified_rank.insert(node.clone(), or);
                        } else {
                            modified_rank.insert(o.clone(), rank);
                        }
                    }
                }
            }
        }
        for (node, rank) in modified_rank.iter() {
            paths_by_rank.entry(*rank).or_default().insert(("root".into(), node.clone()));
        }
    }

    #[derive(Clone, Debug, Default)]
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

        /// containers identifies which nodes are parents of contained nodes
        pub containers: HashSet<V>,

        /// nodes_by_container maps container-nodes to their contents, transitively.
        pub nodes_by_container: HashMap<V, HashSet<V>>,

        /// nesting_depths records how deeply nested each item is.
        pub nesting_depths: HashMap<V, usize>,

        /// nesting_depth_by_container records how many levels of nesting each container contains
        pub nesting_depth_by_container: HashMap<V, usize>,

        /// container_depths records how many ranks each container spans
        pub container_depths: HashMap<V, usize>,
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

    fn walk_body<'s, 't, 'u>(
        queue: &'u mut Vec<(
            &'s Vec<Val<Cow<'t, str>>>, 
            &'s Rel, 
            &'s Vec<eval::Level<Cow<'t, str>>>,
            &'s Option<Cow<'t, str>>,
        )>,
        vcg: &mut Vcg<Cow<'t, str>, Cow<'t, str>>,
        body: &'s Body<Cow<'t, str>>,
        parent: &'s Option<Cow<'t, str>>,
        mut parents: Vec<Cow<'t, str>>,
        mut max_depth: usize,
    ) -> (Vec<Cow<'t, str>>, usize) {
        if let Some(parent) = parent {
            parents.push(parent.clone());
        }
        for chain in body {
            eprintln!("WALK_BODY CHAIN parent: {parent:?}, chain: {chain:#?}");
            match chain {
                Val::Process{label: Some(node), body: None, ..} => {
                    or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, node.clone());
                    vcg.vert_node_labels.insert(node.clone(), node.clone().into());
                    if let Some(parent) = parent {
                        add_contains_edge(vcg, parent, node);
                    }
                    for p in parents.iter() {
                        vcg.nodes_by_container.entry(p.clone()).or_default().insert(node.clone());
                    }
                    vcg.nesting_depths.insert(node.clone(), parents.len());
                },
                Val::Process{label, body: Some(body), ..} => {
                    if let (Some(parent), Some(label)) = (parent.as_ref(), label.as_ref()) {
                        add_contains_edge(vcg, parent, label);
                    }
                    // BUG: need to debruijn-number unlabeled containers
                    if let Some(node) = label.as_ref() {
                        vcg.nesting_depths.insert(node.clone(), parents.len());
                        for p in parents.iter() {
                            vcg.nodes_by_container.entry(p.clone()).or_default().insert(node.clone());
                        }
                        if body.len() > 0 {
                            vcg.containers.insert(node.clone());
                        }
                    }
                    let (new_parents, new_max_depth) = walk_body(queue, vcg, body, label, parents, 0);
                    parents = new_parents;
                    max_depth = std::cmp::max(max_depth, new_max_depth);
                },
                Val::Chain{path, rel, labels, ..} => {
                    queue.push((path, rel, labels, parent));
                    for val in path {
                        if let Val::Process{label: Some(node), ..} = val {
                            for p in parents.iter() {
                                vcg.nodes_by_container.entry(p.clone()).or_default().insert(node.clone());
                            }
                            vcg.nesting_depths.insert(node.clone(), parents.len());
                        }
                    }
                },
                _ => {},
            }
        }
        if let Some(parent) = parent {
            vcg.nesting_depth_by_container.insert(parent.clone(), max_depth);
            max_depth += 1;
            parents.pop();
        }
        (parents, max_depth)
    }

    pub fn add_contains_edge<'s, 't>(vcg: &'t mut Vcg<Cow<'s, str>, Cow<'s, str>>, parent: &'t Cow<'s, str>, node: &'t Cow<'s, str>) {
        let src_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, parent.clone());
        vcg.vert_node_labels.insert(parent.clone(), parent.clone().into());

        let dst_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, node.clone());
        vcg.vert_node_labels.insert(node.clone(), node.clone().into());
        
        vcg.vert.add_edge(src_ix, dst_ix, "contains".into());

        let rels = vcg.vert_edge_labels.entry(parent.clone()).or_default().entry(node.clone()).or_default();
        rels.entry("contains".into()).or_default();

        vcg.containers.insert(parent.clone());
    }

    pub fn calculate_vcg<'s, 't>(process: &'t Val<Cow<'s, str>>) -> Result<Vcg<Cow<'s, str>, Cow<'s, str>>, Error> {
        let vert = Graph::<Cow<str>, Cow<str>>::new();
        let vert_vxmap = HashMap::<Cow<str>, NodeIndex>::new();
        let vert_node_labels = HashMap::new();
        let vert_edge_labels = HashMap::new();
        let containers = HashSet::new();
        let nodes_by_container = HashMap::new();
        let nesting_depth_by_container = HashMap::new();
        let nesting_depths: HashMap<Cow<str>, usize> = HashMap::new();
        let container_depths: HashMap<Cow<str>, usize> = HashMap::new();
        let mut vcg = Vcg{
            vert, 
            vert_vxmap, 
            vert_node_labels, 
            vert_edge_labels, 
            containers, 
            nodes_by_container, 
            nesting_depths, 
            nesting_depth_by_container,
            container_depths,
        };

        let body = if let Val::Process{body: Some(body), ..} = process { body } else { unreachable!(); };

        let mut queue = vec![];

        walk_body(&mut queue, &mut vcg, body, &None, vec![], 0);

        eprintln!("NESTING DEPTH BY CONTAINER: {:#?}", &vcg.nesting_depth_by_container);

        eprintln!("QUEUE: {queue:#?}");

        for (path, rel, labels_by_level, parent) in queue {
            if let (Some(parent), Some(eval::Val::Process{label: Some(node), ..})) = (parent, path.first()) {
                add_contains_edge(&mut vcg, parent, node);
            }
            for node in path {
                let node = if let eval::Val::Process{label: Some(label), ..} = node { label } else { continue; };
                or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, node.clone());
                vcg.vert_node_labels.insert(node.clone(), node.clone().into());
            }
            if *rel == Rel::Horizontal {
                continue
            }
            for n in 0..path.len()-1 {
                let src = &path[n];
                let src = if let Val::Process { label: Some(label), .. } = src { label } else { continue; };
                let dst = &path[n+1];
                let dst = if let Val::Process { label: Some(label), .. } = dst { label } else { continue; };
                let src_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, src.clone());
                let dst_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, dst.clone());

                // TODO: record associated action/percept texts.
                let empty = eval::Level{forward: None, reverse: None};
                let labels = labels_by_level.get(n).unwrap_or(&empty);
                let rels = vcg.vert_edge_labels.entry(src.clone()).or_default().entry(dst.clone()).or_default();
                for action in labels.forward.iter().flatten() {
                    let action = action.clone().trim();
                    if !action.is_empty() {
                        vcg.vert.add_edge(src_ix, dst_ix, "actuates".into());
                        rels.entry("actuates".into()).or_default().push(action);
                    }
                }
                for percept in labels.reverse.iter().flatten() {
                    let percept = percept.clone().trim();
                    if !percept.is_empty() {
                        vcg.vert.add_edge(src_ix, dst_ix, "senses".into());
                        rels.entry("senses".into()).or_default().push(percept);
                    }
                }
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

        let vert_roots = roots(&vcg.vert)?;
        let root_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, "root".into());
        vcg.vert_node_labels.insert("root".into(), "".to_string());
        for node in vert_roots.iter() {
            let node_ix = vcg.vert_vxmap[node];
            vcg.vert.add_edge(root_ix, node_ix, "fake".into());
        }

        let container_depths = &mut vcg.container_depths;
        for vl in vcg.containers.iter() {
            let subdag = vcg.vert.filter_map(|_nx, nl| {
                if vcg.nodes_by_container[vl].contains(nl) {
                    Some(nl.clone())
                } else { 
                    None
                }
            }, |_ex, el|{
                Some(el.clone())
            });
            let subroots = roots(&subdag)?;
            let subpaths_by_rank = rank(&subdag, &subroots, &vcg.containers, &vcg.nodes_by_container, &container_depths)?;
            let depth = std::cmp::max(1, subpaths_by_rank.len());
            container_depths.insert(vl.clone(), depth);
            eprintln!("CONTAINER {vl}");
            eprintln!("SUBROOTS {subroots:#?}");
            eprintln!("SUBDAG {subdag:#?}");
            eprintln!("SUBPATHS {subpaths_by_rank:#?}");
            eprintln!("DEPTH {depth}");
        }

        event!(Level::TRACE, ?vcg, "VCG");
        eprintln!("VCG: {vcg:#?}");

        Ok(vcg)
    }

    /// A "condensed" VCG, in which all parallel edges in the original Vcg 
    /// have been "condensed" into a single compound edge in the Cvcg.
    #[derive(Default)]
    pub struct Cvcg<V: Clone + Debug + Ord + Hash, E: Clone + Debug + Ord> {
        pub condensed: Graph<V, SortedVec<(V, V, E)>>,
        pub condensed_vxmap: HashMap::<V, NodeIndex>
    }
    
    /// Construct a cvcg from a vcg `vert`.
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

    /// Rank a `dag`, starting from `roots`, by finding longest paths
    /// from the roots to each node, e.g., using Floyd-Warshall with
    /// negative weights.
    pub fn rank<'s, V: Graphic, E>(
        dag: &'s Graph<V, E>, 
        roots: &'s SortedVec<V>, 
        containers: &'s HashSet<V>, 
        nodes_by_container: &'s HashMap<V, HashSet<V>>,
        container_depths: &'s HashMap<V, usize>,
    ) -> Result<BTreeMap<VerticalRank, SortedVec<(V, V)>>, Error> {
        let paths_fw = floyd_warshall(&dag, |er| {
            let src = dag.node_weight(er.source()).unwrap();
            let dst = dag.node_weight(er.target()).unwrap();
            if !containers.contains(src) {
                -1
            } else {
                if nodes_by_container[src].contains(dst) {
                    0
                } else {
                    -(container_depths[src] as isize)
                }
            }
        })
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

    /// Methods for graph vertices and edges.f
    pub trait Graphic: Clone + Debug + Eq + Hash + Ord + PartialEq + PartialOrd {}

    impl <T: Clone + Debug + Eq + Hash + Ord + PartialEq + PartialOrd> Graphic for T {}

    /// A graphical object to be positioned relative to other objects
    #[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
    pub enum Loc<V: Graphic + Display, E: Graphic + Display> {
        /// A "box"
        Node(V),
        /// One hop of an "arrow"
        Hop(VerticalRank, E, E),
        /// A vertical border of a nested system of boxes
        Border(Border<V>)
    }

    impl<V: Graphic + Display, E: Graphic + Display> Debug for Loc<V, E> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Node(arg0) => write!(f, "Node({})", arg0),
                Self::Hop(arg0, arg1, arg2) => write!(f, "Hop({}, {}, {})", arg0, arg1, arg2),
                Self::Border(arg0) => write!(f, "Border({})", arg0),
            }
        }
    }

    /// A numeric representation of a hop.
    #[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
    pub struct Hop<V: Graphic> {
        /// The original horizontal rank of the upper endpoint of the hop
        pub mhr: OriginalHorizontalRank,

        /// The original horizontal rank of the lower endpoint of the hop,
        /// or `std::usize::MAX - mhr` if this is an extended hop.
        pub nhr: OriginalHorizontalRank,

        /// The upper node of the hop's edge
        pub vl: V,

        /// The lower node of the hop's edge
        pub wl: V,

        /// The vertical rank of the upper endpoint of the hop
        pub lvl: VerticalRank,
    }

    impl<V: Graphic + Display> Display for Hop<V> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Hop{{{}->{}, {}, {}, {}}}", self.vl, self.wl, self.lvl, self.mhr, self.nhr)
        }
    }

    #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum Node {
        Point(OriginalHorizontalRank),
        Interval(OriginalHorizontalRank, OriginalHorizontalRank),
    }

    #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct Border<V: Graphic + Display> {
        pub vl: V,
        pub ovr: VerticalRank,
        pub ohr: OriginalHorizontalRank,
        pub pair: OriginalHorizontalRank,
    }

    impl<V: Graphic + Display> Display for Border<V> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Border({}, {}, {}, {})", self.vl, self.ovr, self.ohr, self.pair)
        }
    }
    
    #[derive(Clone, Debug, Default)]
    pub struct LayoutProblem<V: Graphic + Display> {
        pub locs_by_level: BTreeMap<VerticalRank, usize>, 
        pub hops_by_level: BTreeMap<VerticalRank, SortedVec<Hop<V>>>,
        pub hops_by_edge: BTreeMap<(V, V), BTreeMap<VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank)>>,
        pub loc_to_node: HashMap<(VerticalRank, OriginalHorizontalRank), Loc<V, V>>,
        pub node_to_loc: HashMap<Loc<V, V>, (VerticalRank, OriginalHorizontalRank)>,
        pub container_borders: HashMap<V, Vec<(VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank))>>,
        pub hcg: Hcg<V>,  
    }

    #[derive(Clone, Debug, Default)]
    pub struct LayoutSolution {
        pub crossing_number: usize,
        pub solved_locs: BTreeMap<VerticalRank, BTreeMap<OriginalHorizontalRank, SolvedHorizontalRank>>,
    }

    pub type RankedPaths<V> = BTreeMap<VerticalRank, SortedVec<(V, V)>>;

    /// Set up a [LayoutProblem] problem
    pub fn calculate_locs_and_hops<'s, V, E>(
        _model: &'s Val<V>,
        dag: &'s Graph<V, E>, 
        paths_by_rank: &'s RankedPaths<V>,
        vcg: &Vcg<V, V>,
        hcg: Hcg<V>,
    ) -> Result<LayoutProblem<V>, Error>
            where 
        V: Display + Graphic, 
        E: Graphic
    {
        let Vcg{containers, nodes_by_container, container_depths, ..} = vcg;

        // Rank vertices by the length of the longest path reaching them.
        let mut vx_rank = BTreeMap::new();
        for (rank, paths) in paths_by_rank.iter() {
            for (_vx, wx) in paths.iter() {
                vx_rank.insert(wx.clone(), *rank);
            }
        }

        eprintln!("PATHS_BY_RANK 0: {paths_by_rank:#?}");
        eprintln!("VX_RANK {vx_rank:#?}");

        let mut loc_to_node = HashMap::new();
        let mut node_to_loc = HashMap::new();
        let mut locs_by_level = BTreeMap::new();

        for (wl, rank) in vx_rank.iter() {
            let level = locs_by_level
                .entry(*rank)
                .or_insert(0);
            let mhr = OriginalHorizontalRank(*level);
            *level += 1;
            if let Some(old) = loc_to_node.insert((*rank, mhr), Loc::Node(wl.clone())) {
                panic!("loc_to_node.insert({rank}, {mhr}) -> {:?}", old);
            };
            node_to_loc.insert(Loc::Node(wl.clone()), (*rank, mhr));
        }

        event!(Level::DEBUG, ?locs_by_level, "LOCS_BY_LEVEL V1");
        let is_contains = |er: &EdgeReference<E>| {
            let src = dag.node_weight(er.source()).unwrap();
            let dst = dag.node_weight(er.target()).unwrap();
            containers.contains(src) && nodes_by_container[src].contains(dst)
        };
        let sorted_condensed_edges = SortedVec::from_unsorted(
            dag
                .edge_references()
                .filter(|x| !is_contains(x))
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
            let (vvr, vhr) = node_to_loc[&Loc::Node(vl.clone())].clone();
            let (wvr, whr) = node_to_loc[&Loc::Node(wl.clone())].clone();
            
            let mut mhrs = vec![vhr];
            for mid_level in (vvr+1).0..(wvr.0) {
                let mid_level = VerticalRank(mid_level); // pending https://github.com/rust-lang/rust/issues/42168
                let mut num_mhrs = locs_by_level.entry(mid_level).or_insert(0);
                let mhr = OriginalHorizontalRank(*num_mhrs);
                *num_mhrs += 1;
                if let Some(old) = loc_to_node.insert((mid_level, mhr), Loc::Hop(mid_level, vl.clone(), wl.clone())) {
                    panic!("loc_to_node.insert({mid_level}, {mhr}) -> {:?}", old);
                };
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

        let mut container_borders: HashMap<V, Vec<(VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank))>> = HashMap::new();
        
        for vl in containers.iter() {
            let (ovr, mut ohr) = node_to_loc[&Loc::Node(vl.clone())];
            let depth = container_depths[vl];
            for vr in 0..depth {
                let vr = VerticalRank(ovr.0 + vr);
                let num_mhrs = locs_by_level.entry(vr).or_insert(0);
                let mhr = OriginalHorizontalRank(*num_mhrs);
                *num_mhrs += 1;
                if vr > ovr {
                    ohr = OriginalHorizontalRank(*num_mhrs);
                    *num_mhrs += 1;
                    if let Some(old) = loc_to_node.insert((vr, ohr), Loc::Border(Border{ vl: vl.clone(), ovr: vr, ohr: mhr, pair: ohr })) {
                        panic!("loc_to_node.insert({vr}, {ohr}) -> {:?}", old);
                    };
                }
                if let Some(old) = loc_to_node.insert((vr, mhr), Loc::Border(Border{ vl: vl.clone(), ovr: vr, ohr, pair: mhr })) {
                    panic!("loc_to_node.insert({vr}, {mhr}) -> {:?}", old);
                };
                container_borders.entry(vl.clone()).or_default().push((vr, (ohr, mhr)));
            }
            
            eprintln!("VERTICAL RANK SPAN: {vl}: {:?}", ovr.0..(ovr.0+depth));
            eprintln!("CONTAINER BORDERS: {vl}: {container_borders:#?}");
            eprintln!("LOCS_BY_LEVEL V3: {vl}: {locs_by_level:#?}");
        }

        eprintln!("NODE_TO_LOC: {node_to_loc:#?}");

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

        Ok(LayoutProblem{locs_by_level, hops_by_level, hops_by_edge, loc_to_node, node_to_loc, container_borders, hcg})
    }

    pub mod debug {
        //! Debug-print helpers for layout problems.
        use std::{collections::{HashMap}, fmt::Display};

        use petgraph::{Graph, dot::Dot};
        use tracing::{event, Level};

        use crate::graph_drawing::{layout::{LayoutProblem, Loc, or_insert, LayoutSolution}};

        use super::Graphic;
        
        /// Print a graphviz "dot" representation of the solution `solved_locs` 
        /// to `layout_problem`
        pub fn debug<V: Display + Graphic>(layout_problem: &LayoutProblem<V>, layout_solution: &LayoutSolution) {
            let LayoutProblem{node_to_loc, hops_by_edge, ..} = layout_problem;
            let LayoutSolution{solved_locs, ..} = &layout_solution;
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
            eprintln!("LAYOUT GRAPH\n{layout_debug_dot:?}");
        }
    }

    pub mod heaps {
        use std::collections::{BTreeMap, HashMap};
        use std::fmt::{Display};

        use crate::graph_drawing::error::{Error, LayoutError, OrErrExt, Kind};
        use crate::graph_drawing::index::{VerticalRank, OriginalHorizontalRank, SolvedHorizontalRank};
        use crate::graph_drawing::layout::{Hop};
        
        use super::{LayoutProblem, Graphic, LayoutSolution, HorizontalConstraint, Loc, Vcg};

        #[inline]
        pub fn is_odd(x: usize) -> bool {
            x & 0b1 == 1
        }

        // https://sedgewick.io/wp-content/themes/sedgewick/papers/1977Permutation.pdf, Algorithm 2
        pub fn search<T>(p: &mut [T], mut process: impl FnMut(&mut [T])) {
            let n = p.len();
            let mut c = vec![0; n];
            let mut i = 0; 
            process(p);
            while i < n {
                if c[i] < i {
                    // let k = if !is_odd(i) { 0 } else { c[i] }
                    // k & 0b1 == 1 if odd, 0 if even.
                    // (k & 0b1) * i = 0 if even, i if odd
                    let k_idx = (i & 0b1_usize) * i;
                    let k = c[k_idx];
                    p.swap(i, k);
                    c[i] += 1;
                    i = 0;
                    process(p)
                } else {
                    c[i] = 0;
                    i += 1
                }
            }
        }

        pub fn multisearch<T>(p: &mut [&mut [T]], mut process: impl FnMut(&mut [&mut [T]]) -> bool) {
            let m = p.len();
            let mut n = vec![];
            let mut c = vec![];
            for q in p.iter() {
                n.push(q.len());
                c.push(vec![0; q.len()]);
            }
            let mut j = 0;
            if process(p) {
                return;
            };
            while j < m {
                let mut i = 0;
                while i < n[j] {
                    if c[j][i] < i {
                        // let k = if !is_odd(i) { 0 } else { c[i] }
                        // k & 0b1 == 1 if odd, 0 if even.
                        // (k & 0b1) * i = 0 if even, i if odd
                        let k_idx = (i & 0b1_usize) * i;
                        let k = c[j][k_idx];
                        p[j].swap(i, k);
                        c[j][i] += 1;
                        i = 0;
                        j = 0;
                        if process(p) {
                            return;
                        }
                    } else {
                        c[j][i] = 0;
                        i += 1
                    }
                }
                j += 1;
            }
        }

        fn crosses<V: Graphic + Display>(h1: &Hop<V>, h2: &Hop<V>, p1: &[usize], p2: &[usize]) -> usize {
            // imagine we have permutations p1, p2 of horizontal ranks for levels l1 and l2
            // and a set of hops spanning l1-l2.
            // we want to know how many of these hops cross.
            // each hop h1, h2, has endpoints (h11, h12) and (h21, h22) on l1, l2 respectively.
            // the crossing number c for h1, h2 is
            let h11 = h1.mhr;
            let h12 = h1.nhr;
            let h21 = h2.mhr;
            let h22 = h2.nhr;
            // if h11 == h21 || h12 == h22 {
            //     return 0
            // }
            let u1 = p1[h11.0];
            let u2 = p2[h12.0];
            let v1 = p1[h21.0];
            let v2 = p2[h22.0];
            let x121 = v1 < u1;
            let x112 = u1 < v1;
            let x221 = v2 < u2;
            let x212 = u2 < v2;
            let c = (x121 && x212) || (x112 && x221);
            
            let lvl = h1.lvl;
            let vl1 = &h1.vl;
            let wl1 = &h1.wl;
            let vl2 = &h2.vl;
            let wl2 = &h2.wl;
            // eprintln!("CROSSES: {h1:?}, {h2:?}, {lvl}, {vl1}->{wl1} X {vl2}->{wl2}, ({h11},{h12}) X ({h21},{h22}) -> ({u1},{u2}) X ({v1},{v2}) -> {c}");
            // if x121 {
            //     eprintln!("{vl2} {vl1}");
            // } else {
            //     eprintln!("{vl1} {vl2}");
            // }
            // if x221 {
            //     eprintln!("{wl2} {wl1}");
            // } else {
            //     eprintln!("{wl1} {wl2}");
            // }
            c as usize
        }

        fn conforms<V: Graphic + Display, E: Graphic>(
            vcg: &Vcg<V, E>, 
            layout_problem: &LayoutProblem<V>, 
            locs_by_level2: &Vec<Vec<&Loc<V, V>>>, 
            nodes_by_container2: &HashMap<V, Vec<(VerticalRank, OriginalHorizontalRank)>>, 
            p: &mut [&mut [usize]]
        ) -> bool {
            let Vcg{nodes_by_container, ..} = vcg;
            let LayoutProblem{node_to_loc, hcg, container_borders, ..} = layout_problem;

            let hcg_satisfied = hcg.iter().all(|constraint| {
                let HorizontalConstraint{a, b} = constraint;
                let an = if let Some(an) = node_to_loc.get(&Loc::Node(a.clone())) { an } else { return true; };
                let bn = if let Some(bn) = node_to_loc.get(&Loc::Node(b.clone())) { bn } else { return true; };
                let aovr = an.0.0;
                let aohr = an.1.0;
                let bovr = bn.0.0;
                let bohr = bn.1.0;
                let ashr = p[aovr][aohr];
                let bshr = p[bovr][bohr];
                // imperfect without rank-spanning constraint edges but maybe a place to start?
                (aovr == bovr && ashr < bshr) || ashr <= bshr
            });

            let nesting_satisfied = container_borders.iter().all(|(container, borders)| {
                let contents = &nodes_by_container[container];
                let all_contents_allowed = borders.iter().all(|border| {
                    let (ovr, (ohr, mhr)) = border;
                    let covr = ovr.0;
                    let cohr1 = ohr.0;
                    let cohr2 = mhr.0;
                    let cshr1 = p[covr][cohr1];
                    let cshr2 = p[covr][cohr2];
                    let clshr = std::cmp::min(cshr1, cshr2);
                    let crshr = std::cmp::max(cshr1, cshr2);
                    let level = &locs_by_level2[ovr.0];
                    (clshr+1..crshr).all(|mid_shr| {
                        let mid_ohr = p[covr].iter().position(|shr| *shr == mid_shr).unwrap();
                        let mid_loc = &level[mid_ohr];
                        match mid_loc {
                            Loc::Node(ml) => {
                                contents.contains(&ml)
                            },
                            Loc::Hop(_mvr, mvl, mwl) => {
                                contents.contains(&mvl) || contents.contains(&mwl)
                            },
                            Loc::Border(mb) => {
                                contents.contains(&mb.vl)
                            },
                        }
                    })
                });
                let all_contents_present = nodes_by_container2[container].iter().copied().all(|(ovr, ohr)| {
                    borders.iter().filter(|(bovr, _)| ovr == *bovr).all(|(_, (bohr1, bohr2))| {
                        let movr = ovr.0;
                        let mohr = ohr.0;
                        let mshr = p[movr][mohr];
                        let bohr10 = bohr1.0;
                        let bohr20 = bohr2.0;
                        let bshr1 = p[movr][bohr10];
                        let bshr2 = p[movr][bohr20];
                        let blshr = std::cmp::min(bshr1, bshr2);
                        let brshr = std::cmp::max(bshr1, bshr2);
                        blshr < mshr && mshr < brshr
                    })
                });
                all_contents_allowed && all_contents_present
            });

            hcg_satisfied && nesting_satisfied
        }

        /// minimize_edge_crossing returns the obtained crossing number and a map of (ovr -> (ohr -> shr))
        #[allow(clippy::type_complexity)]
        pub fn minimize_edge_crossing<V>(
            vcg: &Vcg<V, V>,
            layout_problem: &LayoutProblem<V>
        ) -> Result<LayoutSolution, Error> where
            V: Display + Graphic
        {
            let Vcg{nodes_by_container, ..} = vcg;
            let LayoutProblem{loc_to_node, node_to_loc, locs_by_level, hops_by_level, ..} = layout_problem;

            eprintln!("MINIMIZE");
            eprintln!("LOCS_BY_LEVEL: {locs_by_level:#?}");
            eprintln!("HOPS_BY_LEVEL: {hops_by_level:#?}");
            let mut l2n = loc_to_node.iter().collect::<Vec<_>>();
            l2n.sort();
            eprintln!("LOC_TO_NODE: {l2n:#?}");
            
            if hops_by_level.is_empty() {
                return Ok(LayoutSolution{crossing_number: 0, solved_locs: BTreeMap::new()});
            }

            let mut shrs = vec![];
            for (_rank, locs) in locs_by_level.iter() {
                let n = *locs;
                let shrs_lvl = (0..n).collect::<Vec<_>>();
                shrs.push(shrs_lvl);
            }

            let mut shrs_ref = vec![];
            for shrs_lvl in shrs.iter_mut() {
                shrs_ref.push(&mut shrs_lvl[..]);
            }

            let mut locs_by_level2 = vec![];
            for (ovr, locs) in locs_by_level.iter() {
                let n = *locs;
                let level = (0..n).map(|ohr| { 
                    loc_to_node
                        .get(&(*ovr, OriginalHorizontalRank(ohr)))
                        .or_err(
                            Kind::KeyNotFoundError{key: format!("{ovr}, {ohr}")}
                        )
                }).collect::<Result<Vec<_>, _>>()?;
                locs_by_level2.push(level);
            }

            let mut nodes_by_container2 = HashMap::new();
            for (container, nodes) in nodes_by_container.iter() {
                let nodes = nodes.iter().map(|vl| node_to_loc[&Loc::Node(vl.clone())]).collect::<Vec<_>>();
                nodes_by_container2.insert(container.clone(), nodes);
            }

            let mut crossing_number = usize::MAX;
            let mut solution: Option<Vec<Vec<usize>>> = None;
            multisearch(&mut shrs_ref, |p| {
                // eprintln!("HEAPS PROCESS: ");
                // for (n, s) in p.iter().enumerate() {
                //     eprintln!("{n}: {s:?}");
                // }
                let mut cn = 0;
                for (rank, hops) in hops_by_level.iter() {
                    for h1i in 0..hops.len() {
                        for h2i in 0..hops.len() {
                            if h2i < h1i {
                                // eprintln!("hop: {h1} {h2} -> {}", crosses(h1, h2, p[rank.0], p[rank.0+1]));
                                let h1 = &hops[h1i];
                                let h2 = &hops[h2i];
                                cn += crosses(h1, h2, p[rank.0], p[rank.0+1]);
                            }   
                        }
                    }
                }
                // eprintln!("CN: {cn}");
                if cn < crossing_number {
                    if conforms(vcg, &layout_problem, &locs_by_level2, &nodes_by_container2, p) {
                        crossing_number = cn;
                        solution = Some(p.iter().map(|q| q.to_vec()).collect());
                        if crossing_number == 0 {
                            return true;
                        }
                    }
                }
                false
                // eprintln!("P cn: {cn}: p: {p:?}");
            });

            let solution = solution.or_err(LayoutError::HeapsError{error: "no solution found".into()})?;
            eprintln!("HEAPS CN: {crossing_number}");
            eprintln!("HEAPS SOL: ");
            for (n, s) in solution.iter().enumerate() {
                eprintln!("{n}: {s:?}");
            }
            
            let mut solved_locs = BTreeMap::new();
            for (lvl, shrs) in solution.iter().enumerate() {
                solved_locs.insert(VerticalRank(lvl), shrs
                    .iter()
                    .enumerate()
                    .map(|(a, b)| (OriginalHorizontalRank(a), SolvedHorizontalRank(*b))) // needs mutation testing
                    .collect::<BTreeMap<OriginalHorizontalRank, SolvedHorizontalRank>>());
            }

            eprintln!("SOLVED_LOCS: {solved_locs:#?}");

            Ok(LayoutSolution{crossing_number, solved_locs})
        }
    

        #[cfg(test)]
        mod tests {
            use itertools::Itertools;

            use super::*;
            #[test]
            fn test_is_odd() {
                assert!(!is_odd(0));
                assert!(is_odd(1));
                assert!(!is_odd(2));
                assert!(is_odd(3));
            }

            #[test]
            fn test_crosses() {
                // ???
            }

            #[test]
            fn test_search() {
                for size in 1..6 {      
                    let mut v = (0..size).collect::<Vec<_>>();
                    let mut ps = vec![];
                    search(&mut v, |p| ps.push(p.to_vec()));
                    assert_eq!(ps.len(), (1..=v.len()).product::<usize>());
                    assert_eq!(ps.len(), ps.iter().unique().count());
                }
            }

            #[test]
            fn test_multisearch() {
                for size in 1..5 {  
                    let mut vs = vec![];
                    for size2 in 1..=size {
                        vs.push((0..size2).collect::<Vec<_>>());
                    }
                    let mut vs2 = vec![];
                    for v in vs.iter_mut() {
                        vs2.push(&mut v[..]);
                    }
                    let mut ps = vec![];
                    multisearch(&mut vs2, |p| { ps.push(p.iter().map(|q| q.to_vec()).collect::<Vec<_>>()); false });
                    eprintln!("ps.len {}", ps.len());
                    for (n, p) in ps.iter().enumerate() {
                        eprintln!("n: {}, p: {:?}", n, p);
                    }
                    assert_eq!(ps.len(), vs2.iter().map(|v| (1..=v.len()).product::<usize>()).product());
                    assert_eq!(ps.len(), ps.iter().unique().count()); 
                }
            }
        }
        
    }

    /// Solve for horizontal ranks that minimize edge crossing
    pub use heaps::minimize_edge_crossing;

    use super::eval::Rel;
    use super::index::SolvedHorizontalRank;
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
    //! 3. then, the resulting [osqp_rust::Solution] needs to be destructured so that the resulting solution values can be returned to [`position_sols()`]'s caller as a [GeometrySolution].

    use enum_kinds::EnumKind;
    use ordered_float::OrderedFloat;
    #[cfg(all(not(feature="osqp"), feature="osqp-rust"))]
    use osqp_rust as osqp;
    #[cfg(all(feature="osqp", not(feature="osqp-rust")))]
    use osqp as osqp;

    use petgraph::EdgeDirection::{Outgoing, Incoming};
    use petgraph::visit::EdgeRef;
    use tracing::{event, Level, instrument};
    use tracing_error::InstrumentError;
    use typed_index_collections::TiVec;

    use crate::graph_drawing::layout::{HorizontalConstraint, Border};
    use crate::graph_drawing::osqp::{as_diag_csc_matrix, print_tuples, as_scipy, as_numpy};

    use super::error::{LayoutError};
    use super::osqp::{Constraints, Monomial, Vars, Fresh, Var, Sol, Coeff};

    use super::error::Error;
    use super::index::{VerticalRank, OriginalHorizontalRank, SolvedHorizontalRank, LocSol, HopSol};
    use super::layout::{Loc, Hop, Vcg, LayoutProblem, Graphic, LayoutSolution, Len};

    use std::borrow::Cow;
    use std::cmp::{max, max_by};
    use std::collections::{HashMap, BTreeMap, HashSet};
    use std::fmt::{Debug, Display};
    use std::hash::Hash;
    use std::ops::Mul;

    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, EnumKind)]
    #[enum_kind(AnySolKind)]
    pub enum AnySol {
        L(LocSol),
        R(LocSol),
        S(HopSol),
        T(LocSol),
        B(LocSol),
        F(usize)
    }

    impl Fresh for AnySol {
        fn fresh(index: usize) -> Self {
            Self::F(index)
        }
    }

    impl Display for AnySol {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                AnySol::L(loc) => write!(f, "l{}", loc.0),
                AnySol::R(loc) => write!(f, "r{}", loc.0),
                AnySol::S(hop) => write!(f, "s{}", hop.0),
                AnySol::T(loc) => write!(f, "t{}", loc.0),
                AnySol::B(loc) => write!(f, "b{}", loc.0),
                AnySol::F(idx) => write!(f, "f{}", idx),
            }
        }
    }

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

    #[derive(Clone, Debug, PartialEq, PartialOrd)]
    pub struct NodeSize {
        pub width: f64,
        pub left: f64,
        pub right: f64,
        pub height: f64,
    }

    #[derive(Clone, Debug, PartialEq, PartialOrd)]
    pub struct HopSize {
        pub width: f64,
        pub left: f64,
        pub right: f64,
        pub height: f64,
        pub top: f64,
        pub bottom: f64,
    }
    
    #[derive(Clone, Debug, Default)]
    pub struct GeometryProblem<V: Clone + Debug + Display + Ord + Hash> {
        pub all_locs: Vec<LocRow<V>>,
        pub all_hops0: Vec<HopRow<V>>,
        pub all_hops: Vec<HopRow<V>>,
        pub sol_by_loc: HashMap<(VerticalRank, OriginalHorizontalRank), LocSol>,
        pub sol_by_hop: HashMap<(VerticalRank, OriginalHorizontalRank, V, V), HopSol>,
        pub size_by_loc: HashMap<LocIx, NodeSize>,
        pub size_by_hop: HashMap<(VerticalRank, OriginalHorizontalRank, V, V), HopSize>,
        pub height_scale: Option<f64>,
        pub line_height: Option<f64>,
        pub char_width: Option<f64>,
        pub nesting_top_padding: Option<f64>,
        pub nesting_bottom_padding: Option<f64>,
    }
    
    /// ovr, ohr
    pub type LocIx = (VerticalRank, OriginalHorizontalRank);
    
    /// ovr, ohr -> loc
    pub type LocNodeMap<V> = HashMap<LocIx, Loc<V, V>>;
    
    /// lvl -> (mhr, nhr)
    pub type HopMap = BTreeMap<VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank)>;
    
    pub fn calculate_sols<'s, V>(
        layout_problem: &'s LayoutProblem<V>,
        layout_solution: &'s LayoutSolution,
    ) -> GeometryProblem<V> where
        V: Display + Graphic
    {
        let LayoutProblem{loc_to_node, hops_by_level, hops_by_edge, ..} = layout_problem;
        let LayoutSolution{solved_locs, ..} = layout_solution;

        eprintln!("SOLVED_LOCS {solved_locs:#?}");
        let sorted_loc_to_node = loc_to_node.iter()
            .map(|(loc, node)| 
                ((loc.0, loc.1), node)
            )
            .collect::<BTreeMap<(VerticalRank, OriginalHorizontalRank), _>>();
        eprintln!("LOC_TO_NODE CALC: {sorted_loc_to_node:#?}");

        let all_locs = solved_locs
            .iter()
            .flat_map(|(ovr, nodes)| nodes
                .iter()
                .map(|(ohr, shr)| (*ovr, *ohr, *shr, &loc_to_node[&(*ovr,*ohr)])))
            .enumerate()
            .map(|(n, (ovr, ohr, shr, loc))| LocRow{ovr, ohr, shr, loc: loc.clone(), n: LocSol(n)})
            .collect::<Vec<_>>();

        eprintln!("ALL_LOCS {all_locs:#?}");
    
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
    
        let size_by_loc = HashMap::new();
        let size_by_hop = HashMap::new();

        let height_scale = None;
        let line_height = None;
        let char_width = None;
        let nesting_top_padding = None;
        let nesting_bottom_padding = None;
    
        GeometryProblem{
            all_locs, 
            all_hops0, 
            all_hops, 
            sol_by_loc, 
            sol_by_hop, 
            size_by_loc, 
            size_by_hop, 
            height_scale,
            line_height,
            char_width,
            nesting_top_padding, 
            nesting_bottom_padding
        }
    }
    
    #[derive(Debug, Default)]
    pub struct GeometrySolution {
        pub ls: TiVec<LocSol, f64>,
        pub rs: TiVec<LocSol, f64>,
        pub ss: TiVec<HopSol, f64>,
        pub ts: TiVec<LocSol, f64>,
        pub bs: TiVec<LocSol, f64>,
    }

    fn update_min_width<V: Graphic + Display, E: Graphic>(
        vcg: &Vcg<V, E>, 
        layout_problem: &LayoutProblem<V>,
        layout_solution: &LayoutSolution,
        geometry_problem: &GeometryProblem<V>, 
        min_width: &mut usize, 
        vl: &V
    ) -> Result<(), Error> {
        let Vcg{vert: dag, vert_vxmap: dag_map, ..} = vcg;
        let LayoutProblem{node_to_loc, hops_by_edge, ..} = layout_problem;
        let LayoutSolution{solved_locs, ..} = layout_solution;
        let GeometryProblem{size_by_hop, ..} = geometry_problem;
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
            .filter_map(|(vl, wl)| {
                hops_by_edge.get(&(vl.clone(), wl.clone()))
                    .and_then(|hops| hops.iter().next())
                    .and_then(|(lvl, (mhr, _nhr))| 
                        Some((*lvl, *mhr, vl.clone(), wl.clone())))
            })
            .collect::<Vec<_>>();
        let w_in_last_hops = w_ins
            .iter()
            .filter_map(|(vl, wl)| {
                hops_by_edge.get(&(vl.clone(), wl.clone()))
                    .and_then(|hops| hops.iter().rev().next())
                    .and_then(|(lvl, (mhr, _nhr))|
                        Some((*lvl, *mhr, vl.clone(), wl.clone()))
                    )
            })
            .collect::<Vec<_>>();
        
        let out_width: f64 = v_out_first_hops
            .iter()
            .map(|idx| {
                let sz = &size_by_hop[idx];
                sz.left + sz.right
            })
            .sum();
        let in_width: f64 = w_in_last_hops
            .iter()
            .map(|idx| {
                let sz = &size_by_hop[idx];
                sz.left + sz.right
            })
            .sum();

        let in_width = in_width.round() as usize;
        let out_width = out_width.round() as usize;
        let orig_width = *min_width;
        // min_width += max_by(out_width, in_width, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
        *min_width = max(orig_width, max(in_width, out_width));
        event!(Level::TRACE, %vl, %min_width, %orig_width, %in_width, %out_width, "MIN WIDTH");
        // eprintln!("lvl: {}, vl: {}, wl: {}, hops: {:?}", lvl, vl, wl, hops);
        Ok(())
    }


    fn solve_problem<S: Sol, C: Coeff>(
        v: &Vars<S>,
        c: &Constraints<S, C>, 
        pd: &Vec<Monomial<S, C>>, 
        q: &Vec<Monomial<S, C>>,
        settings: &osqp::Settings,
    ) -> Result<Vec<(Var<S>, f64)>, Error> {
        let n = v.len();

        let sparse_pd = &pd[..];
        eprintln!("sparsePd: {sparse_pd:?}");
        let p2 = as_diag_csc_matrix(Some(n), Some(n), sparse_pd);
        print_tuples("P2", &p2);

        let mut q2 = Vec::with_capacity(n);
        q2.resize(n, 0.);
        for q in q.iter() {
            q2[q.var.index] += q.coeff.into(); 
        }

        let mut l2 = vec![];
        let mut u2 = vec![];
        for (l, _, u) in c.iter() {
            l2.push((*l).into());
            u2.push((*u).into());
        }
        eprintln!("V[{}]: {v}", v.len());
        eprintln!("C[{}]: {c}", &c.len());

        let a2: osqp::CscMatrix = c.clone().into();

        eprintln!("P2[{},{}]: {p2:?}", p2.nrows, p2.ncols);
        eprintln!("Q2[{}]: {q2:?}", q2.len());
        eprintln!("L2[{}]: {l2:?}", l2.len());
        eprintln!("U2[{}]: {u2:?}", u2.len());
        eprintln!("A2[{},{}]: {a2:?}", a2.nrows, a2.ncols);
        eprintln!("NUMPY");
        eprintln!("import osqp");
        eprintln!("import numpy as np");
        eprintln!("import scipy.sparse as sp");
        eprintln!("");
        eprintln!("inf = np.inf");
        eprintln!("np.set_printoptions(precision=1, suppress=True)");
        as_scipy("P", &p2);
        as_numpy("q", &q2);
        as_scipy("A", &a2);
        as_numpy("l", &l2);
        as_numpy("u", &u2);
        eprintln!("m = osqp.OSQP()");
        eprintln!("m.setup(P=P, q=q, A=A, l=l, u=u)");
        eprintln!("r = m.solve()");
        eprintln!("r.info.status");
        eprintln!("r.x");

        let mut prob = osqp::Problem::new(p2, &q2[..], a2, &l2[..], &u2[..], settings)
            .map_err(|e| Error::from(LayoutError::from(e).in_current_span()))?;
        
        let result = prob.solve();
        eprintln!("STATUS {:?}", result);
        let solution = match result {
            osqp::Status::Solved(solution) => Ok(solution),
            osqp::Status::SolvedInaccurate(solution) => Ok(solution),
            osqp::Status::MaxIterationsReached(solution) => Ok(solution),
            osqp::Status::TimeLimitReached(solution) => Ok(solution),
            _ => Err(LayoutError::OsqpError{error: "failed to solve problem".into(),}.in_current_span()),
        }?;
        let x = solution.x();

        // eprintln!("{:?}", x);
        let mut solutions = v.iter().map(|(_sol, var)| (*var, x[var.index])).collect::<Vec<_>>();
        solutions.sort_by_key(|(a, _)| *a);
        // for (var, val) in solutions {
        //     if !matches!(var.sol, AnySol::F(_)) {
        //         eprintln!("{} = {}", var.sol, val);
        //     }
        // }

        Ok(solutions)
    }

    fn extract_variable<Idx: Copy + Debug + Ord, Val: Copy + Debug>(
        v: &Vars<AnySol>,
        solutions: &Vec<(Var<AnySol>, Val)>,
        kind: AnySolKind,
        name: Cow<str>,
        extract_index: impl Fn(AnySol) -> Idx,
    ) -> TiVec<Idx, Val> {
        let mut vs = v.iter()
            .filter_map(|(sol, var)| {
                if AnySolKind::from(sol) == kind {
                    let sol_idx = extract_index(*sol);
                    Some((sol_idx, solutions[var.index].1))
                } else { 
                    None 
                }
            })
            .collect::<Vec<_>>();
        vs.sort_by_key(|(idx, _)| *idx);
        eprintln!("{name}: {vs:?}");
        let vs = vs.iter().map(|(_, v)| *v).collect::<TiVec<Idx, _>>();
        vs
    }

    #[instrument]
    pub fn position_sols<'s, V, E>(
        vcg: &'s Vcg<V, E>,
        layout_problem: &'s LayoutProblem<V>,
        layout_solution: &'s LayoutSolution,
        geometry_problem: &'s GeometryProblem<V>,
    ) -> Result<GeometrySolution, Error> where 
        V: Display + Graphic + Len,
        E: Graphic
    {
        let Vcg{
            vert_edge_labels: dag_edge_labels, 
            containers,
            nodes_by_container,
            nesting_depth_by_container,
            container_depths,
            ..
        } = vcg;
        let LayoutProblem{
            hops_by_edge,
            node_to_loc,
            loc_to_node,
            hcg,
            container_borders,
            ..
        } = layout_problem;
        let LayoutSolution{solved_locs, ..} = &layout_solution;
        let GeometryProblem{
            all_locs, 
            all_hops, 
            sol_by_loc, 
            sol_by_hop, 
            size_by_loc, 
            size_by_hop,
            char_width,
            height_scale,
            line_height,
            nesting_top_padding,
            nesting_bottom_padding,
            ..
        } = geometry_problem;

        let char_width = char_width.unwrap_or(9.);
        let height_scale = height_scale.unwrap_or(100.);
        let line_height = line_height.unwrap_or(20.);
        let nesting_top_padding = nesting_top_padding.unwrap_or(40.);
        let nesting_bottom_padding = nesting_bottom_padding.unwrap_or(10.);
    
        let mut max_edge_label_height_by_rank = BTreeMap::<VerticalRank, usize>::new();
        for (node, loc) in node_to_loc.iter() {
            let (ovr, _ohr) = loc;
            if let Loc::Node(vl) = node {
                let height_max = max_edge_label_height_by_rank.entry(*ovr).or_default();
                for (vl2, dsts) in dag_edge_labels.iter() {
                    if vl == vl2 {
                        let edge_labels = dsts
                            .iter()
                            .flat_map(|(_, rels)| rels
                                .iter()
                                .map(|(_, labels)| labels.len()))
                            .max()
                            .unwrap_or(1);
                        *height_max = max(*height_max, max(0, (edge_labels as i32) - 1) as usize);
                    } 
                }
            } else {
                max_edge_label_height_by_rank.entry(*ovr).or_default();
            }
        }
        // in each row, if there are containers, then there is a biggest container
        let mut max_nesting_span_by_rank = BTreeMap::<VerticalRank, usize>::new();
        let mut max_nesting_depth_by_rank = BTreeMap::<VerticalRank, usize>::new();
        for container in containers.iter() {
            let (ovr, ohr) = node_to_loc[&Loc::Node(container.clone())];
            let nesting_span = container_depths[container];
            let nesting_depth = nesting_depth_by_container[container] + 1;

            let max_span = max_nesting_span_by_rank.entry(ovr).or_default();
            *max_span = max(*max_span, nesting_span);

            let max_depth = max_nesting_depth_by_rank.entry(ovr).or_default();
            *max_depth = max(*max_depth, nesting_depth);
        }
        
        eprintln!("EDGE LABEL HEIGHTS {max_edge_label_height_by_rank:#?}");
        eprintln!("SPAN HEIGHTS {max_nesting_span_by_rank:#?}");
        eprintln!("PADDING HEIGHTS {max_nesting_depth_by_rank:#?}");
        event!(Level::TRACE, ?size_by_hop, "SIZE BY HOP");
        
    
        let sep = 20.0;

        let mut vh: Vars<AnySol> = Vars::new();
        let mut ch: Constraints<AnySol, OrderedFloat<f64>> = Constraints::new();
        let mut pdh: Vec<Monomial<AnySol, OrderedFloat<f64>>> = vec![];
        let mut qh: Vec<Monomial<AnySol, OrderedFloat<f64>>> = vec![];

        let mut vv: Vars<AnySol> = Vars::new();
        let mut cv: Constraints<AnySol, OrderedFloat<f64>> = Constraints::new();
        let mut pdv: Vec<Monomial<AnySol, OrderedFloat<f64>>> = vec![];
        let mut qv: Vec<Monomial<AnySol, OrderedFloat<f64>>> = vec![];

        let l = AnySol::L;
        let r = AnySol::R;
        let s = AnySol::S;
        let t = AnySol::T;
        let b = AnySol::B;

        eprintln!("SOL_BY_LOC: {sol_by_loc:#?}");
        
        let root_n = sol_by_loc[&(VerticalRank(0), OriginalHorizontalRank(0))];
        qh.push(vh.get(r(root_n)));
        qv.push(vv.get(b(root_n)));

        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        enum Loc2<V> {
            Node{vl: V, loc: LocIx, shr: SolvedHorizontalRank, sol: LocSol},
            Hop{vl: V, wl: V, loc: LocIx, shr: SolvedHorizontalRank, sol: HopSol, pshr: Option<SolvedHorizontalRank>},
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
                level_to_object.entry(*lvl).or_default().entry(shr).or_default().insert(Loc2::Hop{vl, wl, loc: (*lvl, *mhr), shr, sol: src_sol, pshr: None});
                level_to_object.entry(lvl1).or_default().entry(shrd).or_default().insert(Loc2::Hop{vl, wl, loc: (lvl1, *nhr), shr: shrd, sol: dst_sol, pshr: Some(shr)});
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
            let node_width = size_by_loc.get(&(ovr, ohr))
                .ok_or_else::<Error,_>(|| LayoutError::OsqpError{error: format!("missing node width: {ovr}, {ohr}")}.in_current_span().into())?;
            let mut min_width = node_width.width.round() as usize;
            let mut min_height = node_width.height.round() as usize;

            if let Loc::Border(_) = loc {
                continue;
            }

            if let Loc::Node(vl) = loc {
                update_min_width(vcg, layout_problem, layout_solution, geometry_problem, &mut min_width, vl)?;
            }

            if let Loc::Hop(_lvl, vl, wl) = loc {
                let ns = sol_by_hop[&(ovr, ohr, vl.clone(), wl.clone())];
                ch.leq(&mut vh, l(n), s(ns));
                ch.leq(&mut vh, s(ns), r(n));
                event!(Level::TRACE, ?loc, %n, %min_width, "X3: l{n} <= s{ns} <= r{n}");
            }
        
            if n != root_n {
                ch.leq(&mut vh, l(root_n), l(n));
                ch.leq(&mut vh, r(n), r(root_n));
            }
            event!(Level::TRACE, ?loc, %n, %min_width, "X0: r{n} >= l{n} + {min_width:.0?}");
            ch.leqc(&mut vh, l(n), r(n), min_width as f64);
            // ch.leqc(&mut vh, l(n), r(n), 40. + min_width as f64);
            // ch.sym(&mut vh, &mut pdh, l(n), r(n), 10000.);

            cv.leqc(&mut vv, t(n), b(n), 26.);
            qv.push(vv.get(b(n)));

            if let Some(ohrp) = locs.iter().position(|(_, shrp)| *shrp+1 == shr).map(OriginalHorizontalRank) {
                let mut process = true;
                if let Loc::Node(vl) = loc {
                    let vll = &loc_to_node[&(ovr, ohrp)];
                    if let Loc::Node(vll) = vll {
                        if containers.contains(vl) && nodes_by_container[vl].contains(vll) 
                        || containers.contains(vll) && nodes_by_container[vll].contains(vl) {
                            process = false;
                        }
                    }
                }
                if process {
                    let np = sol_by_loc[&(ovr, ohrp)];
                    let shrp = locs[&ohrp];
                    let wp = &size_by_loc[&(ovr, ohrp)];
                    let gap = max_by(sep, wp.right + node_width.left, f64::total_cmp);
                    ch.leqc(&mut vh, r(np), l(n), gap);
                    event!(Level::TRACE, ?loc, %ovr, %ohr, %shr, %n, %ovr, %ohrp, %shrp, %np, %gap, "X1: l{n} >= r{np} + ε")
                }
            }
            if let Some(ohrn) = locs.iter().position(|(_, shrn)| *shrn == shr+1).map(OriginalHorizontalRank) {
                let mut process = true;
                if let Loc::Node(vl) = loc {
                    let vlr = &loc_to_node[&(ovr, ohrn)];
                    if let Loc::Node(vlr) = vlr {
                        if containers.contains(vl) && nodes_by_container[vl].contains(vlr) 
                        || containers.contains(vlr) && nodes_by_container[vlr].contains(vl) {
                            process = false;
                        }
                    }
                }
                if process {
                    let nn = sol_by_loc[&(ovr,ohrn)];
                    let shrn = locs[&(ohrn)];
                    let wn = &size_by_loc[&(ovr, ohrn)];
                    let gap = max_by(sep, node_width.right + wn.left, f64::total_cmp);
                    ch.leqc(&mut vh, r(n), l(nn), gap);
                    event!(Level::TRACE, ?loc, %ovr, %ohr, %shr, %n, %ovr, %ohrn, %shrn, %nn, %gap, "X2: r{n} <= l{nn} - ε")
                }
            }
        }

        let mut already_seen = HashSet::new();
        for hop_row in all_hops.iter() {
            let HopRow{lvl, mhr, nhr, vl, wl, ..} = &hop_row;

            let ovr = lvl;
            let shr = &solved_locs[lvl][mhr];
            let n = sol_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())];
            let vloc = node_to_loc[&Loc::Node(vl.clone())].clone();
            let wloc = node_to_loc[&Loc::Node(wl.clone())].clone();
            let vn = sol_by_loc[&vloc];
            let wn = sol_by_loc[&wloc];

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
            let default_hop_size = HopSize{width: 0., left: 20., right: 20., height: 50., top: 0., bottom: 0.};
            let hop_size = &size_by_hop.get(&(*lvl, *mhr, vl.clone(), wl.clone())).unwrap_or(&default_hop_size);
            let (action_width, percept_width, min_hop_height) = (hop_size.left, hop_size.right, hop_size.height);
            // flow_width, flow_rev_width?

            ch.leqc(&mut vh, l(root_n), s(n), action_width);
            ch.leqc(&mut vh, s(n), r(root_n), percept_width);

            if !already_seen.contains(&(vn, wn)) {
                cv.leqc(&mut vv, b(vn), t(wn), min_hop_height);
                already_seen.insert((vn, wn));
            }
            
            // qv.push(vv.get(b(wn)));
            cv.sym(&mut vv, &mut pdv, b(wn), t(vn), 1.);
            

            if !terminal {
                let nd = sol_by_hop[&((*lvl+1), *nhr, (*vl).clone(), (*wl).clone())];
                ch.sym(&mut vh, &mut pdh, s(n), s(nd), 1.);
            }

            event!(Level::TRACE, ?hop_row, ?node, ?all_objects, "POS HOP START");
            if let Some(Loc2::Node{sol: nd, ..}) = node {
                ch.geqc(&mut vh, s(n), l(*nd), sep + action_width);
                ch.leqc(&mut vh, s(n), r(*nd), sep + percept_width);

                if terminal {
                    let mut terminal_hops = all_objects
                        .iter()
                        .filter(|obj| { matches!(obj, Loc2::Hop{wl: owl, pshr, ..} if *owl == wl && pshr.is_some()) })
                        .collect::<Vec<_>>();
                        #[allow(clippy::unit_return_expecting_ord)]

                    terminal_hops.sort_by_key(|hop| {
                        if let Loc2::Hop{pshr: Some(pshr), ..} = hop {
                            pshr
                        } else {
                            unreachable!();
                        }
                    });
                    event!(Level::TRACE, ?hop_row, ?node, ?terminal_hops, "POS HOP TERMINAL");

                    for (ox, hop) in terminal_hops.iter().enumerate() {
                        if let Loc2::Hop{vl: ovl, wl: owl, loc: (oovr, oohr), sol: on, ..} = hop {
                            let owidth = size_by_hop.get(&(*oovr, *oohr, (*ovl).clone(), (*owl).clone())).unwrap_or(&default_hop_size);
                            if let Some(Loc2::Hop{vl: ovll, wl: owll, loc: (oovrl, oohrl), sol: onl, ..}) = ox.checked_sub(1).and_then(|oxl| terminal_hops.get(oxl)) {
                                let owidth_l = size_by_hop.get(&(*oovrl, *oohrl, (*ovll).clone(), (*owll).clone())).unwrap_or(&default_hop_size);
                                ch.leqc(&mut vh, s(*onl), s(*on), sep + owidth_l.right + owidth.left);
                            }
                            if let Some(Loc2::Hop{vl: ovlr, wl: owlr, loc: (ovrr, oohrr), sol: onr, ..}) = terminal_hops.get(ox+1) {
                                let owidth_r = size_by_hop.get(&(*ovrr, *oohrr, (*ovlr).clone(), (*owlr).clone())).unwrap_or(&default_hop_size);
                                ch.leqc(&mut vh, s(*on), s(*onr), sep + owidth_r.left + owidth.right);
                            }
                        }
                    }
                } else {
                    let mut initial_hops = all_objects
                        .iter()
                        .filter(|obj| { matches!(obj, Loc2::Hop{vl: ovl, loc: (_, onhr), ..} if *ovl == vl && onhr.0 <= num_objects) })
                        .collect::<Vec<_>>();

                    #[allow(clippy::unit_return_expecting_ord)]
                    initial_hops.sort_by_key(|hop| {
                        if let Loc2::Hop{vl: hvl, wl: hwl, ..} = hop {
                            #[allow(clippy::unwrap_used)]
                            let (hvr, (_hmhr, hnhr)) = hops_by_edge[&((*hvl).clone(), (*hwl).clone())].iter().next().unwrap();
                            solved_locs[&(*hvr+1)][hnhr]
                        } else {
                            unreachable!()
                        }
                    });
                    
                    event!(Level::TRACE, ?hop_row, ?node, ?initial_hops, "POS HOP INITIAL");

                    for (ox, hop) in initial_hops.iter().enumerate() {
                        if let Loc2::Hop{vl: ovl, wl: owl, loc: (oovr, oohr), sol: on, ..} = hop {
                            let owidth = size_by_hop.get(&(*oovr, *oohr, (*ovl).clone(), (*owl).clone())).unwrap_or(&default_hop_size);
                            if let Some(Loc2::Hop{vl: ovll, wl: owll, loc: (oovrl, oohrl), sol: onl, ..}) = ox.checked_sub(1).and_then(|oxl| initial_hops.get(oxl)) {
                                let owidth_l = size_by_hop.get(&(*oovrl, *oohrl, (*ovll).clone(), (*owll).clone())).unwrap_or(&default_hop_size);
                                ch.leqc(&mut vh, s(*onl), s(*on), sep + owidth_l.right + owidth.left);
                            }
                            if let Some(Loc2::Hop{vl: ovlr, wl: owlr, loc: (ovrr, oohrr), sol: onr, ..}) = initial_hops.get(ox+1) {
                                let owidth_r = size_by_hop.get(&(*ovrr, *oohrr, (*ovlr).clone(), (*owlr).clone())).unwrap_or(&default_hop_size);
                                ch.leqc(&mut vh, s(*on), s(*onr), sep + owidth_r.left + owidth.right);
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
                            ch.geqc(&mut vh, s(n), r(*ln), sep + action_width);
                        },
                        Loc2::Hop{vl: lvl, wl: lwl, loc: (lvr, lhr), sol: ln, ..} => {
                            let hop_size_l = size_by_hop.get(&(*lvr, *lhr, (*lvl).clone(), (*lwl).clone())).unwrap_or(&default_hop_size);
                            ch.geqc(&mut vh, s(n), s(*ln), (2.*sep) + hop_size_l.right + hop_size.left);
                            
                            let lvl = (*lvl).clone();
                            let lwl = (*lwl).clone();
                            let lvloc = &node_to_loc[&Loc::Node(lvl.clone())];
                            let lwloc = &node_to_loc[&Loc::Node(lwl.clone())];
                            if vl != &lvl && lvloc.0 == *ovr {
                                let lvn = sol_by_loc[lvloc];
                                ch.geqc(&mut vh, s(n), r(lvn), sep + action_width);
                            }
                            if wl != &lwl && lwloc.0 == *ovr+1 {
                                let lwn = sol_by_loc[lwloc];
                                ch.geqc(&mut vh, s(n), r(lwn), sep + action_width);
                            }
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
                            ch.leqc(&mut vh, s(n), l(*rn), sep + percept_width);
                        },
                        Loc2::Hop{vl: rvl, wl: rwl, loc: (rvr, rhr), sol: rn, ..} => {
                            let hop_size_r = size_by_hop.get(&(*rvr, *rhr, (*rvl).clone(), (*rwl).clone())).unwrap_or(&default_hop_size);
                            ch.leqc(&mut vh, s(n), s(*rn), (2.*sep) + hop_size_r.left + hop_size.right);

                            let rvl = (*rvl).clone();
                            let rwl = (*rwl).clone();
                            let rvloc = &node_to_loc[&Loc::Node(rvl.clone())];
                            let rwloc = &node_to_loc[&Loc::Node(rwl.clone())];
                            if vl != &rvl && rvloc.0 == *ovr {
                                let rvn = sol_by_loc[rvloc];
                                ch.leqc(&mut vh, s(n), l(rvn), sep + percept_width);
                            }
                            if wl != &rwl && rwloc.0 == *ovr+1 {
                                let rwn = sol_by_loc[rwloc];
                                ch.leqc(&mut vh, s(n), l(rwn), sep + percept_width);
                            }
                        },
                    }
                }
            }
            
            event!(Level::TRACE, ?hop_row, ?node, ?all_objects, "POS HOP END");
        }

        for HorizontalConstraint{a: vl, b: wl} in hcg.constraints.iter() {
            let locl = &node_to_loc[&Loc::Node(vl.clone())];
            let locr = &node_to_loc[&Loc::Node(wl.clone())];
            let nl = sol_by_loc[locl];
            let nr = sol_by_loc[locr];
            // cv.eq(&[vv.get(t(nl)), vv.get(t(nr))]);
            cv.sym(&mut vv, &mut pdv, t(nl), t(nr), 10000.);
            // eprintln!("VSYM {vl} {wl} {nl} {nr}");
        }
        for ((vl, wl), lvl) in &hcg.labels {
            let locl = &node_to_loc[&Loc::Node(vl.clone())];
            let locr = &node_to_loc[&Loc::Node(wl.clone())];
            let nl = sol_by_loc[locl];
            let nr = sol_by_loc[locr];
            let mut left = 0;
            let mut right = 0;
            left = std::cmp::max(left, lvl.reverse.as_ref().and_then(|rs| rs.iter().map(|r| r.len()).max()).unwrap_or(0));
            right = std::cmp::max(right, lvl.forward.as_ref().and_then(|fs| fs.iter().map(|f| f.len()).max()).unwrap_or(0));
            let left_height = lvl.reverse.as_ref().map(|rs| rs.len()).unwrap_or(0);
            let right_height = lvl.forward.as_ref().map(|fs| fs.len()).unwrap_or(0);
            ch.leqc(&mut vh, r(nl), l(nr), sep + char_width * (left + right) as f64);

            let ovrl = locl.0;
            let ovrr = locr.0;
            // forward heights
            for objects in level_to_object.get(&(ovrr-1)).iter() {
                for (shro, objs) in objects.iter() {
                    for obj in objs {
                        let obj_loc = match obj {
                            Loc2::Node{loc: obj_loc, ..} => obj_loc,
                            Loc2::Hop{loc: obj_loc, ..} => obj_loc,
                        };
                        let nobj = sol_by_loc[&obj_loc];
                        let height = height_scale / 2. + line_height * right_height as f64;
                        cv.geqc(&mut vv, t(nl), b(nobj), height);
                        cv.geqc(&mut vv, t(nr), b(nobj), height);
                    }
                }   
            }
            let shrl = solved_locs[&locl.0][&locl.1];
            let shrr = solved_locs[&locr.0][&locr.1];
            let num_objects: usize = level_to_object.iter().flat_map(|row| row.1.iter().map(|cell| cell.1.len())).sum();
            for obj in level_to_object[&ovrr][&shrr].iter() {
                match obj {
                    Loc2::Hop{vl: hvl, wl: hwl, ..} => {
                        if *hwl == wl {
                            // todo!()
                            eprintln!("LOWER RIGHT CORNER {vl}-{wl}, {hvl}->{hwl}");
                        }
                    },
                    _ => {},
                }; 
            }
            for obj in level_to_object[&ovrl][&shrl].iter() {
                match obj {
                    Loc2::Hop{vl: hvl, wl: hwl, ..} => {
                        if *hvl == vl {
                            // todo!()
                            eprintln!("UPPER LEFT CORNER {vl}-{wl}, {hvl}->{hwl}");
                        }
                    },
                    _ => {},
                }; 
            }
            // reverse heights
            for objects in level_to_object.get(&(ovrl+1)).iter() {
                for (shro, objs) in objects.iter() {
                    for obj in objs {
                        let obj_loc = match obj {
                            Loc2::Node{loc: obj_loc, ..} => obj_loc,
                            Loc2::Hop{loc: obj_loc, ..} => obj_loc,
                        };
                        let nobj = sol_by_loc[&obj_loc];
                        let height = height_scale / 2. + line_height * (right_height + left_height) as f64;
                        cv.leqc(&mut vv, t(nl), t(nobj), height);
                        cv.leqc(&mut vv, t(nr), t(nobj), height);
                    }
                }
            }
        }

        for container in containers.iter() {
            let borders = &container_borders[container];
            let (ovr, (ohr, pair)) = borders.iter().next().unwrap();
            let shr1 = solved_locs[ovr][ohr];
            let shr2 = solved_locs[ovr][pair];
            let lohr = if shr1 < shr2 { ohr } else { pair };
            let rohr = if shr1 < shr2 { pair } else { ohr };
            let ln = sol_by_loc[&(*ovr, *lohr)];
            let rn = sol_by_loc[&(*ovr, *rohr)];
            let nc = ln;
            let np = rn;
            ch.leqc(&mut vh, l(nc), r(np), 20.);
            ch.leqc(&mut vh, l(nc), r(nc), 20.);
            ch.leqc(&mut vh, r(nc), l(np), 20.);
            ch.leqc(&mut vh, l(np), r(np), 20.);
            for node in nodes_by_container[container].iter() {
                let locn = &node_to_loc[&Loc::Node(node.clone())];
                let nn = sol_by_loc[locn];
                if containers.contains(node) {
                    let (ovr, (ohr, pair)) = container_borders[container].iter().copied().next().unwrap();
                    let (iovr, (iohr, ipair)) = container_borders[node].iter().copied().next().unwrap();
                    let shr1 = solved_locs[&ovr][&ohr];
                    let shr2 = solved_locs[&ovr][&pair];
                    let lohr = if shr1 < shr2 { ohr } else { pair };
                    let rohr = if shr1 < shr2 { pair } else { ohr };
                    let sol1 = sol_by_loc[&(ovr, lohr)];
                    let sol2 = sol_by_loc[&(ovr, rohr)];

                    let ishr1 = solved_locs[&iovr][&iohr];
                    let ishr2 = solved_locs[&iovr][&ipair];
                    let ilohr = if ishr1 < ishr2 { iohr } else { ipair };
                    let irohr = if ishr1 < ishr2 { ipair } else { iohr };
                    let isol1 = sol_by_loc[&(iovr, ilohr)];
                    let isol2 = sol_by_loc[&(iovr, irohr)];
                    
                    ch.leqc(&mut vh, l(sol1), l(isol1), 20.);
                    ch.leqc(&mut vh, r(isol2), r(sol2), 20.);
                } else {
                    ch.leqc(&mut vh, l(nc), l(nn), 40.);
                    ch.leqc(&mut vh, r(nn), r(np), 40.);
                    // ch.sym(&mut vh, &mut pdh, l(nc), r(nc), 100.);
                    // // ch.sym(&mut vh, &mut pdh, l(nc), r(nn), 1.);
                }
                cv.leqc(&mut vv, t(nc), t(nn), nesting_top_padding);
                cv.leqc(&mut vv, b(nn), b(nc), nesting_bottom_padding);
            }
            for border in container_borders[container].first().iter() {
                let (ovr, (ohr, pair)) = **border;
                let shr1 = solved_locs[&ovr][&ohr];
                let shr2 = solved_locs[&ovr][&pair];
                let lohr = if shr1 < shr2 { ohr } else { pair };
                let rohr = if shr1 < shr2 { pair } else { ohr };
                let sol1 = sol_by_loc[&(ovr, lohr)];
                let sol2 = sol_by_loc[&(ovr, rohr)];
                ch.leq(&mut vh, l(sol1), l(sol2));
                // ch.geq(&mut vh, l(sol1), l(sol2));
                ch.leq(&mut vh, r(sol1), r(sol2));
                // ch.geq(&mut vh, r(sol1), r(sol2));
                // // ch.sym(&mut vh, &mut pdh, l(sol1), r(sol2), 10.);
                // // ch.sym(&mut vh, &mut pdh, l(sol2), r(sol1), 10.);

                cv.leq(&mut vv, t(sol1), t(sol2));
                cv.geq(&mut vv, t(sol1), t(sol2));
                cv.leq(&mut vv, b(sol1), b(sol2));
                cv.geq(&mut vv, b(sol1), b(sol2));
            }
        }


        // add non-negativity constraints for all vars
        for (sol, var) in vh.iter() {
            if !matches!(sol, AnySol::F(_)) {
                ch.push(((0.).into(), vec![var.into()], f64::INFINITY.into()));
            }
            // if matches!(sol, AnySol::R(_)) {
            //     qh.push(10000. as f64 * Monomial::from(var));
            // }
        }
        for (n, (sol, var)) in vv.iter().enumerate() {
            if !matches!(sol, AnySol::F(_)) {
                cv.push(((0.).into(), vec![var.into()], f64::INFINITY.into()));
            }
            // if matches!(sol, AnySol::T(_)) {
            //     qv.push(10000. as f64 * Monomial::from(var));
            // }
        }

        let mut solutions_h;
        loop {
            let mut eps_abs = 1e-4_f64;
            let mut eps_rel = 1e-4_f64;
            eprintln!("SOLVE HORIZONTAL");
            solutions_h = solve_problem(&vh, &ch, &pdh, &qh, &osqp::Settings::default()
                .adaptive_rho(false)
                // .rho(1e2_f64)
                // .check_termination(Some(4000))
                // .adaptive_rho_fraction(1.0) // https://github.com/osqp/osqp/issues/378
                // .adaptive_rho_interval(Some(25))
                // .eps_abs(eps_abs)
                // .eps_rel(eps_rel)
                // .max_iter(128_000_000)
                // .scaled_termination(false)
                // .max_iter(400)
                // .polish(true)
                .verbose(true))?;
            
            let mut constraints_satisfied = true;
            for container in containers.iter() {
                let locc = &node_to_loc[&Loc::Node(container.clone())];
                let nc = sol_by_loc[locc];
                for node in nodes_by_container[container].iter() {
                    let locn = &node_to_loc[&Loc::Node(node.clone())];
                    let nn = sol_by_loc[locn];
                    let vlnc = vh.get::<OrderedFloat<f64>>(l(nc));
                    let vlnn = vh.get::<OrderedFloat<f64>>(l(nn));
                    let vrnn = vh.get::<OrderedFloat<f64>>(r(nn));
                    let vrnc = vh.get::<OrderedFloat<f64>>(r(nc));
                    let lnc = &solutions_h[vlnc.var.index].1;
                    let lnn = &solutions_h[vlnn.var.index].1;
                    let rnc = &solutions_h[vrnc.var.index].1;
                    let rnn = &solutions_h[vrnn.var.index].1;
                    if lnc + 39. >= *lnn {
                        eprintln!("CONSTRAINT VIOLATION: lnc-lnn conflict, {container}, {node}");
                        constraints_satisfied = false;
                    }
                    if rnn + 39. >= *rnc {
                        eprintln!("CONSTRAINT VIOLATION: rnn-rnc conflict, {container}, {node}");
                        constraints_satisfied = false;
                    }
                    if rnc - lnc < 21. {
                        eprintln!("CONSTRAINT VIOLATION: container too narrow, {container}");
                        constraints_satisfied = false;
                    }
                }
            }
            // if constraints_satisfied {
                break
            // } else {
            //     eps_abs = eps_abs / 10.;
            //     eps_rel = eps_rel / 10.;
            // }
        }
        
        eprintln!("SOLVE VERTICAL");
        let solutions_v = solve_problem(&vv, &cv, &pdv, &qv, &osqp::Settings::default()
            .adaptive_rho(false)
            // .check_termination(Some(200))
            // .adaptive_rho_fraction(1.0) // https://github.com/osqp/osqp/issues/378
            // .adaptive_rho_interval(Some(25))
            // .eps_abs(1e-4)
            // .eps_rel(1e-4)
            // .max_iter(128_000)
            // .max_iter(400)
            // .polish(true)
            .verbose(true))?;

        let ls = extract_variable(&vh, &solutions_h, AnySolKind::L, "ls".into(), |s| { 
            if let AnySol::L(l) = s { l } else { panic!() }
        });

        let rs = extract_variable(&vh, &solutions_h, AnySolKind::R, "rs".into(), |s| {
            if let AnySol::R(r) = s { r } else { panic!() }
        });

        let ss = extract_variable(&vh, &solutions_h, AnySolKind::S, "ss".into(), |s| { 
            if let AnySol::S(s) = s { s } else { panic!() }
        });

        let ts = extract_variable(&vv, &solutions_v, AnySolKind::T, "ts".into(), |s| {
            if let AnySol::T(t) = s { t } else { panic!() }
        });

        let bs = extract_variable(&vv, &solutions_v, AnySolKind::B, "bs".into(), |s| {
            if let AnySol::B(b) = s { b } else { panic!() }
        });

        let res = GeometrySolution{ls, rs, ss, ts, bs};
        event!(Level::DEBUG, ?res, "LAYOUT");
        Ok(res)
    }

}

pub mod frontend {
    use std::{fmt::Display, borrow::Cow, collections::HashMap, cmp::max_by};

    use logos::Logos;
    use self_cell::self_cell;
    use sorted_vec::SortedVec;
    use tracing::{event, Level};
    use tracing_error::InstrumentResult;

    use crate::{graph_drawing::{layout::{debug::debug, minimize_edge_crossing, calculate_vcg, condense, rank, calculate_locs_and_hops, calculate_hcg, fixup_hcg_rank, Border}, eval::{eval, index, resolve}, geometry::{calculate_sols, position_sols, HopSize}}, parser::{Item, Parser, Token}};

    use super::{layout::{Vcg, Cvcg, LayoutProblem, Graphic, Len, Loc, RankedPaths, LayoutSolution}, geometry::{GeometryProblem, GeometrySolution, NodeSize}, error::{Error, Kind, OrErrExt}, eval::Val};

    pub fn estimate_widths<I>(
        vcg: &Vcg<I, I>, 
        cvcg: &Cvcg<I, I>,
        layout_problem: &LayoutProblem<I>,
        geometry_problem: &mut GeometryProblem<I>
    ) -> Result<(), Error> where
        I: Graphic + Display + Len + PartialEq<&'static str>,
    {
        // let char_width = 8.67;
        let char_width = 9.0;
        let line_height = geometry_problem.line_height.unwrap_or(20.);
        let arrow_width = 40.0;
        
        let vert_node_labels = &vcg.vert_node_labels;
        let vert_edge_labels = &vcg.vert_edge_labels;
        let size_by_loc = &mut geometry_problem.size_by_loc;
        let size_by_hop = &mut geometry_problem.size_by_hop;
        let hops_by_edge = &layout_problem.hops_by_edge;
        let loc_to_node = &layout_problem.loc_to_node;
        let hcg = &layout_problem.hcg;
        let condensed = &cvcg.condensed;
        let condensed_vxmap = &cvcg.condensed_vxmap;

        eprintln!("LOC_TO_NODE WIDTHS: {loc_to_node:#?}");
        
        for (loc, node) in loc_to_node.iter() {
            let (ovr, ohr) = loc;
            if let Loc::Node(vl) = node {
                let label = vert_node_labels
                    .get(vl)
                    .or_err(Kind::KeyNotFoundError{key: vl.to_string()})?
                    .clone();
                // if !label.is_screaming_snake_case() {
                //     label = label.to_title_case();
                // }
                let mut left = 0;
                let mut right = 0;
                for (hc, lvl) in &hcg.labels {
                    if hc.0 == *vl {
                        right = std::cmp::max(right, lvl.forward.as_ref().and_then(|fs| fs.iter().map(|f| f.len()).max()).unwrap_or(0));
                    }
                    if hc.1 == *vl {
                        left = std::cmp::max(left, lvl.reverse.as_ref().and_then(|rs| rs.iter().map(|r| r.len()).max()).unwrap_or(0));
                    }
                }
                let size = NodeSize{
                    width: char_width * label.len() as f64,
                    left: char_width * left as f64,
                    right: char_width * right as f64,
                    height: 26.,
                };
                size_by_loc.insert((*ovr, *ohr), size);
            }
            if let Loc::Border(border) = node {
                let Border{ovr, ohr, pair, ..} = border;
                size_by_loc.insert((*ovr, *ohr), NodeSize{width: 10., left: 0., right: 0., height: 0.});
                size_by_loc.insert((*ovr, *pair), NodeSize{width: 10., left: 0., right: 0., height: 0.});
            }
        }
    
        for ((vl, wl), hops) in hops_by_edge.iter() {
            let mut action_width = 10.0;
            let mut percept_width = 10.0;
            let mut height = 0.;
            let cex = condensed.find_edge(condensed_vxmap[vl], condensed_vxmap[wl]).unwrap();
            let cew = condensed.edge_weight(cex).unwrap();
            for (vl, wl, ew) in cew.iter() {
                let label_width = vert_edge_labels
                    .get(vl)
                    .and_then(|dsts| dsts
                        .get(wl)
                        .and_then(|rels| rels.get(ew)))
                    .and_then(|labels| labels
                        .iter()
                        .map(|label| label.len())
                        .max()
                    );

                match ew {
                    x if *x == "senses" => {
                        percept_width = label_width.map(|label_width| arrow_width + char_width * label_width as f64).unwrap_or(20.0);
                    }
                    x if *x == "actuates" => {
                        action_width = label_width.map(|label_width| arrow_width + char_width * label_width as f64).unwrap_or(20.0);
                    }
                    _ => {}
                }

                let label_height = vert_edge_labels
                    .get(vl)
                    .and_then(|dsts| dsts
                        .get(wl)
                        .and_then(|rels| rels.get(ew)))
                    .map(|labels| labels.len());

                height = max_by(height, label_height.unwrap_or(1) as f64 * line_height, f64::total_cmp);
            }
            height += 50.;

            for (lvl, (mhr, _nhr)) in hops.iter() {
                size_by_hop.insert((*lvl, *mhr, vl.clone(), wl.clone()), HopSize{
                    width: 0.,
                    left: action_width,
                    right: percept_width,
                    height,
                    top: 0.,
                    bottom: 0.,
                });
                if size_by_loc.get(&(*lvl, *mhr)).is_none() {
                    size_by_loc.insert((*lvl, *mhr), NodeSize{
                        width: action_width + percept_width, 
                        left: 0., 
                        right: 0.,
                        height: 0.,
                    });
                }
            }
        }

        eprintln!("SIZE_BY_LOC: {size_by_loc:#?}");
        eprintln!("SIZE_BY_HOP: {size_by_hop:#?}");
    
        Ok(())
    }

    #[derive(Default)]
    pub struct Depiction<'s> {
        pub items: Vec<Item<'s>>,
        pub val: Val<Cow<'s, str>>,
        pub vcg: Vcg<Cow<'s, str>, Cow<'s, str>>,
        pub cvcg: Cvcg<Cow<'s, str>, Cow<'s, str>>,
        pub roots: SortedVec<Cow<'s, str>>,
        pub paths_by_rank: RankedPaths<Cow<'s, str>>,
        pub layout_problem: LayoutProblem<Cow<'s, str>>,
        pub layout_solution: LayoutSolution,
        pub geometry_problem: GeometryProblem<Cow<'s, str>>,
        pub geometry_solution: GeometrySolution,
    }

    self_cell!{
        pub struct RenderCell<'s> {
            owner: Cow<'s, str>,
            
            #[covariant]
            dependent: Depiction,
        }
    }

    pub fn render<'s>(data: Cow<'s, str>) -> Result<RenderCell, Error> {
        RenderCell::try_new(data, |data| {
            let mut p = Parser::new();
            {
                let lex = Token::lexer(&data);
                let tks = lex.collect::<Vec<_>>();
                event!(Level::TRACE, ?tks, "LEX");
            }
            let mut lex = Token::lexer(data);
            while let Some(tk) = lex.next() {
                p.parse(tk)
                    .map_err(|_| {
                        Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
                    })
                    .in_current_span()?
            }
    
            let items = p.end_of_input()
                .map_err(|_| {
                    Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
                })?;
    
            event!(Level::TRACE, ?items, "PARSE");
            eprintln!("PARSE {items:#?}");
    
            let mut val = eval(&items[..]);

            event!(Level::TRACE, ?val, "EVAL");
            eprintln!("EVAL {val:#?}");

            let mut scopes = HashMap::new();
            let val2 = val.clone();
            index(&val2, &mut vec![], &mut scopes);
            resolve(&mut val, &mut vec![], &scopes);

            eprintln!("SCOPES: {scopes:#?}");
            eprintln!("RESOLVE: {val:#?}");
    
            let vcg = calculate_vcg(&val)?;
            let hcg = calculate_hcg(&val)?;

            event!(Level::TRACE, ?val, "HCG");
            eprintln!("HCG {hcg:#?}");
    
            let Vcg{vert, containers, nodes_by_container, container_depths, ..} = &vcg;
    
            let cvcg = condense(vert)?;
            let Cvcg{condensed, condensed_vxmap: _} = &cvcg;
    
            let roots = crate::graph_drawing::graph::roots(condensed)?;
    
            let mut paths_by_rank = rank(condensed, &roots, containers, nodes_by_container, container_depths)?;

            fixup_hcg_rank(&hcg, &mut paths_by_rank);
    
            let layout_problem = calculate_locs_and_hops(&val, condensed, &paths_by_rank, &vcg, hcg)?;

            // ... adjust problem for horizontal edges
    
            let layout_solution = minimize_edge_crossing(&vcg, &layout_problem)?;
    
            let mut geometry_problem = calculate_sols(&layout_problem, &layout_solution);
    
            estimate_widths(&vcg, &cvcg, &layout_problem, &mut geometry_problem)?;
    
            let geometry_solution = position_sols(&vcg, &layout_problem, &layout_solution, &geometry_problem)?;
    
            debug(&layout_problem, &layout_solution);
            
            Ok(Depiction{
                items,
                val,
                vcg,
                cvcg,
                roots,
                paths_by_rank,
                layout_problem,
                layout_solution,
                geometry_problem,
                geometry_solution,
            })
        })
    }

    pub mod dom {
        use std::{borrow::Cow, cmp::max};

        use tracing::{instrument, event};

        use crate::graph_drawing::{error::{OrErrExt, Kind, Error}, layout::Loc, index::{VerticalRank, OriginalHorizontalRank, LocSol, HopSol}, geometry::{NodeSize, HopSize}};


        #[derive(Clone, Debug, PartialEq, PartialOrd)]
        pub struct Label {
            pub text: String,
            pub hpos: f64,
            pub width: f64,
            pub vpos: f64,
        }

        #[derive(Clone, Debug, PartialEq, PartialOrd)]
        pub enum Node {
            Div { key: String, label: String, hpos: f64, vpos: f64, width: f64, height: f64, z_index: usize, loc: LocSol, estimated_size: NodeSize },
            Svg { key: String, path: String, z_index: usize, rel: String, label: Option<Label>, hops: Vec<HopSol>, estimated_size: HopSize },
        }

        #[derive(Clone, Debug)]
        pub struct Drawing {
            pub crossing_number: Option<usize>,
            pub viewbox_width: f64,
            pub nodes: Vec<Node>,
        }

        impl Default for Drawing {
            fn default() -> Self {
                Self { 
                    crossing_number: Default::default(), 
                    viewbox_width: 1024.0,
                    nodes: Default::default() 
                }
            }
        }

        #[instrument(skip(data))]
        pub fn draw(data: String) -> Result<Drawing, Error> {
            let render_cell = super::render(Cow::Owned(data))?;
            let depiction = render_cell.borrow_dependent();
            
            let rs = &depiction.geometry_solution.rs;
            let ls = &depiction.geometry_solution.ls;
            let ss = &depiction.geometry_solution.ss;
            let ts = &depiction.geometry_solution.ts;
            let bs = &depiction.geometry_solution.bs;
            let sol_by_loc = &depiction.geometry_problem.sol_by_loc;
            let loc_to_node = &depiction.layout_problem.loc_to_node;
            let node_to_loc = &depiction.layout_problem.node_to_loc;
            let vert_node_labels = &depiction.vcg.vert_node_labels;
            let vert_edge_labels = &depiction.vcg.vert_edge_labels;
            let hops_by_edge = &depiction.layout_problem.hops_by_edge;
            let hcg = &depiction.layout_problem.hcg;
            let sol_by_hop = &depiction.geometry_problem.sol_by_hop;
            let size_by_loc = &depiction.geometry_problem.size_by_loc;
            let size_by_hop = &depiction.geometry_problem.size_by_hop;
            let crossing_number = depiction.layout_solution.crossing_number;
            let condensed = &depiction.cvcg.condensed;
            let containers = &depiction.vcg.containers;
            let container_borders = &depiction.layout_problem.container_borders;
            let container_depths = &depiction.vcg.container_depths;
            let nesting_depths = &depiction.vcg.nesting_depths;
            let nesting_depth_by_container = &depiction.vcg.nesting_depth_by_container;
            let solved_locs = &depiction.layout_solution.solved_locs;

            let height_scale = 100.0;
            let vpad = 50.0;
            let line_height = 20.0;
            let char_width = 9.0;
            let nesting_top_padding = 40.;
            let nesting_bottom_padding = 10.;

            let mut texts = vec![];

            let root_n = sol_by_loc[&(VerticalRank(0), OriginalHorizontalRank(0))];
            let root_width = rs[root_n] - ls[root_n];

            for (loc, node) in loc_to_node.iter() {
                let (ovr, ohr) = loc;
                if (*ovr, *ohr) == (VerticalRank(0), OriginalHorizontalRank(0)) { continue; }

                if let Loc::Node(vl) = node {
                    if !containers.contains(vl) {
                        let n = sol_by_loc[&(*ovr, *ohr)];

                        let lpos = ls[n];
                        let rpos = rs[n];

                        let nesting_depth = nesting_depths[vl] as f64;
                        let z_index = nesting_depths[vl];

                        let vpos = ts[n];
                        let width = (rpos - lpos).round();
                        let hpos = lpos.round();
                        let height = bs[n] - ts[n];
                        
                        let key = vl.to_string();
                        let label = vert_node_labels
                            .get(vl)
                            .or_err(Kind::KeyNotFoundError{key: vl.to_string()})?
                            .clone();
                        // if !label.is_screaming_snake_case() {
                        //     label = label.to_title_case();
                        // }
                        let estimated_size = size_by_loc[&(*ovr, *ohr)].clone();
                        texts.push(Node::Div{key, label, hpos, vpos, width, height, z_index, loc: n, estimated_size});
                    } 
                }
            }

            for container in containers {
                for border in container_borders[container].first().iter() {
                    let (ovr, (ohr, pair)) = border;
                    let shr1 = solved_locs[ovr][ohr];
                    let shr2 = solved_locs[ovr][pair];
                    let lohr = if shr1 < shr2 { ohr } else { pair };
                    let rohr = if shr1 < shr2 { pair } else { ohr };
                    let ln = sol_by_loc[&(*ovr, *lohr)];
                    let rn = sol_by_loc[&(*ovr, *rohr)];
                    let lpos = ls[ln];
                    let rpos = rs[rn];
                    let nesting_depth = nesting_depths[container] as f64;
                    let z_index = nesting_depths[container];
                    let vpos = ts[ln];
                    let width = (rpos - lpos).round();
                    let depth = container_depths[container] as f64;
                    let inner_depth = nesting_depth_by_container[container] as f64;
                    let hpos = lpos.round();
                    let height = bs[ln] - ts[ln];
                    eprintln!("HEIGHT: {container} {height}");

                    let key = format!("{}_{}_{}_{}", container, ovr, ohr, pair);
                    let mut label = vert_node_labels
                        .get(container)
                        .or_err(Kind::KeyNotFoundError{key: container.to_string()})?
                        .clone();
                    if label == "_" { label = String::new(); };
                    
                    let estimated_size = size_by_loc[&(*ovr, *ohr)].clone();
                    texts.push(Node::Div{key, label, hpos, vpos, width, height, z_index, loc: ln, estimated_size});
                }
            }

            let mut arrows = vec![];

            for cer in condensed.edge_references() {
                let mut prev_vwe = None;
                for (m, (vl, wl, ew)) in cer.weight().iter().enumerate() {
                    if *vl == "root" { continue; }

                    if prev_vwe == Some((vl, wl, ew)) {
                        continue
                    } else {
                        prev_vwe = Some((vl, wl, ew))
                    }

                    let label_text = vert_edge_labels
                        .get(vl)
                        .and_then(|dsts| dsts
                            .get(wl)
                            .and_then(|rels| rels.get(ew)))
                        .map(|v| v.join("\n"));

                    let hops = hops_by_edge.get(&(vl.clone(), wl.clone()));
                    let hops = if let Some(hops) = hops { hops } else { continue; };
                    // eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

                    let offset = match ew { 
                        x if x == "actuates" => -10.0,
                        x if x == "actuator" => -10.0,
                        x if x == "senses" => 10.0,
                        x if x == "sensor" => 10.0,
                        _ => 0.0,
                    };

                    let z_index = std::cmp::max(nesting_depths[vl], nesting_depths[wl]) + 1;

                    let mut path = vec![];
                    let mut label_hpos = None;
                    let mut label_width = None;
                    let mut label_vpos = None;
                    // use rand::Rng;
                    // let mut rng = rand::thread_rng();
                    let mut hn0 = vec![];
                    let mut estimated_size0 = None;

                    let ndv = nesting_depths[vl] as f64;
                    let ndw = nesting_depths[wl] as f64;

                    let fs = container_depths.get(vl).copied().and_then(|d| d.checked_sub(1)).unwrap_or(0);
                    let hops = if fs == 0 {
                        hops.iter().collect::<Vec<_>>()
                    } else {
                        hops.iter().skip(fs).collect::<Vec<_>>()
                    };
                    let vmin = bs[sol_by_loc[&node_to_loc[&Loc::Node(vl.clone())]]];
                    let vmax = ts[sol_by_loc[&node_to_loc[&Loc::Node(wl.clone())]]];
                    let fh = hops.iter().next().unwrap();
                    let lh = hops.iter().rev().next().unwrap();
                    let nh = hops.len();
                    let vs = (0..=nh).map(|lvl|  {
                        let fraction = lvl as f64 / nh as f64;
                        vmin + fraction * (vmax - vmin)
                    })
                        .collect::<Vec<_>>();
                    eprintln!("vl: {vl}, wl: {wl}, fs: {fs}, vmin: {vmin}, vmax: {vmax}, nh: {nh}, vs: {vs:?}");
                    for (n, hop) in hops.iter().enumerate() {
                        let (lvl, (mhr, nhr)) = *hop;
                        // let hn = sol_by_hop[&(*lvl+1, *nhr, vl.clone(), wl.clone())];
                        let hn = sol_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())];
                        let spos = ss[hn];
                        let hnd = sol_by_hop[&(*lvl+1, *nhr, vl.clone(), wl.clone())];
                        let sposd = ss[hnd];
                        let hpos = (spos + offset).round(); // + rng.gen_range(-0.1..0.1));
                        let hposd = (sposd + offset).round(); //  + 10. * lvl.0 as f64;
                        let lvl_offset = container_depths.get(vl).copied().unwrap_or(0);
                        eprintln!("HOP {vl} {wl} {n} {hop:?} {lvl} {} {}", ndv, lvl_offset);
                        let mut vpos = vs[n];
                        let mut vpos2 = vs[n+1];

                        if n == 0 {
                            hn0.push(hn);
                        }
                        hn0.push(hnd);
                        
                        if n == 0 {
                            estimated_size0 = Some(size_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())].clone());
                            let mut vpos = vpos;
                            if *ew == "senses" {
                                vpos += 7.0; // box height + arrow length
                            } else {
                                // vpos += 26.0;
                            }
                            path.push(format!("M {hpos} {vpos}"));
                        }

                        if n == 0 {
                            let n = sol_by_loc[&((*lvl+1), *nhr)];
                            // let n = sol_by_loc[&((*lvl), *mhr)];
                            label_hpos = Some(match ew {
                                x if x == "senses" => {
                                    // ls[n]
                                    hposd
                                },
                                x if x == "actuates" => {
                                    // ls[n]
                                    hposd
                                },
                                _ => hposd + 9.
                            });
                            label_width = Some(match ew {
                                x if x == "senses" => {
                                    // ls[n]
                                    rs[n] - sposd
                                },
                                x if x == "actuates" => {
                                    // ls[n]
                                    sposd - ls[n]
                                },
                                _ => rs[n] - ls[n]
                            });
                            label_vpos = Some(match ew {
                                x if x == "fake" => {
                                    if containers.contains(vl) {
                                        vpos - 1.
                                    } else {
                                        vpos + 14.
                                    }
                                },
                                _ => vpos - 20.,
                            });
                        }

                        // if n < hops.len() - 1 {
                        //     vpos2 += 26.;
                        // }

                        if n == hops.len() - 1 && *ew == "actuates" { 
                            vpos2 -= 7.0; // arrowhead length
                        }

                        path.push(format!("L {hposd} {vpos2}"));

                    }

                    let key = format!("{vl}_{wl}_{ew}_{m}");
                    let path = path.join(" ");

                    let mut label = None;

                    if let (Some(label_text), Some(label_hpos), Some(label_width), Some(label_vpos)) = (label_text, label_hpos, label_width, label_vpos) {
                        label = Some(Label{text: label_text, hpos: label_hpos, width: label_width, vpos: label_vpos})
                    }
                    arrows.push(Node::Svg{key, path, z_index, rel: ew.to_string(), label, hops: hn0, estimated_size: estimated_size0.unwrap()});
                }
            }
            let forward_voffset = 6.;
            let reverse_voffset = 20.;

            for (m, ((vl, wl), lvl)) in hcg.labels.iter().enumerate() {

                let z_index = std::cmp::max(nesting_depths[vl], nesting_depths[wl]) + 1;
                let ndv = nesting_depths[vl] as f64;
                let ndw = nesting_depths[vl] as f64;

                if let Some(forward) = &lvl.forward {
                    let key = format!("{vl}_{wl}_forward_{m}");
                    let locl = &node_to_loc[&Loc::Node(vl.clone())];
                    let locr = &node_to_loc[&Loc::Node(wl.clone())];
                    let nl = sol_by_loc[locl];
                    let nr = sol_by_loc[locr];
                    let lr = rs[nl];
                    let rl = ls[nr] - 7.;
                    let wl = size_by_loc[locl].right;
                    let wr = size_by_loc[locr].left;
                    let vposl = ts[nl] + forward_voffset;
                    let vposr = ts[nr] + forward_voffset;
                    let path = format!("M {} {} L {} {}", lr, vposl, rl, vposr);
                    let label_text = forward.join("\n");
                    let label_width = char_width * forward.iter().map(|f| f.len()).max().unwrap_or(0) as f64;
                    let label = if !forward.is_empty() {
                        Some(Label{text: label_text, hpos: rl, width: label_width, vpos: vposl })
                    } else {
                        None
                    };
                    let estimated_size = HopSize{ width: 0., left: wl, right: wr, height: 0., top: 0., bottom: 0. };
                    arrows.push(Node::Svg{key, path, z_index, label, rel: "forward".into(), hops: vec![], estimated_size });
                }
                if let Some(reverse) = &lvl.reverse {
                    let key = format!("{vl}_{wl}_reverse_{m}");
                    let locl = &node_to_loc[&Loc::Node(vl.clone())];
                    let locr = &node_to_loc[&Loc::Node(wl.clone())];
                    let nl = sol_by_loc[locl];
                    let nr = sol_by_loc[locr];
                    let lr = rs[nl] + 7.;
                    let rl = ls[nr];
                    let wl = size_by_loc[locl].right;
                    let wr = size_by_loc[locr].left;
                    let vposl = ts[nl] + reverse_voffset;
                    let vposr = ts[nr] + reverse_voffset;
                    let path = format!("M {} {} L {} {}", lr, vposl, rl, vposr);
                    let label_text = reverse.join("\n");
                    let label_width = char_width * reverse.iter().map(|f| f.len()).max().unwrap_or(0) as f64;
                    let label = if !reverse.is_empty() { 
                        Some(Label{text: label_text, hpos: lr, width: label_width, vpos: vposl })
                    } else {
                        None
                    };
                    let estimated_size = HopSize{ width: 0., left: wl, right: wr, height: 0., top: 0., bottom: 0. };
                    arrows.push(Node::Svg{key, path, z_index, label, rel: "reverse".into(), hops: vec![], estimated_size });
                }
            }

            let mut nodes = texts
                .into_iter()
                .chain(arrows.into_iter())
                .collect::<Vec<_>>();
            
            nodes.sort_by_key(|node| match node {
                Node::Div{z_index, ..} => *z_index,
                Node::Svg{z_index, ..} => *z_index,
            });

            event!(tracing::Level::TRACE, %root_width, ?nodes, "NODES");
            eprintln!("NODES: {nodes:#?}");

            Ok(Drawing{
                crossing_number: Some(crossing_number), 
                viewbox_width: root_width,
                nodes
            })
        }
    }

    #[cfg(feature="dioxus")]
    pub mod dioxus {
        use dioxus::prelude::*;

        use super::dom::{Drawing, Node, Label};

        use svg::{Document, node::{element::{Group, Marker, Path, Rectangle, Text as TextElt}, Node as _, Text}};

        use std::io::BufWriter;

        pub fn as_data_svg(drawing: Drawing) -> String {
            let viewbox_width = drawing.viewbox_width;
            let mut nodes = drawing.nodes;
            let viewbox_height = 768f64;

            let mut svg = Document::new()
                .set("viewBox", (0f64, 0f64, viewbox_width, viewbox_height))
                .set("text-depiction", "optimizeLegibility");

            svg.append(Marker::new()
                .set("id", "arrowhead")
                .set("markerWidth", 7)
                .set("markerHeight", 10)
                .set("refX", 0)
                .set("refY", 5)
                .set("orient", "auto")
                .set("viewBox", "0 0 10 10")
                .add(Path::new()
                    .set("d", "M 0 0 L 10 5 L 0 10 z")
                    .set("fill", "black")
                )
            );
            svg.append(Marker::new()
                .set("id", "arrowheadrev")
                .set("markerWidth", 7)
                .set("markerHeight", 10)
                .set("refX", 0)
                .set("refY", 5)
                .set("orient", "auto-start-reverse")
                .set("viewBox", "0 0 10 10")
                .add(Path::new()
                    .set("d", "M 0 0 L 10 5 L 0 10 z")
                    .set("fill", "black")
                )
            );

            nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for node in nodes {
                match node {
                    Node::Div{label, hpos, vpos, width, height, ..} => {
                            svg.append(Group::new()
                                .set("transform", format!("translate({hpos}, {vpos})"))
                                .add(Rectangle::new()
                                    .set("width", width)
                                    .set("height", height)
                                    .set("stroke", "black")
                                    .set("fill", "none"))
                                .add(TextElt::new()
                                    .set("text-anchor", "middle")
                                    .set("transform", format!("translate({}, {})", width / 2., 16.))
                                    .set("font-family", "serif") // 'Times New Roman', Times, serif
                                    .set("fill", "black")
                                    .set("stroke", "none")
                                    .add(Text::new(label)))
                            );
                    },
                    Node::Svg{path, rel, label, ..} => {
                        
                        let mut path_elt = Path::new()
                            .set("d", path)
                            .set("stroke", "black");
                        
                        match rel.as_str() {
                            "actuates" | "forward" => path_elt = path_elt.set("marker-end", "url(%23arrowhead)"),
                            "senses" | "reverse" => path_elt = path_elt.set("marker-start", "url(%23arrowheadrev)"),
                            _ => {},
                        };

                        if let Some(Label{text, hpos, width: _, vpos, ..}) = label {
                            for (lineno, line) in text.lines().enumerate() {
                                let translate = if rel == "actuates" { 
                                    format!("translate({}, {})", hpos-12., vpos + 56. + (20. * lineno as f64))
                                } else { 
                                    format!("translate({}, {})", hpos+12., vpos + 56. + (20. * lineno as f64))
                                };
                                let anchor = if rel == "actuates" {
                                    "end"
                                } else {
                                    "start"
                                };
                                svg.append(Group::new()
                                    .add(TextElt::new()
                                        .set("font-family", "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",\"Courier New\",monospace")
                                        .set("fill", "black")
                                        .set("stroke", "none")
                                        .set("transform", translate)
                                        .set("text-anchor", anchor)
                                        .add(Text::new(line))
                                    )
                                );
                            }
                        }

                        svg.append(Group::new()
                            .add(path_elt)
                        );
                    },
                }
            }

            let mut buf = BufWriter::new(Vec::new());
            svg::write(&mut buf, &svg).unwrap();
            let bytes = buf.into_inner().unwrap();
            let svg_str = String::from_utf8(bytes).unwrap();
            format!("data:image/svg+xml;utf8,{svg_str}")
        }

        pub fn render<P>(cx: Scope<P>, drawing: Drawing)-> Option<VNode> {
            let viewbox_width = drawing.viewbox_width;
            let mut nodes = drawing.nodes;
            let viewbox_height = 768;
            let mut children = vec![];
            nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for node in nodes {
                match node {
                    Node::Div{key, label, hpos, vpos, width, height, z_index, ..} => {
                        children.push(cx.render(rsx! {
                            div {
                                key: "{key}",
                                style: "position: absolute; padding-top: 3px; padding-bottom: 3px; box-sizing: border-box; border: 1px solid black; text-align: center; z-index: 10; background-color: #fff;", // bg-opacity-50
                                top: "{vpos}px",
                                left: "{hpos}px",
                                width: "{width}px",
                                height: "{height}px",
                                z_index: "{z_index}",
                                span {
                                    "{label}"
                                }
                            }
                        }));
                    },
                    Node::Svg{key, path, z_index, rel, label, ..} => {
                        let marker_id = if rel == "actuates" || rel == "forward" { "arrowhead" } else { "arrowheadrev" };
                        let marker_orient = if rel == "actuates" || rel == "forward" { "auto" } else { "auto-start-reverse" };
                        let stroke_dasharray = if rel == "fake" { "5 5" } else { "none" };
                        let stroke_color = if rel == "fake" { "hsl(0, 0%, 50%)" } else { "currentColor" };
                        children.push(cx.render(rsx!{
                            div {
                                key: "{key}",
                                style: "position: absolute;",
                                z_index: "{z_index}",
                                svg {
                                    fill: "none",
                                    stroke: "{stroke_color}",
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "1",
                                    view_box: "0 0 {viewbox_width} {viewbox_height}",
                                    width: "{viewbox_width}px",
                                    height: "{viewbox_height}px",
                                    marker {
                                        id: "{marker_id}",
                                        markerWidth: "7",
                                        markerHeight: "10",
                                        refX: "0",
                                        refY: "5",
                                        orient: "{marker_orient}",
                                        view_box: "0 0 10 10",
                                        path {
                                            d: "M 0 0 L 10 5 L 0 10 z",
                                            fill: "#000",
                                        }
                                    }
                                    { 
                                        match rel.as_str() {
                                            "actuates" | "forward" => {
                                                rsx!(path {
                                                    d: "{path}",
                                                    marker_end: "url(#arrowhead)",
                                                })
                                            },
                                            "senses" | "reverse" => {
                                                rsx!(path {
                                                    d: "{path}",
                                                    "marker-start": "url(#arrowheadrev)",
                                                    // marker_start: "url(#arrowhead)", // BUG: should work, but doesn't.
                                                })
                                            },
                                            _ => {
                                                rsx!(path {
                                                    d: "{path}",
                                                    stroke_dasharray: "{stroke_dasharray}",
                                                })
                                            }
                                        }
                                    }
                                }
                                {match label { 
                                    Some(Label{text, hpos, width: _, vpos}) => {
                                        let translate = match &rel[..] {
                                            "actuates" => "translate(calc(-100% - 1.5ex))",
                                            "forward" => "translate(calc(-100% - 1.5ex), calc(-100% + 20px))",
                                            "senses" | "reverse" => "translate(1.5ex)",
                                            _ => "translate(0px, 0px)",
                                        };
                                        let offset = match &rel[..] {
                                            "actuates" | "senses" => "40px",
                                            "forward" => "-24px",
                                            "reverse" => "4px",
                                            _ => "0px",
                                        };
                                        // let border = match rel.as_str() { 
                                        //     // "actuates" => "border border-red-300",
                                        //     // "senses" => "border border-blue-300",
                                        //     _ => "",
                                        // };
                                        rsx!(div {
                                            style: "position: absolute;",
                                            left: "{hpos}px",
                                            // width: "{width}px",
                                            top: "calc({vpos}px + {offset})",
                                            div {
                                                style: "white-space: pre; z-index: 50; background-color: #fff; box-sizing: border-box; font-size: .875rem; line-height: 1.25rem; font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",\"Courier New\",monospace;",
                                                transform: "{translate}",
                                                "{text}"
                                            }
                                        })
                                    },
                                    _ => rsx!(div {}),
                                }}
                            }
                        }));
                    },
                }
            }
            // dbg!(cx.render(rsx!(children)))
            cx.render(rsx!(children))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{error::Error};
    use crate::{parser::{Parser, Token, Item}, graph_drawing::{layout::{*}, graph::roots, index::{VerticalRank, OriginalHorizontalRank}, geometry::calculate_sols, error::Kind, eval}};
    use tracing_error::InstrumentResult;
    use logos::Logos;

    #[test]
    #[allow(clippy::unwrap_used)]    
    pub fn no_swaps() -> Result<(), Error> {
        let data = "Aa Ab Ac: y / z\nXx Xy Xz: w / x";
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
        
        let val = eval::eval(&v[..]);

        let vcg = calculate_vcg(&val)?;
        let hcg = calculate_hcg(&val)?;

        let Vcg{vert, vert_vxmap, containers, nodes_by_container, container_depths, ..} = &vcg;
        let vx = vert_vxmap["Ab"];
        let wx = vert_vxmap["Ac"];
        assert_eq!(vert.node_weight(vx), Some(&Cow::from("Ab")));
        assert_eq!(vert.node_weight(wx), Some(&Cow::from("Ac")));

        let Cvcg{condensed, condensed_vxmap} = condense(&vert)?;
        let cvx = condensed_vxmap["Ab"];
        let cwx = condensed_vxmap["Ac"];
        assert_eq!(condensed.node_weight(cvx), Some(&Cow::from("Ab")));
        assert_eq!(condensed.node_weight(cwx), Some(&Cow::from("Ac")));

        let roots = roots(&condensed)?;

        let paths_by_rank = rank(&condensed, &roots, &containers, &nodes_by_container, &container_depths)?;
        assert_eq!(paths_by_rank[&VerticalRank(3)][0], (Cow::from("root"), Cow::from("Ac")));

        let layout_problem = calculate_locs_and_hops(&val, &condensed, &paths_by_rank, &vcg, hcg)?;
        let LayoutProblem{hops_by_level, hops_by_edge, loc_to_node, node_to_loc, ..} = &layout_problem;
        let nAa = Loc::Node(Cow::from("Aa"));
        let nAb = Loc::Node(Cow::from("Ab"));
        let nAc = Loc::Node(Cow::from("Ac"));
        let nXx = Loc::Node(Cow::from("Xx"));
        let nXy = Loc::Node(Cow::from("Xy"));
        let nXz = Loc::Node(Cow::from("Xz"));
        let lAa = node_to_loc[&nAa];
        let lAb = node_to_loc[&nAb];
        let lAc = node_to_loc[&nAc];
        let lXx = node_to_loc[&nXx];
        let lXy = node_to_loc[&nXy];
        let lXz = node_to_loc[&nXz];
        // assert_eq!(lv, (2, 1));
        // assert_eq!(lw, (3, 0));
        // assert_eq!(lp, (2, 0));
        // assert_eq!(lq, (3, 1));
        // assert_eq!(lv.1.0 + lw.1.0, 1); // lv.1 != lw.1
        // assert_eq!(lp.1.0 + lq.1.0, 1); // lp.1 != lq.1

        let nAa2 = &loc_to_node[&lAa];
        let nAb2 = &loc_to_node[&lAb];
        let nAc2 = &loc_to_node[&lAc];
        let nXx2 = &loc_to_node[&lXx];
        let nXy2 = &loc_to_node[&lXy];
        let nXz2 = &loc_to_node[&lXz];
        assert_eq!(nAa2, &nAa);
        assert_eq!(nAb2, &nAb);
        assert_eq!(nAc2, &nAc);
        assert_eq!(nXx2, &nXx);
        assert_eq!(nXy2, &nXy);
        assert_eq!(nXz2, &nXz);


        assert_eq!(hops_by_level.len(), 3);
        let h0 = &hops_by_level[&VerticalRank(0)];
        let h1 = &hops_by_level[&VerticalRank(1)];
        let h2 = &hops_by_level[&VerticalRank(2)];
        let ohr = OriginalHorizontalRank;
        let vr = VerticalRank;
        let lRr = &node_to_loc[&Loc::Node("root".into())];
        let h0A: Hop<Cow<str>> = Hop { mhr: lRr.1, nhr: lAa.1, vl: "root".into(), wl: "Aa".into(), lvl: lRr.0 };
        let h0X: Hop<Cow<str>> = Hop { mhr: lRr.1, nhr: lXx.1, vl: "root".into(), wl: "Xx".into(), lvl: lRr.0 };
        let h1A: Hop<Cow<str>> = Hop { mhr: lAa.1, nhr: lAb.1, vl: "Aa".into(), wl: "Ab".into(), lvl: lAa.0 };
        let h1X: Hop<Cow<str>> = Hop { mhr: lXx.1, nhr: lXy.1, vl: "Xx".into(), wl: "Xy".into(), lvl: lXx.0 };
        let h2A: Hop<Cow<str>> = Hop { mhr: lAb.1, nhr: lAc.1, vl: "Ab".into(), wl: "Ac".into(), lvl: lAb.0 };
        let h2X: Hop<Cow<str>> = Hop { mhr: lXy.1, nhr: lXz.1, vl: "Xy".into(), wl: "Xz".into(), lvl: lXy.0 };
        let mut s0 = vec![h0A.clone(), h0X.clone()];
        let mut s1 = vec![h1A.clone(), h1X.clone()];
        let mut s2 = vec![h2A.clone(), h2X.clone()];
        s0.sort();
        s1.sort();
        s2.sort();
        let mut h0 = h0.iter().cloned().collect::<Vec<_>>();
        h0.sort();
        let mut h1 = h1.iter().cloned().collect::<Vec<_>>();
        h1.sort();
        let mut h2 = h2.iter().cloned().collect::<Vec<_>>();
        h2.sort();
        assert_eq!(&h0[..], &s0[..]);
        assert_eq!(&h1[..], &s1[..]);
        assert_eq!(&h2[..], &s2[..]);
        
        let layout_solution = minimize_edge_crossing(&vcg, &layout_problem)?;
        let LayoutSolution{crossing_number, solved_locs} = &layout_solution;
        assert_eq!(*crossing_number, 0);
        // let sv = solved_locs[&2][&1];
        // let sw = solved_locs[&3][&0];
        // let sp = solved_locs[&2][&0];
        // let sq = solved_locs[&3][&1];
        let sAa = solved_locs[&lAa.0][&lAa.1];
        let sAb = solved_locs[&lAb.0][&lAb.1];
        let sAc = solved_locs[&lAc.0][&lAc.1];
        let sXx = solved_locs[&lXx.0][&lXx.1];
        let sXy = solved_locs[&lXy.0][&lXy.1];
        let sXz = solved_locs[&lXz.0][&lXz.1];
        eprintln!("{:<2}: lv0: {}, lv1: {}, sv: {}", "Ab", &lAb.0, &lAb.1, sAb);
        eprintln!("{:<2}: lw0: {}, lw1: {}, sw: {}", "Ac", &lAc.0, &lAc.1, sAc);
        // assert_eq!(sv, 1);
        // assert_eq!(sw, 1);
        // assert_eq!(sp, 0);
        // assert_eq!(sq, 0);
        assert_eq!(sAa, sAb);
        assert_eq!(sAb, sAc); // uncrossing happened
        assert_eq!(sXx, sXy);
        assert_eq!(sXy, sXz);
        

        let geometry_problem = calculate_sols(&layout_problem, &layout_solution);
        let all_locs = &geometry_problem.all_locs;
        let lrAb = all_locs.iter().find(|lr| lr.loc == nAb).unwrap();
        let lrAc = all_locs.iter().find(|lr| lr.loc == nAc).unwrap();
        let lrXy = all_locs.iter().find(|lr| lr.loc == nXy).unwrap();
        let lrXz = all_locs.iter().find(|lr| lr.loc == nXz).unwrap();
        assert_eq!(lrAb.ovr, lAb.0);
        assert_eq!(lrAc.ovr, lAc.0);
        assert_eq!(lrXy.ovr, lXy.0);
        assert_eq!(lrXz.ovr, lXz.0);
        assert_eq!(lrAb.ohr, lAb.1);
        assert_eq!(lrAc.ohr, lAc.1);
        assert_eq!(lrXy.ohr, lXy.1);
        assert_eq!(lrXz.ohr, lXz.1);
        assert_eq!(lrAb.shr, sAb);
        assert_eq!(lrAc.shr, sAc);
        assert_eq!(lrXy.shr, sXy);
        assert_eq!(lrXz.shr, sXz);

        Ok(())
    }
}
