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

    use super::frontend::log::Error as LogError;

    #[cfg(all(feature="osqp", not(feature="osqp-rust")))]
    use osqp;
    #[cfg(all(not(feature="osqp"), feature="osqp-rust"))]
    use osqp_rust as osqp;

    #[non_exhaustive]
    #[derive(Debug, thiserror::Error)]
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
    #[derive(Debug, thiserror::Error)]
    pub enum Error {
        #[error(transparent)]
        TypeError{
            #[from] source: TypeError,
        },
        #[error(transparent)]
        GraphDrawingError{
            #[from] source: Kind,
        },
        #[error(transparent)]
        RankingError{
            #[from] source: RankingError,
        },
        #[error(transparent)]
        LayoutError{
            #[from] source: LayoutError,
        },
        #[error(transparent)]
        LogError{
            #[from] source: LogError,
        }
    }

    /// A trait to use to annotate [Option] values with rich error information.
    pub trait OrErrExt<E> {
        type Item;
        fn or_err(self, error: E) -> Result<Self::Item, Error>;
    }

    impl<V, E> OrErrExt<E> for Option<V> where Error: From<E> {
        type Item = V;
        fn or_err(self, error: E) -> Result<V, Error> {
            self.ok_or_else(|| Error::from(error))
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
                    Err(Error::from(Kind::IndexingError{}))
                },
            }
        }
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

    use super::frontend::log;

    macro_rules! impl_index {
        ($index_name:ident, $tag:literal) => {
            #[derive(Clone, Copy, Eq, From, Hash, Into, Ord, PartialEq, PartialOrd)]
            pub struct $index_name(pub usize);

            impl $index_name {
                pub fn checked_sub(self, rhs: usize) -> Option<Self> {
                    self.0.checked_sub(rhs).map(|s| Self(s))
                }
            }

            impl Debug for $index_name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}{}", self.0, $tag)
                }
            }

            impl Display for $index_name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}", self.0)
                }
            }

            impl log::Name for $index_name {
                fn name(&self) -> String {
                    format!("{:?}", self)
                }
            }

            impl Add<usize> for $index_name {
                type Output = Self;

                fn add(self, rhs: usize) -> Self::Output {
                    Self(self.0 + rhs)
                }
            }

            impl Sub<usize> for $index_name {
                type Output = Self;

                fn sub(self, rhs: usize) -> Self::Output {
                    Self(self.0 - rhs)
                }
            }
        };
    }

    impl_index!(VerticalRank, "v");

    impl_index!(OriginalHorizontalRank, "h");

    impl_index!(SolvedHorizontalRank, "s");

    impl_index!(VarRank, "x");

}

pub mod eval {
    //! The main "value" type of depiction parts.

    use std::{collections::{HashMap}, borrow::{Cow}, vec::IntoIter, ops::Deref, slice::Iter};

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
    #[derive(Clone, Debug, PartialEq)]
    pub struct Level<V> {
        /// What are the labels for the "forward" direction
        /// of this link of the chain?
        pub forward: Option<Vec<V>>,
        /// What are the labels for the "reverse" direction
        /// of this link of the chain?
        pub reverse: Option<Vec<V>>,
    }

    /// How are these parts related to the whole they make up?
    #[derive(Clone, Debug, PartialEq)]
    pub enum Body<V> {
        /// States or modes the process could be in
        Any(Vec<Val<V>>),
        /// Parts or constituents making up the process
        All(Vec<Val<V>>),
    }

    impl<V> AsMut<Vec<Val<V>>> for Body<V> {
        fn as_mut(&mut self) -> &mut Vec<Val<V>> {
            match self {
                Body::Any(b) => b,
                Body::All(b) => b,
            }
        }
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
    #[derive(Clone, Debug, PartialEq)]
    pub enum Val<V> {
        Process {
            /// Maybe this process is named?
            name: Option<V>,
            /// Maybe this process has a label?
            label: Option<V>,
            /// Maybe this process has nested parts?
            body: Option<Body<V>>,
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
            }
        }
    }

    impl<V> Val<V> {
        pub fn name(&self) -> Option<&V> {
            match self {
                Val::Process { name, .. } => name.as_ref(),
                Val::Chain { name, .. } => name.as_ref(),
            }
        }

        pub fn set_name(&mut self, name: V) -> &mut Self {
            match self {
                Val::Process { name: n, .. } => { *n = Some(name); },
                Val::Chain { name: n, .. } => { *n = Some(name); },
            }
            self
        }

        pub fn label(&self) -> Option<&V> {
            match self {
                Val::Process { label, .. } => label.as_ref(),
                Val::Chain { .. } => None,
            }
        }

        pub fn set_label(&mut self, label: Option<V>) -> &mut Self {
            match self {
                Val::Process { label: l, .. } => { *l = label; },
                Val::Chain { .. } => {},
            }
            self
        }

        pub fn set_body(&mut self, body: Option<Body<V>>) -> &mut Self {
            match self {
                Val::Process{ body: b, .. } => { *b = body; }
                _ => {},
            }
            self
        }
    }

    fn as_string1<'s>(o1: &Option<Cow<'s, str>>, o2: &'static str) -> Cow<'s, str> {
        let o2 = Cow::from(o2);
        o1.as_ref().or_else(|| Some(&o2)).unwrap().clone()
    }

    fn as_string2<'s>(o1: &Option<Cow<'s, str>>, o2: &Option<Cow<'s, str>>, o3: &'static str) -> Cow<'s, str> {
        let o3 = Cow::from(o3);
        o1.as_ref().or_else(|| o2.as_ref()).or_else(|| Some(&o3)).unwrap().clone()
    }

    use crate::graph_drawing::frontend::log;
    impl<'s, 't> log::Log<Cow<'s, str>> for &'t Val<Cow<'s, str>> {
        fn log(&self, _cx: Cow<'s, str>, l: &mut log::Logger) -> Result<(), log::Error> {
            match self {
                Val::Process { name, label, body } => {
                    let pname = &as_string2(name, label, "");
                    l.with_group(
                        "Process",
                        pname.clone(),
                        vec![pname.to_string()],
                        |l| {
                            match body {
                                Some(body) if body.len() > 0 => match body {
                                    Body::Any(bs) => {
                                        l.with_group("", "Body: Any", Vec::<String>::new(), |l| {
                                            for b in bs.iter() {
                                                b.log(pname.clone(), l)?
                                            }
                                            Ok(())
                                        })
                                    },
                                    Body::All(bs) => {
                                        l.with_group("", "Body: All", Vec::<String>::new(), |l| {
                                            for b in bs.iter() {
                                                b.log(pname.clone(), l)?
                                            }
                                            Ok(())
                                        })
                                    },
                                },
                                _ => Ok(()),
                            }
                        })
                },
                Val::Chain { name, rel, path, labels } => {
                    let name = as_string1(name, "").clone();
                    l.with_group("Chain", name, Vec::<String>::new(), |l| {
                        l.log_string("rel", rel)?;
                        l.with_group("", "path", Vec::<String>::new(), |l| {
                            for p in path.iter() {
                                p.log("".into(), l)?
                            }
                            Ok(())
                        })?;
                        if !labels.is_empty() {
                            l.log_string("labels", labels)?;
                        }
                        Ok(())
                    })
                },
            }
        }
    }

    fn eval_path<'s, 't>(path: &'t [Item<'s>]) -> Vec<Val<Cow<'s, str>>> {
        // eprintln!("EVAL_PATH: {path:?}");
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

    fn eval_seq<'s, 't>(mut ls: &'t [Item<'s>]) -> Option<Body<Cow<'s, str>>> {
        // eprintln!("EVAL_SEQ: {ls:?}");
        let mut body: Option<Body<_>> = None;

        if ls.len() == 1 {
            if let Item::Comma(ls) = &ls[0] {
                return eval_seq(ls);
            }
        }
        if ls.iter().all(|l| matches!(l, Item::Text(_) | Item::Seq(_))) {
            if ls.len() == 1 {
                body.get_or_insert_with(Default::default).append(&mut eval_path(&ls[..]));
            } else {
                body.get_or_insert_with(Default::default).push(Val::Chain{
                    name: None,
                    rel: eval_rel(&ls[..]),
                    path: eval_path(&ls[..]),
                    labels: vec![],
                });
            }
            return body;
        }
        if ls.len() > 0 {
            if matches!(&ls[ls.len()-1], Item::Comma(v) if v.is_empty()) {
                ls = &ls[0..ls.len()-1];
            }
        }
        if ls.len() > 1 {
            if ls[0..ls.len()-1].iter().all(|l| matches!(l, Item::Text(_) | Item::Seq(_))) {
                match &ls[ls.len()-1] {
                    Item::Sq(nest) | Item::Br(nest) => {
                        if let Val::Process{body: Some(nest_val), ..} = eval(&nest[..]) {
                            let nest_val = if matches!(ls[ls.len()-1], Item::Sq(_)) {
                                Body::All(nest_val.into())
                            } else {
                                Body::Any(nest_val.into())
                            };
                            body.get_or_insert_with(Default::default).push(Val::Process{
                                name: None,
                                label: Some(itertools::join(ls[0..ls.len()-1].iter().filter_map(|i| {
                                    match i {
                                        Item::Text(s) => Some(s.clone()),
                                        Item::Seq(s) => Some(itertools::join(s, " ").into()),
                                        _ => None
                                    }
                                }), " ").into()),
                                body: Some(nest_val),
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
                        });
                    }
                },
                Item::Text(s) => {
                    body.get_or_insert_with(Default::default).push(Val::Process{
                        name: None,
                        label: Some(s.clone()),
                        body: None,
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
        // eprintln!("RESOLVE {current_path:?}");
        let mut resolution = None;
        match val {
            Val::Process { name, body, label, .. } => {
                if let Some(name) = name {
                    current_path.push(name.clone());
                }
                if name.is_none() && body.is_none() {
                    if let Some(label) = label {
                        // eprintln!("RESOLVE {current_path:?} found reference: {label}");
                        let label = label.to_string();
                        let mut base_path = current_path.clone();
                        let path = label.split(".").map(|s| s.to_string()).map(Cow::Owned).collect::<Vec<Cow<'s, str>>>();
                        while !base_path.is_empty() {
                            let mut test_path = base_path.clone();
                            test_path.append(&mut path.clone());
                            // eprintln!("RESOLVE test path: {test_path:?}");
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
        // eprintln!("RESOLVE resolution: {resolution:?}");
        if let Some(resolution) = resolution {
            *val = resolution;
        }
    }

    fn merge<'s>(existing_process: &mut Val<Cow<'s, str>>, rhs: &mut Val<Cow<'s, str>>) {
        // eprintln!("EVAL_MERGE: {existing_process:#?} {rhs:#?}");
        if let (Val::Process { body, .. }, Val::Process { body: rbody, .. }) = (existing_process, rhs) {
            let rbody = <Vec<Val<Cow<'s, str>>> as AsMut<Vec<_>>>::as_mut(rbody.get_or_insert_with(Default::default).as_mut());
            <Vec<Val<Cow<'s, str>>> as AsMut<Vec<_>>>::as_mut(body.get_or_insert_with(Default::default).as_mut()).append(rbody);
        }
    }

    #[derive(Clone, Debug)]
    struct Model<'s> {
        processes: Vec<Val<Cow<'s, str>>>,
        names: HashMap<Cow<'s, str>, usize>,
    }

    impl<'s> Model<'s> {
        fn append(&mut self, rhs: &mut Vec<Val<Cow<'s, str>>>) {
            for val in rhs.drain(..) {
                self.push(val)
            }
        }

        fn push(&mut self, mut rhs: Val<Cow<'s, str>>) {
            use std::collections::hash_map::Entry::*;
            // eprintln!("PUSH {rhs:#?}");
            let rhs_name = match &rhs {
                Val::Process { name, label, .. } => {
                    // the only unnamed process is the administrative / top-level wrapper process
                    if name.is_none() && label.is_none() {
                        self.processes.push(rhs);
                        return
                    }
                    name.clone().or_else(|| label.as_ref().cloned()).unwrap().clone()
                },
                Val::Chain { name, .. } => {
                    if let Some(name) = name {
                        name.clone()
                    } else {
                        self.processes.push(rhs);
                        return
                    }
                },
            };
            let index_entry = self.names.entry(rhs_name.clone());
            match index_entry {
                Occupied(oe) => {
                    let existing_process = self.processes.get_mut(*oe.get()).unwrap();
                    merge(existing_process, &mut rhs);
                },
                Vacant(ve) => {
                    // if we have no names entry for rhs_name, that could be because
                    // rhs is really new, or it could be because rhs is renaming an existing
                    // process.
                    if let Val::Process { name: Some(_), label: Some(rhs_label), .. } = &rhs {
                        let index_entry2 = self.names.entry(rhs_label.clone());
                        match index_entry2 {
                            Occupied(oe2) => {
                                let existing_process_index = *oe2.get();
                                let mut existing_process = self.processes.get_mut(existing_process_index).unwrap();
                                std::mem::swap(existing_process, &mut rhs);
                                merge(&mut existing_process, &mut rhs);
                                self.names.insert(rhs_name, existing_process_index);
                            },
                            Vacant(_) => {
                                self.names.insert(rhs_name, self.processes.len());
                                self.processes.push(rhs);
                            },
                        }
                    } else {
                        ve.insert(self.processes.len());
                        self.processes.push(rhs);
                    }
                },
            }
            // TODO: cases:
            //    rhs has name and we have seen it
            //    rhs has label and we have seen it
            //    rhs has name and we have not seen it
            //    rhs has label and we have not seen it
        }

        fn to_vec(self) -> Vec<Val<Cow<'s, str>>> {
            self.processes
        }

        fn is_empty(&self) -> bool {
            self.processes.is_empty()
        }

        fn new() -> Self {
            Self {
                processes: vec![],
                names: HashMap::new(),
            }
        }
    }

    /// What depiction do the given depict-expressions denote?
    pub fn eval<'s, 't>(exprs: &'t [Item<'s>]) -> Val<Cow<'s, str>> {
        let mut body: Option<Body<_>> = None;
        let mut model = Model::new();

        for expr in exprs {
            match expr {
                Item::Colon(l, r) => {
                    if l.len() == 1 && matches!(l[0], Item::Text(..)){
                        if let Item::Text(name) = &l[0] {
                            let rbody = eval_seq(&r[..]);
                            // eprintln!("BOOM {l:?} {r:?} {rbody:?}");
                            if let Some(rbody) = rbody {
                                let mut rbody: Vec<Val<_>> = rbody.into();
                                rbody.get_mut(0).map(|fst| {
                                    fst.set_name(name.clone());
                                    let label = fst.label().unwrap_or(name);
                                    fst.set_label(Some(label.clone()));
                                });
                                model.append(&mut rbody);
                            } else {
                                model.push(Val::Process{
                                    name: None,
                                    label: Some(name.clone()),
                                    body: None,
                                })
                            }
                        }
                    } else {
                        match &l[..] {
                            [Item::Comma(ls)] => {
                                model.push(Val::Chain{
                                    name: None,
                                    rel: eval_rel(&ls[..]),
                                    path: eval_path(&ls[..]),
                                    labels: eval_labels(None, &r[..]),
                                });
                            }
                            _ => {
                                model.push(Val::Chain{
                                    name: None,
                                    rel: eval_rel(&l[..]),
                                    path: eval_path(&l[..]),
                                    labels: eval_labels(None, &r[..]),
                                });
                            }
                        }
                    }
                },
                Item::Comma(ls) => {
                    if let Some(seq_body) = eval_seq(ls) {
                        model.append(&mut seq_body.into());
                    }
                }
                Item::Seq(ls)  => {
                    match &ls[..] {
                        [Item::Comma(ls)] => {
                            if let Some(seq_body) = eval_seq(ls) {
                                model.append(&mut seq_body.into());
                            }
                        },
                        _ => {
                            if let Some(seq_body) = eval_seq(ls) {
                                model.append(&mut seq_body.into());
                            }
                        }
                    }
                }
                Item::Sq(nest) | Item::Br(nest) => {
                    if let Val::Process{body: Some(nest_val), ..} = eval(&nest[..]) {
                        let nest_val = if matches!(expr, Item::Sq(_)) {
                            Body::All(nest_val.into())
                        } else {
                            Body::Any(nest_val.into())
                        };
                        model.push(Val::Process{
                            name: None,
                            label: None,
                            body: Some(nest_val),
                        });
                    }
                }
                Item::Text(s) => {
                    model.push(Val::Process{
                        name: None,
                        label: Some(s.clone()),
                        body: None,
                    })
                }
                _ => {},
            }
        }

        // eprintln!("EVAL MODEL: {model:#?}");

        if !model.is_empty() {
            body = Some(Body::All(model.to_vec()));
        }
        Val::Process{
            name: Some("root".into()),
            label: None,
            body,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::parser::Item;
        use pretty_assertions::{assert_eq};

        const a: &'static str = "a";
        const b: &'static str = "b";
        const c: &'static str = "c";
        const d: &'static str = "d";
        const dash: &'static str = "-";
        fn r<'s>() -> Val<Cow<'s, str>> { Val::Process{name: Some("root".into()), label: None, body: None} }
        fn p<'s>() -> Val<Cow<'s, str>> { Val::Process{name: None, label: None, body: None} }
        fn l<'s>(x: &'static str) -> Val<Cow<'s, str>> { p().set_label(Some(x.into())).clone() }
        fn mp<'s>(p: &Val<Cow<'s, str>>) -> Val<Cow<'s, str>> { Val::Process{name: Some("root".into()), label: None, body: Some(Body::All(vec![p.clone()]))}}
        fn t<'s>(x: &'static str) -> Item<'s> { Item::Text(Cow::from(x)) }
        fn vi<'s>(x: &[Item<'static>]) -> Vec<Item<'s>> { x.iter().cloned().collect::<Vec<_>>() }
        fn sq<'s>(x: &[Item<'static>]) -> Item<'s>{ Item::Sq(vi(x)) }
        fn seq<'s>(x: &[Item<'static>]) -> Item<'s> { Item::Seq(vi(x)) }
        fn hc<'s>(x: &[Val<Cow<'s, str>>]) -> Val<Cow<'s, str>> { Val::Chain{ name: None, rel: Rel::Horizontal, path: x.iter().cloned().collect::<Vec<_>>(), labels: vec![], }}
        fn col<'s>(x: &[Item<'static>], y: &[Item<'static>]) -> Item<'s> { Item::Colon(vi(x), vi(y)) }

        #[test]
        fn test_eval_empty() {
            //
            assert_eq!(eval(&[]), r());
        }


        #[test]
        fn test_eval_single() {
            // a
            assert_eq!(eval(&[t(a)]), mp(p().set_label(Some(a.into()))));
        }

        #[test]
        fn test_eval_vert_chain() {
            // [ a b ]
            assert_eq!(
                eval(&[sq(&[t(a), t(b)])]),
                mp(p().set_body(Some(Body::All(vec![ l(a), l(b) ]))))
            );
        }

        #[test]
        fn test_eval_horz_chain() {
            // a b -
            assert_eq!(
                eval(&[seq(&[t(a), t(b), t(dash)])]),
                mp(
                    &hc(&[l(a), l(b)])
                )
            );
        }

        #[test]
        fn test_eval_single_nest_horz_chain() {
            // a [ - b c ]
            assert_eq!(
                eval(&[seq(&[t(a), sq(&[seq(&[t(dash), t(b), t(c)])])])]),
                mp(l(a).set_body(Some(Body::All(vec![
                    hc(&[l(b), l(c)])
                ]))))
            );
        }

        #[test]
        fn test_eval_nest_merge() {
            // a: b; a [ c ]
            assert_eq!(
                eval(&[col(&[t(a)], &[t(b)]), seq(&[t(a), sq(&[t(c)])])]),
                mp(l(b)
                    .set_name(a.into())
                    .set_body(Some(Body::All(vec![l(c)]))))
            );
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

    use crate::graph_drawing::frontend::log;

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
            // eprintln!("SYM {t} {lhs} {rhs}");
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
    pub trait Sol: Clone + Copy + Debug + Display + Eq + Fresh + Hash + Ord + PartialEq + PartialOrd + log::Names<'static> {}

    impl<S: Clone + Copy + Debug + Display + Eq + Fresh + Hash + Ord + PartialEq + PartialOrd + log::Names<'static>> Sol for S {}

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

    impl<'n, S: Sol> log::Names<'n> for Var<S> {
        fn names(&self) -> Vec<Box<dyn log::Name + 'n>> {
            self.sol.names()
        }
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

    impl<'n, S: Sol, C: Coeff> log::Names<'n> for Monomial<S, C> {
        fn names(&self) -> Vec<Box<dyn log::Name + 'n>> {
            self.var.names()
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
    //! 2. [`rank()`] the VCG by finding longest-paths
    //! 3. [`calculate_locs_and_hops()`] from the ranked paths of the CVCG to form a [LayoutProblem] by refining edge bundles (i.e., condensed edges) into hops
    //! 4. [`minimize_edge_crossing()`] by direct enumeration, inspired by the integer program described in <cite>[Optimal Sankey Diagrams Via Integer Programming]</cite> ([author's copy])
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

    use petgraph::EdgeDirection::{Incoming};
    use petgraph::algo::{floyd_warshall, dijkstra};
    use petgraph::graph::{Graph, NodeIndex, EdgeReference};
    use petgraph::visit::{EdgeRef};
    use sorted_vec::SortedVec;

    use crate::graph_drawing::error::{Error, Kind, OrErrExt, RankingError};
    use crate::graph_drawing::eval::{Val, self, Body};
    use crate::graph_drawing::frontend::log::{names, Name, Names};

    /// Require a to be left of b
    #[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct HorizontalConstraint<V: Graphic> {
        pub a: V,
        pub b: V,
    }

    fn calculate_hcg_chain<'s, 't, V: Graphic, E: Graphic>(
        vcg: &'t mut Vcg<V, E>,
        chain: &'t Val<V>,
    ) -> Result<(), Error> {
        if let Val::Chain{rel, path, labels, ..} = chain {
            if *rel == Rel::Horizontal {
                for n in 0..path.len()-1 {
                    if let Val::Process{label: Some(al), ..} = &path[n] {
                        if let Val::Process{label: Some(bl), ..} = &path[n+1] {
                            let mut al = al;
                            let mut bl = bl;
                            // bug: needs to be transitive
                            let has_prior_orientation = vcg.horz_constraints.contains(&HorizontalConstraint{a: bl.clone(), b: al.clone()});
                            if !has_prior_orientation {
                                vcg.horz_constraints.insert(
                                    HorizontalConstraint{
                                        a: al.clone(),
                                        b: bl.clone()
                                    }
                                );
                            } else {
                                std::mem::swap(&mut al, &mut bl);
                            }
                            if let Some(level) = labels.get(n) {
                                let eval::Level{mut forward, mut reverse} = level.clone();
                                if has_prior_orientation {
                                    std::mem::swap(&mut forward, &mut reverse);
                                }
                                let hlvl = vcg.horz_edge_labels.entry((al.clone(), bl.clone()))
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

    fn calculate_hcg<'s, 't, V: Graphic, E: Graphic>(
        vcg: &'t mut Vcg<V, E>,
        process: &'t Val<V>,
    ) -> Result<(), Error> {
        match process {
            Val::Chain{..} => {
                calculate_hcg_chain(vcg, process)?;
            },
            Val::Process{body, ..} => {
                for part in body.iter().flatten() {
                    match part {
                        Val::Chain{..} => {
                            calculate_hcg_chain(vcg, part)?;
                        }
                        Val::Process{body, ..} => {
                            for part in body.iter().flatten() {
                                calculate_hcg(vcg, part)?;
                            }
                        },
                    }
                }
            }
        }
        Ok(())
    }

    #[derive(Clone, Debug, Default)]
    pub struct Vcg<V: Graphic, E: Graphic> {
        /// vert is a vertical constraint graph.
        /// Edges (v, w) in vert indicate that v needs to be placed above w.
        /// Node weights must be unique.
        pub vert: Graph<V, E>,

        /// vert_vxmap maps node weights in vert to node-indices.
        pub vert_vxmap: HashMap<V, NodeIndex>,

        /// vert_node_labels maps node weights in vert to display names/labels.
        pub vert_node_labels: HashMap<V, String>,

        /// vert_edge_labels maps (v,w,rel) node weight pairs to display edge labels.
        pub vert_edge_labels: HashMap<(V, V), eval::Level<V>>,

        /// horz_constraints records directed horizontal constraint between pairs of nodes
        pub horz_constraints: HashSet<HorizontalConstraint<V>>,

        // horz_edge_labels maps (v,w) node weight pairs to label pairs
        pub horz_edge_labels: HashMap<(V, V), eval::Level<V>>,

        /// containers identifies which nodes are parents of contained nodes
        pub containers: HashSet<V>,

        /// nodes_by_container maps container-nodes to their immediate contents.
        pub nodes_by_container: HashMap<V, HashSet<V>>,

        /// nodes_by_container_transitive maps container-nodes to their contents, transitively.
        pub nodes_by_container_transitive: HashMap<V, HashSet<V>>,

        /// nesting_depths records how deeply nested each item is.
        pub nesting_depths: HashMap<V, usize>,

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
            // eprintln!("WALK_BODY CHAIN parent: {parent:?}, chain: {chain:#?}");
            match chain {
                Val::Process{label: Some(node), body: None, ..} => {
                    or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, node.clone());
                    vcg.vert_node_labels.insert(node.clone(), node.clone().into());
                    if let Some(parent) = parent {
                        add_contains_edge(vcg, parent, node);
                    }
                    for p in parents.iter() {
                        vcg.nodes_by_container_transitive.entry(p.clone()).or_default().insert(node.clone());
                    }
                },
                Val::Process{label, body: Some(body), ..} => {
                    if let (Some(parent), Some(label)) = (parent.as_ref(), label.as_ref()) {
                        add_contains_edge(vcg, parent, label);
                    }
                    // BUG: need to debruijn-number unlabeled containers
                    if let Some(node) = label.as_ref() {
                        for p in parents.iter() {
                            vcg.nodes_by_container_transitive.entry(p.clone()).or_default().insert(node.clone());
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
                            if let Some(parent) = parent {
                                add_contains_edge(vcg, parent, node);
                            }
                            for p in parents.iter() {
                                vcg.nodes_by_container_transitive.entry(p.clone()).or_default().insert(node.clone());
                            }
                        }
                    }
                },
                _ => {},
            }
        }
        if parent.is_some() {
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

        vcg.containers.insert(parent.clone());
        vcg.nodes_by_container.entry(parent.clone()).or_default().insert(node.clone());
    }

    pub fn calculate_vcg<'s, 't>(process: &'t Val<Cow<'s, str>>, logs: &mut log::Logger) -> Result<Vcg<Cow<'s, str>, Cow<'s, str>>, Error> {
        let vert = Graph::<Cow<str>, Cow<str>>::new();
        let vert_vxmap = HashMap::<Cow<str>, NodeIndex>::new();
        let vert_node_labels = HashMap::new();
        let vert_edge_labels = HashMap::new();
        let horz_constraints = HashSet::new();
        let horz_edge_labels = HashMap::new();
        let containers = HashSet::new();
        let nodes_by_container = HashMap::new();
        let nodes_by_container_transitive = HashMap::new();
        let nesting_depths: HashMap<Cow<str>, usize> = HashMap::new();
        let container_depths: HashMap<Cow<str>, usize> = HashMap::new();
        let mut vcg = Vcg{
            vert,
            vert_vxmap,
            vert_node_labels,
            vert_edge_labels,
            horz_constraints,
            horz_edge_labels,
            containers,
            nodes_by_container,
            nodes_by_container_transitive,
            nesting_depths,
            container_depths,
        };

        calculate_hcg(&mut vcg, process)?;

        let body = &Body::All(vec![process.clone()]);

        let mut queue = vec![];

        walk_body(&mut queue, &mut vcg, body, &None, vec![], 0);

        // eprintln!("QUEUE: {queue:#?}");

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
            if path.is_empty() {
                continue;
            }
            for n in 0..path.len()-1 {
                let src = &path[n];
                let mut src = if let Val::Process { label: Some(label), .. } = src { label } else { continue; };
                let dst = &path[n+1];
                let mut dst = if let Val::Process { label: Some(label), .. } = dst { label } else { continue; };
                let mut src_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, src.clone());
                let mut dst_ix = or_insert(&mut vcg.vert, &mut vcg.vert_vxmap, dst.clone());

                let has_prior_orientation = {
                    let costs = dijkstra::dijkstra(&vcg.vert, dst_ix, Some(src_ix), |_er| 1);
                    costs.contains_key(&src_ix)
                };
                if has_prior_orientation {
                    std::mem::swap(&mut src, &mut dst);
                    std::mem::swap(&mut src_ix, &mut dst_ix);
                }
                vcg.vert.add_edge(src_ix, dst_ix, "vertical".into());

                if let Some(level) = labels_by_level.get(n) {
                    let eval::Level{mut forward, mut reverse} = level.clone();
                    if has_prior_orientation {
                        std::mem::swap(&mut forward, &mut reverse);
                    }
                    let vlvl = vcg.vert_edge_labels.entry((src.clone(), dst.clone()))
                        .or_insert(eval::Level{forward: None, reverse: None});
                    match (&mut vlvl.forward, forward) {
                        (Some(f1), Some(mut f2)) => { f1.append(&mut f2); }
                        (Some(_), None) => {},
                        (None, Some(f2)) => { vlvl.forward.replace(f2); },
                        _ => {},
                    };
                    match (&mut vlvl.reverse, reverse) {
                        (Some(r1), Some(mut r2)) => { r1.append(&mut r2); }
                        (Some(_), None) => {},
                        (None, Some(r2)) => { vlvl.reverse.replace(r2); },
                        _ => {},
                    };
                }
            }
        }

        let container_depths = &mut vcg.container_depths;
        for vl in vcg.containers.iter() {
            let subdag = vcg.vert.filter_map(|_nx, nl| {
                if vcg.nodes_by_container_transitive[vl].contains(nl) {
                    Some(nl.clone())
                } else {
                    None
                }
            }, |_ex, el|{
                Some(el.clone())
            });
            let distance = {
                let containers = &vcg.containers;
                let nodes_by_container = &vcg.nodes_by_container_transitive;
                let container_depths = &container_depths;
                |src: Cow<str>, dst: Cow<str>, l: &mut log::Logger| {
                    if !containers.contains(&src) {
                        l.log_pair(
                            "(V, V)",
                            names![src, dst],
                            format!("{}, {}", src, dst),
                            "isize, &str",
                            vec![],
                            format!("-1, not-container")
                        ).unwrap();
                        -1
                    } else {
                        if nodes_by_container[&src].contains(&dst) {
                            l.log_pair(
                                "(V, V)",
                                names![src, dst],
                                format!("{}, {}", src, dst),
                                "isize, &str",
                                vec![],
                                format!("0, contains")
                            ).unwrap();
                            0
                        } else {
                            l.log_pair(
                                "(V, V)",
                                names![src, dst],
                                format!("{}, {}", src, dst),
                                "isize, &str",
                                vec![],
                                format!("{}, container_depths", -(container_depths[&src] as isize))
                            ).unwrap();
                            -(container_depths[&src] as isize)
                        }
                    }
                }
            };
            let subpaths_by_rank = rank(&subdag, distance, logs)?;
            let depth = std::cmp::max(1, subpaths_by_rank.len());
            container_depths.insert(vl.clone(), depth);
            // eprintln!("CONTAINER {vl}");
            // eprintln!("SUBDAG {subdag:#?}");
            // eprintln!("SUBPATHS {subpaths_by_rank:#?}");
            // eprintln!("DEPTH {depth}");
        }

        let nesting_depths = &mut vcg.nesting_depths;
        for vl in vcg.vert_vxmap.keys() {
            nesting_depths.insert(vl.clone(), vcg.nodes_by_container_transitive.values().filter(|nodes| nodes.contains(vl)).count());
        }

        // eprintln!("VCG: {vcg:#?}");
        // let vcg_dot = Dot::new(&vcg.vert);
        // eprintln!("VCG DOT:\n{vcg_dot:?}");

        Ok(vcg)
    }

    /// Rank a `dag`, starting from its roots, by finding longest paths
    /// from the roots to each node, e.g., using Floyd-Warshall with
    /// negative weights.
    pub fn rank<V, E>(
        dag: &Graph<V, E>,
        distance: impl Fn(V, V, &mut log::Logger) -> isize,
        logs: &mut log::Logger,
    ) -> Result<BTreeMap<VerticalRank, SortedVec<(V, V)>>, Error>
    where
        V: Graphic + From<String>,
        E: Clone + Default,
    {
        let mut dag: Graph<V, E> = dag.clone();

        let roots = dag
            .externals(Incoming)
            .into_iter()
            .collect::<Vec<_>>();

        let root_ix = dag.add_node("root".to_string().into());
        for node_ix in roots.iter() {
            dag.add_edge(root_ix, *node_ix, Default::default());
        }

        let mut paths_fw = Ok(HashMap::new());
        {
            logs.with_group("(V,V)->(isize, reason)", "floyd_warshall", Vec::<String>::new(), {
                |l| {
                    fn constrain<'s, E: 's, F: FnMut(EdgeReference<'s, E>) -> isize>(f: F) -> F { f }
                    paths_fw = floyd_warshall(&dag, constrain(|er| {
                        let src = dag.node_weight(er.source()).unwrap();
                        let dst = dag.node_weight(er.target()).unwrap();
                        let dist = distance(src.clone(), dst.clone(), l);
                        // eprintln!("DISTANCE MUT: {src:?} {dst:?} {dist:?}");
                        dist
                    }));
                    Ok(())
                }
            })?;
        }

        let paths_fw = paths_fw.map_err(|cycle|
            Error::from(RankingError::NegativeCycleError{cycle})
        )?;

        let paths_fw2 = SortedVec::from_unsorted(
            paths_fw
                .iter()
                .map(|((vx, wx), wgt)| {
                    let vl = dag.node_weight(*vx).or_err(Kind::IndexingError{})?.clone();
                    let wl = dag.node_weight(*wx).or_err(Kind::IndexingError{})?.clone();
                    Ok((*wgt, vx, wx, vl, wl))
                })
                .into_iter()
                .collect::<Result<Vec<_>, Error>>()?
        );
        // eprintln!("FLOYD-WARSHALL: {paths_fw2:#?}");

        let paths_from_roots = SortedVec::from_unsorted(
            paths_fw2
                .iter()
                .filter_map(|(wgt, vx, _wx, vl, wl)| {
                    if *wgt <= 0 && roots.contains(vx) {
                        Some((VerticalRank(-(*wgt) as usize), vl.clone(), wl.clone()))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        );
        // eprintln!("PATHS_FROM_ROOTS: {paths_from_roots:#?}");

        let mut paths_by_rank = BTreeMap::new();
        for (wgt, vl, wl) in paths_from_roots.iter() {
            paths_by_rank
                .entry(*wgt)
                .or_insert_with(SortedVec::new)
                .insert((vl.clone(), wl.clone()));
        }
        // eprintln!("RANK_PATHS_BY_RANK: {paths_by_rank:#?}");

        Ok(paths_by_rank)
    }

    use crate::graph_drawing::index::{OriginalHorizontalRank, VerticalRank};

    /// Methods for graph vertices and edges.
    pub trait Graphic: Clone + Debug + Display + Eq + Hash + Ord + PartialEq + PartialOrd + log::Name + PartialEq<&'static str> {}

    impl <T: Clone + Debug + Display + Eq + Hash + Ord + PartialEq + PartialOrd + log::Name + PartialEq<&'static str>> Graphic for T {}


    #[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
    pub struct ObjNode<V: Graphic>{
        pub vl: V,
    }

    impl<'n, V: Graphic> Display for ObjNode<V> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.vl)
        }
    }

    impl<'n, V: Graphic> log::Names<'n> for ObjNode<V> {
        fn names(&self) -> Vec<Box<dyn Name + 'n>> {
            self.vl.to_string().names()
        }
    }

    #[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
    pub struct ObjHop<V: Graphic>{
        pub lvl: VerticalRank,
        pub mhr: OriginalHorizontalRank,
        pub vl: V,
        pub wl: V,
    }

    impl<'n, V: Graphic> Display for ObjHop<V> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} {}->{}", self.lvl, self.vl, self.wl)
        }
    }

    impl<'n, V: Graphic + 'n> log::Names<'n> for ObjHop<V> {
        fn names(&self) -> Vec<Box<dyn Name + 'n>> {
            names![self.lvl, self.vl, self.wl]
        }
    }

    #[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
    pub struct ObjContainer<V: Graphic> {
        pub vl: V,
    }

    impl<'n, V: Graphic> Display for ObjContainer<V> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.vl)
        }
    }

    impl<'n, V: Graphic + 'n> log::Names<'n> for ObjContainer<V> {
        fn names(&self) -> Vec<Box<dyn Name + 'n>> {
            self.vl.names()
        }
    }

    #[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
    pub struct ObjBorder<V: Graphic> {
        pub border: Border<V>,
    }

    impl<'n, V: Graphic> Display for ObjBorder<V> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.border)
        }
    }

    impl<'n, V: Graphic + 'n> log::Names<'n> for ObjBorder<V> {
        fn names(&self) -> Vec<Box<dyn Name + 'n>> {
            self.border.names()
        }
    }

    /// A graphical object to be positioned relative to other objects
    #[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
    pub enum Obj<V: Graphic> {
        /// A leaf "box"
        Node(ObjNode<V>),
        /// One hop of an "arrow"
        Hop(ObjHop<V>),
        /// A container of other objects
        Container(ObjContainer<V>),
        /// A vertical border of a nested system of boxes
        Border(ObjBorder<V>),
    }

    impl<V: Graphic> Obj<V> {
        pub fn from_vl(vl: &V, containers: &HashSet<V>) -> Self {
            if containers.contains(vl) {
                Self::Container(ObjContainer{vl: vl.clone()})
            } else {
                Self::Node(ObjNode{vl: vl.clone()})
            }
        }

        pub fn as_vl(&self) -> Option<&V> {
            match self {
                Obj::Node(ObjNode{vl}) => Some(vl),
                Obj::Hop(ObjHop{vl, ..}) => Some(vl),
                Obj::Container(ObjContainer{vl}) => Some(vl),
                Obj::Border(_) => None,
            }
        }

        pub fn as_wl(&self) -> Option<&V> {
            match self {
                Obj::Node(ObjNode{vl}) => Some(vl),
                Obj::Hop(ObjHop{wl, ..}) => Some(wl),
                Obj::Container(ObjContainer{vl}) => Some(vl),
                Obj::Border(_) => None,
            }
        }
    }

    impl<'n, V> log::Names<'n> for Obj<V>
    where
        V: Graphic + 'n,
    {
        fn names(&self) -> Vec<Box<dyn log::Name + 'n>> {
            match self {
                Obj::Node(ObjNode{vl}) => {
                    names![vl]
                },
                Obj::Hop(ObjHop{lvl, mhr, vl, wl}) => {
                    names![lvl, mhr, vl, wl]
                },
                Obj::Container(ObjContainer{ vl }) => {
                    names![vl]
                },
                Obj::Border(ObjBorder{border: Border{ vl, ovr, ohr, pair }}) => {
                    names![vl, ovr, ohr, pair]
                },
            }
        }
    }

    impl<V: Graphic> Display for Obj<V> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Obj::Node(inner) => write!(f, "{inner}"),
                Obj::Hop(inner) => write!(f, "{inner}"),
                Obj::Container(inner) => write!(f, "{inner}"),
                Obj::Border(inner) => write!(f, "{inner}"),
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

    impl<'n, V: Graphic + 'n> log::Names<'n> for Border<V> {
        fn names(&self) -> Vec<Box<dyn Name + 'n>> {
            let Border{ vl, ovr, ohr, pair } = self;
            names![vl, ovr, ohr, pair]
        }
    }

    impl<V: Graphic + Display + log::Name> log::Log for HashMap<(VerticalRank, OriginalHorizontalRank), Obj<V>> {
        fn log(&self, _cx: (), l: &mut log::Logger) -> Result<(), log::Error> {
            l.with_map("loc_to_node", "LocToNode", self.iter(), |loc_ix, loc, l| {
                match loc {
                    Obj::Node(loc) => {
                        l.log_pair(
                            "Loc",
                            loc_ix.names(),
                            format!("{:?}, {:?}", loc_ix.0, loc_ix.1),
                            "Obj::Node",
                            loc.names(),
                            format!("{loc}"),
                        )
                    },
                    Obj::Hop(loc@ObjHop{lvl, vl, wl, ..}) => {
                        l.log_pair(
                            "Loc",
                            loc_ix.names(),
                            format!("{:?}, {:?}", loc_ix.0, loc_ix.1),
                            "Obj::Hop",
                            loc.names(),
                            format!("{lvl}, {vl}->{wl}")
                        )
                    }
                    Obj::Container(loc) => {
                        l.log_pair(
                            "Loc",
                            loc_ix.names(),
                            format!("{:?}, {:?}", loc_ix.0, loc_ix.1),
                            "Obj::Container",
                            loc.names(),
                            format!("{loc}")
                        )
                    },
                    Obj::Border(loc) => {
                        l.log_pair(
                            "Loc",
                            loc_ix.names(),
                            format!("{:?}, {:?}", loc_ix.0, loc_ix.1),
                            "Obj::Border",
                            loc.names(),
                            format!("{loc}")
                        )
                    }
                }
            })
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct LayoutProblem<V: Graphic + Display + log::Name> {
        pub locs_by_level: BTreeMap<VerticalRank, usize>,
        pub hops_by_level: BTreeMap<VerticalRank, SortedVec<Hop<V>>>,
        pub hops_by_edge: BTreeMap<(V, V), BTreeMap<VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank)>>,
        pub loc_to_node: HashMap<(VerticalRank, OriginalHorizontalRank), Obj<V>>,
        pub node_to_loc: HashMap<Obj<V>, (VerticalRank, OriginalHorizontalRank)>,
        pub container_borders: HashMap<V, Vec<(VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank))>>,
    }

    #[derive(Clone, Debug, Default)]
    pub struct LayoutSolution {
        pub crossing_number: usize,
        pub solved_locs: BTreeMap<VerticalRank, BTreeMap<OriginalHorizontalRank, SolvedHorizontalRank>>,
    }

    /// Marker trait for closures mapping LocIx to names
    pub(crate) trait L2n : Fn(VerticalRank, OriginalHorizontalRank) -> Vec<Box<dyn Name>> {}
    impl<CX: Fn(VerticalRank, OriginalHorizontalRank) -> Vec<Box<dyn Name>>> L2n for CX {}

    /// Marker trait for closures mapping LayoutSol to names
    pub(crate) trait V2n : Fn(VarRank) -> Vec<Box<dyn Name>> {}
    impl<CX: Fn(VarRank) -> Vec<Box<dyn Name>>> V2n for CX {}

    impl<CX: L2n> log::Log<CX> for BTreeMap<VerticalRank, BTreeMap<OriginalHorizontalRank, SolvedHorizontalRank>> {
        fn log(&self, cx: CX, l: &mut log::Logger) -> Result<(), log::Error> {
            l.with_map("solved_locs", "SolvedLocs", self.iter(), |lvl, row, l| {
                l.with_map(format!("solved_locs[{lvl}]"), "SolvedLocs[i]", row.iter(), |ohr, shr, l| {
                    let mut src_names = names![lvl, ohr];
                    src_names.append(&mut cx(*lvl, *ohr));
                    l.log_pair(
                        "Loc",
                        src_names,
                        format!("{lvl}v, {ohr}h"),
                        "SolvedHorizontalRank",
                        names![shr],
                        format!("{lvl}v, {shr}s")
                    )
                })
            })
        }
    }

    pub type RankedPaths<V> = BTreeMap<VerticalRank, SortedVec<(V, V)>>;

    /// Set up a [LayoutProblem] problem
    pub fn calculate_locs_and_hops<'s, V: Graphic>(
        _model: &'s Val<V>,
        paths_by_rank: &'s RankedPaths<V>,
        vcg: &Vcg<V, V>,
        logs: &mut log::Logger,
    ) -> Result<LayoutProblem<V>, Error> {
        let Vcg{vert, containers, container_depths, ..} = vcg;
        let dag = vert;

        // Rank vertices by the length of the longest path reaching them.
        let mut vx_rank = BTreeMap::new();
        for (rank, paths) in paths_by_rank.iter() {
            for (_vx, wx) in paths.iter() {
                vx_rank.insert(wx.clone(), *rank);
            }
        }

        // eprintln!("PATHS_BY_RANK 0: {paths_by_rank:#?}");
        // eprintln!("VX_RANK {vx_rank:#?}");

        logs.with_map("vx_rank", "BTreeMap<V, VerticalRank>", vx_rank.iter(), |wl, rank, l| {
            l.log_pair(
                "V",
                names![wl],
                format!("{wl}"),
                "usize",
                names![rank],
                format!("{rank}v")
            )
        })?;

        let mut loc_to_node = HashMap::new();
        let mut node_to_loc = HashMap::new();
        let mut locs_by_level = BTreeMap::new();

        for (wl, rank) in vx_rank.iter() {
            let level = locs_by_level
                .entry(*rank)
                .or_insert(0);
            let mhr = OriginalHorizontalRank(*level);
            *level += 1;
            if let Some(old) = loc_to_node.insert((*rank, mhr), Obj::from_vl(wl, containers)) {
                panic!("loc_to_node.insert({rank}, {mhr}) -> {:?}", old);
            };
            node_to_loc.insert(Obj::from_vl(wl, containers), (*rank, mhr));
        }

        logs.with_map("node_to_loc", "HashMap<Loc<V, V>, (VerticalRank, OriginalHorizontalRank)>", node_to_loc.iter(), |loc, loc_ix, l| {

            l.log_pair(
                "Loc<V, V>",
                loc.names(),
                format!("{loc:?}"),
                "(VerticalRank, OriginalHorizontalRank)",
                loc_ix.names(),
                format!("{loc_ix:?}")
            )
        })?;

        let mut hops_by_edge = BTreeMap::new();
        let mut hops_by_level = BTreeMap::new();
        for er in dag.edge_references() {
            let vx = er.source();
            let wx = er.target();
            let vl = dag.node_weight(vx).unwrap();
            let wl = dag.node_weight(wx).unwrap();
            let (vvr, vhr) = node_to_loc[&Obj::from_vl(vl, containers)].clone();
            let (wvr, whr) = node_to_loc[&Obj::from_vl(wl, containers)].clone();
            assert_ne!((vvr, vhr), (wvr, whr));

            let mut mhrs = vec![vhr];
            for mid_level in (vvr+1).0..(wvr.0) {
                let mid_level = VerticalRank(mid_level); // pending https://github.com/rust-lang/rust/issues/42168
                let num_mhrs = locs_by_level.entry(mid_level).or_insert(0);
                let mhr = OriginalHorizontalRank(*num_mhrs);
                *num_mhrs += 1;
                if let Some(old) = loc_to_node.insert((mid_level, mhr), Obj::Hop(ObjHop{lvl: mid_level, mhr: mhr, vl: vl.clone(), wl: wl.clone()})) {
                    panic!("loc_to_node.insert({mid_level}, {mhr}) -> {:?}", old);
                };
                node_to_loc.insert(Obj::Hop(ObjHop{lvl: mid_level, mhr, vl: vl.clone(), wl: wl.clone()}), (mid_level, mhr)); // BUG: what about the endpoints?
                mhrs.push(mhr);
            }
            mhrs.push(whr);

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

        logs.with_map("hops_by_edge", "HashMap<(V, V), (VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank))>", hops_by_edge.iter(), |edge, hops, l| {
            l.with_set(format!("hops_by_edge[{}->{}]", edge.0, edge.1), "Hops", hops.iter(), |(lvl, (mhr, nhr)), l| {
                l.log_pair(
                    "VerticalRank",
                    names![lvl],
                    format!("{lvl}"),
                    "(OriginalHorizontalRank, OriginalHorizontalRank)",
                    names![mhr, nhr],
                    format!("{mhr}, {nhr}")
                )
            })
        })?;

        // eprintln!("NODE_TO_LOC: {node_to_loc:#?}");

        let mut container_borders: HashMap<V, Vec<(VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank))>> = HashMap::new();

        for vl in containers.iter() {
            let (ovr, mut ohr) = node_to_loc[&Obj::Container(ObjContainer{vl: vl.clone()})];
            let depth = container_depths[vl];
            for vr in 0..depth {
                let vr = VerticalRank(ovr.0 + vr);
                let num_mhrs = locs_by_level.entry(vr).or_insert(0);
                let mhr = OriginalHorizontalRank(*num_mhrs);
                *num_mhrs += 1;
                if vr > ovr {
                    ohr = OriginalHorizontalRank(*num_mhrs);
                    *num_mhrs += 1;
                    let border = Obj::Border(ObjBorder{border: Border{ vl: vl.clone(), ovr: vr, ohr: mhr, pair: ohr }});
                    node_to_loc.insert(border.clone(), (vr, ohr));
                    if let Some(old) = loc_to_node.insert((vr, ohr), border) {
                        panic!("loc_to_node.insert({vr}, {ohr}) -> {:?}", old);
                    };
                }
                let border = Obj::Border(ObjBorder{ border: Border{vl: vl.clone(), ovr: vr, ohr, pair: mhr }});
                node_to_loc.insert(border.clone(), (vr, mhr));
                if let Some(old) = loc_to_node.insert((vr, mhr), border) {
                    panic!("loc_to_node.insert({vr}, {mhr}) -> {:?}", old);
                };
                container_borders.entry(vl.clone()).or_default().push((vr, (ohr, mhr)));
            }

            // eprintln!("VERTICAL RANK SPAN: {vl}: {:?}", ovr.0..(ovr.0+depth));
            // eprintln!("CONTAINER BORDERS: {vl}: {container_borders:#?}");
            // eprintln!("LOCS_BY_LEVEL V3: {vl}: {locs_by_level:#?}");
        }

        // eprintln!("NODE_TO_LOC: {node_to_loc:#?}");

        Ok(LayoutProblem{locs_by_level, hops_by_level, hops_by_edge, loc_to_node, node_to_loc, container_borders})
    }

    #[cfg(test)]
    mod tests {
        use crate::graph_drawing::frontend::log::Logger;

        use super::*;


        #[test]
        fn test_rank() {
            let mut vert: Graph<Cow<str>, Cow<str>> = Graph::new();
            let mut vx_map = HashMap::new();

            let a: Cow<str> = "a".into();
            let b: Cow<str> = "b".into();
            let actuates: Cow<str> = "actuates".into();

            let ax = or_insert(&mut vert, &mut vx_map, a.clone());
            let bx = or_insert(&mut vert, &mut vx_map, b.clone());
            vert.add_edge(ax, bx, actuates.clone());

            let mut logs = Logger::new();

            let paths_by_rank = rank(&vert, |a, b, l| -1, &mut logs).unwrap();
            assert_eq!(paths_by_rank[&VerticalRank(0)], SortedVec::from_unsorted(vec![(a.clone(), a.clone())]));
            assert_eq!(paths_by_rank[&VerticalRank(1)], SortedVec::from_unsorted(vec![(a.clone(), b.clone())]));
        }
    }

    pub mod heaps {
        use std::collections::{BTreeMap, HashMap};
        use std::fmt::{Display};

        use crate::graph_drawing::error::{Error, LayoutError, OrErrExt, Kind};
        use crate::graph_drawing::index::{VerticalRank, OriginalHorizontalRank, SolvedHorizontalRank};
        use crate::graph_drawing::layout::{Hop};

        use super::{LayoutProblem, Graphic, LayoutSolution, HorizontalConstraint, Obj, Vcg, ObjNode, ObjHop, ObjBorder, ObjContainer};

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

            // let lvl = h1.lvl;
            // let vl1 = &h1.vl;
            // let wl1 = &h1.wl;
            // let vl2 = &h2.vl;
            // let wl2 = &h2.wl;
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

        use crate::graph_drawing::frontend::log;

        fn conforms<V: Graphic + Display + log::Name, E: Graphic>(
            vcg: &Vcg<V, E>,
            layout_problem: &LayoutProblem<V>,
            locs_by_level2: &Vec<Vec<&Obj<V>>>,
            nodes_by_container2: &HashMap<V, Vec<(VerticalRank, OriginalHorizontalRank)>>,
            p: &mut [&mut [usize]]
        ) -> bool {
            let Vcg{nodes_by_container_transitive: nodes_by_container, ..} = vcg;
            let LayoutProblem{node_to_loc, container_borders, ..} = layout_problem;

            // eprintln!("HCG: {:#?}", hcg.iter().collect::<Vec<_>>());
            let hcg_satisfied = vcg.horz_constraints.iter().all(|constraint| {
                let HorizontalConstraint{a, b} = constraint;
                // let an = if let Some(an) = node_to_loc.get(&Obj::Node(ObjNode{vl: a.clone()})) { an } else { return true; };
                // let bn = if let Some(bn) = node_to_loc.get(&Obj::Node(ObjNode{vl: b.clone()})) { bn } else { return true; };
                let aobj = &Obj::from_vl(a, &vcg.containers);
                let bobj = &Obj::from_vl(b, &vcg.containers);
                let aloc = node_to_loc[aobj];
                let bloc = node_to_loc[bobj];
                let aovr = aloc.0.0;
                let aohr = aloc.1.0;
                let bovr = bloc.0.0;
                let bohr = bloc.1.0;
                let ashr = p[aovr][aohr];
                let bshr = p[bovr][bohr];
                // let ret = (aovr == bovr && ashr < bshr) || ashr <= bshr;
                // eprintln!("HCG CONFORMS: constraint: {constraint:?}, aobj: {aobj}, bobj: {bobj}, aloc: {aloc:?}, bloc: {bloc:?}, ashr: {ashr}, bshr: {bshr}, ret: {ret}");
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
                            Obj::Node(ObjNode{vl: ml}) => {
                                contents.contains(&ml)
                            },
                            Obj::Hop(ObjHop{vl: mvl, wl: mwl, ..}) => {
                                contents.contains(&mvl) || contents.contains(&mwl)
                            },
                            Obj::Container(ObjContainer{vl: ml}) => {
                                contents.contains(&ml)
                            },
                            Obj::Border(ObjBorder{border: mb}) => {
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
        pub fn minimize_edge_crossing<V>(
            vcg: &Vcg<V, V>,
            layout_problem: &LayoutProblem<V>
        ) -> Result<LayoutSolution, Error> where
            V: Display + Graphic + log::Name
        {
            let Vcg{containers, nodes_by_container_transitive: nodes_by_container, ..} = vcg;
            let LayoutProblem{loc_to_node, node_to_loc, locs_by_level, hops_by_level, ..} = layout_problem;

            // eprintln!("MINIMIZE");
            // eprintln!("LOCS_BY_LEVEL: {locs_by_level:#?}");
            // eprintln!("HOPS_BY_LEVEL: {hops_by_level:#?}");
            let mut l2n = loc_to_node.iter().collect::<Vec<_>>();
            l2n.sort();
            // eprintln!("LOC_TO_NODE: {l2n:#?}");

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
                let nodes = nodes.iter()
                    .map(|vl| node_to_loc[&Obj::from_vl(vl, containers)])
                    .collect::<Vec<_>>();
                nodes_by_container2.insert(container.clone(), nodes);
            }

            let mut crossing_number = usize::MAX;
            let mut solution: Option<Vec<Vec<usize>>> = None;
            // eprintln!("MULTISEARCH {:#?}", shrs_ref);
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
                                let h1 = &hops[h1i];
                                let h2 = &hops[h2i];
                                // eprintln!("hop: {h1} {h2} -> {}", crosses(h1, h2, p[rank.0], p[rank.0+1]));
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
                // eprintln!("P cn: {cn}: p: {p:?}");
                false
            });

            let solution = solution.or_err(LayoutError::HeapsError{error: "no solution found".into()})?;
            // eprintln!("HEAPS CN: {crossing_number}");
            // eprintln!("HEAPS SOL: ");
            // for (n, s) in solution.iter().enumerate() {
            //     eprintln!("{n}: {s:?}");
            // }

            let mut solved_locs = BTreeMap::new();
            for (lvl, shrs) in solution.iter().enumerate() {
                solved_locs.insert(VerticalRank(lvl), shrs
                    .iter()
                    .enumerate()
                    .map(|(a, b)| (OriginalHorizontalRank(a), SolvedHorizontalRank(*b))) // needs mutation testing
                    .collect::<BTreeMap<OriginalHorizontalRank, SolvedHorizontalRank>>());
            }

            // eprintln!("SOLVED_LOCS: {solved_locs:#?}");

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
    use super::frontend::log;
    use super::index::{SolvedHorizontalRank, VarRank};
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
    use petgraph::Graph;
    use petgraph::algo::is_cyclic_directed;
    use petgraph::visit::{EdgeRef};

    use crate::graph_drawing::layout::ObjNode;
    #[allow(unused_imports)]
    use crate::graph_drawing::osqp::{as_diag_csc_matrix, print_tuples, as_scipy, as_numpy};

    use super::error::{LayoutError};
    use super::frontend::log::Name;
    use super::osqp::{Constraints, Monomial, Vars, Fresh, Var, Sol, Coeff};

    use super::error::Error;
    use super::index::{VerticalRank, OriginalHorizontalRank, SolvedHorizontalRank, VarRank};
    use super::layout::{Obj, Hop, Vcg, LayoutProblem, Graphic, LayoutSolution, Len, L2n, V2n, or_insert, ObjBorder, Border, ObjHop, ObjContainer};

    use std::borrow::Cow;
    use std::cmp::{max};
    use std::collections::{HashMap, BTreeMap};
    use std::fmt::{Debug, Display};
    use std::hash::Hash;

    #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, EnumKind)]
    #[enum_kind(AnySolKind)]
    pub enum AnySol {
        L(VarRank),
        R(VarRank),
        S(VarRank),
        T(VarRank),
        B(VarRank),
        H(VerticalRank),
        V(SolvedHorizontalRank),
        F(usize),
        #[default]
        Default,
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
                AnySol::H(ovr) => write!(f, "h{}", ovr.0),
                AnySol::V(shr) => write!(f, "v{}", shr.0),
                AnySol::F(idx) => write!(f, "f{}", idx),
                AnySol::Default => write!(f, "def"),
            }
        }
    }

    impl Name for AnySol {
        fn name(&self) -> String {
            format!("{}", self)
        }
    }

    #[derive(Clone, Debug)]
    pub struct LocRow<V: Graphic> {
        pub ovr: VerticalRank,
        pub ohr: OriginalHorizontalRank,
        pub shr: SolvedHorizontalRank,
        pub loc: Obj<V>,
        pub n: VarRank,
    }

    #[derive(Clone, Debug)]
    pub struct HopRow<V: Graphic> {
        pub lvl: VerticalRank,
        pub mhr: OriginalHorizontalRank,
        pub nhr: OriginalHorizontalRank,
        pub vl: V,
        pub wl: V,
        pub n: VarRank,
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
    pub struct GeometryProblem<V: Graphic> {
        pub varrank_by_obj: HashMap<Obj<V>, VarRank>,
        pub loc_by_varrank: HashMap<VarRank, LocIx>,
        pub size_by_loc: HashMap<LocIx, NodeSize>,
        pub size_by_hop: HashMap<(VerticalRank, OriginalHorizontalRank, V, V), HopSize>,
        pub height_scale: Option<f64>,
        pub line_height: Option<f64>,
        pub char_width: Option<f64>,
        pub nesting_top_padding: Option<f64>,
        pub nesting_bottom_padding: Option<f64>,
    }

    use crate::graph_drawing::frontend::log::{self, names, Names};

    impl<V: Graphic> log::Log for HashMap<Obj<V>, VarRank> {
        fn log(&self, _cx: (), l: &mut log::Logger) -> Result<(), log::Error> {
            l.with_map("obj_by_varrank", "ObjByVarrank", self.iter(), |obj, sol, l| {
                let src_names = obj.names();
                l.log_pair(
                    "Varrank",
                    names![sol],
                    format!("{sol}"),
                    "Obj",
                    src_names,
                    format!("{obj}"),
                )
            })
        }
    }

    impl<CX: L2n> log::Log<CX> for HashMap<LocIx, NodeSize> {
        fn log(&self, cx: CX, l: &mut log::Logger) -> Result<(), log::Error> {
            l.with_map("size_by_loc", "SizeByLoc", self.iter(), |loc_ix, size, l| {
                let mut src_names = loc_ix.names();
                src_names.append(&mut cx(loc_ix.0, loc_ix.1));
                l.log_pair(
                    "Loc",
                    src_names,
                    format!("{:?}, {:?}", loc_ix.0, loc_ix.1),
                    "NodeSize",
                    vec![],
                    format!("{size:?}"))
            })
        }
    }

    impl<V: Graphic + Display + log::Name, CX: L2n> log::Log<CX> for HashMap<HopIx<V>, HopSize> {
        fn log(&self, cx: CX, l: &mut log::Logger) -> Result<(), log::Error> {
            l.with_map("size_by_hop", "SizeByHop", self.iter(), |(ovr, ohr, vl, wl), size, l| {
                let mut src_names = names![ovr, ohr, vl, wl];
                src_names.append(&mut cx(*ovr, *ohr));
                l.log_pair(
                    "Loc",
                    src_names,
                    format!("{ovr}v, {ohr}h"),
                    "HopSize",
                    vec![],
                    format!("{size:?}")
                )
            })
        }
    }

    /// ovr, ohr
    pub type LocIx = (VerticalRank, OriginalHorizontalRank);

    pub type HopIx<V> = (VerticalRank, OriginalHorizontalRank, V, V);

    /// ovr, ohr -> loc
    pub type LocNodeMap<V> = HashMap<LocIx, Obj<V>>;

    /// lvl -> (mhr, nhr)
    pub type HopMap = BTreeMap<VerticalRank, (OriginalHorizontalRank, OriginalHorizontalRank)>;

    pub fn calculate_sols<'s, V>(
        layout_problem: &'s LayoutProblem<V>,
        layout_solution: &'s LayoutSolution,
    ) -> GeometryProblem<V> where
        V: Display + Graphic + log::Name
    {
        let LayoutProblem{loc_to_node, hops_by_level, hops_by_edge, ..} = layout_problem;
        let LayoutSolution{solved_locs, ..} = layout_solution;

        // eprintln!("SOLVED_LOCS {solved_locs:#?}");

        let mut var_counter = 0;
        let mut var_rank = || {
            let n = var_counter;
            var_counter += 1;
            VarRank(n)
        };

        let mut varrank_by_obj = HashMap::new();
        let mut loc_by_varrank = HashMap::new();

        for (ovr, ohr, _shr, obj) in solved_locs
            .iter()
            .flat_map(|(ovr, nodes)| nodes
                .iter()
                .map(|(ohr, shr)| (*ovr, *ohr, *shr, &loc_to_node[&(*ovr,*ohr)]))) {
            let rank = var_rank();
            varrank_by_obj.insert(obj.clone(), rank);
            loc_by_varrank.insert(rank, (ovr, ohr));
        }

        for (mhr, _nhr, vl, wl, lvl) in hops_by_level
            .iter()
            .flat_map(|h|
                h.1.iter().map(|Hop{mhr, nhr, vl, wl, lvl}| {
                    (*mhr, *nhr, vl.clone(), wl.clone(), *lvl)
                })
            )
            .chain(
                hops_by_edge.iter().map(|((vl, wl), hops)| {
                        let (lvl, (mhr, nhr)) = {
                            #[allow(clippy::unwrap_used)] // an edge with no hops really should panic
                            let kv = hops.last_key_value().unwrap();
                            (*kv.0, *kv.1)
                        };
                        (nhr, OriginalHorizontalRank(std::usize::MAX - mhr.0), vl.clone(), wl.clone(), lvl+1)
                })
            ) {
            let rank = var_rank();
            varrank_by_obj.insert(Obj::Hop(ObjHop{lvl: lvl, mhr, vl: vl.clone(), wl: wl.clone()}), rank);
            loc_by_varrank.insert(rank, (lvl, mhr));
        }

        let size_by_loc = HashMap::new();
        let size_by_hop = HashMap::new();

        let height_scale = None;
        let line_height = None;
        let char_width = None;
        let nesting_top_padding = None;
        let nesting_bottom_padding = None;

        GeometryProblem{
            varrank_by_obj,
            loc_by_varrank,
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
    pub struct OptimizationProblem<S: Sol, C: Coeff> {
        v: Vars<S>,
        c: Constraints<S, C>,
        pd: Vec<Monomial<S, C>>,
        q: Vec<Monomial<S, C>>,
    }

    use std::fmt::Write;

    impl<C: Coeff, CX: V2n> log::Log<(String, CX)> for OptimizationProblem<AnySol, C> {
        fn log(&self, cx: (String, CX), l: &mut log::Logger) -> Result<(), log::Error> {
            let names = |monomial: &Monomial<AnySol, C>| {
                let mut ret = monomial.names();
                match monomial.var.sol {
                    AnySol::L(ls) | AnySol::R(ls) | AnySol::T(ls) | AnySol::B(ls) | AnySol::S(ls) => {
                        ret.append(&mut (cx.1)(ls));
                    },
                    _ => {},
                }
                ret
            };
            l.with_group("OptimizationProblem", cx.0.clone(), Vec::<String>::new(), |l| {
                l.with_map(cx.0.clone(), "Vars", self.v.iter(), |sol, var, l| {
                    let mut src_names = sol.names();
                    match sol {
                        AnySol::L(ls) | AnySol::R(ls) | AnySol::T(ls) | AnySol::B(ls) | AnySol::S(ls) => {
                            src_names.append(&mut (cx.1)(*ls));
                        },
                        _ => {},
                    }
                    l.log_pair(
                        "AnySol",
                        src_names,
                        format!("{sol}"),
                        "Var",
                        var.names(),
                        format!("v{}", var.index)
                    )
                })?;
                l.with_set(cx.0.clone(), "Constraints", self.c.iter(), |(lower, monomials, upper), l| {
                    let mut s = String::new();
                    let mut monomial_names = vec![];
                    write!(s, "{lower} <= ")?;
                    for term in monomials.iter() {
                        monomial_names.append(&mut names(term));
                        write!(s, "{term} ")?;
                    }
                    if *upper != f64::INFINITY {
                        writeln!(s, "<= {upper}")?;
                    }
                    let monomial_names = monomial_names.into_iter().map(|n| n.name()).collect::<Vec<_>>();
                    l.log_element("Constraint", monomial_names, s)?;
                    Ok(())
                })?;
                l.with_set(cx.0.clone(), "Quadratic Objective", self.pd.iter(), |monomial, l| {
                    l.log_element("Monomial", names(monomial).into_iter().map(|n| n.name()).collect::<Vec<_>>(), format!("{monomial}"))
                })?;
                l.with_set(cx.0.clone(), "Linear Objective", self.q.iter(), |monomial, l| {
                    l.log_element("Monomial", names(monomial).into_iter().map(|n| n.name()).collect::<Vec<_>>(), format!("{monomial}"))
                })?;
                Ok(())
            })
        }
    }

    #[derive(Debug, Default)]
    pub struct GeometrySolution {
        pub ls: BTreeMap<VarRank, f64>,
        pub rs: BTreeMap<VarRank, f64>,
        pub ss: BTreeMap<VarRank, f64>,
        pub ts: BTreeMap<VarRank, f64>,
        pub bs: BTreeMap<VarRank, f64>,
        pub status_h: OSQPStatusKind,
        pub status_v: OSQPStatusKind,
    }

    impl<CX: V2n> log::Log<CX> for GeometrySolution {
        fn log(&self, cx: CX, l: &mut log::Logger) -> Result<(), log::Error> {
            l.with_group("Coordinates", "", Vec::<String>::new(), |l| {
                l.with_map("ls", "BTreeMap<Sol, f64>", self.ls.iter(), |ls, coord, l| {
                    let mut src_names = ls.names();
                    src_names.append(&mut cx(*ls));
                    l.log_pair(
                        "Sol",
                        src_names,
                        format!("{ls}"),
                        "f64",
                        vec![],
                        format!("{:.0}", coord.round())
                    )
                })?;
                l.with_map("rs", "BTreeMap<Sol, f64>", self.rs.iter(), |ls, coord, l| {
                    let mut src_names = ls.names();
                    src_names.append(&mut cx(*ls));
                    l.log_pair(
                        "Sol",
                        src_names,
                        format!("{ls}"),
                        "f64",
                        vec![],
                        format!("{:.0}", coord.round())
                    )
                })?;
                l.with_map("ts", "BTreeMap<Sol, f64>", self.ts.iter(), |ls, coord, l| {
                    let mut src_names = ls.names();
                    src_names.append(&mut cx(*ls));
                    l.log_pair(
                        "Sol",
                        src_names,
                        format!("{ls}"),
                        "f64",
                        vec![],
                        format!("{:.0}", coord.round())
                    )
                })?;
                l.with_map("bs", "BTreeMap<Sol, f64>", self.bs.iter(), |ls, coord, l| {
                    let mut src_names = ls.names();
                    src_names.append(&mut cx(*ls));
                    l.log_pair(
                        "Sol",
                        src_names,
                        format!("{ls}"),
                        "f64",
                        vec![],
                        format!("{:.0}", coord.round())
                    )
                })?;
                l.with_map("ss", "BTreeMap<Sol, f64>", self.ss.iter(), |hs, coord, l| {
                    let mut src_names = hs.names();
                    src_names.append(&mut cx(*hs));
                    l.log_pair(
                        "HopSol",
                        src_names,
                        format!("{hs}"),
                        "f64",
                        vec![],
                        format!("{:.0}", coord.round())
                    )
                })?;
                Ok(())
            })
        }
    }

    fn update_min_width<V: Graphic + Display + Len + log::Name, E: Graphic>(
        vcg: &Vcg<V, E>,
        layout_problem: &LayoutProblem<V>,
        layout_solution: &LayoutSolution,
        geometry_problem: &GeometryProblem<V>,
        min_width: &mut OrderedFloat<f64>,
        vl: &V
    ) -> Result<(), Error> {
        let Vcg{vert: dag, vert_vxmap: dag_map, containers, ..} = vcg;
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
                    .ok_or_else::<Error, _>(|| LayoutError::OsqpError{error: "missing node weight".into()}.into())
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let mut w_srcs = w_ers
            .iter()
            .map(|er| {
                dag
                    .node_weight(er.source())
                    .map(Clone::clone)
                    .ok_or_else::<Error, _>(|| LayoutError::OsqpError{error: "missing node weight".into()}.into())
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        v_dsts.sort(); v_dsts.dedup();
        v_dsts.sort_by_key(|dst| {
            let (ovr, ohr) = node_to_loc[&Obj::from_vl(dst, containers)];
            let (svr, shr) = (ovr, solved_locs[&ovr][&ohr]);
            (shr, -(svr.0 as i32))
        });
        let v_outs = v_dsts
            .iter()
            .map(|dst| { (vl.clone(), dst.clone()) })
            .collect::<Vec<_>>();

        w_srcs.sort(); w_srcs.dedup();
        w_srcs.sort_by_key(|src| {
            let (ovr, ohr) = node_to_loc[&Obj::from_vl(src, containers)];
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
            .filter_map(|idx| {
                size_by_hop.get(idx).map(|sz| sz.left + sz.right)
            })
            .sum();
        let in_width: f64 = w_in_last_hops
            .iter()
            .filter_map(|idx| {
                size_by_hop.get(idx).map(|sz| sz.left + sz.right)
            })
            .sum();

        let of = OrderedFloat::<f64>::from;

        let out_width = of(out_width);
        let in_width = of(in_width);

        let orig_width = max(max(of(4.), *min_width), of(9. * vl.len() as f64));
        // min_width += max_by(out_width, in_width, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
        *min_width = max(orig_width, max(in_width, out_width));
        // eprintln!("lvl: {}, vl: {}, wl: {}, hops: {:?}", lvl, vl, wl, hops);
        Ok(())
    }

    fn left(sloc: (VerticalRank, SolvedHorizontalRank)) -> Option<(VerticalRank, SolvedHorizontalRank)> {
        sloc.1.checked_sub(1).map(|shrl| (sloc.0, shrl))
    }

    #[derive(Clone, Copy, Debug)]
    enum Direction {
        Vertical,
        Horizontal,
    }

    impl Display for Direction {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Direction::Vertical => write!(f, "Vertical"),
                Direction::Horizontal => write!(f, "Horizontal"),
            }
        }
    }

    #[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
    pub struct GeomLabel<V: Graphic, E: Graphic> {
        vl: V,
        wl: V,
        rel: E,
    }

    impl<V: Graphic, E: Graphic> Display for GeomLabel<V, E> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Label({}, {}, {})", self.vl, self.wl, self.rel)
        }
    }

    #[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
    pub enum Geom<V: Graphic, E: Graphic> {
        Node(Obj<V>),
        Hop(Obj<V>),
        Container(Obj<V>),
        Border(Obj<V>),
        Label(GeomLabel<V, E>),
    }

    impl<V: Graphic, E: Graphic> From<Obj<V>> for Geom<V, E> {
        fn from(value: Obj<V>) -> Self {
            match value {
                Obj::Node(_) => Geom::Node(value),
                Obj::Hop(_) => Geom::Hop(value),
                Obj::Container(_) => Geom::Container(value),
                Obj::Border(_) => Geom::Border(value),
            }
        }
    }

    impl<V: Graphic, E: Graphic> Geom<V, E> {
        pub fn as_vl(&self) -> Option<&V> {
            match self {
                Geom::Node(v) | Geom::Hop(v) | Geom::Container(v) | Geom::Border(v) => v.as_vl(),
                Geom::Label(label) => Some(&label.vl),
            }
        }

        pub fn as_wl(&self) -> Option<&V> {
            match self {
                Geom::Node(v) | Geom::Hop(v) | Geom::Container(v) | Geom::Border(v) => v.as_wl(),
                Geom::Label(label) => Some(&label.wl),
            }
        }
    }

    impl<V: Graphic, E: Graphic> Display for Geom<V, E> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Geom::Node(v) | Geom::Hop(v) | Geom::Container(v) | Geom::Border(v) => write!(f, "{v}"),
                Geom::Label(label) => write!(f, "{label}"),
            }
        }
    }

    #[derive(Clone, Debug)]
    struct ObjEdge {
        name: usize,
        reason: String,
        dir: Direction,
        margin: OrderedFloat<f64>,
    }

    impl Display for ObjEdge {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "no: {}, \nreason: {}, \ndir: {}, \nmargin: {}", self.name, self.reason, self.dir, self.margin)
        }
    }


    #[derive(Clone, Debug)]
    struct ConEdgeMargin {
        margin: OrderedFloat<f64>,
    }

    #[derive(Clone, Debug)]
    enum ConEdgeFlavor {
        Margin(ConEdgeMargin),
        Hop(),
    }

    impl Display for ConEdgeFlavor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ConEdgeFlavor::Margin(ConEdgeMargin{margin}) => write!(f, "margin: {}", margin.0),
                ConEdgeFlavor::Hop() => write!(f, "hop"),
            }
        }
    }

    #[derive(Clone, Debug)]
    struct ConEdge {
        name: usize,
        reason: String,
        flavor: ConEdgeFlavor,
    }

    impl Display for ConEdge {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "no: {}, \nreason: {}, {}", self.name, self.reason, self.flavor)
        }
    }


    pub fn position_sols<V, E>(
        vcg: &Vcg<V, E>,
        layout_problem: &LayoutProblem<V>,
        layout_solution: &LayoutSolution,
        geometry_problem: &GeometryProblem<V>,
        logs: &mut log::Logger,
    ) -> Result<(OptimizationProblem<AnySol, OrderedFloat<f64>>, OptimizationProblem<AnySol, OrderedFloat<f64>>), Error>
    where
        V: Graphic + Len + std::cmp::PartialEq<str>,
        E: Graphic + std::cmp::PartialEq<str>,
    {
        let containers = &vcg.containers;
        let nodes_by_container = &vcg.nodes_by_container;
        let loc_to_node = &layout_problem.loc_to_node;
        let node_to_loc = &layout_problem.node_to_loc;
        let hops_by_edge = &layout_problem.hops_by_edge;
        let solved_locs = &layout_solution.solved_locs;
        let size_by_loc = &geometry_problem.size_by_loc;
        let size_by_hop = &geometry_problem.size_by_hop;
        let varrank_by_obj = &geometry_problem.varrank_by_obj;

        let of = OrderedFloat::<f64>::from;

        // let obj_count = all_locs.len() + all_hops.len();
        // 1. Record how all objects are positioned relative to one another.
        let mut obj_graph = Graph::<Geom<V, E>, ObjEdge>::new();
        let mut obj_vxmap = HashMap::new();
        let mut solved_vxmap = HashMap::new();
        for (loc_ix, obj) in loc_to_node.iter() {
            let ix = or_insert(&mut obj_graph, &mut obj_vxmap, obj.clone().into());
            let shr = solved_locs[&loc_ix.0][&loc_ix.1];
            solved_vxmap.insert((loc_ix.0, shr), ix);
        }

        let solved_to_orig = solved_locs.iter()
            .flat_map(|(ovr, row)|
                row.iter().map(|(ohr, shr)| (*ovr, *ohr, *shr))
            )
            .map(|(ovr, ohr, shr)| ((ovr, shr), (ovr, ohr)))
            .collect::<HashMap<_, _>>();

        let mut obj_edges = 0;
        let mut obj_edge = |dir: Direction, reason: String, margin: f64| {
            let e = ObjEdge{name: obj_edges, reason, dir, margin: of(margin)};
            obj_edges += 1;
            e
        };

        for (solved, orig) in solved_to_orig.iter() {
            if let Some(left) = left(*solved) {
                let left_ix = solved_vxmap[&left];
                let solved_ix = solved_vxmap[solved];
                let left_obj = &loc_to_node[orig];
                let solved_obj = &loc_to_node[&solved_to_orig[&left]];
                if !obj_graph.contains_edge(left_ix, solved_ix) &&
                    !matches!(left_obj, Obj::Border(_)) &&
                    !matches!(solved_obj, Obj::Border(_)) &&
                    !matches!((left_obj, solved_obj),
                        (Obj::Container(ObjContainer{vl: container}),
                        Obj::Node(ObjNode{vl: wl}) | Obj::Container(ObjContainer{vl: wl}))
                        if nodes_by_container[container].contains(wl)) {
                    obj_graph.add_edge(left_ix, solved_ix, obj_edge(Direction::Horizontal, "solved-left".into(), 40.));
                }
            }
            // todo: hop edges, nested edges, ...
        }

        for er in vcg.vert.edge_references() {
            let ew = er.weight();
            if ew != "vertical" { continue; }
            let src = vcg.vert.node_weight(er.source()).unwrap();
            let dst = vcg.vert.node_weight(er.target()).unwrap();
            let src_obj = Obj::from_vl(src, containers);
            let dst_obj = Obj::from_vl(dst, containers);
            let src_loc = node_to_loc[&src_obj];
            let dst_loc = node_to_loc[&dst_obj];
            let src_ix = solved_vxmap[&(src_loc.0, solved_locs[&src_loc.0][&src_loc.1])];
            let dst_ix = solved_vxmap[&(dst_loc.0, solved_locs[&dst_loc.0][&dst_loc.1])];

            let hops = &hops_by_edge.get(&(src.clone(), dst.clone()));
            let hops = if hops.is_none() {
                vec![src_loc, dst_loc]
            } else {
                let mut hops = hops.unwrap().iter().map(|(ovr, (ohr, mhr))| (*ovr, *ohr, *mhr)).collect::<Vec<_>>();
                let last = hops.last().unwrap();
                hops.push((last.0 + 1, last.2, OriginalHorizontalRank(usize::MAX - last.1.0)));
                hops.into_iter().map(|(ovr, ohr, _)| (ovr, ohr)).collect::<Vec<_>>()
            };
            let mut prev_hop_ix = None;
            for hop in hops.iter() {
                let initial = hop == &src_loc;
                let terminal = hop == &dst_loc;
                let hop_ix = or_insert(&mut obj_graph, &mut obj_vxmap, Geom::Hop(Obj::Hop(ObjHop{vl: src.clone(), wl: dst.clone(), lvl: hop.0, mhr: hop.1})));
                if initial {
                    obj_graph.add_edge(src_ix, hop_ix, obj_edge(Direction::Vertical, format!("vert-edge: {}", ew), 0.));
                }
                if terminal {
                    obj_graph.add_edge(hop_ix, dst_ix, obj_edge(Direction::Vertical, format!("vert-edge: {}", ew), 0.));
                }
                if let Some(prev_hop_ix) = prev_hop_ix {
                    obj_graph.add_edge(prev_hop_ix, hop_ix, obj_edge(Direction::Vertical, format!("vert-edge: {}", ew), 26.));
                }
                prev_hop_ix = Some(hop_ix);
            }
        }

        // eprintln!("obj graph: {}", Dot::new(&obj_graph));

        if let Some(obj_svg) = as_svg(&obj_graph) {
            logs.log_svg(Some("obj_graph"), None::<String>, Vec::<String>::new(), obj_svg).unwrap();
        }


        // 2. Map objects to their corresponding positioning variables and
        // constraints between those variables.
        let mut con_graph = Graph::<AnySol, ConEdge>::new();
        let mut con_vxmap = HashMap::new();

        let mut con_edges = 0;
        let mut con_edge = |reason: String, flavor: ConEdgeFlavor| {
            let e = ConEdge{name: con_edges, reason, flavor};
            con_edges += 1;
            e
        };

        for geom in obj_graph.node_weights() {
            match geom {
                Geom::Node(obj@Obj::Node(ObjNode{vl})) => {
                    let ls = varrank_by_obj[&obj];
                    let left = or_insert(&mut con_graph, &mut con_vxmap, AnySol::L(ls));
                    let right = or_insert(&mut con_graph, &mut con_vxmap, AnySol::R(ls));
                    let top = or_insert(&mut con_graph, &mut con_vxmap, AnySol::T(ls));
                    let bottom = or_insert(&mut con_graph, &mut con_vxmap, AnySol::B(ls));
                    let node_width = &size_by_loc[&node_to_loc[obj]];
                    let mut width = of(node_width.width);
                    update_min_width(vcg, layout_problem, layout_solution, geometry_problem, &mut width, vl)?;
                    let height = 26.;
                    con_graph.add_edge(left, right, con_edge("node-width".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: width})));
                    con_graph.add_edge(top, bottom, con_edge("node-height".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(height)})));
                },
                Geom::Hop(obj@Obj::Hop(ObjHop{vl, wl, lvl, mhr})) => {
                    let hs = varrank_by_obj[&obj];
                    let guide_sol = or_insert(&mut con_graph, &mut con_vxmap, AnySol::S(hs));
                    let top = or_insert(&mut con_graph, &mut con_vxmap, AnySol::T(hs));
                    let bottom = or_insert(&mut con_graph, &mut con_vxmap, AnySol::B(hs));
                    let left = or_insert(&mut con_graph, &mut con_vxmap, AnySol::L(hs));
                    let right = or_insert(&mut con_graph, &mut con_vxmap, AnySol::R(hs));
                    let hop_size = &size_by_hop.get(&(*lvl, *mhr, vl.clone(), wl.clone())).unwrap_or_else(|| {
                        eprintln!("WARNING: con_graph: no size for hop: {obj}");
                        &HopSize{width: 10., left: 5., right: 5., height: 20., top: 0., bottom: 0.}
                    });
                    con_graph.add_edge(top, bottom, con_edge("hop-height".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(hop_size.height)})));
                    con_graph.add_edge(left, guide_sol, con_edge("hop-left".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(hop_size.left)})));
                    con_graph.add_edge(guide_sol, right, con_edge("hop-right".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(hop_size.right)})));
                },
                Geom::Container(obj@Obj::Container(ObjContainer{vl})) => {
                    let container = vl;
                    let container_sol = varrank_by_obj[&obj];
                    let left = or_insert(&mut con_graph, &mut con_vxmap, AnySol::L(container_sol));
                    let right = or_insert(&mut con_graph, &mut con_vxmap, AnySol::R(container_sol));
                    let top = or_insert(&mut con_graph, &mut con_vxmap, AnySol::T(container_sol));
                    let bottom = or_insert(&mut con_graph, &mut con_vxmap, AnySol::B(container_sol));
                    let node_width = &size_by_loc[&node_to_loc[obj]];
                    let mut width = of(node_width.width);
                    update_min_width(vcg, layout_problem, layout_solution, geometry_problem, &mut width, vl)?;
                    let height = 20.;
                    con_graph.add_edge(left, right, con_edge("container-width".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: width})));
                    con_graph.add_edge(top, bottom, con_edge("container-height".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(height)})));

                    for node in &nodes_by_container[container] {
                        let child_sol = varrank_by_obj[&Obj::from_vl(node, containers)];
                        // XXX: child might be a container too..., or on a deeper rank
                        let child_left = or_insert(&mut con_graph, &mut con_vxmap, AnySol::L(child_sol));
                        let child_right = or_insert(&mut con_graph, &mut con_vxmap, AnySol::R(child_sol));
                        let child_top = or_insert(&mut con_graph, &mut con_vxmap, AnySol::T(child_sol));
                        let child_bottom = or_insert(&mut con_graph, &mut con_vxmap, AnySol::B(child_sol));
                        let padding = 10.;
                        con_graph.add_edge(left, child_left, con_edge("child-padding-left".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(padding)})));
                        con_graph.add_edge(child_right, right, con_edge("child-padding-right".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(padding)})));
                        con_graph.add_edge(top, child_top, con_edge("child-padding-top".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(2.*padding + 6.)})));
                        con_graph.add_edge(child_bottom, bottom, con_edge("child-padding-bottom".into(), ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(padding)})));
                    }
                },
                _ => {}
            }
        }

        for er in obj_graph.edge_references() {
            let edge = er.weight();
            // eprintln!("EDGE IX: {:?} -> {:?}, {}", er.source(), er.target(), edge);
            let src = obj_graph.node_weight(er.source()).unwrap();
            let dst = obj_graph.node_weight(er.target()).unwrap();
            if matches!((src, dst),
                (Geom::Container(Obj::Container(ObjContainer{vl: container})),
                    Geom::Node(Obj::Node(ObjNode{vl: wl})) | Geom::Container(Obj::Container(ObjContainer{vl: wl})))
                    if nodes_by_container[container].contains(wl)) {
                continue;
            }

            macro_rules! sol_for {
                ($node:ident) => {
                    match $node {
                        Geom::Node(v) => {varrank_by_obj[v]},
                        Geom::Hop(v) => {varrank_by_obj[v]},
                        Geom::Container(v) => {varrank_by_obj[v]},
                        Geom::Border(Obj::Border(ObjBorder{border: Border{vl, ..}})) => {varrank_by_obj[&Obj::from_vl(vl, containers)]},
                        _ => unimplemented!(),
                    }
                };
            }
            macro_rules! impl_con {
                ($src:ident, $src_sol_con:ident, $dst:ident, $dst_sol_con:ident, $edge:ident) => {
                    {
                        let src_guide_sol = AnySol::$src_sol_con(sol_for!($src));
                        let dst_guide_sol = AnySol::$dst_sol_con(sol_for!($dst));
                        let src_ix = *con_vxmap.get(&src_guide_sol).expect(&format!("no entry for src key: {src_guide_sol}, src: ({src}) -> dst: ({dst}), edge: {edge}, in\n{varrank_by_obj:#?}\ncon_vxmap: {con_vxmap:#?}\n"));
                        let dst_ix = *con_vxmap.get(&dst_guide_sol).expect(&format!("no entry for dst key: {dst_guide_sol}, src: ({src}) -> dst: ({dst}), edge: {edge}, in\n{varrank_by_obj:#?}\ncon_vxmap: {con_vxmap:#?}\n"));
                        con_graph.add_edge(src_ix, dst_ix, con_edge(format!("obj-edge: {edge}"), $edge.clone()));
                    }
                };
            }

            enum Flavor {
                Box,
                Arrow,
            }
            impl<V: Graphic, E: Graphic> From<&Geom<V, E>> for Flavor {
                fn from(value: &Geom<V, E>) -> Self {
                    match value {
                        Geom::Node(_) | Geom::Container(_) | Geom::Border(_) | Geom::Label(_) => Flavor::Box,
                        Geom::Hop(_) => Flavor::Arrow,
                    }
                }
            }
            let src_flavor: Flavor = src.into();
            let dst_flavor: Flavor = dst.into();

            let margin = ConEdgeFlavor::Margin(ConEdgeMargin{margin: of(edge.margin.0)});
            let symmetrize = ConEdgeFlavor::Hop();
            match (src_flavor, dst_flavor, edge.dir) {
                (Flavor::Box, Flavor::Box, Direction::Vertical) => {
                    impl_con![src, B, dst, T, margin];
                },
                (Flavor::Box, Flavor::Box, Direction::Horizontal) => {
                    impl_con![src, R, dst, L, margin];
                },
                (Flavor::Box, Flavor::Arrow, Direction::Vertical) => {
                    impl_con![src, B, dst, T, margin];
                    impl_con![src, L, dst, L, margin];
                    impl_con![dst, L, dst, S, margin];
                    impl_con![dst, S, dst, R, margin];
                    impl_con![dst, R, src, R, margin];
                },
                (Flavor::Box, Flavor::Arrow, Direction::Horizontal) => {
                    impl_con![src, R, dst, S, margin];
                },
                (Flavor::Arrow, Flavor::Box, Direction::Vertical) => {
                    impl_con![src, B, dst, T, margin];
                    impl_con![dst, L, src, L, margin];
                    impl_con![src, L, src, S, margin];
                    impl_con![src, S, dst, R, margin];
                    impl_con![src, R, dst, R, margin];
                },
                (Flavor::Arrow, Flavor::Box, Direction::Horizontal) => {
                    impl_con![src, S, dst, L, margin];
                },
                (Flavor::Arrow, Flavor::Arrow, Direction::Vertical) => {
                    impl_con![src, B, dst, T, margin];
                    impl_con![src, S, dst, S, symmetrize];
                }, //{eprintln!("WARNING: arrow->arrow vertical constraint: {src} -> {dst}, edge: {edge}");},
                (Flavor::Arrow, Flavor::Arrow, Direction::Horizontal) => {
                    eprintln!("WARNING: arrow->arrow, {src} -> {dst}, edge: {edge}");
                    continue
                }, //impl_con![S, S],
            }
        }

        // eprintln!("con graph: {}", Dot::new(&con_graph));

        if let Some(con_svg) = as_svg(&con_graph) {
            logs.log_svg(Some("con_graph"), None::<String>, Vec::<String>::new(), con_svg).unwrap();
        }

        #[cfg(debug_assertions)]
        if is_cyclic_directed(&con_graph) {
            eprintln!("ERROR: CYCLIC CONSTRAINT GRAPH DETECTED");
        }

        // 3. Translate those variables and constraints into optimization problems to be solved.
        let mut vertical_problem = OptimizationProblem { v: Vars::new(), c: Constraints::new(), pd: vec![], q: vec![] };
        let mut horizontal_problem = OptimizationProblem { v: Vars::new(), c: Constraints::new(), pd: vec![], q: vec![] };

        for (sol, _ix) in con_vxmap.iter() {
            match sol {
                AnySol::L(_) | AnySol::R(_) | AnySol::S(_) | AnySol::V(_) => {
                    let var = horizontal_problem.v.get(*sol);
                    horizontal_problem.c.push((of(0.), vec![var.clone()], of(f64::INFINITY)));
                    horizontal_problem.q.push(var.clone());
                },
                AnySol::T(_) | AnySol::B(_) | AnySol::H(_) => {
                    let var = vertical_problem.v.get(*sol);
                    vertical_problem.c.push((of(0.), vec![var.clone()], of(f64::INFINITY)));
                    vertical_problem.q.push(var.clone());
                },
                _ => {},
            }
        }
        for er in con_graph.edge_references() {
            let src = con_graph.node_weight(er.source()).unwrap();
            let tgt = con_graph.node_weight(er.target()).unwrap();
            let wgt = er.weight();
            match (src, &wgt.flavor) {
                (AnySol::L(_) | AnySol::R(_) | AnySol::S(_) | AnySol::V(_), ConEdgeFlavor::Margin(ConEdgeMargin{margin})) => {
                    horizontal_problem.c.leqc(&mut horizontal_problem.v, *src, *tgt, *margin);
                },
                (AnySol::T(_) | AnySol::B(_) | AnySol::H(_), ConEdgeFlavor::Margin(ConEdgeMargin{margin})) => {
                    vertical_problem.c.leqc(&mut vertical_problem.v, *src, *tgt, *margin);
                },
                (_, ConEdgeFlavor::Hop()) => {
                    horizontal_problem.c.sym(&mut horizontal_problem.v, &mut horizontal_problem.pd, *src, *tgt, 10.);
                }
                _ => {},
            }
        }

        Ok((horizontal_problem, vertical_problem))
    }

    #[cfg(feature="desktop")]
    fn as_svg<V: Display, E: Display>(graph: &Graph<V, E>) -> Option<String> {
        use std::{process::{Stdio, Command}, io::Write};

        let dot = format!("{}", petgraph::dot::Dot::new(graph));
        let mut child = Command::new("dot")
            .arg("-Tsvg")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("failed to execute dot");
        let mut stdin = child.stdin.take().expect("Failed to open stdin");
        std::thread::spawn(move || {
            stdin.write_all(dot.as_bytes()).expect("Failed to write to stdin");
        });
        let output = child.wait_with_output().expect("Failed to read stdout");
        let svg = String::from_utf8_lossy(&output.stdout);
        Some(format!("data:image/svg+xml;utf8,{svg}"))
    }

    #[cfg(not(feature="desktop"))]
    fn as_svg<V, E>(_graph: &Graph<V, E>) -> Option<String> {
        None
    }

    #[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
    pub enum OSQPStatusKind {
        Solved,
        SolvedInaccurate,
        MaxIterationsReached,
        TimeLimitReached,
        #[default] Default,
    }

    fn solve_problem<S: Sol, C: Coeff>(
        optimization_problem: &OptimizationProblem<S, C>
    ) -> Result<(Vec<(Var<S>, f64)>, OSQPStatusKind), Error> {
        let OptimizationProblem{v, c, pd, q} = optimization_problem;

        let settings = osqp::Settings::default()
            .verbose(false)
            // .adaptive_rho(false)
            // .rho(6.)
            // .check_termination(Some(200))
            // .adaptive_rho_fraction(1.0) // https://github.com/osqp/osqp/issues/378
            // .adaptive_rho_interval(Some(25))
            .eps_abs(1.)
            .eps_rel(1e-2)
            // .max_iter(128_000)
            // .max_iter(400)
            // .polish(true)
        ;
        #[cfg(debug_assertions)]
        let settings = settings.verbose(true);
        let settings = &settings;

        let n = v.len();

        let sparse_pd = &pd[..];
        // eprintln!("sparsePd: {sparse_pd:?}");
        let p2 = as_diag_csc_matrix(Some(n), Some(n), sparse_pd);
        // print_tuples("P2", &p2);

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
        // eprintln!("V[{}]: {v}", v.len());
        // eprintln!("C[{}]: {c}", &c.len());

        let a2: osqp::CscMatrix = c.clone().into();

        // eprintln!("P2[{},{}]: {p2:?}", p2.nrows, p2.ncols);
        // eprintln!("Q2[{}]: {q2:?}", q2.len());
        // eprintln!("L2[{}]: {l2:?}", l2.len());
        // eprintln!("U2[{}]: {u2:?}", u2.len());
        // eprintln!("A2[{},{}]: {a2:?}", a2.nrows, a2.ncols);
        // eprintln!("NUMPY");
        // eprintln!("import osqp");
        // eprintln!("import numpy as np");
        // eprintln!("import scipy.sparse as sp");
        // eprintln!("");
        // eprintln!("inf = np.inf");
        // eprintln!("np.set_printoptions(precision=1, suppress=True)");
        // as_scipy("P", &p2);
        // as_numpy("q", &q2);
        // as_scipy("A", &a2);
        // as_numpy("l", &l2);
        // as_numpy("u", &u2);
        // eprintln!("m = osqp.OSQP()");
        // eprintln!("m.setup(P=P, q=q, A=A, l=l, u=u)");
        // eprintln!("r = m.solve()");
        // eprintln!("r.info.status");
        // eprintln!("r.x");

        let mut prob = osqp::Problem::new(p2, &q2[..], a2, &l2[..], &u2[..], settings)
            .map_err(|e| Error::from(LayoutError::from(e)))?;

        let result = prob.solve();
        // eprintln!("STATUS {:?}", result);
        let solution = match result {
            osqp::Status::Solved(solution) => Ok((solution, OSQPStatusKind::Solved)),
            osqp::Status::SolvedInaccurate(solution) => Ok((solution, OSQPStatusKind::SolvedInaccurate)),
            osqp::Status::MaxIterationsReached(solution) => Ok((solution, OSQPStatusKind::MaxIterationsReached)),
            osqp::Status::TimeLimitReached(solution) => Ok((solution, OSQPStatusKind::TimeLimitReached)),
            _ => Err(LayoutError::OsqpError{error: "failed to solve problem".into(),}),
        }?;
        let x = solution.0.x();

        // eprintln!("{:?}", x);
        let mut solutions = v.iter().map(|(_sol, var)| (*var, x[var.index])).collect::<Vec<_>>();
        solutions.sort_by_key(|(a, _)| *a);
        // for (var, val) in solutions {
        //     if !matches!(var.sol, AnySol::F(_)) {
        //         eprintln!("{} = {}", var.sol, val);
        //     }
        // }

        Ok((solutions, solution.1))
    }

    fn extract_variable<Idx: Copy + Debug + Ord, Val: Copy + Debug>(
        v: &Vars<AnySol>,
        solutions: &Vec<(Var<AnySol>, Val)>,
        kind: AnySolKind,
        _name: Cow<str>,
        extract_index: impl Fn(AnySol) -> Idx,
    ) -> BTreeMap<Idx, Val> {
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
        // eprintln!("{name}: {vs:?}");
        let vs = vs.iter().copied().collect::<BTreeMap<Idx, Val>>();
        vs
    }

    pub fn solve_optimization_problems(
        horizontal_problem: &OptimizationProblem<AnySol, OrderedFloat<f64>>,
        vertical_problem: &OptimizationProblem<AnySol, OrderedFloat<f64>>
    ) -> Result<GeometrySolution, Error> {
        let OptimizationProblem{v: vh, ..} = horizontal_problem;
        let OptimizationProblem{v: vv, ..} = vertical_problem;

        // eprintln!("SOLVE HORIZONTAL");
        let (solutions_h, status_h) = solve_problem(&horizontal_problem)?;

        // eprintln!("SOLVE VERTICAL");
        let (solutions_v, status_v) = solve_problem(&vertical_problem)?;

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

        let res = GeometrySolution{ls, rs, ss, ts, bs, status_h, status_v};
        Ok(res)
    }

}

pub mod frontend {
    use std::{borrow::Cow, collections::HashMap, cmp::{max}};

    use logos::Logos;
    use ordered_float::OrderedFloat;
    use self_cell::self_cell;

    use crate::{graph_drawing::{layout::{minimize_edge_crossing, calculate_vcg, rank, calculate_locs_and_hops, Border, ObjNode, ObjBorder}, eval::{eval, index, resolve}, geometry::{calculate_sols, position_sols, HopSize}}, parser::{Item, Parser, Token}};

    use super::{layout::{Vcg, LayoutProblem, Graphic, Len, Obj, RankedPaths, LayoutSolution}, geometry::{GeometryProblem, GeometrySolution, NodeSize, OptimizationProblem, AnySol, solve_optimization_problems}, error::{Error, Kind, OrErrExt}, eval::Val, index::OriginalHorizontalRank};

    use log::{names};

    pub fn estimate_widths<I: Graphic>(
        vcg: &Vcg<I, I>,
        layout_problem: &LayoutProblem<I>,
        geometry_problem: &mut GeometryProblem<I>
    ) -> Result<(), Error> where
        I: Graphic + Len,
    {
        // let char_width = 8.67;
        let char_width = 9.0;
        let line_height = geometry_problem.line_height.unwrap_or(20.);
        let arrow_width = 40.0;

        let vert = &vcg.vert;
        let vert_node_labels = &vcg.vert_node_labels;
        let vert_edge_labels = &vcg.vert_edge_labels;
        let horz_edge_labels = &vcg.horz_edge_labels;
        let size_by_loc = &mut geometry_problem.size_by_loc;
        let size_by_hop = &mut geometry_problem.size_by_hop;
        let hops_by_edge = &layout_problem.hops_by_edge;
        let loc_to_node = &layout_problem.loc_to_node;

        // eprintln!("LOC_TO_NODE WIDTHS: {loc_to_node:#?}");

        for (loc, node) in loc_to_node.iter() {
            let (ovr, ohr) = loc;
            if let Obj::Node(ObjNode{vl}) = node {
                let label = vert_node_labels
                    .get(vl)
                    .or_err(Kind::KeyNotFoundError{key: vl.to_string()})?
                    .clone();
                // if !label.is_screaming_snake_case() {
                //     label = label.to_title_case();
                // }
                let mut left = 0;
                let mut right = 0;
                for (hc, lvl) in horz_edge_labels.iter() {
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
            if let Obj::Border(ObjBorder{border}) = node {
                let Border{ovr, ohr, pair, ..} = border;
                size_by_loc.insert((*ovr, *ohr), NodeSize{width: 10., left: 0., right: 0., height: 0.});
                size_by_loc.insert((*ovr, *pair), NodeSize{width: 10., left: 0., right: 0., height: 0.});
            }
        }

        for ((vl, wl), hops) in hops_by_edge.iter() {
            let mut action_width = 10.0;
            let mut percept_width = 10.0;
            let ex = vert.find_edge(vcg.vert_vxmap[vl], vcg.vert_vxmap[wl]).unwrap();
            let ew = vert.edge_weight(ex).unwrap();
            if *ew != "vertical" { continue };

            let labels = vert_edge_labels.get(&(vl.clone(), wl.clone()));
            let mut height = 10.;
            if let Some(labels) = labels {
                let forward_label_width = labels.forward
                    .as_ref()
                    .and_then(|forward| forward
                        .iter()
                        .map(|label| label.len())
                        .max()
                    );

                let reverse_label_width = labels.reverse
                    .as_ref()
                    .and_then(|reverse| reverse
                        .iter()
                        .map(|label| label.len())
                        .max()
                    );

                let level_height = max(
                    labels.forward
                        .as_ref()
                        .map(|f| f.len())
                        .unwrap_or(0),
                    labels.reverse
                        .as_ref()
                        .map(|r| r.len())
                        .unwrap_or(0)
                );

                action_width = forward_label_width.map(|label_width| arrow_width + char_width * label_width as f64).unwrap_or(20.);
                percept_width = reverse_label_width.map(|label_width| arrow_width + char_width * label_width as f64).unwrap_or(20.);

                height = level_height as f64 * line_height;
            }

            let mut hops = hops.clone();
            let (lvl, (mhr, nhr)) = {
                #[allow(clippy::unwrap_used)] // an edge with no hops really should panic
                let kv = hops.last_key_value().unwrap();
                (*kv.0, *kv.1)
            };
            hops.insert(lvl+1, (nhr, OriginalHorizontalRank(std::usize::MAX - mhr.0)));

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

        // eprintln!("SIZE_BY_LOC: {size_by_loc:#?}");
        // eprintln!("SIZE_BY_HOP: {size_by_hop:#?}");

        Ok(())
    }

    #[derive(Default)]
    pub struct Depiction<'s> {
        pub items: Vec<Item<'s>>,
        pub val: Val<Cow<'s, str>>,
        pub vcg: Vcg<Cow<'s, str>, Cow<'s, str>>,
        pub paths_by_rank: RankedPaths<Cow<'s, str>>,
        pub layout_problem: LayoutProblem<Cow<'s, str>>,
        pub layout_solution: LayoutSolution,
        pub geometry_problem: GeometryProblem<Cow<'s, str>>,
        pub horizontal_problem: OptimizationProblem<AnySol, OrderedFloat<f64>>,
        pub vertical_problem: OptimizationProblem<AnySol, OrderedFloat<f64>>,
        pub geometry_solution: GeometrySolution,
    }

    self_cell!{
        pub struct RenderCell<'s> {
            owner: Cow<'s, str>,

            #[covariant]
            dependent: Depiction,
        }
    }

    pub fn render<'s, 't>(data: Cow<'s, str>, logs: &'t mut log::Logger) -> Result<RenderCell<'s>, Error> {
        RenderCell::try_new(data, |data| {
            let mut p = Parser::new();
            let mut lex = Token::lexer(data);
            while let Some(tk) = lex.next() {
                p.parse(tk)
                    .map_err(|_| {
                        Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
                    })?
            }

            let items = p.end_of_input()
                .map_err(|_| {
                    Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
                })?;

            // eprintln!("PARSE {items:#?}");

            let mut val = eval(&items[..]);

            // eprintln!("EVAL {val:#?}");

            let mut scopes = HashMap::new();
            let val2 = val.clone();
            index(&val2, &mut vec![], &mut scopes);
            resolve(&mut val, &mut vec![], &scopes);

            // eprintln!("SCOPES: {scopes:#?}");
            // eprintln!("RESOLVE: {val:#?}");

            let vcg = calculate_vcg(&val, logs)?;

            // eprintln!("HCG {hcg:#?}");

            let Vcg{vert, containers, nodes_by_container_transitive: nodes_by_container, container_depths, ..} = &vcg;

            let distance = {
                let containers = &containers;
                let nodes_by_container = &nodes_by_container;
                let container_depths = &container_depths;
                |src: Cow<str>, dst: Cow<str>, l: &mut log::Logger| {
                    if !containers.contains(&src) {
                        // if hcg.constraints.contains(&HorizontalConstraint{a: src.clone(), b: dst.clone()})
                        // || hcg.constraints.contains(&HorizontalConstraint{a: dst.clone(), b: src.clone()}) {
                        //     0
                        // } else {
                        //     -1
                        // }
                        l.log_pair(
                            "(V, V)",
                            names![src, dst],
                            format!("{}, {}", src, dst),
                            "isize, &str",
                            vec![],
                            format!("-1, src-not-container")
                        ).unwrap();
                        -1
                    } else {
                        if nodes_by_container[&src].contains(&dst) {
                            l.log_pair(
                                "(V, V)",
                                names![src, dst],
                                format!("{}, {}", src, dst),
                                "isize, &str",
                                vec![],
                                format!("0, src-contains-dst")
                            ).unwrap();
                            0
                        } else {
                            l.log_pair(
                                "(V, V)",
                                names![src, dst],
                                format!("{}, {}", src, dst),
                                "isize, &str",
                                vec![],
                                format!("{}, container_depths", -(container_depths[&src] as isize))
                            ).unwrap();
                            -(container_depths[&src] as isize)
                        }
                    }
                }
            };
            let paths_by_rank = rank(vert, distance, logs)?;

            logs.with_map("paths_by_rank", "BTreeMap<VerticalRank, SortedVec<(V, V)>>", paths_by_rank.iter(), |rank, paths, l| {
                l.with_map(format!("paths_by_rank[{rank}]"), "SortedVec<(V, V)>", paths.iter().map(|p| (&p.0, &p.1)), |from, to, l| {
                    l.log_pair(
                        "V",
                        names![from],
                        format!("{from}"),
                        "V",
                        names![to],
                        format!("{to}"),
                    )
                })
            })?;

            let layout_problem = calculate_locs_and_hops(&val, &paths_by_rank, &vcg, logs)?;

            // ... adjust problem for horizontal edges

            let layout_solution = minimize_edge_crossing(&vcg, &layout_problem)?;

            let mut geometry_problem = calculate_sols(&layout_problem, &layout_solution);

            estimate_widths(&vcg, &layout_problem, &mut geometry_problem)?;

            let (horizontal_problem, vertical_problem) = position_sols(&vcg, &layout_problem, &layout_solution, &geometry_problem, logs)?;

            let geometry_solution = solve_optimization_problems(&horizontal_problem, &vertical_problem)?;

            Ok(Depiction{
                items,
                val,
                vcg,
                paths_by_rank,
                layout_problem,
                layout_solution,
                geometry_problem,
                horizontal_problem,
                vertical_problem,
                geometry_solution,
            })
        })
    }

    pub mod log {

        #[derive(Clone, Debug, PartialEq, PartialOrd)]
        pub enum Record {
            String { name: Option<String>, ty: Option<String>, names: Vec<String>, val: String, },
            Group { name: Option<String>, ty: Option<String>, names: Vec<String>, val: Vec<Record>, },
            Svg { name: Option<String>, ty: Option<String>, names: Vec<String>, val: String, }
        }

        #[derive(Clone, Debug, thiserror::Error)]
        #[error("LogError")]
        pub enum Error {
            FmtError{ #[from] source: std::fmt::Error }
        }

        pub trait Log<Cx = ()> {
            fn log(&self, cx: Cx, l: &mut Logger) -> Result<(), Error>;
        }

        pub trait Name {
            fn name(&self) -> String;
        }

        impl Name for String {
            fn name(&self) -> String {
                self.clone()
            }
        }

        impl<'s> Name for std::borrow::Cow<'s, str> {
            fn name(&self) -> String {
                self.to_string()
            }
        }

        pub trait Names<'n> {
            fn names(&self) -> Vec<Box<dyn Name + 'n>>;
        }

        #[macro_export]
        macro_rules! names {
            ($($x:expr),*) => {{
                let mut v = Vec::new();
                $(
                    v.push(Box::new(($x).clone()) as Box<dyn crate::graph_drawing::frontend::log::Name>);
                )*
                v
            }}
        }

        pub use names;

        impl<'n, A: Clone + Name + 'n> Names<'n> for A {
            fn names(&self) -> Vec<Box<dyn Name + 'n>> {
                names![self]
            }
        }

        impl<'n, A1: Clone + Name + 'n, A2: Clone + Name + 'n> Names<'n> for (A1, A2) {
            fn names(&self) -> Vec<Box<dyn Name + 'n>> {
                names![self.0, self.1]
            }
        }

        #[derive(Clone, Debug, Default)]
        pub struct Logger {
            logs: Vec<Record>,
        }

        impl Logger {
            pub fn new() -> Self {
                Self {
                    logs: vec![],
                }
            }

            pub fn with_map<K, V, F>(
                &mut self,
                name: impl Into<String>,
                ty: impl Into<String>,
                pairs: impl IntoIterator<Item=(K, V)>,
                mut f: F
            ) -> Result<(), Error> where
                F: FnMut(K, V, &mut Logger) -> Result<(), Error>
            {
                self.with_group(ty.into(), name.into(), Vec::<String>::new(), |l| {
                    for (k, v) in pairs.into_iter() {
                        f(k, v, l)?
                    }
                    Ok(())
                })?;
                Ok(())
            }

            pub fn with_set<V, F>(
                &mut self,
                name: impl Into<String>,
                ty: impl Into<String>,
                elements: impl IntoIterator<Item=V>,
                mut f: F
            ) -> Result<(), Error> where
                F: FnMut(V, &mut Logger) -> Result<(), Error>
            {
                self.with_group(ty.into(), name.into(), Vec::<String>::new(), |l| {
                    for v in elements.into_iter() {
                        f(v, l)?
                    }
                    Ok(())
                })?;
                Ok(())
            }

            pub fn log_pair<'n>(
                &mut self,
                src_ty: impl Into<String>,
                src_names: Vec<Box<dyn Name + 'n>>,
                src_val: impl Into<String>,
                dst_ty: impl Into<String>,
                dst_names: Vec<Box<dyn Name + 'n>>,
                dst_val: impl Into<String>
            ) -> Result<(), Error> {
                let src_ty = src_ty.into();
                let mut src_names = src_names.into_iter().map(|n| n.name()).collect::<Vec<String>>();
                let src_val = src_val.into();
                let dst_ty = dst_ty.into();
                let mut dst_names = dst_names.into_iter().map(|n| n.name()).collect::<Vec<String>>();
                let dst_val = dst_val.into();
                src_names.append(&mut dst_names);
                self.logs.push(Record::String{
                    name: None,
                    ty: Some(format!("{src_ty}->{dst_ty}")),
                    names: src_names,
                    val: format!("{src_val} -> {dst_val}")
                });
                Ok(())
            }

            pub fn log_element(
                &mut self,
                ty: impl Into<String>,
                names: Vec<impl Into<String>>,
                val: impl Into<String>
            ) -> Result<(), Error> {
                let ty = ty.into();
                let names = names.into_iter().map(|n| n.into()).collect::<Vec<String>>();
                let val = val.into();
                self.logs.push(Record::String{
                    name: None,
                    ty: Some(ty),
                    names,
                    val,
                });
                Ok(())
            }

            pub fn log_string(&mut self, name: impl Into<String>, val: impl std::fmt::Debug) -> Result<(), Error> {
                self.logs.push(Record::String{
                    name: Some(name.into()),
                    ty: None,
                    names: vec![],
                    val: format!("{val:#?}")
                });
                Ok(())
            }

            pub fn log_svg(&mut self, name: Option<impl Into<String>>, ty: Option<impl Into<String>>, names: Vec<impl Into<String>>, val: String) -> Result<(), Error> {
                let name = name.map(|name| name.into());
                let ty = ty.map(|ty| ty.into());
                let names = names.into_iter().map(|n| n.into()).collect::<Vec<String>>();
                self.logs.push(Record::Svg{
                    name,
                    ty,
                    names,
                    val,
                });
                Ok(())
            }

            pub fn with_group<F>(
                &mut self,
                ty: impl Into<String>,
                name: impl Into<String>,
                names: Vec<impl Into<String>>,
                f: F
            ) -> Result<(), Error> where
                F: FnOnce(&mut Logger) -> Result<(), Error>
            {
                let mut nested_logger = Logger::new();
                f(&mut nested_logger)?;
                self.logs.push(Record::Group{
                    name: Some(name.into()),
                    ty: Some(ty.into()),
                    names: names.into_iter().map(|n| n.into()).collect::<Vec<_>>(),
                    val: nested_logger.logs
                });
                Ok(())
            }

            pub fn to_vec(self) -> Vec<Record> {
                self.logs
            }
        }
    }

    pub mod dom {
        use std::{borrow::Cow};

        use petgraph::visit::EdgeRef;

        use crate::{graph_drawing::{error::{OrErrExt, Kind, Error}, layout::{Obj, Border, ObjContainer, ObjNode, ObjHop, ObjBorder}, index::{VarRank}, geometry::{NodeSize, HopSize, OSQPStatusKind}, frontend::log::{Names}}, names};

        use super::log::{self, Log};


        #[derive(Clone, Debug, PartialEq, PartialOrd)]
        pub struct Label {
            pub text: String,
            pub hpos: f64,
            pub width: f64,
            pub vpos: f64,
        }

        #[derive(Clone, Debug, PartialEq, PartialOrd)]
        pub enum Node {
            Div { key: String, label: String, hpos: f64, vpos: f64, width: f64, height: f64, z_index: usize, loc: VarRank, estimated_size: NodeSize },
            Svg { key: String, path: String, z_index: usize, dir: String, rel: String, label: Option<Label>, hops: Vec<VarRank>, classes: String, estimated_size: HopSize, control_points: Vec<(f64, f64)> },
        }

        #[derive(Clone, Debug)]
        pub struct Drawing {
            pub crossing_number: Option<usize>,
            pub status_v: OSQPStatusKind,
            pub status_h: OSQPStatusKind,
            pub viewbox_width: f64,
            pub viewbox_height: f64,
            pub nodes: Vec<Node>,
            pub logs: Vec<log::Record>,
        }

        impl Default for Drawing {
            fn default() -> Self {
                Self {
                    crossing_number: Default::default(),
                    status_v: Default::default(),
                    status_h: Default::default(),
                    viewbox_width: 1024.0,
                    viewbox_height: 400.0,
                    nodes: Default::default(),
                    logs: vec![],
                }
            }
        }

        pub fn draw(data: String) -> Result<Drawing, Error> {
            let mut logs = log::Logger::new();

            let render_cell = super::render(Cow::Owned(data), &mut logs)?;
            let depiction = render_cell.borrow_dependent();

            let val = &depiction.val;
            let geometry_solution = &depiction.geometry_solution;
            let rs = &depiction.geometry_solution.rs;
            let ls = &depiction.geometry_solution.ls;
            let ss = &depiction.geometry_solution.ss;
            let ts = &depiction.geometry_solution.ts;
            let bs = &depiction.geometry_solution.bs;
            let status_v = depiction.geometry_solution.status_v;
            let status_h = depiction.geometry_solution.status_h;
            let varrank_by_obj = &depiction.geometry_problem.varrank_by_obj;
            let loc_by_varrank = &depiction.geometry_problem.loc_by_varrank;
            let loc_to_node = &depiction.layout_problem.loc_to_node;
            let node_to_loc = &depiction.layout_problem.node_to_loc;
            let vert = &depiction.vcg.vert;
            let vert_node_labels = &depiction.vcg.vert_node_labels;
            let vert_edge_labels = &depiction.vcg.vert_edge_labels;
            let horz_edge_labels = &depiction.vcg.horz_edge_labels;
            let hops_by_edge = &depiction.layout_problem.hops_by_edge;
            let size_by_loc = &depiction.geometry_problem.size_by_loc;
            let size_by_hop = &depiction.geometry_problem.size_by_hop;
            let crossing_number = depiction.layout_solution.crossing_number;
            let containers = &depiction.vcg.containers;
            let container_borders = &depiction.layout_problem.container_borders;
            let container_depths = &depiction.vcg.container_depths;
            let nesting_depths = &depiction.vcg.nesting_depths;
            let solved_locs = &depiction.layout_solution.solved_locs;
            let horizontal_problem = &depiction.horizontal_problem;
            let vertical_problem = &depiction.vertical_problem;

            let char_width = &depiction.geometry_problem.char_width.unwrap_or(9.);

            let mut texts = vec![];

            // Log the resolved value
            // logs.log_string("VAL", val);
            logs.with_group("Eval", "", Vec::<String>::new(), |mut logs| {
                val.log("VAL".into(), &mut logs)
            })?;

            let l2n = |ovr, ohr| match &loc_to_node[&(ovr, ohr)] {
                Obj::Node(ObjNode{vl: node}) => node.to_string().names(),
                Obj::Hop(ObjHop{lvl: ovr, mhr, vl, wl}) => names![ovr, mhr, vl.to_string(), wl.to_string()],
                Obj::Container(ObjContainer{vl}) => vl.to_string().names(),
                Obj::Border(ObjBorder{border: Border{vl, ovr, ohr, pair}}) => names![vl.to_string(), ovr, ohr, pair],
            };
            let v2n = |varrank| {
                let loc_ix = loc_by_varrank[&varrank];
                l2n(loc_ix.0, loc_ix.1)
            };


            logs.with_group("Layout", "", Vec::<String>::new(), |mut logs| {
                loc_to_node.log((), &mut logs)?;
                // sol_by_loc.log(l2n, &mut logs)?;
                // sol_by_hop.log(l2n, &mut logs)?;
                varrank_by_obj.log((), &mut logs)?;
                // loc_by_varrank.log(l2n, &mut logs)?;
                solved_locs.log(l2n, &mut logs)?;
                size_by_loc.log(l2n, &mut logs)?;
                size_by_hop.log(l2n, &mut logs)
            })?;
            horizontal_problem.log(("horizontal_problem".into(), v2n), &mut logs)?;
            vertical_problem.log(("vertical_problem".into(), v2n), &mut logs)?;
            geometry_solution.log(v2n, &mut logs)?;

            let mut logs = logs.to_vec();
            logs.reverse();

            // Render Nodes
            let viewbox_width = (rs.values().copied().map(ordered_float::OrderedFloat).max().unwrap_or_default() - ls.values().copied().map(ordered_float::OrderedFloat).min().unwrap_or_default()).0;
            let viewbox_height = (bs.values().copied().map(ordered_float::OrderedFloat).max().unwrap_or_default() - ts.values().copied().map(ordered_float::OrderedFloat).min().unwrap_or_default()).0;

            for (loc, node) in loc_to_node.iter() {
                let (ovr, ohr) = loc;

                if let Obj::Node(ObjNode{vl}) = node {
                    let n = varrank_by_obj[node];

                    // eprintln!("TEXT {vl} {ovr} {ohr} {n}");

                    let lpos = ls[&n];
                    let rpos = rs[&n];

                    let z_index = nesting_depths[vl];

                    let vpos = ts[&n];
                    let width = (rpos - lpos).round();
                    let hpos = lpos.round();
                    let height = bs[&n] - ts[&n];

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

            for container in containers {
                let container_obj = Obj::Container(ObjContainer{ vl: container.clone() });
                let container_loc_ix = &node_to_loc[&container_obj];
                let (ovr, ohr) = container_loc_ix;
                let cn = varrank_by_obj[&container_obj];
                let lpos = ls[&cn];
                let rpos = rs[&cn];
                let z_index = nesting_depths[container];
                let vpos = ts[&cn];
                let width = (rpos - lpos).round();
                let hpos = lpos.round();
                let height = bs[&cn] - ts[&cn];
                // eprintln!("HEIGHT: {container} {height}");

                let key = format!("{container}");
                let mut label = vert_node_labels
                    .get(container)
                    .or_err(Kind::KeyNotFoundError{key: container.to_string()})?
                    .clone();
                if label == "_" { label = String::new(); };

                let estimated_size = size_by_loc[&(*ovr, *ohr)].clone();
                texts.push(Node::Div{key, label, hpos, vpos, width, height, z_index, loc: cn, estimated_size});
            }

            let mut arrows = vec![];

            for er in vert.edge_references() {
                let vl = vert.node_weight(er.source()).unwrap();
                let wl = vert.node_weight(er.target()).unwrap();
                let ew = er.weight();

                let Some(level) = vert_edge_labels.get(&(vl.clone(), wl.clone())) else {continue};

                for (dir, labels) in &[("forward", level.forward.as_ref()), ("reverse", level.reverse.as_ref())] {
                    let label_text = labels.map(|labels| labels.join("\n"));
                    let hops = hops_by_edge.get(&(vl.clone(), wl.clone()));
                    let hops = if let Some(hops) = hops { hops } else { continue; };
                    // eprintln!("vl: {}, wl: {}, hops: {:?}", vl, wl, hops);

                    let offset = match *dir {
                        x if x == "forward" => -10.0,
                        x if x == "forward" => -10.0,
                        x if x == "reverse" => 10.0,
                        x if x == "reverse" => 10.0,
                        _ => 0.0,
                    };

                    let z_index = std::cmp::max(nesting_depths[vl], nesting_depths[wl]) + 1;

                    let mut path = vec![];
                    let mut control_points = vec![];
                    let mut label_hpos = None;
                    let mut label_width = None;
                    let mut label_vpos = None;
                    // use rand::Rng;
                    // let mut rng = rand::thread_rng();
                    let mut hn0 = vec![];
                    let mut estimated_size0 = None;

                    let fs = container_depths.get(vl).copied().and_then(|d| d.checked_sub(1)).unwrap_or(0);
                    let hops = if fs == 0 {
                        hops.iter().collect::<Vec<_>>()
                    } else {
                        hops.iter().skip(fs).collect::<Vec<_>>()
                    };
                    let vmin = bs[&varrank_by_obj[&Obj::from_vl(vl, containers)]];
                    let vmax = ts[&varrank_by_obj[&&Obj::from_vl(wl, containers)]];
                    let nh = hops.len();
                    let vs = (0..=nh).map(|lvl|  {
                        let fraction = lvl as f64 / nh as f64;
                        vmin + fraction * (vmax - vmin)
                    })
                        .collect::<Vec<_>>();
                    // eprintln!("vl: {vl}, wl: {wl}, fs: {fs}, vmin: {vmin}, vmax: {vmax}, nh: {nh}, vs: {vs:?}");
                    for (n, hop) in hops.iter().enumerate() {
                        let (lvl, (mhr, nhr)) = *hop;
                        // let hn = sol_by_hop[&(*lvl+1, *nhr, vl.clone(), wl.clone())];
                        let hn = varrank_by_obj[&Obj::Hop(ObjHop{lvl: *lvl, mhr: *mhr, vl: vl.clone(), wl: wl.clone()})];
                        let spos = ss[&hn];
                        let hnd = varrank_by_obj[&Obj::Hop(ObjHop{lvl: *lvl+1, mhr: *nhr, vl: vl.clone(), wl: wl.clone()})];
                        let sposd = ss[&hnd];
                        let hpos = (spos + offset).round(); // + rng.gen_range(-0.1..0.1));
                        let hposd = (sposd + offset).round(); //  + 10. * lvl.0 as f64;
                        // eprintln!("HOP {vl} {wl} {n} {hop:?} {lvl} {} {}", ndv, lvl_offset);
                        let vpos = vs[n];
                        let mut vpos2 = vs[n+1];

                        if n == 0 {
                            hn0.push(hn);
                        }
                        hn0.push(hnd);

                        if n == 0 {
                            estimated_size0 = Some(size_by_hop[&(*lvl, *mhr, vl.clone(), wl.clone())].clone());
                            let mut vpos = vpos;
                            if *dir == "reverse" {
                                vpos += 7.0; // box height + arrow length
                                label_vpos = Some(vpos);
                            } else {
                                // vpos += 26.0;
                            }
                            path.push(format!("M {hpos} {vpos}"));
                            control_points.push((hpos, vpos));
                        }

                        if n == 0 {
                            let n = varrank_by_obj[&Obj::Hop(ObjHop{lvl: *lvl+1, mhr: *nhr, vl: vl.clone(), wl: wl.clone()})];
                            // sol_by_loc[&((*lvl+1), *nhr)];
                            // let n = sol_by_loc[&((*lvl), *mhr)];
                            label_hpos = Some(match *dir {
                                "reverse" => {
                                    // ls[n]
                                    hposd
                                },
                                "forward" => {
                                    // ls[n]
                                    hposd
                                },
                                _ => hposd + 9.
                            });
                            label_width = Some(match *dir {
                                "reverse" => {
                                    // ls[n]
                                    rs[&n] - sposd
                                },
                                "forward" => {
                                    // ls[n]
                                    sposd - ls[&n]
                                },
                                _ => rs[&n] - ls[&n]
                            });
                        }

                        // if n < hops.len() - 1 {
                        //     vpos2 += 26.;
                        // }

                        if n == hops.len() - 1 && *dir == "forward" {
                            vpos2 -= 7.0; // arrowhead length
                            if *dir == "forward" {
                                label_vpos = Some(vpos2 - estimated_size0.as_ref().unwrap().height);
                            }
                        }

                        path.push(format!("L {hposd} {vpos2}"));
                        control_points.push((hpos, vpos2));
                    }

                    let key = format!("{vl}_{wl}_{ew}_{dir}");
                    let path = path.join(" ");

                    let mut label = None;

                    if let (Some(label_text), Some(label_hpos), Some(label_width), Some(label_vpos)) = (label_text, label_hpos, label_width, label_vpos) {
                        label = Some(Label{text: label_text, hpos: label_hpos, width: label_width, vpos: label_vpos})
                    }
                    let classes = format!("arrow vertical {ew} {vl}_{wl} {vl}_{wl}_{ew}");

                    arrows.push(Node::Svg{key, path, z_index, dir: "vertical".into(), rel: dir.to_string(), label, hops: hn0, classes, estimated_size: estimated_size0.unwrap(), control_points});
                }
            }
            let forward_voffset = 6.;
            let reverse_voffset = 20.;

            for (m, ((vl, wl), lvl)) in horz_edge_labels.iter().enumerate() {

                let z_index = std::cmp::max(nesting_depths[vl], nesting_depths[wl]) + 1;

                if let Some(forward) = &lvl.forward {
                    let key = format!("{vl}_{wl}_forward_{m}");
                    let classes = format!("arrow horizontal forward {vl}_{wl} {vl}_{wl}_forward");
                    let locl = if !containers.contains(vl) {
                        node_to_loc[&Obj::Node(ObjNode{vl: vl.clone()})]
                    } else {
                        let (ovr, (ohr, pair)) = container_borders[vl].first().unwrap();
                        let shr1 = solved_locs[ovr][ohr];
                        let shr2 = solved_locs[ovr][pair];
                        // let lohr = if shr1 < shr2 { ohr } else { pair };
                        let rohr = if shr1 < shr2 { pair } else { ohr };
                        (*ovr, *rohr)
                    };
                    let locr = if !containers.contains(wl) {
                        node_to_loc[&Obj::Node(ObjNode{vl: wl.clone()})]
                    } else {
                        let (ovr, (ohr, pair)) = container_borders[wl].first().unwrap();
                        let shr1 = solved_locs[ovr][ohr];
                        let shr2 = solved_locs[ovr][pair];
                        let lohr = if shr1 < shr2 { ohr } else { pair };
                        // let rohr = if shr1 < shr2 { pair } else { ohr };
                        (*ovr, *lohr)
                    };
                    let nl = varrank_by_obj[&Obj::from_vl(vl, containers)];
                    let nr = varrank_by_obj[&Obj::from_vl(wl, containers)];
                    let lr = rs[&nl];
                    let rl = ls[&nr] - 7.;
                    let wl = size_by_loc[&locl].right;
                    let wr = size_by_loc[&locr].left;
                    let vposl = ts[&nl] + forward_voffset;
                    let vposr = ts[&nr] + forward_voffset;
                    let path = format!("M {} {} L {} {}", lr, vposl, rl, vposr);
                    let control_points = vec![(lr, vposl), (rl, vposr)];
                    let label_text = forward.join("\n");
                    let label_width = char_width * forward.iter().map(|f| f.len()).max().unwrap_or(0) as f64;
                    let label = if !forward.is_empty() {
                        Some(Label{text: label_text, hpos: rl, width: label_width, vpos: vposl })
                    } else {
                        None
                    };
                    let estimated_size = HopSize{ width: 0., left: wl, right: wr, height: 0., top: 0., bottom: 0. };
                    arrows.push(Node::Svg{key, path, z_index, label, dir: "horizontal".into(), rel: "forward".into(), hops: vec![], classes, estimated_size, control_points });
                }
                if let Some(reverse) = &lvl.reverse {
                    let key = format!("{vl}_{wl}_reverse_{m}");
                    let classes = format!("arrow horizontal reverse {vl}_{wl} {vl}_{wl}_reverse");
                    let locl = if !containers.contains(vl) {
                        node_to_loc[&Obj::Node(ObjNode{vl: vl.clone()})]
                    } else {
                        let (ovr, (ohr, pair)) = container_borders[vl].first().unwrap();
                        let shr1 = solved_locs[ovr][ohr];
                        let shr2 = solved_locs[ovr][pair];
                        // let lohr = if shr1 < shr2 { ohr } else { pair };
                        let rohr = if shr1 < shr2 { pair } else { ohr };
                        (*ovr, *rohr)
                    };
                    let locr = if !containers.contains(wl) {
                        node_to_loc[&Obj::Node(ObjNode{vl: wl.clone()})]
                    } else {
                        let (ovr, (ohr, pair)) = container_borders[wl].first().unwrap();
                        let shr1 = solved_locs[ovr][ohr];
                        let shr2 = solved_locs[ovr][pair];
                        let lohr = if shr1 < shr2 { ohr } else { pair };
                        // let rohr = if shr1 < shr2 { pair } else { ohr };
                        (*ovr, *lohr)
                    };
                    let nl = varrank_by_obj[&Obj::from_vl(vl, containers)];
                    let nr = varrank_by_obj[&Obj::from_vl(wl, containers)];
                    let lr = rs[&nl] + 7.;
                    let rl = ls[&nr];
                    let wl = size_by_loc[&locl].right;
                    let wr = size_by_loc[&locr].left;
                    let vposl = ts[&nl] + reverse_voffset;
                    let vposr = ts[&nr] + reverse_voffset;
                    let path = format!("M {} {} L {} {}", lr, vposl, rl, vposr);
                    let control_points = vec![(lr, vposl), (rl, vposr)];
                    let label_text = reverse.join("\n");
                    let label_width = char_width * reverse.iter().map(|f| f.len()).max().unwrap_or(0) as f64;
                    let label = if !reverse.is_empty() {
                        Some(Label{text: label_text, hpos: lr, width: label_width, vpos: vposl })
                    } else {
                        None
                    };
                    let estimated_size = HopSize{ width: 0., left: wl, right: wr, height: 0., top: 0., bottom: 0. };
                    arrows.push(Node::Svg{key, path, z_index, label, dir: "horizontal".into(), rel: "reverse".into(), hops: vec![], classes, estimated_size, control_points });
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

            // eprintln!("NODES: {nodes:#?}");

            Ok(Drawing{
                crossing_number: Some(crossing_number),
                status_v,
                status_h,
                viewbox_width,
                viewbox_height,
                nodes,
                logs,
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
                    Node::Svg{path, dir, rel, label, classes, ..} => {

                        let mut path_elt = Path::new()
                            .set("class", classes)
                            .set("d", path)
                            .set("stroke", "black");

                        match rel.as_str() {
                            "forward" => path_elt = path_elt.set("marker-end", "url(%23arrowhead)"),
                            "reverse" => path_elt = path_elt.set("marker-start", "url(%23arrowheadrev)"),
                            _ => {},
                        };

                        if let Some(Label{text, hpos, width: _, vpos, ..}) = label {
                            for (lineno, line) in text.lines().enumerate() {
                                let translate = match (dir.as_ref(), rel.as_ref()) {
                                    ("vertical", "forward") => format!("translate({}, {})", hpos-12., vpos + 56. + (20. * lineno as f64)),
                                    ("vertical", "reverse") => format!("translate({}, {})", hpos+12., vpos + 56. + (20. * lineno as f64)),
                                    ("horizontal", "forward") => format!("translate({}, {})", hpos, vpos - 10. - 20. * lineno as f64),
                                    ("horizontal", "reverse") => format!("translate({}, {})", hpos, vpos + 20. + 20. * lineno as f64),
                                    ("vertical", "fake") => format!("translate({}, {})", hpos, vpos + 20.),
                                    _ => format!("translate({}, {})", hpos, (vpos + 20. * lineno as f64)),
                                };
                                let anchor = match rel.as_ref() {
                                    "forward" => "end",
                                    _ => "start",
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
            // eprintln!("SVG {svg_str}");
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
                                class: "box highlight_{label}",
                                style: "position: absolute; top: {vpos}px; left: {hpos}px; width: {width}px; height: {height}px; z-index: {z_index}; padding-top: 3px; padding-bottom: 3px; box-sizing: border-box; border: 1px solid black; text-align: center; z-index: 10;", // bg-opacity-50
                                span {
                                    "{label}"
                                }
                            }
                        }));
                    },
                    Node::Svg{key, path, z_index, dir, rel, label, classes, ..} => {
                        let marker_id = if rel == "forward" { "arrowhead" } else { "arrowheadrev" };
                        let marker_orient = if rel == "forward" { "auto" } else { "auto-start-reverse" };
                        // let stroke_dasharray = if rel == "fake" { "5 5" } else { "none" };
                        // let stroke_color = if rel == "fake" { "hsl(0, 0%, 50%)" } else { "currentColor" };
                        children.push(cx.render(rsx!{
                            div {
                                key: "{key}",
                                class: "{classes}",
                                style: "position: absolute; z-index: {z_index};",
                                svg {
                                    fill: "none",
                                    // stroke: "{stroke_color}",
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    // stroke_width: "1",
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
                                            "forward" => {
                                                rsx!(path {
                                                    d: "{path}",
                                                    marker_end: "url(#arrowhead)",
                                                })
                                            },
                                            "reverse" => {
                                                rsx!(path {
                                                    d: "{path}",
                                                    "marker-start": "url(#arrowheadrev)",
                                                    // marker_start: "url(#arrowhead)", // BUG: should work, but doesn't.
                                                })
                                            },
                                            _ => {
                                                rsx!(path {
                                                    d: "{path}",
                                                    // stroke_dasharray: "{stroke_dasharray}",
                                                })
                                            }
                                        }
                                    }
                                }
                                {match label {
                                    Some(Label{text, hpos, width: _, vpos}) => {
                                        let translate = match (dir.as_ref(), rel.as_ref()) {
                                            ("vertical", "forward") => "translate(calc(-100% - 1.5ex))",
                                            ("horizontal", "forward") => "translate(calc(-100% - 1.5ex), calc(-100% + 20px))",
                                            (_, "reverse") => "translate(1.5ex)",
                                            _ => "translate(0px, 0px)",
                                        };
                                        let offset = match (dir.as_ref(), rel.as_ref()) {
                                            ("horizontal", "forward") => "-24px",
                                            ("horizontal", "reverse") => "4px",
                                            _ => "0px",
                                        };
                                        // let border = match rel.as_str() {
                                        //     // "actuates" => "border border-red-300",
                                        //     // "senses" => "border border-blue-300",
                                        //     _ => "",
                                        // };
                                        rsx!(div {
                                            style: "position: absolute; left: {hpos}px; top: calc({vpos}px + {offset});", // width: "{width}px",
                                            div {
                                                style: "white-space: pre; z-index: 50; background-color: #fff; box-sizing: border-box; transform: {translate}; font-size: .875rem; line-height: 1.25rem; font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",\"Courier New\",monospace;",
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
            Some(cx.render(rsx!(children.into_iter())).unwrap())
        }

        pub fn syntax_guide<P>(cx: Scope<P>) -> Option<VNode> {
            cx.render(rsx!{
                div {
                    div {
                        style: "display: flex; flex-direction: row; justify-content: space-between; font-size: 0.875rem; line-height: 1.25rem;",
                        div {
                            details {
                                // open: "true",
                                summary {
                                    span {
                                        style: "color: #000;",
                                        "Syntax + Examples"
                                    }
                                }
                                div {
                                    p {
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "nodes"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            ":"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "actions"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            "/"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "feedback"
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            class: "example",
                                            "person microwave food: open start stop / beep : heat"
                                        },
                                        br {},
                                        span {
                                            class: "example",
                                            "person food: eat"
                                        },
                                    }
                                    p {
                                        span {
                                            class: "keyword",
                                            "-"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "nodes"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            ":"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "flows"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            "/"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "responses"
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            class: "example",
                                            "- left right: input / reply"
                                        },
                                    }
                                    p {
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "container"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            "["
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "components"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            "]"
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            class: "example",
                                            "plane [ pilot navigator ]"
                                        },
                                    }
                                    p {
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "abbreviation"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            ":"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "..."
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            class: "example",
                                            "c: controller; p: process; c p: setpoint / feedback"
                                        },
                                    }
                                    p {
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "nodes"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            ":"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "long-label"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            ","
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "labels"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            "/"
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "long-label"
                                        }
                                        " "
                                        span {
                                            class: "keyword",
                                            ","
                                        }
                                        " "
                                        span {
                                            style: "font-style: italic; background-color: rgba(156, 163, 175, 0.2);",
                                            "labels"
                                        }
                                        " "
                                    }
                                    p {
                                        style: "text-align: right;",
                                        span {
                                            class: "example",
                                            "controller process: a long action, / a long feedback, another feedback"
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            })
        }
    }


    #[cfg(test)]
    mod tests {
        use std::fmt::Display;

        use crate::graph_drawing::{error::Error, frontend::dom::{Node, Label}};

        use super::dom::Drawing;

        trait Check {
            fn check(&self, drawing: &Result<Drawing, Error>);
        }

        struct NoCollisions {}


        #[derive(Debug)]
        struct Rect { id: String, l: f64, r: f64, t: f64, b: f64 }

        impl Display for Rect {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Rect {{ l: {:7.2}, r: {:7.2}, t: {:7.2}, b: {:7.2}, id: {:?} }}", self.l, self.r, self.t, self.b, self.id)
            }
        }

        fn collides(a: &Rect, b: &Rect) -> bool {
            a.r > b.l &&
            a.l < b.r &&
            a.b > b.t &&
            a.t < b.b
        }

        impl Check for NoCollisions {
            fn check(&self, drawing: &Result<Drawing, Error>) {
                let Drawing{nodes, ..} = drawing.as_ref().unwrap();
                let rects = nodes
                    .iter()
                    .flat_map(|n| {
                        match n {
                            Node::Div { key, label, hpos, vpos, width, height, z_index, loc, estimated_size } => {
                                vec![Rect { id: key.clone(), l: *hpos, r: hpos + width, t: *vpos, b: vpos + height }]
                            },
                            Node::Svg { key, path, z_index, dir, rel, label, hops, classes, estimated_size, control_points } => {
                                let mut res = vec![];
                                if let Some(Label{text, hpos, width, vpos}) = label {
                                    if rel == "forward" {
                                        res.push(Rect { id: key.clone(), l: *hpos - width, r: *hpos, t: *vpos, b: vpos + estimated_size.height });
                                    } else {
                                        res.push(Rect { id: key.clone(), l: *hpos, r: hpos + width, t: *vpos, b: vpos + estimated_size.height });
                                    }
                                }
                                for (n, window) in control_points.windows(2).enumerate() {
                                    let [cur, nxt, ..] = window else { continue };
                                    res.push(Rect {
                                        id: format!("{key},{n}"),
                                        l: f64::min(cur.0, nxt.0),
                                        r: f64::max(cur.0, nxt.0),
                                        t: f64::min(cur.1, nxt.1),
                                        b: f64::max(cur.1, nxt.1),
                                    })
                                }
                                res
                            },
                        }
                    })
                    .collect::<Vec<_>>();
                let mut collisions = false;
                for i in 0..rects.len() {
                    for j in i+1..rects.len() {
                        let ri = &rects[i];
                        let rj = &rects[j];
                        let collides = collides(ri, rj);
                        if collides {
                            eprintln!("COLLISION: {i}, {j}, \n{ri}, \n{rj}\n");
                        }
                        collisions |= collides;
                    }
                }
                assert!(!collisions);
            }
        }

        fn check(model: &str, checks: Vec<&dyn Check>) {
            let drawing = super::dom::draw(model.into());
            for check in checks {
                check.check(&drawing);
            }
        }

        #[test]
        pub fn test_long_hop() {
            check("a b c; a c", vec![&NoCollisions{}]);
        }

        #[test]
        pub fn test_simple_labels() {
            check("a b: c / d", vec![&NoCollisions{}]);
        }

        #[test]
        pub fn test_microwave() {
            check("person microwave food: open start stop / beep : heat; person food: eat",
                vec![] //vec![&NoCollisions{}]
            );
        }

        #[test]
        pub fn test_simple_containment() {
            check("a [ b ]", vec![]);
        }

        #[test]
        pub fn test_container_vertical_hop() {
            check("a [ b c ]", vec![]);
        }

        #[test]
        pub fn test_objc() {
            check(r#"
                devs status-type: add property,
                devs services: add service, plumb data
                services status-type: read property, check protocol conformance, check selector responsiveness
                "#,
                vec![
                    // &NoCollisions{}
                ]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{error::Error};
    use crate::{parser::{Parser, Token, Item}, graph_drawing::{layout::{*}, index::{VerticalRank, OriginalHorizontalRank}, geometry::calculate_sols, error::Kind, eval, frontend::log::Logger}};

    use logos::Logos;

    #[test]
    #[allow(clippy::unwrap_used)]
    pub fn no_swaps() -> Result<(), Error> {
        let mut logs = Logger::new();
        let data = "Aa Ab Ac: y / z\nXx Xy Xz: w / x";
        let mut p = Parser::new();
        let mut lex = Token::lexer(data);
        while let Some(tk) = lex.next() {
            p.parse(tk)
                .map_err(|_| {
                    Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
                })?
        }

        let v: Vec<Item> = p.end_of_input()
            .map_err(|_| {
                Kind::PomeloError{span: lex.span(), text: lex.slice().into()}
            })?;

        let val = eval::eval(&v[..]);

        let vcg = calculate_vcg(&val, &mut logs)?;

        let Vcg{vert, vert_vxmap, containers, nodes_by_container_transitive: nodes_by_container, container_depths, ..} = &vcg;
        let vx = vert_vxmap["Ab"];
        let wx = vert_vxmap["Ac"];
        assert_eq!(vert.node_weight(vx), Some(&Cow::from("Ab")));
        assert_eq!(vert.node_weight(wx), Some(&Cow::from("Ac")));

        let distance = {
            let containers = &containers;
            let nodes_by_container = &nodes_by_container;
            let container_depths = &container_depths;
            |src: Cow<str>, dst: Cow<str>, logs: &mut Logger| {
                if !containers.contains(&src) {
                    -1
                } else {
                    if nodes_by_container[&src].contains(&dst) {
                        0
                    } else {
                        -(container_depths[&src] as isize)
                    }
                }
            }
        };
        let paths_by_rank = rank(&vcg.vert, distance, &mut logs)?;
        assert_eq!(paths_by_rank[&VerticalRank(2)][0], (Cow::from("Aa"), Cow::from("Ac")));

        let layout_problem = calculate_locs_and_hops(&val, &paths_by_rank, &vcg, &mut logs)?;
        let LayoutProblem{hops_by_level, loc_to_node, node_to_loc, ..} = &layout_problem;
        let nAa = Obj::Node(ObjNode{ vl: Cow::from("Aa") });
        let nAb = Obj::Node(ObjNode{ vl: Cow::from("Ab") });
        let nAc = Obj::Node(ObjNode{ vl: Cow::from("Ac") });
        let nXx = Obj::Node(ObjNode{ vl: Cow::from("Xx") });
        let nXy = Obj::Node(ObjNode{ vl: Cow::from("Xy") });
        let nXz = Obj::Node(ObjNode{ vl: Cow::from("Xz") });
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


        assert_eq!(hops_by_level.len(), 2);
        let h0 = &hops_by_level[&VerticalRank(0)];
        let h1 = &hops_by_level[&VerticalRank(1)];
        let ohr = OriginalHorizontalRank;
        let vr = VerticalRank;
        let h0A: Hop<Cow<str>> = Hop { mhr: lAa.1, nhr: lAb.1, vl: "Aa".into(), wl: "Ab".into(), lvl: lAa.0 };
        let h0X: Hop<Cow<str>> = Hop { mhr: lXx.1, nhr: lXy.1, vl: "Xx".into(), wl: "Xy".into(), lvl: lXx.0 };
        let h1A: Hop<Cow<str>> = Hop { mhr: lAb.1, nhr: lAc.1, vl: "Ab".into(), wl: "Ac".into(), lvl: lAb.0 };
        let h1X: Hop<Cow<str>> = Hop { mhr: lXy.1, nhr: lXz.1, vl: "Xy".into(), wl: "Xz".into(), lvl: lXy.0 };
        let mut s0 = vec![h0A.clone(), h0X.clone()];
        let mut s1 = vec![h1A.clone(), h1X.clone()];
        s0.sort();
        s1.sort();
        let mut h0 = h0.iter().cloned().collect::<Vec<_>>();
        h0.sort();
        let mut h1 = h1.iter().cloned().collect::<Vec<_>>();
        h1.sort();
        assert_eq!(&h0[..], &s0[..]);
        assert_eq!(&h1[..], &s1[..]);

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
        // ...

        Ok(())
    }
}
