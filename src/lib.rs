
//! depict is library for automatically drawing beautiful, readable pictures of 
//! models of systems, processes, and concepts of operation (ConOps).
//! 
//! # Summary
//! 
//! dpict may be best understood as a compiler from a textual language of 
//! "depict-expressions" ("depictions") to "pictures". It is implemented as a 
//! library for easy use by downstream packages like [depict_desktop], [depict_web], 
//! [depict_server], [depict_tikz], and [depict_parse].
//! 
//! [depict_desktop]: ../depict_desktop/index.html
//! [depict_parse]: ../depict_parse/index.html
//! [depict_server]: ../depict_server/index.html
//! [depict_tikz]: ../depict_tikz/index.html
//! [depict_web]: ../depict_web/index.html
pub mod printer {
    //! A pretty-printer for "depiction" parse trees
    //! 
    //! (The main purpose of the pretty-printer is to help test the 
    //! [parser](super::parser) via [proptest].)
    use std::borrow::Cow;

    use itertools::Itertools;

    use super::parser::{Item};

    pub fn print(model: &[Item]) -> String {
        model.iter().map(print1).join("\n")
    }

    pub fn print1(i: &Item) -> String {
        let mut v = Vec::new();
        match i {
            Item::Text(s) => v.push(s.clone()),
            Item::Seq(s) => v.extend(s.iter().map(|i| Cow::from(print1(i)))),
            Item::Comma(s) => {
                v.extend(itertools::intersperse(s.iter().map(|i| Cow::from(print1(i))), Cow::from(",")));
                if s.len() <= 1 {
                    v.push(Cow::from(","));
                }
            },
            Item::Colon(l, r) => {
                v.extend(l.iter().map(|i| Cow::from(print1(i))));
                v.push(Cow::from(":"));
                v.extend(r.iter().map(|i| Cow::from(print1(i))));
            },
            Item::Slash(l, r) => {
                v.extend(l.iter().map(|i| Cow::from(print1(i))));
                v.push(Cow::from("/"));
                v.extend(r.iter().map(|i| Cow::from(print1(i))));
            },
            Item::Sq(s) => {
                v.push(Cow::from("["));
                v.extend(s.iter().map(|i| Cow::from(print1(i))));
                v.push(Cow::from("]"));
            },
            Item::Br(s) => {
                v.push(Cow::from("{"));
                v.extend(s.iter().map(|i| Cow::from(print1(i))));
                v.push(Cow::from("}"));
            },
        }
        v.join(" ")
    }

    #[cfg(test)]
    mod test {
        use std::borrow::Cow;

        use proptest::prelude::*;
        use logos::Logos;
        use crate::parser::{Item, Parser, Token};

        fn deseq(v: Vec<Item>) -> Vec<Item> {
            if let [Item::Seq(v)] = &v[..] { 
                v.clone()
            } else { 
                v
            }
        }
        
        /// Generate an arbitrary [Item]
        /// 
        /// (Note: one challenge in this area is that in normal use, [Item] has 
        /// associativity and precedence invariants enforced by [Parser] and, 
        /// as a consequence, "arbitary" items need to be carefully constructed 
        /// to enforce these invariants.)
        fn arb_item() -> impl Strategy<Value = Item<'static>> {
            let leaf = "[a-z]+".prop_map(|s| Item::Text(Cow::from(s)));
            let leaf2 = leaf.clone().prop_recursive(1, 4, 3, |inner| {
                prop::collection::vec(inner.clone(), 2..3).prop_map(Item::Seq)
            });
            let leaf3 = prop_oneof![
                leaf.clone(),
                leaf2.clone(),
                prop::collection::vec(leaf.clone(), 0..3).prop_map(Item::Comma),
            ];
            leaf3.prop_recursive(
                1, 4, 3, |inner| {
                    prop_oneof![
                        // note: this pair of cases, plus the depth limitation above, 
                        // is a crude work-around for needing to generate only right-
                        // associated trees of colons
                        (inner.clone(), inner.clone()).prop_map(|(i, j)| Item::Colon(deseq(vec![i]), deseq(vec![j]))),
                        (inner.clone(), inner.clone(), inner.clone()).prop_map(|(i, j, k)| 
                            Item::Colon(
                                deseq(vec![i]), 
                                vec![Item::Colon(deseq(vec![j]), deseq(vec![k]))]
                            )),

                        (inner.clone(), inner.clone()).prop_map(|(i, j)| Item::Slash(deseq(vec![i]), deseq(vec![j]))),

                        inner.clone().prop_map(|i| Item::Br(vec![i])),

                        inner.clone().prop_map(|i| Item::Sq(vec![i])),
                    ]
                }
            )
        }

        proptest! {
            #[test]
            fn doesnt_crash(s in "\\PC*") {
                let mut lex = Token::lexer(&s);
                let mut p = Parser::new();

                for tk in lex.by_ref() {
                    if p.parse(tk).is_err() {
                        return Ok(());
                    }
                }
                let v = p.end_of_input();
                if v.is_err() {
                    return Ok(());
                }
            }

            #[test]
            fn has_partial_inverse(i in arb_item()) {
                let s = super::print1(&i);
                let mut lex = Token::lexer(&s);
                let mut p = Parser::new();

                for tk in lex.by_ref() {
                    p.parse(tk).unwrap();
                }
                let v = p.end_of_input().unwrap();
                assert!(i == v[0] || i == (if let Item::Seq(v) = &v[0] { v[0].clone() } else { v[0].clone() }), "\n\ni: {i:#?}\ns: {s:?}\no: {v:#?}\n\n");
            }
        }
    }
}

pub mod parser {
    //! The parser for "depictions"
    //! 
    //! # Summary
    //! 
    //! The language of depictions loosely consists of:
    //! 
    //! * definitions ::= *name* **:** *expr*,
    //! * relations ::=  *name* *name* ... (**:** *labels* (**/** */ *labels*)?)*
    //! * labels ::= *label*... for single-word labels or *label* (**,** *label*)* for multi-word labels
    //! * nesting ::= **[** *model* **]**
    //! * alternatives ::= **{** *model* **}**
    //! 
    //! # Links
    //! 
    //! [Model] and [Item] values can be pretty-printed by [`print()`](crate::printer::print) and [`print1()`](crate::printer::print1), respectively.
    //! 
    use enum_kinds::EnumKind;
    use std::borrow::Cow;
    use std::hash::Hash;

    use pomelo::pomelo;

    /// Depictions consist of [Item]s.
    pub type Model<'s> = Vec<Item<'s>>;

    /// Items are the main "expression" type of depictions.
    #[derive(Clone, Debug, EnumKind, PartialEq)]
    #[enum_kind(ItemKind)]
    pub enum Item<'s> {
        Text(Cow<'s, str>),
        Seq(Vec<Item<'s>>),
        Comma(Vec<Item<'s>>),
        Colon(Vec<Item<'s>>, Vec<Item<'s>>),
        Slash(Vec<Item<'s>>, Vec<Item<'s>>),
        Sq(Vec<Item<'s>>),
        Br(Vec<Item<'s>>),
    }

    // A couple of relations guide the merging process.
    // 
    // Textual items are Texts, Tildes, and Seqs.
    // Binary items are Colons and Slashes.
    // Unary items are Sq and Br.
    // 
    // First, eats: a < b or a > b for "a eats b" or "b eats a".
    // 
    // In general, binary items eat textual items.
    // Colons eat Slashes on their right.
    // Colons eat textual items to their left.
    // While a right colon is eating the *hs of a seq, 
    // the right colon will eat the right-most textual elements, 
    // but the seq will eat the colon thereafter unless it has been emptied.
    // While a right colon is eating the rhs of a left-colon, 
    // the right colon will eat the right-most textual elements, 
    // but if another left colon appears, recurse.
    // Slashes eat textual items on their left, and steal them from colon-rights.
    // Colons do not eat commas on their right; Commas make sequences.
    // Otherwise, when a left colon is eating an item to the right, the eating is delegated
    // to the colon's rhs' end if any, or to the colon's rhs otherwise
    // When fully eaten, Seqs dissolve.
    // Non-seq textual items are conceptually wrapped in a Seq for eating purposes.
    
    pub fn seq<'s>(i: Item<'s>, j: Item<'s>) -> Item<'s> {
        Item::Seq(vec![i, j])
    }

    impl<'s> Item<'s> {
        /// Fold the item on the right (`self` or `j`) with an item on the left (`i`).
        // when eating a left item, eat as much as you can.
        // if you ate the whole item, then only you remain.
        // otherwise, what's left of the item eats you.
        fn eat_left(mut self, mut i: Item<'s>) -> Self {
            if matches!(i, Item::Text(..)  | Item::Sq(..) | Item::Br(..)) {
                i = Item::Seq(vec![i]);
            }
            if matches!(i, Item::Comma(..)) && matches!(self, Item::Slash(..) | Item::Colon(..)) {
                self.left().insert(0, i);
                return self;
            }
            let ikind = ItemKind::from(&i);
            let jkind = ItemKind::from(&self);
            let mut comma_buffer = vec![];
            use ItemKind::*;
            while !i.right().is_empty() {
                let mut end = i.right().pop().unwrap();
                let ekind = ItemKind::from(&end);
                match (ikind, ekind, jkind) {
                    (Seq  , Text  | Br | Sq        , Slash | Colon | Seq) | 
                    (Seq  , Slash | Colon          , Colon              ) |
                    (Colon, Text  | Br | Sq | Comma, Slash | Colon | Seq) => { 
                        // we eat end; i eats us.
                        self.left().insert(0, end);
                    },
                    (_, Comma, Comma) => {
                        end.right().append(self.left());
                        i.right().push(end);
                        return i;
                    },
                    (_, Text, Comma) => {
                        comma_buffer.push(end);
                    },
                    (Seq, _, Slash | Colon) => { 
                        // i eats us.
                        i.right().push(end);
                    },
                    (Seq, _, Seq) => {
                        // i eats us and we dissolve.
                        i.right().push(end);
                    },
                    (_, Colon, _) | (_, Slash, Comma) => {
                        let end = merge_item(end, self);
                        i.right().push(end);
                        return i
                    },
                    (_, Slash, Colon) => {
                        self.left().insert(0, end);
                    },
                    (_, Slash, _) => {
                        end.right().push(self);
                        i.right().push(end);
                        return i
                    },
                    _ => {
                        i.right().push(end);
                        break
                    }
                }
            }
            if jkind == Comma {
                match comma_buffer.len() {
                    0 => {},
                    1 => {
                        self.left().insert(0, comma_buffer.pop().unwrap());
                        if ikind == Seq && i.right().is_empty() {
                            return self;
                        }
                    },
                    _n => {
                        self.left().insert(0, Item::Seq(comma_buffer.into_iter().rev().collect::<Vec<_>>()));
                    },
                }
                i.right().push(self);
                i
            } else if jkind == Seq && ikind != Comma {
                i.right().append(self.left());
                i
            } else if ikind == Seq && i.right().is_empty() { 
                self
            } else {
                i.right().push(self);
                i
            }
        }

        /// Get a &mut reference to `self`'s right-most sequence if one exists, or panic.
        fn right(&mut self) -> &mut Vec<Item<'s>> {
            match self {
                Item::Seq(ref mut r) => r,
                Item::Comma(ref mut r) => r,
                Item::Colon(_, ref mut r) => r,
                Item::Slash(_, ref mut r) => r,
                _ => unreachable!(),
            }
        }

        /// Get a &mut reference to `self`'s left-most sequence if one exists, or panic.
        fn left(&mut self) -> &mut Vec<Item<'s>> {
            match self {
                Item::Seq(ref mut l) => l,
                Item::Comma(ref mut l) => l,
                Item::Colon(ref mut l, _) => l,
                Item::Slash(ref mut l, _) => l,
                _ => unreachable!(),
            }
        }
    }

    /// Combine the two right-most items.
    fn merge_item<'s>(i: Item<'s>, j: Item<'s>) -> Item<'s> {
        eprint!("MERGE {i:?} {j:?}");
        let r = j.eat_left(i);
        eprintln!(" -> {r:?}");
        r
    }
    
    pomelo! {
        %module fact;
        // %parser #[derive(Clone)] pub struct Parser<'s> {};
        // %stack_type 
        %include {
            use std::borrow::Cow;
            use super::{Model, Item, merge_item};
            use logos::{Logos};
        }
        %token #[derive(Copy, Clone, Debug, Logos)] pub enum Token<'s> {};
        %type #[error] #[regex(r#"[\p{Pattern_White_Space}&&[^\r\n]]+"#, logos::skip)] Error;
        %type #[token("{")] Lbr;
        %type #[token("}")] Rbr;
        %type #[token("[")] Lsq;
        %type #[token("]")] Rsq;
        // %type #[token(";")] Semi;
        %type #[token(",")] Comma;
        %type #[token(":")] Colon;
        // %type #[token("~")] Tilde;
        %type #[token("/")] Slash;
        // %type #[token("|")] Pipe;
        // %type #[token("-")] Dash;
        %type #[token("!")] Bang;
        %type #[regex("[\r\n]+")] Nl;
        %type #[regex(r#"[\p{XID_Start}$<()][\p{XID_Continue}().\->&&[^:/]]*(\\/[\p{XID_Continue}().\->&&[^:/]]*)*"#)] Text &'s str;
        %type start Model<'s>;
        %type model Vec<Item<'s>>;
        %type item Item<'s>;
        %type expr1 Item<'s>;
        %type expr2 Item<'s>;
        %type expr2a Item<'s>;
        %type expr3 Item<'s>;
        %type expr4 Item<'s>;
        %type expr5 Item<'s>;
        %right Bang;
        %left Nl;
        %right Slash;
        %right Colon;
        %left Dash;
        %right Comma;
        %right Lsq Lbr;
        %right Rsq Rbr;
        %left Semi;
        %right Text;
        // %left Tilde;
        // %verbose;
        // %trace;

        start ::= model;
        model ::= model?(i) Nl expr1?(j) { 
            let mut i = i.unwrap_or_default();
            if let Some(j) = j { i.push(j) }; 
            i 
        };
        model ::= expr1(j) { vec![j] };

        expr1 ::= Lsq model(j) Rsq { Item::Sq(j) };
        expr1 ::= Lbr model(j) Rbr { Item::Br(j) };
        expr1 ::= expr3(i) [Bang] { i };

        expr3 ::= Text(t) { Item::Text(Cow::Borrowed(t)) };
        expr3 ::= Slash { Item::Slash(vec![], vec![]) };
        expr3 ::= Colon { Item::Colon(vec![], vec![]) };
        expr3 ::= Comma { Item::Comma(vec![]) };
        expr3 ::= expr1(i) expr1(j) [Text] { merge_item(i, j) };
    }

    /// The [pomelo!]-generated depiction parser
    pub use fact::Parser;

    /// The [pomelo!]-generated depiction lexer.
    /// 
    /// To use, please bring the [Logos] trait into scope like so:
    /// ```ignore
    /// use logos::Logos;
    /// ```
    pub use fact::Token;


    pub type Labels<I> = Vec<Option<I>>;

    /// The intermediate representation (IR) of depictions
    /// 
    /// In depict, models are viewed as asserting a claimed set of "facts" 
    /// to be depicted.
    /// 
    /// These "facts" are represented by [Fact]s, each of which record a 
    /// claim like "the sequence `path` model entities are related with 
    /// labels for the forward and backward dimensions of each such 
    /// atomic relationship in the corresponding entries of `labels_by_level`."
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct Fact<I> {
        pub path: Vec<I>,
        pub labels_by_level: Vec<(Labels<I>, Labels<I>)>,
    }
}

#[cfg(all(feature="minion", feature="osqp"))]
pub mod graph_drawing;

#[cfg(any(feature="client", feature="server"))]
pub mod rest {
    //! Message types and codecs for client-server implementations of depict APIs
    use serde::{Deserialize, Serialize};
    use petgraph::Graph;

    /// Labels describe positioned boxes of text.
    #[derive(Clone, Debug, PartialEq, PartialOrd, Deserialize, Serialize)]
    pub struct Label {
        pub text: String,
        pub hpos: f64,
        pub width: f64,
        pub vpos: f64,
    }

    /// Positioned graphical elements, with unique keys.
    #[derive(Clone, Debug, PartialEq, PartialOrd, Deserialize, Serialize)]
    pub enum Node {
        /// Boxes
        Div { key: String, label: String, hpos: f64, vpos: f64, width: f64 },
        /// Arrows with optional textual labels
        Svg { key: String, path: String, rel: String, label: Option<Label> },
    }

    /// The data of a drawing of a "depiction".
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Drawing {
        pub crossing_number: Option<usize>,
        pub viewbox_width: f64,
        pub layout_debug: Graph<String, String>,
        pub nodes: Vec<Node>,
    }

    /// A drawing request containing a "depiction" to draw.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Draw {
        pub text: String
    }

    /// A drawing response containing a [Drawing].
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct DrawResp {
        pub drawing: Drawing
    }

}
