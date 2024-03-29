
//! depict is library for drawing beautiful, readable pictures of
//! models of systems, processes, and concepts of operation (ConOps).
//!
//! # Summary
//!
//! depict may be best understood as a compiler from a textual language of
//! "depict-expressions" ("depictions") to "pictures". It is implemented as a
//! library for easy use by downstream packages like depict's desktop and web
//! front-ends.
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
            Item::At(l, r) => {
                v.extend(l.iter().map(|i| Cow::from(print1(i))));
                v.push(Cow::from("@"));
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

                        (inner.clone(), inner.clone()).prop_map(|(i, j)| Item::At(deseq(vec![i]), deseq(vec![j]))),

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
    //! # Guide-level Explanation
    //!
    //! (tbd.)
    //!
    //! # Reference-level Explanation
    //!
    //! The parser for the loose grammar shown above is actually produced by the [pomelo]
    //! LALR(1) parser-generator proc-macro.
    //!
    //! The two key ideas of the LALR(1) grammar given to pomelo are
    //!
    //! 1. to build the parse tree by giving merge rules for how to combine adjacent
    //! parse-tree fragments to be driven by a single "juxtaposition rule" that,
    //! combined with precedence information, drives the merging process backward from
    //! right to left according to the shift-reduce conflict resolutions specified by
    //! the precedence rules
    //!
    //! 2. to check that the resulting parser produces desirable parse trees by
    //! demanding that it be a partial inverse to the [printer](crate::printer)
    //! pretty-printer.
    use enum_kinds::EnumKind;
    use std::borrow::Cow;
    use std::fmt::Display;
    use std::fmt::Formatter;

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
        At(Vec<Item<'s>>, Vec<Item<'s>>),
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
            if matches!(i, Item::Comma(..)) && matches!(self, Item::Slash(..) | Item::Colon(..) | Item::At(..)) {
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
                eprintln!("\nCASE {ikind:?} {ekind:?} {jkind:?}");
                match (ikind, ekind, jkind) {
                    (Seq  , Text  | Br | Sq            , Slash | Colon | At | Seq) |
                    (Seq  , Slash | Colon | Comma , Colon                   ) |
                    (Colon | At, Text  | Br | Sq | Comma, Slash | Colon | Seq    ) => {
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
                    (Colon, _, At) => {
                        i.right().push(end);
                        self.left().insert(0, i);
                        return self;
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
            } else if ikind == Colon && jkind == At {
                self.left().insert(0, i);
                self
            } else {
                i.right().push(self);
                i
            }
        }

        /// Get a &mut reference to `self`'s right-most sequence if one exists, or panic.
        pub(crate) fn right(&mut self) -> &mut Vec<Item<'s>> {
            match self {
                Item::Seq(ref mut r) => r,
                Item::Comma(ref mut r) => r,
                Item::Colon(_, ref mut r) => r,
                Item::Slash(_, ref mut r) => r,
                Item::At(_, ref mut r) => r,
                _ => unreachable!(),
            }
        }

        /// Get a &mut reference to `self`'s left-most sequence if one exists, or panic.
        pub(crate) fn left(&mut self) -> &mut Vec<Item<'s>> {
            match self {
                Item::Seq(ref mut l) => l,
                Item::Comma(ref mut l) => l,
                Item::Colon(ref mut l, _) => l,
                Item::Slash(ref mut l, _) => l,
                Item::At(ref mut l, _) => l,
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
        %type #[token("@")] At;
        %type #[token("!")] Bang;
        %type #[regex("[\r\n;]+")] Nl;
        %type #[regex(r#"[\p{XID_Start}$<>\-\*()_0-9][\p{XID_Continue}().\-\*_>&&[^:/@]]*(\\/[\p{XID_Continue}().\-\*_>&&[^:/@]]*)*"#)] Text &'s str;
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
        %right At;
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
        expr3 ::= At { Item::At(vec![], vec![])};
        expr3 ::= Comma { Item::Comma(vec![]) };
        expr3 ::= expr1(i) expr1(j) [Text] { merge_item(i, j) };
    }

    impl<'s> Display for Item<'s> {
        fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(fmt, "{}", crate::printer::print1(self))
        }
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

    pub mod visit {

        use crate::parser::{Item, Model};
        use std::borrow::Cow;

        pub trait Visit<'s, 't> {
            fn visit_model(&mut self, model: &'t Vec<Item<'s>>) {
                visit_model(self, model);
            }

            fn visit_items(&mut self, items: &'t Vec<Item<'s>>) {
                visit_items(self, items);
            }

            fn visit_item(&mut self, item: &'t Item<'s>) {
                visit_item(self, item);
            }

            fn visit_text(&mut self, text: &'t Cow<'s, str>) {
                visit_text(self, text);
            }

            fn visit_seq(&mut self, seq: &'t Vec<Item<'s>>) {
                visit_seq(self, seq);
            }

            fn visit_comma(&mut self, comma: &'t Vec<Item<'s>>) {
                visit_comma(self, comma);
            }

            fn visit_colon(&mut self, lhs: &'t Vec<Item<'s>>, rhs: &'t Vec<Item<'s>>) {
                visit_colon(self, lhs, rhs);
            }

            fn visit_colon_lhs(&mut self, lhs: &'t Vec<Item<'s>>) {
                visit_colon_lhs(self, lhs);
            }

            fn visit_colon_rhs(&mut self, rhs: &'t Vec<Item<'s>>) {
                visit_colon_rhs(self, rhs);
            }

            fn visit_slash(&mut self, lhs: &'t Vec<Item<'s>>, rhs: &'t Vec<Item<'s>>) {
                visit_slash(self, lhs, rhs);
            }

            fn visit_slash_lhs(&mut self, lhs: &'t Vec<Item<'s>>) {
                visit_slash_lhs(self, lhs);
            }

            fn visit_slash_rhs(&mut self, rhs: &'t Vec<Item<'s>>) {
                visit_slash_rhs(self, rhs);
            }

            fn visit_at(&mut self, lhs: &'t Vec<Item<'s>>, rhs: &'t Vec<Item<'s>>) {
                visit_at(self, lhs, rhs);
            }

            fn visit_at_lhs(&mut self, lhs: &'t Vec<Item<'s>>) {
                visit_at_lhs(self, lhs);
            }

            fn visit_at_rhs(&mut self, rhs: &'t Vec<Item<'s>>) {
                visit_at_rhs(self, rhs);
            }

            fn visit_sq(&mut self, sq: &'t Vec<Item<'s>>) {
                visit_sq(self, sq);
            }

            fn visit_br(&mut self, br: &'t Vec<Item<'s>>) {
                visit_br(self, br);
            }
        }

        pub fn visit_model<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, model: &'t Vec<Item<'s>>) {
            v.visit_items(model);
        }

        pub fn visit_items<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, items: &'t Vec<Item<'s>>) {
            for item in items {
                v.visit_item(item);
            }
        }

        pub fn visit_item<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, item: &'t Item<'s>) {
            match item {
                Item::Text(s) => v.visit_text(s),
                Item::Seq(s) => v.visit_seq(s),
                Item::Comma(s) => v.visit_comma(s),
                Item::Colon(lhs, rhs) => v.visit_colon(lhs, rhs),
                Item::Slash(lhs, rhs) => v.visit_slash(lhs, rhs),
                Item::At(lhs, rhs) => v.visit_at(lhs, rhs),
                Item::Sq(s) => v.visit_sq(s),
                Item::Br(s) => v.visit_br(s),
            }
        }

        pub fn visit_text<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, text: &'t Cow<'s, str>) {

        }

        pub fn visit_seq<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, seq: &'t Vec<Item<'s>>) {
            v.visit_items(seq);
        }

        pub fn visit_comma<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, comma: &'t Vec<Item<'s>>) {
            v.visit_items(comma);
        }

        pub fn visit_colon<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, lhs: &'t Vec<Item<'s>>, rhs: &'t Vec<Item<'s>>) {
            v.visit_colon_lhs(lhs);
            v.visit_colon_rhs(rhs);
        }

        pub fn visit_colon_lhs<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, lhs: &'t Vec<Item<'s>>) {
            v.visit_items(lhs);
        }

        pub fn visit_colon_rhs<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, rhs: &'t Vec<Item<'s>>) {
            v.visit_items(rhs);
        }

        pub fn visit_slash<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, lhs: &'t Vec<Item<'s>>, rhs: &'t Vec<Item<'s>>) {
            v.visit_slash_lhs(lhs);
            v.visit_slash_rhs(rhs);
        }

        pub fn visit_slash_lhs<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, lhs: &'t Vec<Item<'s>>) {
            v.visit_items(lhs);
        }

        pub fn visit_slash_rhs<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, rhs: &'t Vec<Item<'s>>) {
            v.visit_items(rhs);
        }

        pub fn visit_at<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, lhs: &'t Vec<Item<'s>>, rhs: &'t Vec<Item<'s>>) {
            v.visit_at_lhs(lhs);
            v.visit_at_rhs(rhs);
        }

        pub fn visit_at_lhs<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, lhs: &'t Vec<Item<'s>>) {
            v.visit_items(lhs);
        }

        pub fn visit_at_rhs<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, rhs: &'t Vec<Item<'s>>) {
            v.visit_items(rhs);
        }

        pub fn visit_sq<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, sq: &'t Vec<Item<'s>>) {
            v.visit_items(sq);
        }

        pub fn visit_br<'s, 't, V: Visit<'s, 't> + ?Sized>(v: &mut V, br: &'t Vec<Item<'s>>) {
            v.visit_items(br);
        }

    }

    #[cfg(test)]
    mod tests {
        use pretty_assertions::{assert_eq};
        use super::{Parser, Item, Token, Model};
        use std::fmt::Debug;
        use logos::Logos;
        use std::borrow::Cow;

        fn check(model: &str, goal: Model) {
            let mut p = Parser::new();
            let mut lex = Token::lexer(model);

            while let Some(tk) = lex.next() {
                p.parse(tk).unwrap();
            }

            let items = p.end_of_input().unwrap();

            assert_eq!(goal, items);
        }

        const a: &'static str = "a";
        const b: &'static str = "b";
        const c: &'static str = "c";
        const d: &'static str = "d";
        const e: &'static str = "e";
        fn t<'s>(x: &'static str) -> Item<'s> { Item::Text(Cow::from(x)) }
        fn vi<'s>(x: &[Item<'static>]) -> Vec<Item<'s>> { x.iter().cloned().collect::<Vec<_>>() }
        fn sq<'s>(x: &[Item<'static>]) -> Item<'s>{ Item::Sq(vi(x)) }
        fn br<'s>(x: &[Item<'static>]) -> Item<'s>{ Item::Br(vi(x)) }
        fn seq<'s>(x: &[Item<'static>]) -> Item<'s> { Item::Seq(vi(x)) }
        fn col<'s>(x: &[Item<'static>], y: &[Item<'static>]) -> Item<'s> { Item::Colon(vi(x), vi(y)) }
        fn at<'s>(x: &[Item<'static>], y: &[Item<'static>]) -> Item<'s> { Item::At(vi(x), vi(y)) }
        fn sl<'s>(x: &[Item<'static>], y: &[Item<'static>]) -> Item<'s> { Item::Slash(vi(x), vi(y)) }

        #[test]
        pub fn test_parse() {
            let tests: Vec<(&str, Model)> = vec![
                ("a : b @ c", vi(&[at(&[col(&[t(a)], &[t(b)])], &[t(c)])])),
                ("@: a", vi(&[at(&[], &[col(&[], &[t(a)])])])),
                ("@a: b", vi(&[at(&[], &[col(&[t(a)], &[t(b)])])])),
                ("a b: c / d @ e", vi(&[at(&[col(&[t(a), t(b)], &[sl(&[t(c)], &[t(d)])])], &[t(e)])])),
                ("@ { a }", vi(&[at(&[], &[br(&[t(a)])])])),
                ("a b: @ c", vi(&[at(&[col(&[t(a), t(b)], &[])], &[t(c)])]))
            ];
            for (prompt, goal) in tests {
                eprintln!("PROMPT: {prompt}. GOAL: {goal:?}");
                check(prompt, goal);
            }
        }
    }
}

#[cfg(any(feature="osqp", feature="osqp-rust"))]
pub mod graph_drawing;

#[cfg(any(feature="client", feature="server"))]
pub mod rest {
    //! Message types and codecs for client-server implementations of depict APIs
    use serde::{Deserialize, Serialize};

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
        Div {
            key: String,
            label: String,
            hpos: f64,
            vpos: f64,
            width: f64,
            height: f64,
            z_index: usize,
        },
        /// Arrows with optional textual labels
        Svg {
            key: String,
            path: String,
            z_index: usize,
            rel: String,
            label: Option<Label>,
        },
    }

    /// The data of a drawing of a "depiction".
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Drawing {
        pub crossing_number: Option<usize>,
        pub viewbox_width: f64,
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


pub mod licenses {
    use include_dir::{Dir, include_dir};
    pub const LICENSES: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/licenses");
}