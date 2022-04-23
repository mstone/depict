pub mod parser {
    // use crate::data::*;
    use nom::{IResult, InputLength, InputIter, InputTake, InputTakeAtPosition, FindToken, Slice};
    use nom::error::{VerboseError, context};
    use nom::bytes::complete::{take_while, take_while1, is_not, is_a};
    // use nom::character::{is_space};
    use nom::character::complete::{char};
    use nom::combinator::{map, opt};
    use nom::multi::{many1, separated_list0};
    use nom::sequence::{preceded, terminated, tuple, separated_pair};
    use std::hash::Hash;
    use std::ops::RangeFrom;

    use pomelo::pomelo;

    pub type Model<'s> = Vec<Item<'s>>;


    #[derive(Debug)]
    pub enum Item<'s> {
        Path(Vec<&'s str>, Vec<Item<'s>>),
        Body(Vec<Body<'s>>)
    }

    #[derive(Debug)]
    pub enum Body<'s> {
        Item(Vec<Item<'s>>),
        Sq(Item<'s>),
        Br(Item<'s>),
        Slash(Item<'s>, Item<'s>),
    }
    
    pub fn merge_colon<'s>(i: Item<'s>, j: Item<'s>) -> Item<'s> {
        eprint!("MERGE COLON {i:?} {j:?}");
        let r = match i { 
            Item::Path(p, mut ps) => {
                ps.push(j);
                Item::Path(p, ps)
            },
            Item::Body(mut i2) => {
                match j {
                    Item::Body(mut js) => {
                        i2.append(&mut js);
                    },
                    _ => {
                        i2.push(Body::Item(vec![j]));
                    }
                };
                Item::Body(i2)
            }
        };
        eprintln!(" -> {r:?}");
        r
    }

    pub fn merge_body<'s>(i: Item<'s>, b: Body<'s>) -> Item<'s> {
        eprint!("MERGE BODY {i:?} {b:?}");
        let r = match i {
            Item::Path(p, mut ps) => {
                if ps.is_empty() {
                    ps.push(Item::Body(vec![b]));
                } else {
                    let last = ps.pop().unwrap();
                    eprintln!(" -> ");
                    let last = merge_body(last, b);
                    ps.push(last);
                }
                Item::Path(p, ps)
            },
            Item::Body(mut ibs) => {
                ibs.push(b);
                Item::Body(ibs)
            },
        };
        eprintln!(" -> {r:?}");
        r
    }

    pub fn merge_slash<'s>(i: Item<'s>, j: Item<'s>) -> Item<'s> {
        Item::Body(vec![
            Body::Slash(i, j)
        ])
    }

    pub fn merge_text<'s>(i: Item<'s>, t: &'s str) -> Item<'s> {
        eprint!("MERGE TEXT {i:?} {t:?}");
        let r = match i {
            Item::Path(mut p, mut ps) => {
                if ps.is_empty() {
                    p.push(t);
                    Item::Path(p, vec![])
                } else if p.is_empty() {
                    ps.push(Item::Path(vec![t], vec![]));
                    Item::Path(p, ps)
                } else {
                    Item::Path(vec![], vec![Item::Path(p, ps), Item::Path(vec![t], vec![])])
                }
            },
            Item::Body(_) => todo!(),
        };
        eprintln!(" -> {r:?}");
        r
    }
    // %type #[regex(r#"\p{Pattern_Syntax}+"#)] Punctuation;
    // %type #[token(r#"\p{XID_Start}\p{XID_Continue}*"#)] Ident;
    
    pomelo! {
        %module fact;
        %include {
            use super::{Model, Item, Body, merge_body, merge_colon, merge_slash, merge_text};
            use logos::{Logos};

            // fn parse_str(lex: &Lexer<Token>) -> &str {
            //     lex.slice()
            // }
        }
        %token #[derive(Debug, Logos)] pub enum Token<'s> {};
        %type #[error] #[regex(r#"[\p{Pattern_White_Space}&&[^\n\r]]+"#, logos::skip)] Error;
        %type #[regex(r#"[\n\r]+"#)] Nl;
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
        %type #[regex(r#"[\p{XID_Start}$][\p{XID_Continue}.-]*"#)] Text &'s str;
        %type start Model<'s>;
        %type model Vec<Item<'s>>;
        %type item Item<'s>;
        %type body Body<'s>;
        %right Colon;
        %right Lsq;
        %right Lbr;
        %right Slash;
        %right Comma;
        // %left Semi;
        %right Text;
        // %verbose;

        start ::= model;

        model ::= model(mut m) Nl item?(i) { if let Some(i) = i { m.push(i); }; m };
        model ::= item(i) { vec![i] };

        item ::= item(i) body(b) [Colon] { merge_body(i, b) };
        item ::= item(i) Colon item(j) { merge_colon(i, j) };
        item ::= item(i) Slash item(j) { merge_slash(i, j) };
        item ::= item(i) Comma Text(t) { merge_text(i, t) };
        item ::= item(i) Text(t) { merge_text(i, t) };
        item ::= Text(t) { Item::Path(vec![t], vec![]) };
        item ::= body(b) [Colon] { Item::Body(vec![b]) };

        body ::= Lsq item(i) Rsq { Body::Sq(i) };
        body ::= Lbr item(i) Rbr { Body::Br(i) };
    }

    pub use fact::Parser;
    pub use fact::Token;


    pub type Labels<I> = Vec<Option<I>>;

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct Fact<I> {
        pub path: Vec<I>,
        pub labels_by_level: Vec<(Labels<I>, Labels<I>)>,
    }

    pub fn is_ws<'s, I>(chr: char) -> bool where 
        I: Clone + InputLength + InputIter<Item=char> + InputTake + InputTakeAtPosition<Item=char>,
        &'s str: nom::FindToken<char> 
    {
        chr == ' '
    }

    pub fn is_wst(chr: char) -> bool {
        matches!(chr, ' ' | '\n' | '\r' | '.')
    }

    pub fn nl<'s, I, Item>(s: I) -> IResult<I, I, VerboseError<I>> where 
        I: Clone + InputLength + InputIter<Item=Item> + InputTake + InputTakeAtPosition<Item=Item>,
        &'s str: nom::FindToken<<I as nom::InputIter>::Item> 
    {
        is_a("\r\n.")(s)
    }

    pub fn ws<'s, I>(s: I) -> IResult<I, I, VerboseError<I>> where 
        I: Clone + InputLength + InputIter<Item=char> + InputTake + InputTakeAtPosition<Item=char>,
        &'s str: FindToken<char>
    {
        take_while(is_ws::<'s, I>)(s)
    }

    pub fn ws1<'s, I>(s: I) -> IResult<I, I, VerboseError<I>> where
        I: Clone + InputLength + InputIter<Item=char> + InputTake + InputTakeAtPosition<Item=char>,
        &'s str: FindToken<char>
    {
        take_while1(is_ws::<'s, I>)(s)
    }

    pub fn normal<'s, I>(s: I) -> IResult<I, I, VerboseError<I>> where
        I: Clone + InputLength + InputIter<Item=char> + InputTake + InputTakeAtPosition<Item=char>,
        &'s str: FindToken<char>
    {
        is_not(" \n\r:.,/")(s)
    }

    pub fn fact<I>(s: I) -> IResult<I, Fact<I>, VerboseError<I>> where
        I: Clone + InputLength + InputIter<Item=char> + InputTake + InputTakeAtPosition<Item=char> + Slice<RangeFrom<usize>>
    {
        map(
            context(
                "fact",
                tuple((
                    many1(preceded(ws, normal)),
                    opt(preceded(ws, char(':'))),
                    map(opt(separated_list0(
                        preceded(ws, char(':')),
                            separated_pair(
                                separated_list0(
                                    preceded(ws, char(',')),
                                    preceded(ws, opt(is_not("\n\r:.,/"))),
                                ),
                                preceded(ws, opt(char('/'))),
                                separated_list0(
                                    preceded(ws, char(',')),
                                    preceded(ws, opt(is_not("\n\r:.,/"))),
                                ),
                            ),
                    )), |x| x.unwrap_or_default())
                ))
            ),
            |(path, _, labels_by_level)| Fact{path, labels_by_level}
        )(s)
    }

    pub fn parse<'s, I>(s: I) -> IResult<I, Vec<Fact<I>>, VerboseError<I>> where
        I: 's + Clone + InputLength + InputIter<Item=char> + InputTake + InputTakeAtPosition<Item=char> + Slice<RangeFrom<usize>>
    {
        terminated(many1(
            preceded(take_while(is_wst), fact),
        ), take_while(is_wst))(s)
    }

    #[cfg(test)]
    mod tests {
        use nom::error::{convert_error};

        #[test]
        fn fact_works() {
            let s = "hello: bar / baz ";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Fact{path: vec!["hello"], labels_by_level: vec![(vec![Some("bar ")], vec![Some("baz ")])]}
            ])));
        }

        #[test]
        fn action_works() {
            let s = "hello: bar /  ";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Fact{path: vec!["hello"], labels_by_level: vec![(vec![Some("bar ")], vec![None])]}
            ])));
        }

        #[test]
        fn percept_works() {
            let s = "hello:  / baz ";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Fact{path: vec!["hello"], labels_by_level: vec![(vec![None], vec![Some("baz ")])]}
            ])));
        }

        #[test]
        fn multiple_levels_works() {
            let s = "hello: bar / baz : foo / quux";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Fact{path: vec!["hello"], labels_by_level: vec![(vec![Some("bar ")], vec![Some("baz ")]), (vec![Some("foo ")], vec![Some("quux")])]}
            ])));
        }

        #[test]
        fn multiple_labels_works() {
            let s = "hello: bar, foo / baz, quux";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Fact{path: vec!["hello"], labels_by_level: vec![(vec![Some("bar"), Some("foo ")], vec![Some("baz"), Some("quux")])]}
            ])));
        }
    }
}

#[cfg(all(feature="minion", feature="cvxpy"))]
pub mod graph_drawing;

#[cfg(any(feature="client", feature="server"))]
pub mod rest {
    use serde::{Deserialize, Serialize};
    use petgraph::Graph;

    #[derive(Clone, Debug, PartialEq, PartialOrd, Deserialize, Serialize)]
    pub struct Label {
        pub text: String,
        pub hpos: f64,
        pub width: f64,
        pub vpos: f64,
    }

    #[derive(Clone, Debug, PartialEq, PartialOrd, Deserialize, Serialize)]
    pub enum Node {
    Div { key: String, label: String, hpos: f64, vpos: f64, width: f64 },
    Svg { key: String, path: String, rel: String, label: Option<Label> },
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Drawing {
        pub crossing_number: Option<usize>,
        pub viewbox_width: f64,
        pub layout_debug: Graph<String, String>,
        pub nodes: Vec<Node>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Draw {
        pub text: String
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct DrawResp {
        pub drawing: Drawing
    }

}