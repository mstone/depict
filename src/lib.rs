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


    #[derive(Clone, Debug)]
    pub enum Item<'s> {
        Text(&'s str),
        Tilde(),
        Seq(Vec<Item<'s>>),
        Colon(Vec<Item<'s>>),
        Dash(Vec<Item<'s>>),
        Slash(Vec<Item<'s>>, Vec<Item<'s>>),
        Sq(Vec<Item<'s>>),
        Br(Vec<Item<'s>>),
    }

    pub fn merge_item<'s>(i: Item<'s>, j: Item<'s>) -> Item<'s> {
        eprint!("MERGE {i:?} {j:?}");
        // the problem to solve is that 
        // a) slashes also need to eat their RHS.
        // b) slashes need to bind tighter than colons 
        //    despite the fact that colons come first.
        let r = match (i.clone(), j.clone()) {
            (Item::Seq(mut lhs), Item::Slash(mut sl, mut sr)) => {
                while lhs.len() > 0 {
                    let end = lhs.pop().unwrap();
                    if matches!(end, Item::Text(_)) {
                        sl.insert(0, end);
                    } else {
                        lhs.push(end);
                        break
                    }
                }
                lhs.push(Item::Slash(sl, sr));
                Some(Item::Seq(lhs))
            },
            (Item::Seq(mut lhs), j2) => {
                if lhs.len() > 0 {
                    let end = lhs.pop().unwrap();
                    if let Item::Slash(mut sl, mut sr) = end {
                        sr.push(j2);
                        lhs.push(Item::Slash(sl, sr));
                        Some(Item::Seq(lhs)) 
                    } else {
                        lhs.push(end);
                        lhs.push(j2);
                        Some(Item::Seq(lhs))
                    }
                } else {
                    lhs.push(j2);
                    Some(Item::Seq(lhs))
                }
            },
            _ => None,
        };
        let r = r.or_else(|| Some(match (i.clone(), j.clone()) {
            (_, Item::Text(_)       ) => Item::Seq(vec![i, j]),
            (_, Item::Tilde()       ) => Item::Seq(vec![i, j]),
            (_, Item::Seq(mut rhs)  ) => { rhs.insert(0, i); Item::Seq(rhs) },
            (_, Item::Colon(_)      ) => { Item::Seq(vec![i, j]) },
            (_, Item::Dash(mut rhs) ) => { rhs.insert(0, i); Item::Dash(rhs) },
            // (_, Item::Slash(mut rhs)) => { rhs.insert(0, i); Item::Slash(rhs) },
            (_, Item::Slash(_, _)         ) => { Item::Seq(vec![i, j]) },
            (_, Item::Sq(_)         ) => { Item::Seq(vec![i, j]) },
            (_, Item::Br(_)         ) => { Item::Seq(vec![i, j]) },   
    }));
        eprintln!(" -> {r:?}");
        r.unwrap()
    }
    
    pomelo! {
        %module fact;
        // %parser #[derive(Clone)] pub struct Parser<'s> {};
        // %stack_type 
        %include {
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
        %type #[token("~")] Tilde;
        %type #[token("/")] Slash;
        // %type #[token("|")] Pipe;
        %type #[token("-")] Dash;
        %type #[token("!")] Bang;
        %type #[regex("[\r\n]+")] Nl;
        %type #[regex(r#"[\p{XID_Start}$<][\p{XID_Continue}.\-\\&&[^:/]]*"#)] Text &'s str;
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
        %right Lsq Lbr;
        %right Rsq Rbr;
        %right Comma;
        %left Semi;
        %right Text;
        %left Tilde;
        // %verbose;
        // %trace;

        start ::= model;
        model ::= model(mut i) Nl expr1?(j) { if let Some(j) = j { i.push(j) }; i };
        model ::= expr1(j) { vec![j] };

        expr1 ::= expr1(j) Colon [Tilde] { if let Item::Seq(lhs) = j { Item::Colon(lhs) } else { Item::Colon(vec![j]) } };
        expr1 ::= expr1(j) Dash { Item::Dash(vec![j]) };
        expr1 ::= expr1(j) Comma { j };
        expr1 ::= Lsq expr1(j) Rsq { Item::Sq(vec![j])};
        expr1 ::= Lsq expr1(j) [Rsq] { Item::Sq(vec![j]) };
        expr1 ::= Lbr expr1(j) Rbr { Item::Br(vec![j])};
        expr1 ::= Lbr expr1(j) [Rbr] { Item::Br(vec![j]) };
        // expr1 ::= expr1(i) expr3(j) { merge_item(i, j) };
        // expr1 ::= expr2(i) { i };
        expr1 ::= expr3(i) [Bang] { i };
        // expr1 ::= expr1(i) expr1(j) [Tilde] { merge_item(i, j) };

        // expr2 ::= expr3(i) Slash expr2(j) { Item::Slash(vec![i, j]) };
        // expr2 ::= expr3(i) [Bang] { i };

        expr3 ::= Text(t) { Item::Text(t) };
        expr3 ::= Tilde { Item::Tilde() };
        expr3 ::= Slash { Item::Slash(vec![], vec![]) };
        expr3 ::= expr1(i) expr1(j) [Tilde] { merge_item(i, j) };
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