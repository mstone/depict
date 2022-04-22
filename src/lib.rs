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
    pub struct Item<'s>(pub Vec<&'s str>, pub Option<Box<Body<'s>>>);

    #[derive(Debug)]
    pub enum Body<'s> {
        Colon(Item<'s>),
        Sq(Item<'s>),
        Br(Item<'s>),
        Slash(Item<'s>, Item<'s>),
    }
    

    // %type #[regex(r#"\p{Pattern_Syntax}+"#)] Punctuation;
    // %type #[token(r#"\p{XID_Start}\p{XID_Continue}*"#)] Ident;
    
    pomelo! {
        %module fact;
        %include {
            use super::{Model, Item, Body};
            use logos::{Logos, Lexer};

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

        // item ::= item(mut i) item(b) Slash item(j) { i.1 = Some(Box::new(Body::Slash(b, j))); i };
        item ::= item(mut i) body(b) [Colon] { i.1 = Some(Box::new(b)); i };
        // item ::= item(mut i) Slash item(j) { i.1 = Some(Box::new(Body::Slash(j))); i };
        item ::= item(mut i) Comma Text(t) { i.0.push(t); i };
        item ::= item(mut i) Text(t) { i.0.push(t); i };
        item ::= Text(t) { Item(vec![t], None) };
        // item ::= body(b) Slash body(j) { Item(vec![], Some(Box::new(Body::Slash(b, j)))) };
        item ::= body(b) [Colon] { Item(vec![], Some(Box::new(b))) };
        
        body ::= Colon item(i) { Body::Colon(i) };
        body ::= Lsq item(i) Rsq { Body::Sq(i) };
        body ::= Lbr item(i) Rbr { Body::Br(i) };
        body ::= item(i) Slash item(j) { Body::Slash(i, j) };
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