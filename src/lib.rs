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
    pub struct Literal<'s> {
        pub label: Option<&'s str>,
        pub body: Option<Body<'s>>,
    }

    #[derive(Debug)]
    pub enum Item<'s> {
        Literal{ 
            literal: Literal<'s>,
        },
        Binding {
            binding: &'s str,
            expr: Box<Item<'s>>,
        },
        Relating {
            lhs: Vec<&'s str>,
            rhs: Vec<(Vec<&'s str>, Vec<&'s str>)>
        }
    }

    #[derive(Debug)]    
    pub enum Body<'s> { 
        And(Vec<Item<'s>>),
        Or(Vec<Item<'s>>),
    }

    // %type #[regex(r#"\p{Pattern_Syntax}+"#)] Punctuation;
    // %type #[token(r#"\p{XID_Start}\p{XID_Continue}*"#)] Ident;
    
    pomelo! {
        %module fact;
        %include {
            use super::{Model, Item, Body, Literal};
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
        %type #[token(";")] Semi;
        %type #[token(",")] Comma;
        %type #[token(":")] Colon;
        %type #[token("~")] Tilde;
        %type #[token("/")] Slash;
        %type #[token("|")] Pipe;
        %type #[regex(r#"\p{XID_Start}\p{XID_Continue}*"#)] Text &'s str;
        %type binding Item<'s>;
        %type literal Literal<'s>;
        %type relating Item<'s>;
        %type binding_body Body<'s>;
        %type start Model<'s>;
        %type model Vec<Item<'s>>;
        %type bindings Vec<Item<'s>>;
        %type bindings_comma Vec<Item<'s>>;
        %type bindings_pipe Vec<Item<'s>>;
        %type model_semi Vec<Item<'s>>;
        %type model_comma Vec<Item<'s>>;
        %type model_sp Vec<Item<'s>>;
        %type model_nl Vec<Item<'s>>;
        %type labels_down Vec<&'s str>;
        %type labels_up Vec<&'s str>;
        %type label_level (Vec<&'s str>, Vec<&'s str>);
        %type label &'s str;
        %type labels Vec<(Vec<&'s str>, Vec<&'s str>)>;
        %right Colon;
        %right Text;
        %right Lsq;
        %right Lbr;
        // %verbose;

        start ::= model;

        model ::= model(mut m) Nl binding(i) { m.push(i); m };
        model ::= model(mut m) Nl relating(i) { m.push(i); m };
        // model ::= model(mut m) Nl literal(i) { m.push(Item::Literal{literal: i}); m };
        model ::= binding(i) { vec![i] };
        model ::= relating(i) { vec![i] };
        // model ::= literal(i) { vec![Item::Literal{literal: i}] };

        binding ::= Text(binding) Colon literal(l) {Item::Binding{binding, expr: Box::new(Item::Literal{literal: l})}};
        binding ::= Text(binding) Colon relating(r) {Item::Binding{binding, expr: Box::new(r)}};
        binding ::= literal(i) { Item::Binding{binding: "", expr: Box::new(Item::Literal{literal: i})}};

        literal ::= Text(label) Lsq bindings?(body) Rsq { Literal{ label: Some(label), body: Some(Body::And(body.unwrap_or_default())) }};
        literal ::= Text(label) Lbr bindings?(body) Rbr { Literal{ label: Some(label), body: Some(Body::Or(body.unwrap_or_default())) }};
        literal ::= Lsq bindings?(body) Rsq { Literal{ label: None, body: Some(Body::And(body.unwrap_or_default())) }};
        literal ::= Lbr bindings?(body) Rbr { Literal{ label: None, body: Some(Body::Or(body.unwrap_or_default())) }};
        literal ::= Text(label) { Literal{ label: Some(label), body: None} };

        bindings ::= bindings(mut bs) binding(b) { bs.push(b); bs };
        bindings ::= binding(b) { vec![b] };

        relating ::= Text(t1) Text(t2) Colon labels(labels) { Item::Relating{lhs: vec![t1, t2], rhs: labels}};
        relating ::= Text(t1) Text(t2) { Item::Relating{lhs: vec![t1, t2], rhs: vec![]} };

        labels ::= labels(mut l) Colon label_level(lvl) { l.push(lvl); l };
        labels ::= label_level(lvl) { vec![lvl] };

        label_level ::= labels_down(d) Slash labels_up(u) { (d, u) };
        label_level ::= Slash labels_up(u) { (vec![], u) };
        label_level ::= labels_down(d) { (d, vec![]) };
        // label_level ::= { (vec![], vec![]) };

        labels_down ::= labels_down(mut ls) Comma label(l) { ls.push(l); ls };
        labels_up ::= labels_up(mut ls) Comma label(l) { ls.push(l); ls };
        labels_down ::= label(l) { vec![l] };
        labels_up ::= label(l) { vec![l] };
        label ::= Text;
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