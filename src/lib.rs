pub mod parser {
    // use crate::data::*;
    use nom::{IResult};
    use nom::error::{VerboseError, context};
    use nom::bytes::complete::{take_while, take_while1, is_not, is_a};
    // use nom::character::{is_space};
    use nom::character::complete::{char};
    use nom::combinator::{map, opt};
    use nom::multi::{many1};
    use nom::sequence::{preceded, terminated, tuple};
    use std::hash::Hash;

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct Fact<'s> {
        pub path: Vec<&'s str>,
        pub action: Option<&'s str>,
        pub percept: Option<&'s str>,
    }

    pub fn is_ws(chr: char) -> bool {
        matches!(chr, ' ')
    }

    pub fn is_wst(chr: char) -> bool {
        matches!(chr, ' ' | '\n' | '\r' | '.')
    }

    pub fn nl(s: &str) -> IResult<&str, &str, VerboseError<&str>> {
        is_a("\r\n.")(s)
    }

    pub fn ws(s: &str) -> IResult<&str, &str, VerboseError<&str>> {
        take_while(is_ws)(s)
    }

    pub fn ws1(s: &str) -> IResult<&str, &str, VerboseError<&str>> {
        take_while1(is_ws)(s)
    }

    pub fn normal(s: &str) -> IResult<&str, &str, VerboseError<&str>> {
        is_not(" \n\r:./")(s)
    }

    pub fn fact(s: &str) -> IResult<&str, Fact, VerboseError<&str>> {
        map(
            context(
                "fact",
                tuple((
                    many1(preceded(ws, normal)),
                    opt(tuple((
                        ws, 
                        char(':'), 
                        preceded(ws, opt(is_not("\n\r:./"))),
                        opt(preceded(char('/'),
                        preceded(ws, opt(is_not("\n\r:./")))))))),
                ))
            ),
            |(path, tail)| Fact{path, action: tail.and_then(|t| t.2), percept: tail.and_then(|t| t.3.flatten())}
        )(s)
    }

    pub fn parse(s: &str) -> IResult<&str, Vec<Fact>, VerboseError<&str>> {
        terminated(many1(
            preceded(take_while(is_wst), fact),
        ), take_while(is_wst))(s)
    }

    #[cfg(test)]
    mod tests {
        use nom::error::convert_error;

        #[test]
        fn fact_works() {
            let s = "hello: bar / baz ";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Fact{path: vec!["hello"], action: Some("bar "), percept: Some("baz ")}
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
                super::Fact{path: vec!["hello"], action: Some("bar "), percept: None}
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
                super::Fact{path: vec!["hello"], action: None, percept: Some("baz ")}
            ])));
        }
    }
}

pub mod graph_drawing;