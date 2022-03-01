pub mod parser {
    use futures::TryFutureExt;
    use nom::branch::alt;
    // use crate::data::*;
    use nom::{IResult};
    use nom::error::{VerboseError, context};
    use nom::bytes::complete::{take_while, take_while1, is_not, is_a};
    // use nom::character::{is_space};
    use nom::character::complete::{char};
    use nom::combinator::{map, opt};
    use nom::multi::{many1, separated_list0, separated_list1};
    use nom::sequence::{preceded, terminated, tuple, separated_pair, pair};
    use std::hash::Hash;

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct Fact<'s> {
        pub path: Vec<&'s str>,
        pub labels_by_level: Vec<Vec<(Option<&'s str>, Option<&'s str>)>>,
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
        is_not(" \n\r:.,/")(s)
    }

    pub fn fact(s: &str) -> IResult<&str, Fact, VerboseError<&str>> {
        map(
            context(
                "fact",
                tuple((
                    many1(preceded(ws, normal)),
                    opt(char(':')),
                    map(opt(separated_list0(
                        preceded(ws, char(':')),
                        separated_list0(
                            preceded(ws, char(',')),
                            separated_pair(
                                preceded(ws, opt(is_not("\n\r:.,/"))),
                                preceded(ws, opt(char('/'))),
                                preceded(ws, opt(is_not("\n\r:.,/"))),
                            ),
                            // alt((
                            //     map(
                            //         separated_pair(
                            //             preceded(ws, is_not("\n\r:.,/")),
                            //             preceded(ws, char('/')),
                            //             preceded(ws, is_not("\n\r:.,/")),
                            //         ),
                            //         |(action, percept)| (Some(action), Some(percept))
                            //     ),
                            //     map(
                            //         pair(
                            //             preceded(ws, char('/')),
                            //             preceded(ws, is_not("\n\r:.,/")),
                            //         ),
                            //         |(_, percept)| (None, Some(percept))
                            //     ),
                            //     map(
                            //         pair(
                            //             preceded(ws, is_not("\n\r:.,/")),
                            //             preceded(ws, opt(char('/'))),
                            //         ),
                            //         |(action, _)| (Some(action), None)
                            //     ),
                            // ))
                        )
                    )), |x| x.unwrap_or_default())
                ))
            ),
            |(path, _, labels_by_level)| Fact{path, labels_by_level}
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
                super::Fact{path: vec!["hello"], labels_by_level: vec![vec![(Some("bar "), Some("baz "))]]}
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
                super::Fact{path: vec!["hello"], labels_by_level: vec![vec![(Some("bar "), None)]]}
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
                super::Fact{path: vec!["hello"], labels_by_level: vec![vec![(None, Some("baz "))]]}
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
                super::Fact{path: vec!["hello"], labels_by_level: vec![vec![(Some("bar "), Some("baz "))], vec![(Some("foo "), Some("quux"))]]}
            ])));
        }

        #[test]
        fn multiple_labels_works() {
            let s = "hello: bar / baz, foo / quux";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Fact{path: vec!["hello"], labels_by_level: vec![vec![(Some("bar "), Some("baz")), (Some("foo "), Some("quux"))]]}
            ])));
        }
    }
}

pub mod graph_drawing;