#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

mod data {
    pub struct World {
        names: Vec<String>,
        directives: Vec<String>,
    }
}

pub mod parser {
    // use crate::data::*;
    use nom::IResult;
    use nom::error::{VerboseError, context};
    use nom::branch::{alt};
    use nom::bytes::complete::{take_while, take_while1, is_not, is_a};
    // use nom::character::{is_space};
    use nom::character::complete::{char, one_of};
    use nom::combinator::{map, not, peek};
    use nom::multi::{many0, many1};
    use nom::sequence::{preceded, terminated, separated_pair, tuple};

    #[derive(Clone, Debug, PartialEq)]
    pub enum Syn<I> where I: Clone + Ord + PartialEq {
        Ident(I),
        Directive(I, Vec<Self>),
        Fact(Box<Self>, Vec<Self>)
    }

    pub fn is_ws(chr: char) -> bool {
        match chr {
            ' ' => true,
            '\t' => true,
            _ => false,
        }
    }

    pub fn is_wst(chr: char) -> bool {
        match chr {
            ' ' => true,
            '\t' => true,
            '\n' => true,
            '\r' => true,
            '.' => true,
            _ => false,
        }
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
        is_not(" \t\n\r:@.")(s)
    }

    pub fn ident(s: &str) -> IResult<&str, Syn<&str>, VerboseError<&str>> {
        map(context("ident", normal), Syn::<&str>::Ident)(s)
    }

    pub fn guarded_ident(s: &str) -> IResult<&str, Syn<&str>, VerboseError<&str>> {
        map(context("guarded_ident", terminated(normal, not(char(':')))), Syn::<&str>::Ident)(s)
    }

    pub fn directive(s: &str) -> IResult<&str, Syn<&str>, VerboseError<&str>> {
        map(
            context(
                "directive", 
                tuple((
                    char('@'), 
                    normal,
                    many1(preceded(take_while(is_wst), guarded_ident))
                ))
            ), 
            |(_, v1, v2)| Syn::<&str>::Directive(v1, v2)
        )(s)
    }

    pub fn fact(s: &str) -> IResult<&str, Syn<&str>, VerboseError<&str>> {
        map(
            context(
                "fact", 
                tuple(
                    (
                        ident,
                        char(':'),
                        alt((
                            many1(preceded(tuple((ws, char('\n'), ws1)), fact)), // ws1, for indentation
                            many1(preceded(ws, fact)),
                            many0(preceded(ws, guarded_ident)), // many0, for empty facts.
                        )),
                    )
                )
            ),
            |(v1, _, v2)| Syn::<&str>::Fact(Box::new(v1), v2)
        )(s)
    }

    pub fn parse(s: &str) -> IResult<&str, Vec<Syn<&str>>, VerboseError<&str>> {
        // many1(tag("hello"))(s)
        terminated(many1(alt((
            preceded(take_while(is_wst), directive),
            preceded(take_while(is_wst), fact),
            // ident
        ))), take_while(is_wst))(s)
    }

    // let w = World{};
    // w

    #[cfg(test)]
    mod tests {
        use nom::error::convert_error;

        #[test]
        fn parse_works() {
            let s = "@hello hello";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Syn::<&str>::Directive(
                    "hello", 
                    vec![
                        super::Syn::<&str>::Ident("hello")
                    ]
                )
            ])));
        }

        #[test]
        fn fact_works() {
            let s = "hello: bar";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Syn::<&str>::Fact(
                    Box::new(super::Syn::<&str>::Ident("hello")), 
                    vec![
                        super::Syn::<&str>::Ident("bar")
                    ]
                ),
            ])));
        }

        #[test]
        fn nested_fact_works() {
            let s = "hello: bar: baz bar2: baz2";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Syn::<&str>::Fact(
                    Box::new(super::Syn::<&str>::Ident("hello")),
                    vec![
                        super::Syn::<&str>::Fact(
                            Box::new(super::Syn::<&str>::Ident("bar")),
                            vec![
                                super::Syn::<&str>::Ident("baz")
                            ]
                        ),
                        super::Syn::<&str>::Fact(
                            Box::new(super::Syn::<&str>::Ident("bar2")),
                            vec![
                                super::Syn::<&str>::Ident("baz2")
                            ]
                        )
                    ]
                ),
            ])));
        }

        #[test]
        fn expanded_fact_works() {
            let s = "hello:\n\tbar: baz";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Syn::<&str>::Fact(
                    Box::new(super::Syn::<&str>::Ident("hello")),
                    vec![
                        super::Syn::<&str>::Fact(
                            Box::new(super::Syn::<&str>::Ident("bar")),
                            vec![
                                super::Syn::<&str>::Ident("baz")
                            ]
                        )
                    ]
                ),
            ])));
        }
    }
}