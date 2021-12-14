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

mod parser {
    // use crate::data::*;
    use nom::IResult;
    use nom::error::{VerboseError, context};
    use nom::branch::{alt};
    use nom::bytes::complete::{take_while, is_not};
    // use nom::character::{is_space};
    use nom::character::complete::{char, one_of};
    use nom::combinator::map;
    use nom::multi::many1;
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
            '\n' => true,
            '\t' => true,
            _ => false,
        }
    }

    pub fn not_ws(s: &str) -> IResult<&str, &str, VerboseError<&str>> {
        is_not(" \t\n")(s)
    }

    pub fn normal(s: &str) -> IResult<&str, &str, VerboseError<&str>> {
        is_not(" \t\n:@.")(s)
    }

    pub fn ident(s: &str) -> IResult<&str, Syn<&str>, VerboseError<&str>> {
        map(context("ident", preceded(take_while(is_ws), normal)), Syn::<&str>::Ident)(s)
    }

    pub fn directive(s: &str) -> IResult<&str, Syn<&str>, VerboseError<&str>> {
        map(
            context(
                "directive", 
                tuple((
                    take_while(is_ws), 
                    char('@'), 
                    normal,
                    many1(ident)
                ))
            ), 
            |(_, _, v1, v2)| Syn::<&str>::Directive(v1, v2)
        )(s)
    }

    pub fn fact(s: &str) -> IResult<&str, Syn<&str>, VerboseError<&str>> {
        map(
            context(
                "fact", 
                tuple(
                    (
                        take_while(is_ws), 
                        ident,
                        char(':'),
                        alt((
                            terminated(many1(fact), one_of(".\n")),
                            many1(ident),
                        )),
                    )
                )
            ),
            |(_, v1, _, v2)| Syn::<&str>::Fact(Box::new(v1), v2)
        )(s)
    }

    pub fn parse(s: &str) -> IResult<&str, Vec<Syn<&str>>, VerboseError<&str>> {
        // many1(tag("hello"))(s)
        many1(alt((directive, fact, ident)))(s)
    }

    // let w = World{};
    // w

    #[cfg(test)]
    mod tests {
        use nom::error::convert_error;

        #[test]
        fn parse_works() {
            let s = " @hello hello";
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
            let s = "hello: bar: baz.";
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

        #[test]
        fn expanded_fact_works() {
            let s = "hello:\n\tbar: baz\n";
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