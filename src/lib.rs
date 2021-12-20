pub mod parser {
    // use crate::data::*;
    use nom::IResult;
    use nom::error::{VerboseError, context};
    use nom::branch::{alt};
    use nom::bytes::complete::{take_while, take_while1, take_while_m_n, is_not, is_a};
    // use nom::character::{is_space};
    use nom::character::complete::{char};
    use nom::combinator::{map, not};
    use nom::multi::{many0, many1};
    use nom::sequence::{preceded, terminated, tuple};
    use std::hash::Hash;
    use std::fmt::Display;


    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct Ident<I>(pub I);

    impl<I: Display> Display for Ident<I> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct Directive<I>(pub I, pub Vec<I>);

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub enum Fact<I> {
        Atom(I),
        Fact(I, Vec<Self>)
    }

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub enum Syn<I> where I: Clone + std::fmt::Debug + PartialEq {
        Ident(Ident<I>),
        Directive(Directive<Ident<I>>),
        Fact(Fact<Ident<I>>),
    }

    impl<'a, I: Clone + std::fmt::Debug + Eq + Hash + PartialEq> TryFrom<&'a Syn<I>> for &'a Fact<Ident<I>> {
        type Error = ();

        fn try_from(value: &'a Syn<I>) -> Result<Self, ()> {
            match value {
                Syn::Fact(f) => Ok(f),
                _ => Err(())
            }
        }
    }

    pub fn is_ws(chr: char) -> bool {
        match chr {
            ' ' => true,
            _ => false,
        }
    }

    pub fn is_wst(chr: char) -> bool {
        match chr {
            ' ' => true,
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
        is_not(" \n\r:@.")(s)
    }

    pub fn ident(s: &str) -> IResult<&str, Ident<&str>, VerboseError<&str>> {
        map(context("ident", normal), Ident::<&str>)(s)
    }

    pub fn guarded_ident(s: &str) -> IResult<&str, Ident<&str>, VerboseError<&str>> {
        map(context("guarded_ident", terminated(normal, not(char(':')))), Ident::<&str>)(s)
    }

    pub fn directive(s: &str) -> IResult<&str, Directive<Ident<&str>>, VerboseError<&str>> {
        map(
            context(
                "directive", 
                tuple((
                    char('@'), 
                    map(normal, Ident),
                    many1(preceded(take_while(is_wst), guarded_ident))
                ))
            ), 
            |(_, v1, v2)| Directive(v1, v2)
        )(s)
    }

    pub fn fact(indent: usize) -> impl Fn(&str) -> IResult<&str, Fact<Ident<&str>>, VerboseError<&str>> {
        let n = (indent + 1) * 4;
        move |s: &str| {
            map(
                context(
                    "fact",
                    tuple(
                        (
                            ident,
                            char(':'),
                            alt((
                                many1(preceded(tuple((ws, char('\n'), take_while_m_n(n, n, is_ws), ws)), fact(indent+1))), // for indentation
                                many1(preceded(ws, fact(indent))),
                                many0(preceded(ws, map(guarded_ident, |i| Fact::<Ident<&str>>::Atom(i)))), // many0, for empty facts.
                            )),
                        )
                    )
                ),
                |(v1, _, v2)| Fact::<Ident<&str>>::Fact(v1, v2)
            )(s)
        }
    }

    pub fn parse(s: &str) -> IResult<&str, Vec<Syn<&str>>, VerboseError<&str>> {
        // many1(tag("hello"))(s)
        terminated(many1(alt((
            preceded(take_while(is_wst), map(directive, Syn::<&str>::Directive)),
            preceded(take_while(is_wst), map(fact(0), Syn::<&str>::Fact)),
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
                    super::Directive(
                        super::Ident("hello"), 
                        vec![
                            super::Ident("hello")
                        ]
                    )
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
                    super::Fact::Fact(
                        super::Ident("hello"), 
                        vec![
                            super::Fact::Atom(
                                super::Ident("bar"),
                            )
                        ]
                    )
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
                    super::Fact::Fact(
                        super::Ident("hello"),
                        vec![
                            super::Fact::Fact(
                                super::Ident("bar"),
                                vec![
                                    super::Fact::Atom(
                                        super::Ident("baz"),
                                    )
                                ]
                            ),
                            super::Fact::Fact(
                                super::Ident("bar2"),
                                vec![
                                    super::Fact::Atom(
                                        super::Ident("baz2"),
                                    )
                                ]
                            )
                        ]
                    )
                ),
            ])));
        }

        #[test]
        fn expanded_fact_works() {
            let s = "hello:\n    bar: baz";
            let y = super::parse(s);
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok(("", vec![
                super::Syn::<&str>::Fact(
                    super::Fact::Fact(
                        super::Ident("hello"),
                        vec![
                            super::Fact::Fact(
                                super::Ident("bar"),
                                vec![
                                    super::Fact::Atom(
                                        super::Ident("baz"),
                                    )
                                ]
                            )
                        ]
                    )
                ),
            ])));
        }
    }
}