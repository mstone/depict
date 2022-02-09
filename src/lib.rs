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

    impl<'a, I> From<&&'a Fact<Ident<I>>> for &'a Fact<Ident<I>> {
        fn from(value: &&'a Fact<Ident<I>>) -> Self {
            *value
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

pub mod render {
    use auto_enums::auto_enum;

    pub trait Render {
        fn header();
        fn footer();
    }

    pub type Syn<'a> = crate::parser::Syn::<&'a str>;
    pub type Ident<'a> = crate::parser::Ident<&'a str>;
    pub type Fact<'a> = crate::parser::Fact<Ident<'a>>;

    // pub fn filter_directives<'a, I: Iterator<Item = Syn<'a>>>(v: I) -> Vec<&'a Directive<'a>> {
    //     v
    //         .filter_map(|e| if let Syn::Directive(d) = e { Some(d) } else { None })
    //         .collect()
    // }

    // pub fn filter_fact<'a>(v: &'a Vec<Syn>, i: &'a Ident) -> impl Iterator<Item = &'a Fact<'a>> {
    // pub fn filter_fact<'a, I: Iterator<Item = &'a Syn<'a>>>(v: I, i: &'a Ident) -> impl Iterator<Item = &'a Fact<'a>> {
    pub fn filter_fact<'a, I: Iterator<Item = Item>, Item: TryInto<&'a Fact<'a>, Error=E>, E>(v: I, q: &'a Ident) -> impl Iterator<Item = &'a Fact<'a>> {
        v
            .filter_map(move |e| match e.try_into() { Ok(Fact::Fact(ref i, f)) if q == i => Some(f), _ => None, })
            .flatten()
    }

    // pub fn resolve<'a>(v: &'a Vec<Syn>, r: &'a Fact<'a>) -> Vec<&'a Fact<'a>> {
    #[auto_enum(Iterator)]
    pub fn resolve<'a, I: Iterator<Item = Item>, Item: TryInto<&'a Fact<'a>, Error=E>, E>(v: I, r: &'a Fact<'a>) -> impl Iterator<Item = &'a Fact<'a>> {
        match r {
            Fact::Atom(i) => {
                return filter_fact(v, i);
            },
            Fact::Fact(_i, fs) => {
                return fs.iter();
            },
        }
    }

    pub fn unwrap_atom<'a>(a: &'a Fact<'a>) -> Option<&'a str> {
        match a {
            Fact::Atom(crate::parser::Ident(i)) => Some(*i),
            _ => None,
        }
    }

    pub fn to_ident<'a>(a: &'a Fact<'a>) -> &'a Ident<'a> {
        match a {
            Fact::Atom(i) => i,
            Fact::Fact(i, _fs) => i,
        }
    }

    pub fn first_ident<'a, I: Iterator<Item = &'a Fact<'a>>>(mut v: I) -> Option<&'a str> {
        v
            .find_map(unwrap_atom)
    }

    /// Given a database `v`, returns facts of type `q1` about the entity identified by `q2`.
    /// 
    /// Example: 
    /// 
    /// Suppose person: actuates: lever. Then
    /// 
    /// ```rust
    /// let resolved_actuates = find_parent(v.iter(), &Ident("actuates"), to_ident(item)).next(); 
    /// ```
    /// 
    /// Then `resolved_actuates == Some(Ident("lever"))`.
    pub fn find_parent<'a, I: Iterator<Item = Item>, Item: PartialEq + TryInto<&'a Fact<'a>, Error=E>, E>(v: I, q1: &'a Ident, q2: &'a Ident) -> impl Iterator<Item = &'a Ident<'a>> {
        v
            .filter_map(move |item|
                match item.try_into() {
                    Ok(Fact::Fact(i, fs)) => {
                        let mut candidates = filter_fact(fs.iter(), q1);
                        candidates.find(|c| match c {
                            Fact::Atom(a) if a == q2 => true,
                            _ => false,
                        }).map(|_| i)
                    },
                    _ => None,
                }
            )
    }

    pub fn as_string<'a, I: IntoIterator<Item = Item>, Item: PartialEq + TryInto<&'a Fact<'a>, Error=E>, E>(v: I, q: &'a Ident, default: String) -> String {
        let default = vec![Fact::Atom(crate::parser::Ident(&default))];
        let mut subfacts = v
            .into_iter()
            .filter_map(move |e| match e.try_into() { Ok(Fact::Fact(ref i, f)) if q == i => Some(f), _ => None, })
            .collect::<Vec<&Vec<Fact>>>();
        if subfacts.is_empty() {
            subfacts.push(&default);
        }
        subfacts
            .iter()
            .map(|f| f
                .iter()
                .map(|ia| match ia {
                    Fact::Atom(a2) => a2.0,
                    Fact::Fact(ref i2, _f2) => i2.0,
                })
                .collect::<Vec<&str>>()
                .join(" "))
            .collect::<Vec<String>>()
            .join(" ")
    }

    #[cfg(test)]
    mod tests {
        use crate::{render::{as_string}, parser::{Ident, parse}};

        #[test]
        fn as_string_works() {
            let y: String = "baz baz2".into();
            let s = "bar: baz bar: baz2";
            let p = parse(s);
            let q = p.unwrap();
            let r = q.1.iter();
            assert_eq!(y, as_string(r, &Ident("bar"), "".into()));
        }

        #[test]
        fn as_string_default_works() {
            let y: String = "quux".into();
            let s = "bar: baz bar: baz2";
            let p = parse(s);
            let q = p.unwrap();
            let r = q.1.iter();
            assert_eq!(y, as_string(r, &Ident("foo"), "quux".into()))
        }
    }
}

pub mod graph_drawing;