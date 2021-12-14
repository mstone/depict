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
    use crate::data::*;
    use nom::IResult;
    use nom::error::{VerboseError, context};
    use nom::branch::{alt};
    use nom::bytes::complete::{take_while, is_not, tag};
    use nom::character::{is_space};
    use nom::character::complete::{char};
    use nom::combinator::map;
    use nom::multi::many1;
    use nom::sequence::preceded;

    #[derive(Clone, Debug, PartialEq)]
    pub enum Syn<I> where I: Clone + Ord + PartialEq {
        Ident(I),
        Directive(I),
    }

    pub fn is_ws(chr: char) -> bool {
        match chr {
            ' ' => true,
            _ => false,
        }
    }

    pub fn not_ws(s: &str) -> IResult<&str, &str, VerboseError<&str>> {
        is_not(" \t")(s)
    }

    pub fn ident(s: &str) -> IResult<&str, Syn<&str>, VerboseError<&str>> {
        map(context("ident", preceded(take_while(is_ws), not_ws)), Syn::<&str>::Ident)(s)
    }

    pub fn directive(s: &str) -> IResult<&str, Syn<&str>, VerboseError<&str>> {
        map(context("directive", preceded(take_while(is_ws), preceded(char('@'), not_ws))), Syn::<&str>::Directive)(s)
    }

    pub fn parse(s: &str) -> IResult<&str, Vec<Syn<&str>>, VerboseError<&str>> {
        // many1(tag("hello"))(s)
        many1(alt((directive, ident)))(s)
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
            assert_eq!(y, Ok(("", vec![super::Syn::<&str>::Directive("hello"), super::Syn::<&str>::Ident("hello")])));
        }
    }
}