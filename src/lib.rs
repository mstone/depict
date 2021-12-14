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

    }
}

mod parser {
    use crate::data::*;
    use nom::IResult;
    use nom::error::VerboseError;
    use nom::bytes::complete::tag;

    pub fn parse(s: &str) -> IResult<&str, &str, VerboseError<&str>> {
        tag("hello")(s)
    }

    // let w = World{};
    // w

    #[cfg(test)]
    mod tests {
        use nom::error::convert_error;

        #[test]
        fn parse_works() {
            let s = "hello";
            let y = super::parse(s);
            { 
                match y { 
                    Err(nom::Err::Error(ref y2)) => println!("{}", convert_error(s, y2.clone())),
                    _ => (),
                }
            }
            assert_eq!(y, Ok(("", "hello")));
        }
    }
}