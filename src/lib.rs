pub mod parser {
    // use crate::data::*;
    use nom::{IResult, InputLength, InputIter, InputTake, InputTakeAtPosition, FindToken, Needed, Slice};
    use nom::error::{VerboseError, context};
    use nom::bytes::complete::{take_while, take_while1, is_not, is_a};
    // use nom::character::{is_space};
    use nom::character::complete::{char};
    use nom::combinator::{map, opt};
    use nom::multi::{many1, separated_list0};
    use nom::sequence::{preceded, terminated, tuple, separated_pair};
    use std::hash::Hash;
    use std::ops::RangeFrom;

    use std::ops::Deref;

    #[derive(Clone, Debug, PartialEq)]
    pub struct Rc<T>(pub std::rc::Rc<T>);

    impl<T> Rc<T> {
        pub fn new(t: T) -> Self {
            Rc(std::rc::Rc::new(t))
        }
    }

    impl<D, T> Deref for Rc<T> where D: ?Sized, T: Deref<Target=D> {
        type Target = D;

        fn deref(&self) -> &Self::Target {
            self.0.deref()
        }
    }

    impl<T: InputLength> InputLength for Rc<T> {
        fn input_len(&self) -> usize { 
            self.0.input_len()
        }
    }

    impl<T: InputIter> InputIter for Rc<T> {
        type Item = T::Item;

        type Iter = T::Iter;

        type IterElem = T::IterElem;

        fn iter_indices(&self) -> Self::Iter { 
            self.0.iter_indices() 
        }

        fn iter_elements(&self) -> Self::IterElem {
            self.0.iter_elements()
        }

        fn position<P>(&self, predicate: P) -> Option<usize>
        where
            P: Fn(Self::Item) -> bool 
        {
            self.0.position(predicate)
        }

        fn slice_index(&self, count: usize) -> Result<usize, nom::Needed> {
            self.0.slice_index(count)
        }
    }

    impl<T: InputTake> InputTake for Rc<T> {
        fn take(&self, count: usize) -> Self {
            Rc::new(self.0.take(count))
        }

        fn take_split(&self, count: usize) -> (Self, Self) {
            let (a, b) = self.0.take_split(count);
            (Rc::new(a), Rc::new(b))
        }
    }

    impl<T: Slice<RangeFrom<usize>>> Slice<RangeFrom<usize>> for Rc<T> {
        fn slice(&self, range: RangeFrom<usize>) -> Self {
            Rc::new(self.0.slice(range))
        }
    }

    impl<T: InputLength + InputIter + InputTake + InputTakeAtPosition> InputTakeAtPosition for Rc<T> {
        type Item = <T as InputIter>::Item;

        fn split_at_position<P, E: nom::error::ParseError<Self>>(&self, predicate: P) -> nom::IResult<Self, Self, E>
        where
            P: Fn(Self::Item) -> bool 
        {
            match self.position(predicate) {
                Some(n) => Ok(self.take_split(n)),
                None => Err(nom::Err::Incomplete(Needed::new(1))),
            }
        }

        fn split_at_position1<P, E: nom::error::ParseError<Self>>(
            &self,
            predicate: P,
            e: nom::error::ErrorKind,
        ) -> nom::IResult<Self, Self, E>
        where
            P: Fn(Self::Item) -> bool 
        {
            match self.position(predicate) {
                Some(0) => Err(nom::Err::Error(E::from_error_kind(Rc(self.0.clone()), e))),
                Some(n) => Ok(self.take_split(n)),
                None => Err(nom::Err::Incomplete(Needed::new(1))),
            }
        }

        fn split_at_position_complete<P, E: nom::error::ParseError<Self>>(
            &self,
            predicate: P,
        ) -> nom::IResult<Self, Self, E>
        where
            P: Fn(Self::Item) -> bool 
        {
            match self.split_at_position(predicate) {
                Err(nom::Err::Incomplete(_)) => Ok(self.take_split(self.input_len())),
                res => res,
            }
        }

        fn split_at_position1_complete<P, E: nom::error::ParseError<Self>>(
            &self,
            predicate: P,
            e: nom::error::ErrorKind,
        ) -> nom::IResult<Self, Self, E>
        where
            P: Fn(Self::Item) -> bool 
        {
            match self.split_at_position1(predicate, e) {
                Err(nom::Err::Incomplete(_)) => {
                    if self.input_len() == 0 {
                        Err(nom::Err::Error(E::from_error_kind(Rc(self.0.clone()), e)))
                    } else {
                        Ok(self.take_split(self.input_len()))
                    }
                }
                res => res,
            }
        }
    }

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
                    opt(char(':')),
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
        use super::Rc;

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

        #[test]
        fn fact_rc() {
            let s = Rc::new("hello: bar / baz ");
            let y = super::parse(s.clone());
            if let Err(nom::Err::Error(ref y2)) = y {
                println!("{}", convert_error(s, y2.clone()))
            }
            assert_eq!(y, Ok((Rc::new(""), vec![
                super::Fact{path: vec![Rc::new("hello")], labels_by_level: vec![(vec![Some(Rc::new("bar "))], vec![Some(Rc::new("baz "))])]}
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