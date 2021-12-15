use diagrams::parser::*;
use std::fs::read_to_string;
use std::env::args;
use std::io;
use nom::error::convert_error;

pub fn main() -> io::Result<()> {
    for path in args().skip(1) {
        let contents = read_to_string(path)?;
        println!("{}\n\n", &contents);
        let v = parse(&contents[..]);
        if let Err(nom::Err::Error(v2)) = v {
            println!("{}", convert_error(&contents[..], v2))
        } else {
            println!("{:#?}", v);
        }
    }
    Ok(())
}