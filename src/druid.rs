use diagrams::parser::{parse};
use diagrams::render::{Syn};
use std::fs::read_to_string;
use std::env::args;
use std::io;
use std::process::exit;
use nom::error::convert_error;

use druid::{AppLauncher, WindowDesc, Data};
use druid::{widget::Label, Widget};

#[derive(Clone, Data)]
pub struct AppState {}

pub fn build_ui() -> impl Widget<AppState> {
    Label::new("Hello")
}

pub fn render(_v: Vec<Syn>) {
    let main_window = WindowDesc::new(build_ui)
    .title("Todo Tutorial")
    .window_size((400.0, 400.0));

    let initial_state = AppState {};

    AppLauncher::with_window(main_window)
        .launch(initial_state)
        .expect("Failed to launch application");
}

pub fn main() -> io::Result<()> {
    for path in args().skip(1) {
        let contents = read_to_string(path)?;
        // println!("{}\n\n", &contents);
        let v = parse(&contents[..]);
        match v {
            Err(nom::Err::Error(v2)) => {
                println!("{}", convert_error(&contents[..], v2));
                exit(1);
            },
            Ok(("", v2)) => {
                render(v2);
            }
            _ => {
                println!("{:#?}", v);
                exit(2);
            }
        }
    }
    Ok(())
}