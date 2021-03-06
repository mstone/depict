<div align="center">
  <h1>Depict</h1>
</div>

<div align="center">
  <!-- Crates version -->
  <a href="https://crates.io/crates/depict">
    <img src="https://img.shields.io/crates/v/depict.svg?style=flat-square"
    alt="Crates.io version" />
  </a>
  <!-- Downloads -->
  <a href="https://crates.io/crates/depict">
    <img src="https://img.shields.io/crates/d/depict.svg?style=flat-square"
      alt="Download" />
  </a>
  <!-- docs -->
  <a href="https://docs.rs/depict">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square"
      alt="docs.rs docs" />
  </a>
  <!-- CI
  <a href="https://github.com/mstone/depict/actions">
    <img src="https://github.com/mstone/depict/actions/workflows/main.yml/badge.svg"
      alt="CI status" />
  </a> -->
  <!-- Discord -->
  <a href="https://discord.gg/UpWYZ5dN">
    <img src="https://img.shields.io/discord/973591045881360414.svg?logo=discord&style=flat-square" alt="Discord Link" />
  </a>
</div>

*Depict* helps people communicate about complex systems with pictures. ([demo](https://mstone.info/depict/))

[![Depict live demo, showing a model of a microwave](https://raw.githubusercontent.com/mstone/depict/main/doc/microwave.gif)](https://mstone.info/depict/)

People who need to communicate about complex systems often draw pictures with boxes and arrows. 

Unfortunately, many people find it hard to make these drawings quickly and legibly with conventional tools. They often struggle to uncross arrows or to keep parts of their drawing from colliding, especially while editing text labels. These challenges also make the drawings hard to reuse and to maintain over time.

*Depict* can help:
* concisely describe processes, systems, and concepts of operations
* automatically draw pretty, legible, maintainable pictures
* extract and reuse portions of previous descriptions

thereby helping you to analyze and tell powerful stories about such systems.

## Installation

The simplest way to try depict is to use [nix](https://nixos.org/nix/) with flakes enabled to run:

```bash
nix run github:mstone/depict#desktop
```

This should produce a window similar to the one shown in the screenshot above.

Alternately, if you'd like to run depict without with nix, you'll need to

1. install a recent Rust compiler, and any dependencies necessary for your platform
2. use `cargo` to build or run one of Depict's sub-packages, like:

```bash
cargo run -p depict-desktop
```


## Usage/Examples

*Depict* models systems as hierarchies of interacting processes expressed as partial orders. Each input line describes a chain in this order, which will be drawn as a downward-directed path with labels through this graph. Hence the input line:

```
person microwave food: open, start, stop / beep : heat
person food: eat
```

says: 

* there is a path downward in our model from a process (controller) named `person` to a process named `microwave` to a process named `food`, 
* in the space between `person` and `microwave`, there are three actions, `open`, `start`, and `stop`, and one feedback, `beep`, 
* in the space between `microwave` and `food`, there is one action, `heat`.
* finally, there is also a direct relationship between `person` and `food` consisting of the action: `stir`.


## Syntax

The language of depictions loosely consists of 

| production |   | syntax                                                  |
|------------|---|---------------------------------------------------------|
|definitions |::=| *name* **:** *expr*
|relations   |::=| *name* *name* ... (**:** *labels* (**/** */ *labels*)?)*
|labels      |::=| *label*... for single-word labels 
|labels      |::=| *label* (**,** *label*)* for multi-word labels
|nesting     |::=| **[** *model* **]**
|alternatives|::=| **{** *model* **}**


## License

This project is licensed under the [MIT license].

[MIT license]: https://github.com/mstone/depict/blob/main/LICENSE

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Depict by you, shall be licensed as MIT, without any additional terms or conditions.
