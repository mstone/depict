Execute Command Macro
====================

[![CI](https://github.com/magiclen/execute/actions/workflows/ci.yml/badge.svg)](https://github.com/magiclen/execute/actions/workflows/ci.yml)

Create `Command` instances using the `command!` macro or the `command_args!` macro.

Also see [`execute`](https://crates.io/crates/execute).

## Examples

```rust
#[macro_use] extern crate execute_command_macro;

let command = command!("program arg1 arg2 'arg 3' -opt1 -opt2");
```

```rust
#[macro_use] extern crate execute_command_macro;

let command = command_args!("program", "arg1", "arg2", "-opt1", "-opt2");
```

## Crates.io

https://crates.io/crates/execute

## Documentation

https://docs.rs/execute

## License

[MIT](LICENSE)
