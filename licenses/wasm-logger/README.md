# wasm-logger

A logger that sends a message with its Rust source's line and filename to the browser console. 

![screenshot](Screenshot.png)

## Usage

**Note**: For more information about how to use loggers in Rust, see [log](https://crates.io/crates/log).

`Cargo.toml`
```
[dependencies]
log = "0.4.6"
wasm-logger = "0.2.0"
```

Initialize `wasm-logger` when your app start:
```rust
wasm_logger::init(wasm_logger::Config::default());

// Logging
log::info!("Some info");
log::error!("Error message");
```

### Log for a specific module only

You can provide a path prefix:
```rust
wasm_logger::init(wasm_logger::Config::default().module_prefix("some::module"));
```

then, `wasm-logger` only logs message from `some::module` 

## Mapping from `log` to console's methods
`log::error!`, `log::warn!` and `log::info!` call theirs equivalent methods of the browser console. The `console.trace` method outputs some extra trace from the generated JS glue code which we don't want. Therefore, we choose to map:
* `log::debug!` to `console.log`
* `log::trace!` to `console.debug`.

## Note for Chromium/Chrome users

Chromium/Chrome filters out `console.debug` (execute by `log::trace!`) by default. You must check the `Verbose` filter in your browser console to see trace entries.

## License
[MIT](http://opensource.org/licenses/MIT)
or
[Apache-2.0](http://www.apache.org/licenses/LICENSE-2.0)
