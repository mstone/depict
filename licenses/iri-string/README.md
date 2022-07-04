# iri-string

[![Build Status](https://travis-ci.com/lo48576/iri-string.svg?branch=develop)](https://travis-ci.com/lo48576/iri-string)
[![Latest version](https://img.shields.io/crates/v/iri-string.svg)](https://crates.io/crates/iri-string)
[![Documentation](https://docs.rs/iri-string/badge.svg)](https://docs.rs/iri-string)
![Minimum rustc version: 1.48](https://img.shields.io/badge/rustc-1.48+-lightgray.svg)

String types for [IRI](https://tools.ietf.org/html/rfc3987)s (Internationalized Resource
Identifiers) and [URI](https://tools.ietf.org/html/rfc3986)s (Uniform Resource Identifiers).

See the [documentation](https://docs.rs/iri-string) for details.

## Features

* `no_std` support.
* String types (both owned and borrowed) for IRIs.
* IRI reference resolution algorithm.

### Feature flags

#### Direct
* `alloc` (enabled by default)
    + Enables types and functions which require memory allocation.
    + Requires `std` or `alloc` crate available.
* `std` (enabled by default)
    + Enables all `std` features (such as memory allocations and `std::error::Error` trait).
    + Requires `std` crate available.
    + This automatically enables `alloc` feature.

#### nom
* `nom-std` (enabled by default)
    + Enable optimization for internal parsers, using std power.
    + This automatically enables `std` feature.

#### serde
* `serde`
    + Implements `Serialize` and `Deserialize` traits for string types.
* `serde-alloc`
    + Enables deriving serialization and deserialization impls which requires heap memory allocation.
    + If you enable `alloc` and `serde` features at the same time, you should also enable `serde-alloc` feature.
    + This automatically enables `alloc` feature.
* `serde-std`
    + Enables deriving serialization and deserialization impls which requires std stuff.
    + If you enable `std` and `serde` features at the same time, you should also enable `serde-std` feature.
    + This automatically enables `std` feature.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE.txt](LICENSE-APACHE.txt) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT.txt](LICENSE-MIT.txt) or
  <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
