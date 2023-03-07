Execute
====================

[![CI](https://github.com/magiclen/execute/actions/workflows/ci.yml/badge.svg)](https://github.com/magiclen/execute/actions/workflows/ci.yml)

This library is used for extending `Command` in order to execute programs more easily.

## Usage

```rust
use std::process::Command;

use execute::Execute;

// ...
```

### Verify the Program

Since `Command` is used for spawning a process of a command and the executed progrom is external which may not exist or may not be the program that we expected, we usually need to verify the external program at runtime.

The `execute_check_exit_status_code` method can be used to execute a command and check its exit status. For example,

```rust
use std::process::Command;

use execute::Execute;

const FFMPEG_PATH: &str = "/path/to/ffmpeg";

let mut first_command = Command::new(FFMPEG_PATH);

first_command.arg("-version");

if first_command.execute_check_exit_status_code(0).is_err() {
    eprintln!("The path `{}` is not a correct FFmpeg executable binary file.", FFMPEG_PATH);
}
```

### Execute and Get the Exit Status

```rust
use std::process::Command;

use execute::Execute;

const FFMPEG_PATH: &str = "/path/to/ffmpeg";

let mut command = Command::new(FFMPEG_PATH);

command.arg("-i");
command.arg("/path/to/media-file");
command.arg("/path/to/output-file");

if let Some(exit_code) = command.execute().unwrap() {
    if exit_code == 0 {
        println!("Ok.");
    } else {
        eprintln!("Failed.");
    }
} else {
    eprintln!("Interrupted!");
}
```

### Execute and Get the Output

#### Output to the Screen

```rust
use std::process::Command;

use execute::Execute;

const FFMPEG_PATH: &str = "/path/to/ffmpeg";

let mut command = Command::new(FFMPEG_PATH);

command.arg("-i");
command.arg("/path/to/media-file");
command.arg("/path/to/output-file");

let output = command.execute_output().unwrap();

if let Some(exit_code) = output.status.code() {
    if exit_code == 0 {
        println!("Ok.");
    } else {
        eprintln!("Failed.");
    }
} else {
    eprintln!("Interrupted!");
}
```

#### Output to Memory (Captured)

```rust
use std::process::{Command, Stdio};

use execute::Execute;

const FFMPEG_PATH: &str = "/path/to/ffmpeg";

let mut command = Command::new(FFMPEG_PATH);

command.arg("-i");
command.arg("/path/to/media-file");
command.arg("/path/to/output-file");

command.stdout(Stdio::piped());
command.stderr(Stdio::piped());

let output = command.execute_output().unwrap();

if let Some(exit_code) = output.status.code() {
    if exit_code == 0 {
        println!("Ok.");
    } else {
        eprintln!("Failed.");
    }
} else {
    eprintln!("Interrupted!");
}

println!("{}", String::from_utf8(output.stdout).unwrap());
println!("{}", String::from_utf8(output.stderr).unwrap());
```

### Execute and Input Data

#### Input In-memory Data

```rust
use std::process::{Command, Stdio};

use execute::Execute;

let mut bc_command = Command::new("bc");

bc_command.stdout(Stdio::piped());

let output = bc_command.execute_input_output("2^99\n").unwrap();

println!("Answer: {}", String::from_utf8(output.stdout).unwrap().trim_end());
```

#### Input from a Reader

```rust
use std::process::{Command, Stdio};
use std::fs::File;

use execute::Execute;

let mut cat_command = Command::new("cat");

cat_command.stdout(Stdio::piped());

let mut file = File::open("Cargo.toml").unwrap();

let output = cat_command.execute_input_reader_output(&mut file).unwrap();

println!("{}", String::from_utf8(output.stdout).unwrap());
```

By default, the buffer size is 256 bytes. If you want to change that, you can use the `_reader_output2` or `_reader2` methods and define a length explicitly.

For example, to change the buffer size to 4096 bytes,

```rust
use std::process::{Command, Stdio};
use std::fs::File;

use execute::generic_array::typenum::U4096;
use execute::Execute;

let mut cat_command = Command::new("cat");

cat_command.stdout(Stdio::piped());

let mut file = File::open("Cargo.toml").unwrap();

let output = cat_command.execute_input_reader_output2::<U4096>(&mut file).unwrap();

println!("{}", String::from_utf8(output.stdout).unwrap());
```

### Execute Multiple Commands and Pipe Them Together

```rust
use std::process::{Command, Stdio};

use execute::Execute;

let mut command1 = Command::new("echo");
command1.arg("HELLO WORLD");

let mut command2 = Command::new("cut");
command2.arg("-d").arg(" ").arg("-f").arg("1");

let mut command3 = Command::new("tr");
command3.arg("A-Z").arg("a-z");

command3.stdout(Stdio::piped());

let output = command1.execute_multiple_output(&mut [&mut command2, &mut command3]).unwrap();

assert_eq!(b"hello\n", output.stdout.as_slice());
```

### Run a Command String in the Current Shell

The `shell` function can be used to create a `Command` instance with a single command string instead of a program name and scattered arguments.

```rust
use std::process::{Command, Stdio};

use execute::{Execute, shell};

let mut command = shell("cat /proc/meminfo");

command.stdout(Stdio::piped());

let output = command.execute_output().unwrap();

println!("{}", String::from_utf8(output.stdout).unwrap());
```

### Parse a Command String at Runtime

The `command` function can be used to create a `Command` instance with a single command string instead of a program name and scattered arguments. The difference between the `shell` function and the `command` function is that the former is interpreted by the current shell while the latter is parsed by this crate.

```rust
use std::process::{Command, Stdio};

use execute::{Execute, command};

let mut command = command("cat '/proc/meminfo'");

command.stdout(Stdio::piped());

let output = command.execute_output().unwrap();

println!("{}", String::from_utf8(output.stdout).unwrap());
```

### Parse a Command String at Compile Time

The `command!` macro can be used to create a `Command` instance with a single command string literal instead of a program name and scattered arguments.

```rust
use std::process::{Command, Stdio};

use execute::Execute;

let mut command = execute::command!("cat '/proc/meminfo'");

command.stdout(Stdio::piped());

let output = command.execute_output().unwrap();

println!("{}", String::from_utf8(output.stdout).unwrap());
```

### Create a `Command` Instance by Providing Arguments Separately

The `command_args!` macro can be used to create a `Command` instance with a program name and arguments separately. The program name and arguments can be non-literal.

```rust
use std::process::{Command, Stdio};

use execute::Execute;

let mut command = execute::command_args!("cat", "/proc/meminfo");

command.stdout(Stdio::piped());

let output = command.execute_output().unwrap();

println!("{}", String::from_utf8(output.stdout).unwrap());
```

## Crates.io

https://crates.io/crates/execute

## Documentation

https://docs.rs/execute

## License

[MIT](LICENSE)
