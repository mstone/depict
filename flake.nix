{
  description = "github.com/mstone/diagrams";

  inputs.import-cargo.url = "git+https://github.com/edolstra/import-cargo";
  inputs.nixpkgs.url = "nixpkgs";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay";
  inputs.rust-overlay.inputs.flake-utils.follows = "flake-utils";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";

  outputs = {self, nixpkgs, import-cargo, rust-overlay, flake-utils}:
    flake-utils.lib.simpleFlake {
      inherit self nixpkgs;
      name = "diagrams";
      systems = flake-utils.lib.allSystems;
      preOverlays = [ rust-overlay.overlay ];
      overlay = final: prev: {
        diagrams = rec {
          diagrams = lib.diagrams { isShell = false; };
          devShell = lib.diagrams { isShell = true; };
          defaultPackage = diagrams;
          lib.diagrams = { isShell }: with final; with pkgs; stdenv.mkDerivation {
            name = "diagrams";

            src = self;

            buildInputs = [
              rust-bin.stable.latest.rust
              texlive.combined.scheme-full
              (python39.withPackages (ps: with ps; [cvxpy]))
            ] ++ (if isShell then [
              entr
              wasm-pack
            ] else [
              (import-cargo.builders.importCargo {
                lockFile = ./Cargo.lock;
                inherit pkgs;
              }).cargoHome
            ]) ++ final.lib.optionals stdenv.isDarwin [
              darwin.apple_sdk.frameworks.AppKit
            ];

            buildPhase = ''
              cargo build --frozen --offline
            '';

            doCheck = true;

            checkPhase = ''
              cargo test --frozen --offline
            '';

            installPhase = ''
              mkdir -p $out/bin
              cargo install --frozen --offline --path bin/diagrams --root $out
              rm $out/.crates.toml
            '';
          };
        };
      };
    };
}
