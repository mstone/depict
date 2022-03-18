{
  description = "github.com/mstone/diagrams";

  inputs.crane.url = "github:ipetkov/crane";
  inputs.crane.inputs.nixpkgs.follows = "nixpkgs";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:mstone/nixpkgs";
  inputs.nix-filter.url = "github:numtide/nix-filter";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay";
  inputs.rust-overlay.inputs.flake-utils.follows = "flake-utils";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";

  outputs = {self, nixpkgs, crane, rust-overlay, flake-utils, nix-filter}:
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
          lib.diagrams = { isShell }: with final; with pkgs; crane.lib.${final.system}.buildPackage {
            pname = "diagrams";
            version = "1.0";

            src = nix-filter.lib.filter {
              root = self;
              include = [
                "Cargo.lock"
                "Cargo.toml"
                "src"
                "web"
              ];
            };

            cargoLock = self + "/Cargo.lock";

            buildInputs = [
              (rust-bin.stable.latest.minimal.override { targets = [ "wasm32-unknown-unknown" ]; })
              #texlive.combined.scheme-full
              (python39.withPackages (ps: with ps; [cvxpy]))
            ] ++ final.lib.optionals isShell [
              entr
              wasm-pack
              trunk
            ] ++ final.lib.optionals stdenv.isDarwin (with darwin.apple_sdk.frameworks; [
              AppKit
              Security
              CoreServices
              CoreFoundation
              Foundation
              AppKit
              WebKit
              Cocoa
            ]);

            doCheck = false;
          };
        };
      };
    };
}
