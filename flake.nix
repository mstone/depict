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
  inputs.minionSrc.url = "github:minion/minion";
  inputs.minionSrc.flake = false;

  outputs = {self, nixpkgs, crane, minionSrc, rust-overlay, flake-utils, nix-filter}:
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

          minion = with final; with pkgs; stdenv.mkDerivation {
            pname = "minion";
            version = "2.0.0-rc1";
            src = minionSrc;
            buildInputs = [ python2 ];
            buildPhase = ''
              mkdir build
              cd build
              python2 $src/configure.py
              make minion
            '';
            installPhase = ''
              mkdir -p $out/bin
              cp -a ./minion $out/bin
            '';
          };

          server = with final; with pkgs; let
            serverBin = (lib.diagrams { isShell = false; subdir = "server"; });
          in stdenv.mkDerivation { 
            pname = "server";
            version = "1.0";
            buildInputs = [ makeWrapper ];
            phases = [ "installPhase" ];
            installPhase = ''
              mkdir -p $out/bin
              cp ${serverBin}/bin/server $out/bin/server
              wrapProgram $out/bin/server \
                --prefix PATH : "${minion}/bin/" \
                --set PYTHONPATH ${python3.pkgs.makePythonPath [python3.pkgs.cvxpy]}
            '';
          };

          web = with final; with pkgs; stdenv.mkDerivation {
            pname = "web";
            version = "1.0";
            src = self;
            buildInputs = [ 
              (rust-bin.stable.latest.minimal.override { targets = [ "wasm32-unknown-unknown" ]; })
              trunk 
            ];
            buildPhase = ''
              mkdir home
              mkdir cargo
              mkdir trunk
              export HOME="$(pwd)/home";
              export CARGO_HOME="$(pwd)/cargo";
              export TRUNK_STAGING_DIR="$(pwd)/trunk";
              (cd web; trunk build)
            '';
            installPhase = ''
              mkdir -p $out
              (cd web/dist; cp -a * $out)
            '';
          };

          sketch-desktop = with final; with pkgs; let
            pkgName = "diadym-sketch-desktop";
            pkg = (lib.diagrams { isShell = false; subdir = pkgName; });
          in stdenv.mkDerivation { 
            pname = pkgName;
            version = "1.0";
            buildInputs = [ makeWrapper ];
            phases = [ "installPhase" ];
            installPhase = ''
              mkdir -p $out/bin
              cp ${pkg}/bin/${pkgName} $out/bin/${pkgName}
              wrapProgram $out/bin/${pkgName} \
                --prefix PATH : "${minion}/bin/" \
                --set PYTHONPATH ${python3.pkgs.makePythonPath [python3.pkgs.cvxpy]}
            '';
          };

          lib.diagrams = { isShell, subdir ? "." }: 
            let 
              pnameSuffix = if subdir == "." then "" else "-${subdir}";
            in with final; with pkgs; crane.lib.${final.system}.buildPackage {
            pname = "diagrams${pnameSuffix}";
            version = "1.0";

            src = self;
            # src = nix-filter.lib.filter {
            #   root = self;
            #   include = [
            #     "Cargo.lock"
            #     "Cargo.toml"
            #     "src"
            #     "web"
            #   ];
            # };

            cargoLock = self + "/Cargo.lock";
            cargoCheckCommand = "";
            cargoBuildCommand = "cargo build -p ${subdir} --release";

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
