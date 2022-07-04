{
  description = "github.com/mstone/depict";

  inputs.crane.url = "github:ipetkov/crane";
  inputs.crane.inputs.nixpkgs.follows = "nixpkgs";

  inputs.deploy-rs.url = "github:serokell/deploy-rs";
  inputs.deploy-rs.inputs.nixpkgs.follows = "nixpkgs";
  inputs.deploy-rs.inputs.flake-utils.follows = "flake-utils";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  inputs.nixpkgs.url = "nixpkgs/nixpkgs-unstable";

  inputs.nix-filter.url = "github:numtide/nix-filter";

  inputs.rust-overlay.url = "github:oxalica/rust-overlay";
  inputs.rust-overlay.inputs.flake-utils.follows = "flake-utils";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";

  inputs.nixbom.url = "github:mstone/nixbom";
  inputs.nixbom.inputs.crane.follows = "crane";
  inputs.nixbom.inputs.flake-utils.follows = "flake-utils";
  inputs.nixbom.inputs.nixpkgs.follows = "nixpkgs";
  inputs.nixbom.inputs.nix-filter.follows = "nix-filter";
  inputs.nixbom.inputs.rust-overlay.follows = "rust-overlay";

  outputs = {self, nixpkgs, crane, deploy-rs, nixbom, rust-overlay, flake-utils, nix-filter}:
    flake-utils.lib.simpleFlake {
      inherit self nixpkgs;
      name = "depict";
      systems = flake-utils.lib.allSystems;
      preOverlays = [ 
        rust-overlay.overlay
      ];
      overlay = final: prev: {
        depict = rec {

          depictVersion = "0.2";
          depict = lib.depict { isShell = false; };
          devShell = lib.depict { isShell = true; };
          defaultPackage = depict;

          server = with final; with pkgs; let
            subpkg = "depict-server";
            serverBin = (lib.depict { isShell = false; subpkg = subpkg; subdir = "server"; });
          in stdenv.mkDerivation { 
            pname = "${subpkg}";
            version = depictVersion;
            buildInputs = [ makeWrapper ];
            phases = [ "installPhase" ];
            installPhase = ''
              mkdir -p $out/bin
              cp ${serverBin}/bin/${subpkg} $out/bin/${subpkg}
              wrapProgram $out/bin/${subpkg} \
                --set WEBROOT ${web}
            '';
          };

          web = with final; with pkgs; let
            subpkg = "depict-web";
            webBin = (lib.depict { isShell = false; subpkg = subpkg; subdir = "web"; isWasm = true; });
            indexHtml = writeText "index.html" ''
              <!DOCTYPE html><html><head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <link rel="preload" href="/web_bg.wasm" as="fetch" type="application/wasm" crossorigin="">
              <link rel="modulepreload" href="/web.js"></head>
              <body>
                <div id="main"> </div>
                <script type="module">import init from '/web.js';init('/web_bg.wasm');</script>
              </body>
              </html>
            '';
          in stdenv.mkDerivation { 
            pname = "${subpkg}";
            version = depictVersion;
            phases = [ "buildPhase" "installPhase" ];
            buildInputs = [
              wasm-bindgen-cli 
            ];
            buildPhase = ''
              cp ${webBin}/bin/${subpkg}.wasm web.wasm
              mkdir pkg
              wasm-bindgen --target web --out-dir pkg web.wasm
            '';
            installPhase = ''
              mkdir $out;
              cp -a pkg/* $out
              cp ${indexHtml} $out/index.html
            '';
          };

          desktop = with final; with pkgs; let
            subpkg = "depict-desktop";
            pkg = (lib.depict { isShell = false; subpkg = subpkg; subdir = "dioxus"; });
          in stdenv.mkDerivation { 
            pname = subpkg;
            version = depictVersion;
            buildInputs = [ makeWrapper ];
            phases = [ "installPhase" ];
            installPhase = ''
              mkdir -p $out/bin
              cp ${pkg}/bin/${subpkg} $out/bin/${subpkg}
            '';
          };

          bintools = prev.bintools.overrideAttrs (old: {
            postFixup = 
              if prev.stdenv.isDarwin then 
                builtins.replaceStrings ["-no_uuid"] [""] old.postFixup
              else 
                old.postFixup;
          });

          cc = prev.stdenv.cc.overrideAttrs (old: {
            inherit bintools;
          });

          stdenv = prev.overrideCC prev.stdenv cc;

          # rust from rust-overlay adds stdenv.cc to propagatedBuildInputs 
          # and depsHostHostPropagated; therefore, to ensure that the correct
          # cc is present in downstream consumers, we need to override both these 
          # attrs.
          rust = with final; with pkgs; 
            #(rust-bin.stable.latest.minimal.override { targets = [ "wasm32-unknown-unknown" ]; })
            #(rust-bin.nightly.latest.minimal.override { extensions = [ "rustfmt" ]; targets = [ "wasm32-unknown-unknown" ]; })
            (rust-bin.selectLatestNightlyWith (toolchain: toolchain.minimal.override {
              extensions = [ "rustfmt" ];
              targets = [ "wasm32-unknown-unknown" ];
            })).overrideAttrs (old: {
              inherit stdenv;
              propagatedBuildInputs = [ stdenv.cc ];
              depsHostHostPropagated = [ stdenv.cc ];
            });

          # crane provides a buildPackage helper that calls stdenv.mkDerivation
          # which provides a default builder that sources a "setup" file defined
          # by the stdenv itself (passed as the environment variable "stdenv" that 
          # in turn defines a defaultNativeBuildInputs variable that gets added to 
          # PATH via the genericBuild initialization code. Therefore, we override
          # crane's stdenv to use our modified cc-wrapper.
          cranelib = crane.lib.${final.system}.overrideScope' (final: prev: {
            inherit stdenv;
          });

          tex = with final; with pkgs; texlive.combined.scheme-full;

          lib.depict = { isShell, isWasm ? false, subpkg ? "depict", subdir ? "." }: 
            let 
              buildInputs = with final; with pkgs; [
                rust
                tex
                cmake
              ] ++ final.lib.optionals isShell [
                entr
                trunk
                wasm-bindgen-cli
                wabt
                deploy-rs.packages.${final.system}.deploy-rs
                (terraform_1.withPlugins (p: with p; [aws gandi vultr]))
                nixbom.legacyPackages.${final.system}.nixbom
                cargo-expand
                cargo-license
                cargo-outdated
                cargo-udeps
                rustfmt
              ] ++ final.lib.optionals stdenv.isDarwin (with darwin.apple_sdk.frameworks; [
                AppKit
                Security
                CoreServices
                CoreFoundation
                Foundation
                AppKit
                WebKit
                Cocoa
              ]) ++ final.lib.optionals stdenv.isLinux ([
                pkg-config
                glib
                gdk-pixbuf
                gtk3
                webkitgtk
                libayatana-appindicator-gtk3
                libappindicator-gtk3
              ]);
            in with final; with pkgs; cranelib.buildPackage {
            pname = "${subpkg}";
            version = depictVersion;

            src = self;

            cargoArtifacts = cranelib.buildDepsOnly { 
              src = self;

              inherit buildInputs;
              dontUseCmakeConfigure = true;

              cargoCheckCommand = if isWasm then "" else 
                if final.lib.hasSuffix "darwin" final.system then ''
                  cargo check --release -p depict-desktop -p depict-server; 
                  cargo check --release -p depict-web --target wasm32-unknown-unknown
                '' else ''
                  cargo check --release -p depict-server
                '';
              cargoBuildCommand = if isWasm then "cargo build --release -p depict-web --target wasm32-unknown-unknown" else
                if final.lib.hasSuffix "darwin" final.system then ''
                  cargo build --release -p depict-desktop -p depict-server; 
                  cargo build --release -p depict-web --target wasm32-unknown-unknown
                '' else ''
                  cargo build --release -p depict-server
                '';
              cargoTestCommand = if isWasm then "" else
                if final.lib.hasSuffix "darwin" final.system then ''
                  cargo test --release -p depict-desktop -p depict-server;
                '' else ''
                  cargo test --release -p depict-server
                '';
            };
            cargoCheckCommand = if isWasm then "" else "cargo check --release -p ${subpkg}";
            cargoBuildCommand = if isWasm then "cargo build --release -p ${subpkg} --target wasm32-unknown-unknown" else "cargo build --release -p ${subpkg}";

            inherit buildInputs;
            dontUseCmakeConfigure = true;

            doCheck = false;
          };
        };
      };
    };
}
