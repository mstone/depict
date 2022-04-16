{
  description = "github.com/mstone/diagrams";

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
  inputs.minionSrc.url = "github:minion/minion";
  inputs.minionSrc.flake = false;

  outputs = {self, nixpkgs, crane, deploy-rs, minionSrc, rust-overlay, flake-utils, nix-filter}:
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

          diagramsSrc = nix-filter.lib.filter {
            root = self;
            include = [
              "Cargo.lock"
              "Cargo.toml"
              "src"
              "dioxus/src"
              "server/src"
              "tikz/src"
              "objc/src"
              "web/src"
              (nix-filter.lib.matchExt "rs")
              (nix-filter.lib.matchExt "toml")
              (nix-filter.lib.matchExt "lock")
            ];
          };

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
                --set PYTHONPATH ${python3.pkgs.makePythonPath [python3.pkgs.cvxpy]} \
                --set WEBROOT ${web}
            '';
          };

          web = with final; with pkgs; let
            webBin = (lib.diagrams { isShell = false; subdir = "web"; isWasm = true; });
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
            pname = "web";
            version = "1.0";
            phases = [ "buildPhase" "installPhase" ];
            buildInputs = [
              wasm-bindgen-cli 
            ];
            buildPhase = ''
              cp ${webBin}/bin/web.wasm .
              mkdir pkg
              wasm-bindgen --target web --out-dir pkg web.wasm
            '';
            installPhase = ''
              mkdir $out;
              cp -a pkg/* $out
              cp ${indexHtml} $out/index.html
            '';
          };

          #web = with final; with pkgs; stdenv.mkDerivation {
          #  pname = "web";
          #  version = "1.0";
          #  src = self;
          #  buildInputs = [ 
          #    (rust-bin.stable.latest.minimal.override { targets = [ "wasm32-unknown-unknown" ]; })
          #    wasm-pack 
          #  ];
          #  buildPhase = ''
          #    
          #    (cd web; RUSTFLAGS="-C linker=lld" cargo build --release --target wasm32-unknown-unknown)
          #  '';
          #  installPhase = ''
          #    mkdir -p $out
          #    (cd web/dist; cp -a * $out)
          #  '';
          #};

          desktop = with final; with pkgs; let
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

          lib.diagrams = { isShell, isWasm ? false, subdir ? "." }: 
            let 
              pnameSuffix = if subdir == "." then "" else "-${subdir}";
              python = with final; with pkgs; python39.withPackages (ps: with ps; [cvxpy]);
              buildInputs = with final; with pkgs; [
                (rust-bin.stable.latest.minimal.override { targets = [ "wasm32-unknown-unknown" ]; })
                #texlive.combined.scheme-full
                python
              ] ++ final.lib.optionals isShell [
                entr
                trunk
                deploy-rs.packages.${final.system}.deploy-rs
                (terraform_1.withPlugins (p: with p; [aws gandi vultr]))
                cargo-outdated
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
            in with final; with pkgs; crane.lib.${final.system}.buildPackage {
            pname = "diagrams${pnameSuffix}";
            version = "1.0";

            # src = self;
            src = diagramsSrc;

            cargoArtifacts = crane.lib.${final.system}.buildDepsOnly { 
              src = diagramsSrc; 
              inherit buildInputs;
              cargoLock = diagramsSrc + "/Cargo.lock";
              cargoToml = diagramsSrc + "/Cargo.toml";
              #cargoExtraArgs = if isWasm then "-p web --target wasm32-unknown-unknown" else null;
            };
            cargoLock = diagramsSrc + "/Cargo.lock";
            cargoToml = diagramsSrc + "/Cargo.toml";
            cargoCheckCommand = "";
            cargoBuildCommand = "cargo build --release -p ${subdir}" + final.lib.optionalString isWasm " --target wasm32-unknown-unknown";

            inherit buildInputs;

            doCheck = false;

            shellHook = if isShell then ''
              export PYTHONPATH=''${PYTHONPATH:+''${PYTHONPATH}:}${python}/${python.sitePackages}
            '' else null;
          };
        };
      };
    };
}
