{
  description = "deploy scripts for github.com/mstone/diagrams";

  inputs.diagrams.url = "path:..";

  inputs.nixpkgs.follows = "diagrams/nixpkgs";

  inputs.deploy-rs.url = "github:serokell/deploy-rs";
  inputs.deploy-rs.inputs.nixpkgs.follows = "nixpkgs";
  inputs.deploy-rs.inputs.flake-utils.follows = "flake-utils";

  outputs = {self, nixpkgs, deploy-rs, diagrams}: {

    nixosConfigurations.cold = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [ cold/configuration.nix ];
    };

    deploy.nodes.cold = {
      hostname = "cold";
      profiles.system = {
        user = "root";
        path = deploy-rs.lib.x86_64-linux.activate.nixos self.nixosConfigurations.cold;
      };
    };
    
    deploy.sshUser = "root";

    checks = builtins.mapAttrs (system: deployLib: deployLib.deployChecks self.deploy) deploy-rs.lib;
  };
}
