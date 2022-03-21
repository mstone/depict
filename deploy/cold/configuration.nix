{ config, pkgs, ... }:
{
  imports =
    [ 
      ./hardware-configuration.nix
    ];

  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  boot.kernelParams = [ "nomodeset" ];

  networking.hostName = "cold"; # Define your hostname.

  time.timeZone = "US/Eastern";

  networking.useDHCP = false;
  networking.interfaces.eno1.useDHCP = true;
  networking.interfaces.wlp7s0.useDHCP = false;

  i18n.defaultLocale = "en_US.UTF-8";
  console = {
    font = "Lat2-Terminus16";
    keyMap = "us";
  };

  nix = {
    package = pkgs.nixVersions.nix_2_7;
    extraOptions = "experimental-features = nix-command flakes";
  };

  users.users."root".openssh.authorizedKeys.keys = [ "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG1bbOqaW9rAtfhclAFi35fcGbMaIvQDbWbMBGXPRE6O mstone@MacBook-Pro.local" ];

  environment.systemPackages = with pkgs; [
    vim
  ];

  programs.mosh.enable = true;

  services.getty.autologinUser = "root";

  services.openssh.enable = true;

  system.stateVersion = "21.11";
}
