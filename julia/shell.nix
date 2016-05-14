{ pkgs ? import <nixpkgs> {} }:

pkgs.stdenv.mkDerivation {
  name = "julia-env";
  buildInputs = with pkgs; [ gcc gnumake julia git python3Packages.jupyter autoconf automake m4 zeromq ];
}
