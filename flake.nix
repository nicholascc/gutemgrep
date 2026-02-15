{
  description = "gutemgrep python dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3.withPackages (ps: [
          ps.hnswlib
          ps.lxml
          ps.nltk
          ps.numpy
          ps.pandas
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
          ];
        };
      });
}
