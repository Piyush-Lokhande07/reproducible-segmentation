{
  description = "Hermetic environment for reproducible semantic segmentation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = false; # Set to true on a CUDA-supported Linux machine
          };
        };

        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          torch
          torchvision
          numpy
          scikit-learn
          matplotlib
          seaborn
          tqdm
          pyyaml
          albumentations
          opencv4
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.libGL
            pkgs.glib
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.libGL pkgs.glib ]}:$LD_LIBRARY_PATH"
            echo "Nix reproducible environment active. Python: $(python --version)"
          '';
        };
      }
    );
}
