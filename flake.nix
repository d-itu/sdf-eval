{
  description = "Pinned environment for preparing the SDF benchmark dataset";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/4975466d324710c576dc11ad614684e6bd8cad8e";
    sdf-view = {
      url = "github:d-itu/sdf-view/51ff7f4f9e026d852520263374056a6af0740bbc";
      flake = false;
    };
  };

  outputs =
    { nixpkgs, sdf-view, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              rustc
              rustfmt
              clippy
              rust-analyzer
              pkg-config
              git
              curl
              (python3.withPackages (ps: [ ps.pillow ]))
            ];

            RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
            SDF_VIEW_SOURCE = "${sdf-view}";

            shellHook = ''
              export CARGO_TARGET_DIR="$PWD/target"
              export LD_LIBRARY_PATH="${
                pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ]
              }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            '';
          };
        }
      );
    };
}
