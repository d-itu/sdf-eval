{
  description = "Pinned environment for preparing the SDF benchmark dataset";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/4975466d324710c576dc11ad614684e6bd8cad8e";
    sdf-view = {
      url = "github:d-itu/sdf-view/cfb01c7535db99cec2351408672d069101789596";
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
          mesaIcdDir = "${pkgs.mesa}/share/vulkan/icd.d";
          mesaIcdArch = builtins.head (pkgs.lib.splitString "-" system);
          sdfViewPackage = pkgs.rustPlatform.buildRustPackage {
            pname = "sdf-view";
            version = "0.2.0";
            src = sdf-view;
            cargoLock.lockFile = "${sdf-view}/Cargo.lock";
            doCheck = false;

            nativeBuildInputs = with pkgs; [
              pkg-config
              makeWrapper
            ];
            buildInputs = with pkgs; [
              vulkan-loader
              wayland
              libxkbcommon
            ];
            postFixup = ''
              wrapProgram $out/bin/sdf-view \
                --prefix LD_LIBRARY_PATH : ${
                  pkgs.lib.makeLibraryPath [
                    pkgs.vulkan-loader
                    pkgs.wayland
                    pkgs.libxkbcommon
                  ]
                }
            '';
          };
          sdfViewSoftware = pkgs.writeShellScriptBin "sdf-view-software" ''
            export WGPU_BACKEND=vulkan
            export VK_DRIVER_FILES="${mesaIcdDir}/lvp_icd.${mesaIcdArch}.json"
            exec ${sdfViewPackage}/bin/sdf-view "$@"
          '';
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              sdfViewPackage
              sdfViewSoftware
              cargo
              rustc
              rustfmt
              clippy
              rust-analyzer
              pkg-config
              git
              curl
            ];

            RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
            SDF_VIEW_SOURCE = "${sdf-view}";

            shellHook = ''
              export CARGO_TARGET_DIR="$PWD/target"
              export UV_PYTHON_INSTALL_DIR="$PWD/.work/python"
              export UV_PYTHON_DOWNLOADS=never
              export WGPU_BACKEND=vulkan
              export LD_LIBRARY_PATH="${
                pkgs.lib.makeLibraryPath [
                  pkgs.vulkan-loader
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                ]
              }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              VK_DRIVER_FILES=
              for icd in ${mesaIcdDir}/*_icd.*.json; do
                case "$icd" in
                  */lvp_icd.*) continue ;;
                esac
                VK_DRIVER_FILES="''${VK_DRIVER_FILES:+$VK_DRIVER_FILES:}$icd"
              done
              export VK_DRIVER_FILES
            '';
          };
        }
      );
    };
}
