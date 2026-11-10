{
  description = "Pinned environment for preparing the SDF benchmark dataset";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/4975466d324710c576dc11ad614684e6bd8cad8e";
    sdf-view = {
      url = "github:d-itu/sdf-view/fe59aa944e81337a4e81725118b83795f47ef182";
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
            cargoBuildFlags = [
              "-p"
              "sdf-view-cli"
            ];
            doCheck = false;

            nativeBuildInputs = with pkgs; [
              pkg-config
              makeWrapper
            ];
            buildInputs = with pkgs; [
              vulkan-loader
              wayland
              libxkbcommon
              libX11
              libXcursor
              libXi
              libXrandr
              libxcb
            ];
            postFixup = ''
              wrapProgram $out/bin/sdf-view \
                --prefix LD_LIBRARY_PATH : ${
                  pkgs.lib.makeLibraryPath [
                    pkgs.vulkan-loader
                    pkgs.wayland
                    pkgs.libxkbcommon
                    pkgs.libX11
                    pkgs.libXcursor
                    pkgs.libXi
                    pkgs.libXrandr
                    pkgs.libxcb
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
            packages = [
              sdfViewPackage
              sdfViewSoftware
            ];

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
              for icd in \
                /run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json \
                /usr/share/vulkan/icd.d/nvidia_icd.json \
                /etc/vulkan/icd.d/nvidia_icd.json; do
                if [ -r "$icd" ]; then
                  VK_DRIVER_FILES="$icd"
                  break
                fi
              done
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
