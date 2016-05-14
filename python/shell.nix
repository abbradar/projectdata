{ pkgs ? import <nixpkgs> {} }:

(pkgs.python3.buildEnv.override {
  extraLibs = with pkgs.python3Packages;
    [ scikitlearn xgboost pandas matplotlib pyqt5 ipython
    ];
}).env
