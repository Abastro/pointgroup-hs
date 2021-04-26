#!/usr/bin/env bash

pushd hasktorch
source setenv   # Shell environment
./setup-cabal.sh  # Setup local for hasktorch
popd
# Copies local file into current project
cp -f ./hasktorch/cabal.project.local ./cabal.project.local
