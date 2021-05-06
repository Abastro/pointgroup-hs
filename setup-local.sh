#!/usr/bin/env bash

./hasktorch/setup-cabal.sh  # Setup local for hasktorch
# Copies local file into current project
cp -f ./hasktorch/cabal.project.local ./cabal.project.local
