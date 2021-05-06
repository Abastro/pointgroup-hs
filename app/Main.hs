module Main where

import System.Environment (getArgs)
import Preprocess

main :: IO ()
main = do
  l <- getArgs
  case l of
    [path] -> putStrLn "TODO: Main process"
    [path, "--prep=train"] -> do
      handleScannet path False "scannetv2_train" "scans" "train"
    [path, "--prep=val"] -> do
      handleScannet path False "scannetv2_val" "scans" "val"
    [path, "--prep=test"] -> do
      handleScannet path True "scannetv2_test" "scans_test" "test"
    _ -> do
      putStrLn "pointgroup-hs [data_path] <flags>"
      putStrLn "--prep=[train|val|test] for preparation"
