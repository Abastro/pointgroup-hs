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
    [pathSrc, "--link", pathTar] -> do
      linkScannet pathSrc "scannetv2_train" "scans" pathTar "train"
      linkScannet pathSrc "scannetv2_val" "scans" pathTar "val"
      linkScannet pathSrc "scannetv2_test" "scans_test" pathTar "test"
    _ -> do
      putStrLn "pointgroup-hs [data_path] <flags>"
      putStrLn "--prep=[train|val|test] for preparation"
      putStrLn "--link [target_path] to put dataset links into the target"
