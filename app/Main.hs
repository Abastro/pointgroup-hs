module Main where

import System.Environment (getArgs)
import System.FilePath ( (</>) )
import System.Directory
import Data.List ( isPrefixOf )
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import Control.Monad.Cont (ContT (..))
import Control.Monad (when)
import Text.Printf (printf)
import Pipes
import qualified Pipes.Prelude as P
import Torch as T hiding ( div )
import Torch.Serialize ( loadParams, saveParams )
import PointGroup
import Preprocess
import Models
import Util
import PointNet

pgConfig :: PGCfg
pgConfig = PGCfg{
  clusterCfg = Cluster{
    clRadius = 0.03, ptThreshold = 50
  }
, scThres = (0.25, 0.75)
, nmsThres = 0.3
}

pgSpec :: PGSpec PNSpec PointNet
pgSpec = PGSpec{
  nDim = 3
, nClass = 20
, nFeature = 16
, backSpec = undefined -- TODO
, pgSpCfg = pgConfig
, scFeature = 16
, scBackSpec = undefined
}

-- Train configs
tsTrain :: TrainSet
tsTrain = TrainSet{
  train = True
, bnMomentum = 0.1
, bnEps = 1e-4
, learningRate = 1e-3
}
datTrain :: PGDataSet
datTrain = PGDataSet{
  nBatch = 4
, dataPath = undefined
, withGT = undefined
, maxNum = 150000
, defScale = 512
, posScale = 50
}

-- Test configs
tsTest :: TrainSet
tsTest = tsTrain{ train = False }

main :: IO ()
main = do
  l <- getArgs
  case l of
    [path, "--train"] -> do
      dPaths <- getDataPaths path "train"
      let dat = datTrain{ dataPath = dPaths }
      model <- trainModel @PointNet @PNSpec tsTrain dat pgSpec
      putStrLn "Done"
    [path, "--val"] -> do
      dPaths <- getDataPaths path "val"
      let dat = datTrain{ dataPath = dPaths }
      valModel @PointNet tsTest dat pgSpec
      putStrLn "Evaluated"
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
      putStrLn "--train for training"
      putStrLn "--val for evaluation"
      putStrLn "--prep=[train|val|test] for preparation"
      putStrLn "--link [target_path] to put dataset links into the target"

loadLast :: Parameterized m => FilePath -> m -> IO m
loadLast path model = do
  lists <- listDirectory path
  case filter ("model_" `isPrefixOf`) lists of
    [] -> pure model
    ls -> do
      let toLoad = maximum ls
      printf "Loading %s..\n" toLoad
      loadParams model toLoad

saveTo :: Parameterized m => FilePath -> Int -> m -> IO ()
saveTo path iter model = saveParams model (path </> printf "model_%d" iter)

pathResult :: FilePath
pathResult = "result"

trainModel :: (Parameterized n, BackSpec s, Backbone n, Randomizable s n)
  => TrainSet -> PGDataSet -> PGSpec s n -> IO (PointGroup n)
trainModel ts cfg spec = do
  init <- sample spec
  loaded <- loadLast pathResult init
  let optimizer = mkAdam 0 0.9 0.999 (flattenParameters loaded)
  let dataset = streamFromMap (datasetOpts 16) cfg{ withGT = True }
  let iteration model iter = trainLoop ts model optimizer iter . fst
  foldLoop loaded 300 $ \m -> runContT dataset . iteration m

trainLoop :: (Parameterized n, Backbone n, Optimizer o)
  => TrainSet -> PointGroup n -> o -> Int
  -> ListT IO (Irreg PGInput, Maybe PGGroundTruth) -> IO (PointGroup n)
trainLoop ts init opt iter = P.foldM step begin done . enumerateData
  where
    step (model, _, sumLoss) ((input, gt), phase) = do
      let (_, Just losses) = pointGroup ts model input gt
      let loss = case losses of
            PGLoss{..} -> lossSem + fst lossOff + snd lossOff + lossScore
      printf "Phase %d | Loss %f\n" phase (toDouble loss)
      -- Runs the optimizer
      (newModel, _) <- runStep model opt loss (learningRate ts)
      pure (newModel, succ phase, sumLoss + loss)
    begin = pure (init, 0, 0)
    done (model, nPhase, sumLoss) = do
      when (succ iter `mod` 10 == 0) $ saveTo pathResult iter model
      let total = 300 :: Int
      let avgLoss = toDouble sumLoss / fromIntegral nPhase
      printf "Iteration %d/%d | Avg Loss %f\n" iter total avgLoss
      putStrLn ""
      pure model

valModel :: (Parameterized n, BackSpec s, Backbone n, Randomizable s n)
  => TrainSet -> PGDataSet -> PGSpec s n -> IO ()
valModel ts cfg spec = do
  init <- sample spec -- Easier to initialize this way
  loaded <- loadLast pathResult init
  let dataset = streamFromMap (datasetOpts 16) cfg{ withGT = True }
  runContT dataset (go loaded . fst)
  where
    go model = P.foldM step begin done . enumerateData where
      step (_, sumAP) ((input, gt), phase) = do
        let (predicted, _) = pointGroup ts model input Nothing
        printf "Phase %d | AP50 \n" phase
        putStrLn ""
        -- TODO Data Accumulation
        pure (succ phase, sumAP)
      begin = pure (0, 0)
      done (nPhase, sumAP) = do
        -- TODO Data Print
        printf "AP50\n"
        putStrLn ""
