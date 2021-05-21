module Main where

import System.Environment (getArgs)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import Data.Set
import Control.Monad.Cont (ContT (..))
import Control.Monad (when)
import Text.Printf (printf)
import Pipes
import qualified Pipes.Prelude as P
import Torch as T hiding ( div )
import PointGroup
import Preprocess
import Util

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


data PGDataSet = PGDataSet{
  nBatch :: Int -- ^ Number of batches
  , dataPath :: V.Vector FilePath -- ^ Data path
  , withGT :: Bool -- ^ Does the data contain GT?
  , maxNum :: Int
  , nScale :: Float
}
instance Dataset IO PGDataSet Int (Irreg PGInput, Maybe PGGroundTruth) where
  -- | Gets item, merging the entries
  getItem PGDataSet{..} index = do
    let toFetch = V.toList $ V.take nBatch . V.drop (index * nBatch) $ dataPath
    pr <- traverse process toFetch
    let irreg = asTensor $ fst <$> pr -- TODO apply cumulative sum
    let inp = catInputs $ fst . snd <$> pr
    let gt = catGTs $ snd . snd <$> pr
    pure (Irreg irreg inp, if withGT then Just gt else Nothing)
    where
      process path = do
        verts <- readPVFromFile path
        noised <- SV.mapM (applyNoise nScale) verts
        -- TODO Apply Offset?
        let entry i = noised SV.! i
        sels <- cropRegion maxNum noised
        let insts = groupInstance noised sels
        let pgInp = PGInput{  -- MAYBE This is too slow
          pgPos = asTensor $ v3ToList . pCoord . entry <$> SV.toList sels
        , pgColor = asTensor $ v3ToList . pColor . entry <$> SV.toList sels
        }
        let pgGT = PGGroundTruth{
          gtSem = asTensor $ pSemLabel . entry <$> SV.toList sels
        , gtIns = undefined -- TODO Fix this
        }
        pure (SV.length sels, (pgInp, pgGT))

  keys PGDataSet{..} = fromList [0 .. (length dataPath - 1) `div` nBatch]

train :: (Parameterized n, BackSpec s, Backbone n, Randomizable s n)
  => PGSpec s n -> IO (PointGroup n)
train spec = do
  let optimizer = GD -- TODO < Need change ofc
  init <- sample spec
  let dataset = streamFromMap (datasetOpts 16) PGDataSet{ withGT = True }
  let step model = trainLoop undefined model optimizer . fst
  foldLoop init 300 $ \m _ -> runContT dataset . step $ m

trainLoop :: (Parameterized n, Backbone n, Optimizer o) => TrainSet
  -> PointGroup n -> o -> ListT IO (Irreg PGInput, Maybe PGGroundTruth) -> IO (PointGroup n)
trainLoop ts init opt = P.foldM step begin done . enumerateData
  where
    step (model, sumLoss) ((input, gt), iter) = do
      let (_, Just losses) = pointGroup ts model input gt
      let loss = case losses of
            PGLoss{..} -> lossSem + fst lossOff + snd lossOff + lossScore
      let log = iter > 0 && iter `mod` 50 == 0
      when log $ do
        printf "Iteration: %d | Loss: %s" iter (show $ sumLoss / 50)
      -- Runs the optimizer
      (newModel, _) <- runStep model opt loss 1e-3
      pure (newModel, (if log then 0 else sumLoss) + loss)
    begin = pure (init, 0)
    done = pure . fst
