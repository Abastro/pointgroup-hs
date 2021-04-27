module PointGroup where

import Torch as T
import Util
import Models
import PointNet

data PGInput = PGInput{
  pgColor :: Tensor -- ^ colors [batch, nP, 3]
  , pgPos :: Tensor -- ^ positions [batch, nP, 3]
}
data PGGroundTruth = PGGroundTruth{
  gtSem :: Tensor       -- ^ semantic labels [batch, nP]
  , gtIns :: Instances  -- ^ instances
}
-- | PointGroup Losses
data PGLoss = PGLoss {
  lossSem :: Tensor
  , lossOff :: (Tensor, Tensor) -- ^ (loss_reg, loss_dir)
  , lossScore :: Tensor
}

data PGSpec = PGSpec{
  backSpec :: PNSpec
  , semSpec :: SemSpec
  , offSpec :: OffSpec
  , scoreSpec :: ScSpec PNSpec PointNet
}
data PointGroup = PointGroup{
  backNet :: PointNet
  , semBranch :: Semantics
  , offBranch :: Offsets
  , scoreNet :: ScoreNet PointNet
}
instance Randomizable PGSpec PointGroup where
  sample PGSpec{..} = PointGroup
    <$> sample backSpec
    <*> sample semSpec
    <*> sample offSpec
    <*> sample scoreSpec

-- | PointGroup. pass Nothing for Ground Truth for tests
pointGroup :: TrainSet -> PointGroup -> PGInput -> Maybe PGGroundTruth -> (Tensor, Maybe PGLoss)
pointGroup ts PointGroup{..} PGInput{..} gt = (res, losses) where
  inp = cat (Dim 2) [pgColor, pgPos]
  ptFeat = backbone ts backNet inp
  -- Semantic branch
  semSc = semScore semBranch ptFeat
  predSem = semLabel semSc
  lossSem = do
    PGGroundTruth{..} <- gt
    return $ semLoss gtSem semSc

  -- Offset branch
  predOff = offsetPred ts offBranch ptFeat
  lossRegDir = do
    PGGroundTruth{..} <- gt
    let m = undefined
    let cent = centroid gtIns pgPos
    let reg = regLoss m predOff cent pgPos
    let dir = dirLoss m predOff cent pgPos
    return (reg, dir)

  -- Clustering
  ins = cluster undefined predSem pgPos

  -- ScoreNet
  score = insScore scoreNet ins pgPos ptFeat
  lossScore = do
    PGGroundTruth{..} <- gt
    return $ scoreLoss (0.25, 0.75) ins gtIns score

  -- Non-Max Suppression
  res = nonMaxSup 0.3 ins score
  losses = PGLoss <$> lossSem <*> lossRegDir <*> lossScore
