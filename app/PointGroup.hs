module PointGroup where

import Torch as T
import Util
import Models
import GHC.Generics (Generic)

data PGInput = PGInput{
  pgColor :: Tensor -- ^ colors [N, nDim]
  , pgPos :: Tensor -- ^ positions [N, nDim]
}

data PGGroundTruth = PGGroundTruth{
  gtSem :: Vect -- ^ semantic labels [N]
  , gtIL :: Vect -- ^ instance labels [N]
  , gtIns :: Instances  -- ^ instances
}

-- | PointGroup Losses
data PGLoss = PGLoss {
  lossSem :: Loss
  , lossOff :: (Loss, Loss) -- ^ (loss_reg, loss_dir)
  , lossScore :: Loss
}

data PGCfg = PGCfg {
  clusterCfg :: Cluster
  , scThres :: (Float, Float)
  , nmsThres :: Float
} deriving Generic
instance Parameterized PGCfg
data PGSpec s n = PGSpec{
  nDim :: Int, nClass :: Int
  , nFeature :: Int
  , backSpec :: s
  , pgSpCfg :: PGCfg
  , scFeature :: Int -- ^ feature # for hidden layer
  , scBackSpec :: s
}
data PointGroup n = PointGroup{
  backNet :: n
  , semBranch :: Semantics
  , offBranch :: Offsets
  , pgCfg :: PGCfg
  , scoreNet :: ScoreNet n
} deriving Generic
instance Parameterized n => Parameterized (PointGroup n)
instance (BackSpec s, Randomizable s n) => Randomizable (PGSpec s n) (PointGroup n) where
  sample PGSpec{..} = PointGroup
    <$> sample (modBackSpec (BackSpecBase (nDim + nDim) nFeature) backSpec)
    <*> sample (SemSpec nFeature nClass)
    <*> sample (OffSpec nFeature nDim)
    <*> return pgSpCfg
    <*> sample (ScSpec nDim nFeature scBackSpec scFeature)

-- LATER Instead of Maybe, use a containment (Proxy & Identity)
-- | PointGroup. Pass Nothing for Ground Truth for tests
pointGroup :: Backbone n => TrainSet -> PointGroup n
  -> Irreg PGInput -> Maybe PGGroundTruth -> (Instances, Maybe PGLoss)
pointGroup ts PointGroup{..} pInput gt = (res, losses) where
  basis x = cat (Dim 2) [pgColor x, pgPos x]
  pos = pgPos <$> pInput

  -- Backbone
  ptFeat = backbone ts backNet $ basis <$> pInput

  -- Semantic branch
  semSc = semScore semBranch <$> ptFeat
  predSem = semLabel (irregData semSc)
  lossSem = do
    PGGroundTruth{..} <- gt
    return $ semLoss gtSem (irregData semSc)

  -- Offset branch
  predOff = offsetPred ts offBranch $ irregData ptFeat
  lossRegDir = do
    PGGroundTruth{..} <- gt
    let m = gtIL /=. asTensor ignoreLabel
    let pos = pgPos (irregData pInput)
    let cent = indexSelect 0 gtIL $ centroid gtIns pos -- get corresopnding instance
    return (regLoss m predOff cent pos, dirLoss m predOff cent pos)

  -- Clustering
  clusterWith = cluster (clusterCfg pgCfg) . fmap (predSem, )
  insP = clusterWith pos
  insQ = clusterWith $ (predOff +) <$> pos
  ins = mergeInst insP insQ

  -- ScoreNet
  score = insScore ts scoreNet ins (irregData pos) (irregData ptFeat)
  lossScore = do
    PGGroundTruth{..} <- gt
    return $ scoreLoss (scThres pgCfg) ins gtIns score

  -- Non-Max Suppression
  res = nonMaxSup (nmsThres pgCfg) ins score
  losses = PGLoss <$> lossSem <*> lossRegDir <*> lossScore
