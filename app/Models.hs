module Models where

import Data.Function ((&))
import GHC.Generics (Generic)
import Torch as T
import Util

-- TODO Write with COO

-- * Conventions
--
-- $Conventions
-- N : total number of points over several batches (sum nP)
-- nFeature : total number of features
-- nDim : dimension of coordinates (= 3)
-- nIns : number of instances
-- sum x : sum of x over batches

-- | Denotes an instance segmantation. Batches included
data Instances = Instances{
  insPts :: Tensor -- ^ Instance-sorted point indices, [sum nA]
  , insBegin :: Irreg Tensor -- ^ Instance begin offset, pad last, [sum nIns + 1]
  , insOfPt :: Irreg Tensor -- ^ Instance of a point, [N]
}
-- TODO instance construction from (insPts, insBegin)
insLOffset :: Instances -> Irreg Tensor
insLOffset Instances{..} = slice 0 0 (nIns-1) 1 <$> insBegin where nIns = size 0 (irregData insBegin)
insROffset :: Instances -> Irreg Tensor
insROffset Instances{..} = slice 0 1 nIns 1 <$> insBegin where nIns = size 0 (irregData insBegin)


data SemSpec = SemSpec Int Int -- ^ SemSpec nFeature nClass

-- |Semantic Segmentation
newtype Semantics = Semantics{ semFc1 :: Linear } deriving Generic
instance Parameterized Semantics
instance Randomizable SemSpec Semantics where
  sample (SemSpec nFeature nClass) = Semantics
    <$> sample (LinearSpec nFeature nClass)

-- | Semantic scores. F[N, nFeature] -> Sc[N, nFeature]
semScore :: Semantics -> Tensor -> Tensor
semScore Semantics{..} = linear semFc1

-- | Semantic labels. Sc[N, nFeature] -> S[N]
semLabel :: Tensor -> Tensor
semLabel = argmax (Dim 1) RemoveDim

-- | Average semantic loss. GT[N] -> Sc[N, nFeature] -> Loss
semLoss :: Tensor -> Tensor -> Tensor
semLoss gt = nllLoss' gt . logSoftmax (Dim 1)


data OffSpec = OffSpec Int Int -- ^ OffSpec nFeature nDim

-- |Offset Segmentation
newtype Offsets = Offsets{ offMlp :: MLP } deriving Generic
instance Parameterized Offsets
instance Randomizable OffSpec Offsets where
  sample (OffSpec nFeature dim) = Offsets
    <$> sample (MLPSpec [nFeature, nFeature, dim])

-- | Offset. F[N, nFeature] -> O[N, nDim]
offsetPred :: TrainSet -> Offsets -> Tensor -> Tensor
offsetPred ts Offsets{..} = applyMLP ts offMlp

-- | Centroid of instance. p[N, nDim] -> c[N, nDim]
centroid :: Instances -> Tensor -> Tensor
centroid ins p = ind (irregData $ insOfPt ins) insCen where
  ind = indexSelect 0
  ipos = ind (insPts ins) p -- instance positions in contiguous manner
  sums = cat (Dim 1) [zerosLike (slice 0 0 1 1 ipos), cumsum 0 Float ipos]

  insLeft = irregData $ insLOffset ins; insRight = irregData $ insROffset ins
  insS = ind insRight sums `T.sub` ind insLeft sums
  insN = insRight `T.sub` insLeft
  insCen = insS `T.div` insN

-- | L1 Regression Loss.
--
-- m[N] -> o[N, nDim] -> c[N, nDim] -> p[N, nDim] -> Loss
--
-- m: mask, o: offset, c: centroid, p: position
regLoss :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
regLoss m o c p = out where
  diff = o `sub` (c `sub` p)
  out = m `mul` normDim 1.0 (Dim 1) RemoveDim diff
    & sumDim (Dim 0) RemoveDim Float
    & divScalar (toDouble $ sumAll m)

-- | Direction Loss.
--
-- m[N] -> o[N, nDim] -> c[N, nDim] -> p[N, nDim] -> Loss
--
-- m: mask, o: offset, c: centroid, p: position
dirLoss :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
dirLoss m o c p = out where
  r = p `sub` c -- Negation of c-p
  s = normDim 2.0 (Dim 1) RemoveDim o `mul` normDim 2.0 (Dim 1) RemoveDim r
  cosL = sumDim (Dim 1) RemoveDim Float (o `mul` r) `T.div` s
  out = m `mul` cosL
    & sumDim (Dim 0) RemoveDim Float
    & divScalar (toDouble $ sumAll m)


-- | Clustering information
data Cluster = Cluster{
  clRadius :: Double
  , ptThreshold :: Int
} deriving Generic
instance Parameterized Cluster
-- | Clustering. No gradient. Irreg (sem[N], pos[N, nDim]) -> instances
cluster :: Cluster -> Irreg (Tensor, Tensor) -> Instances
cluster cl semPos = undefined where
  -- TODO write this without CUDA first

data ScSpec s n = ScSpec Int Int s Int -- ^ ScSpec nDim nFeat backSpec nHidden
data ScoreNet n = ScoreNet{
  scBack :: n, scFc1 :: Linear
} deriving Generic
instance Parameterized n => Parameterized (ScoreNet n)
instance (BackSpec s, Randomizable s n) => Randomizable (ScSpec s n) (ScoreNet n) where
  sample (ScSpec nDim nFeat backSpec nHidden) = ScoreNet
    <$> sample (modBackSpec (BackSpecBase (nDim + nFeat) nHidden) backSpec)
    <*> sample (LinearSpec nHidden 1)

-- | Predicts score of each instance.
-- ins -> pos[N, nDim] -> feat[N, nFeature] -> score[sum nIns]
insScore :: Backbone n => TrainSet -> ScoreNet n -> Instances -> Tensor -> Tensor -> Tensor
insScore ts ScoreNet{..} Instances{..} pos feat = score where
  -- Concat & Sort along the instances
  preF = indexSelect 0 insPts $ cat (Dim 2) [feat, pos] -- [sum nA, dim + nFeat]
  scores = backbone ts scBack (Irreg (irregData insBegin) preF) -- [sum nA, hidN]
  (feat, _) = irregMax scores -- [sum nIns, hidN]
  score = squeezeDim 2 . sigmoid . linear scFc1 $ feat -- [sum nIns]

-- TODO How to IoU per batch properly
-- | Calculates IoU for instances. No gradient.
-- insA -> insB -> IoU[batch, nIns_A, nIns_B]
calcIoU :: Instances -> Instances -> Tensor
calcIoU = undefined where

-- | Calcualtes score loss.
-- (thres-low, thres-high); ins -> gt -> score[sum nIns] -> Loss
scoreLoss :: (Float, Float) -> Instances -> Instances -> Tensor -> Tensor
scoreLoss (thL, thH) ins gt score = loss where
  -- TODO Proper max within IoU
  (iouMax, _) = maxDim (Dim 2) RemoveDim $ calcIoU ins gt -- iouMax[sum nIns]
  gtSc = clamp 0.0 1.0 $ divScalar (thH - thL) . subScalar thL $ iouMax
  loss = binaryCrossEntropyLoss' gtSc score

-- | Non-max suppresion. No gradient.
-- iouThreshold; ins -> score[sum nIns] -> G
nonMaxSup :: Float -> Instances -> Tensor -> Instances
nonMaxSup iouTh ins = undefined where
  selfIoU = calcIoU ins ins
