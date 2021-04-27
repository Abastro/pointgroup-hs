module Models where

import Data.Function ((&))
import Torch as T
import Torch.Functional.Internal ( normDim, instance_norm, nll_loss )
import Util

data Instances = Instances{
  insPts :: Tensor -- ^ List of point indices, [batch, nActive]
  , insBegin :: Tensor -- ^ Instance begin offset (last = nActive). [batch, nIns]
  , insOfPt :: Tensor -- ^ Instance of a point, [batch, nP]
}

data SemSpec = SemSpec{
  semFeature :: Int
  , semClass :: Int
}
-- |Semantic Segmentation
data Semantics = Semantics {
  semFc1 :: Linear
}
instance Randomizable SemSpec Semantics where
  sample SemSpec{..} = Semantics
    <$> sample (LinearSpec semFeature semClass)

-- | Semantic scores. F[batch, nP, nFeature] -> Sc[batch, nP, nFeature]
semScore :: Semantics -> Tensor -> Tensor
semScore Semantics{..} = linear semFc1

-- | Sc[batch, nP, nFeature] -> S[batch, nP]
semLabel :: Tensor -> Tensor
semLabel = argmax (Dim 1) RemoveDim

-- | Semantic loss. GT[batch, nP] -> Sc[batch, nP, nFeature] -> Loss
semLoss :: Tensor -> Tensor -> Tensor
semLoss gt = nllLoss' gt . logSoftmax (Dim 2)


data OffSpec = OffSpec {
  offFeature :: Int
  , offDim :: Int
}
-- |Offset Segmentation
newtype Offsets = Offsets{ offMlp :: MLP }
instance Randomizable OffSpec Offsets where
  sample OffSpec{..} = Offsets
    <$> sample (MLPSpec [offFeature, offFeature, offDim])

-- | Offset. F[batch, nP, nFeature] -> O[batch, nP, nDim]
offsetPred :: TrainSet -> Offsets -> Tensor -> Tensor
offsetPred ts Offsets{..} = applyMLP ts offMlp

-- | Centroid of instance. p[batch, nP, nDim] -> c[batch, nP, nDim]
centroid :: Instances -> Tensor -> Tensor
centroid Instances{..} p = ind insOfPt insCen where
  ind = indexSelect 1
  ipos = ind insPts p -- instance positions in contiguous manner
  padZ = zerosLike (slice 1 0 1 1 ipos)
  sums = cat (Dim 1) [padZ, cumsum 1 Float ipos]

  nIns = size 1 insBegin
  slleft = slice 1 0 (nIns-1) 1; slright = slice 1 1 nIns 1;
  insS = ind (slright insBegin) sums `T.sub` ind (slleft insBegin) sums
  insN = slright insBegin `T.sub` slleft insBegin
  insCen = insS `T.div` insN

-- | L1 Regression Loss.
--
-- m[batch, nP]
-- -> o[batch, nP, nDim]
-- -> c[batch, nP, nDim]
-- -> p[batch, nP, nDim]
-- -> Loss
--
-- m: mask, o: offset, c: centroid, p: position
regLoss :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
regLoss m o c p = out where
  diff = o `sub` (c `sub` p)
  normed = normDim diff 1.0 2 False
  out = m `mul` normed
    & sumDim (Dim 1) RemoveDim Float
    & divScalar (toDouble $ sumAll m)

-- | Direction Loss.
--
-- m[batch, nP]
-- -> o[batch, nP, nDim]
-- -> c[batch, nP, nDim]
-- -> p[batch, nP, nDim]
-- -> Loss
--
-- m: mask, o: offset, c: centroid, p: position
dirLoss :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
dirLoss m o c p = out where
  r = p `sub` c -- Negation of c-p
  s = normDim o 2.0 2 False `mul` normDim r 2.0 2 False
  cosL = sumDim (Dim 2) RemoveDim Float (o `mul` r) `T.div` s
  out = m `mul` cosL
    & sumDim (Dim 1) RemoveDim Float
    & divScalar (toDouble $ sumAll m)


data Cluster = Cluster{
  clRadius :: Double
  , ptThreshold :: Int
}

-- | Clustering. sem[batch, nP] -> crd[batch, nP, 3] -> instances
cluster :: Cluster -> Tensor -> Tensor -> Instances
cluster cl sem crd = undefined where
  -- TODO write this without CUDA first


data ScSpec s n = ScSpec{
  scBSpec :: s
  , nscHidden :: Int
}
data ScoreNet n = ScoreNet{
  scBack :: n
  , scFc1 :: Linear
}
instance Randomizable s n => Randomizable (ScSpec s n) (ScoreNet n) where
  sample ScSpec{..} = ScoreNet
    <$> sample scBSpec
    <*> sample (LinearSpec nscHidden 1)

-- | Predicts score of each instance.
-- ins -> p[batch, nP, dim] -> feat[batch, nP, nFeat] -> score[batch, nIns, 1]
insScore :: Backbone n => ScoreNet n -> Instances -> Tensor -> Tensor -> Tensor
insScore ScoreNet{..} ins p feat = score where
  preF = cat (Dim 2) [feat, p]
  -- TODO Backbone here, need to do cluster-wise feature processing
  -- TODO Cluster-aware Max-pooling
  -- [batch, nIns, nP, hidN]
  (feat, _) = maxDim (Dim 2) RemoveDim undefined -- [batch, nIns, hidN]
  score = sigmoid . linear scFc1 $ feat -- [batch, nIns, 1] -> remove 2nd dim

-- | Calcualtes score loss.
-- (thres-low, thres-high); ins -> gt -> score[batch, nIns] -> Loss
scoreLoss :: (Float, Float) -> Instances -> Instances -> Tensor -> Tensor
scoreLoss (thL, thH) ins gt score = loss where
  iou = undefined -- [batch, nIns]
  gtSc = divScalar (thH - thL) . subScalar thL $ iou
  loss = binaryCrossEntropyLoss' gtSc score

-- TODO G is not this, need to fix!
-- | Non-max suppresion.
-- iouThreshold; ins -> sc[batch, nIns] -> G[batch, fin]
nonMaxSup :: Double -> Instances -> Tensor -> Tensor
nonMaxSup iouTh ins = undefined
