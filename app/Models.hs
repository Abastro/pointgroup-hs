module Models where

import Data.Function ((&))
import Torch as T
import Torch.Functional.Internal ( normDim )

data SemSpec = SemSpec{
  nFeature :: Int
  , nClass :: Int
}
-- |Semantic Segmentation
data Semantics = Semantics {
  fc1 :: Linear, fc1bn :: BatchNorm
  , fc2 :: Linear, fc2bn :: BatchNorm
}

-- |Offset Segmentation
newtype Offsets = Offsets Parameter -- This is so wrong..
offset :: Offsets -> Tensor
offset (Offsets p) = toDependent p

-- | Offset. p[batch, nP, nDim] -> q[batch, nP, nDim]
applyOffset :: Offsets -> Tensor -> Tensor
applyOffset off = add (offset off)

-- | Regularization Loss.
--
-- m[batch, nP]
-- -> c[batch, nP, nDim]
-- -> p[batch, nP, nDim]
-- -> Loss
--
-- m: mask, o: offset, c: centroid, p: position
regLoss :: Offsets -> Tensor -> Tensor -> Tensor -> Tensor
regLoss off m c p = out where
  o = offset off
  diff = o `sub` (c `sub` p)
  normed = normDim diff 2.0 2 False
  out = m `mul` normed
    & sumDim (Dim 1) RemoveDim Float
    & divScalar (toDouble $ sumAll m)

-- | Direction Loss.
--
-- m[batch, nP]
-- -> c[batch, nP, nDim]
-- -> p[batch, nP, nDim]
-- -> Loss
--
-- m: mask, o: offset, c: centroid, p: position
dirLoss :: Offsets -> Tensor -> Tensor -> Tensor -> Tensor
dirLoss off m c p = undefined where
  o = offset off
  r = p `sub` c -- Negated of c-p
  s = normDim o 2.0 2 False `mul` normDim r 2.0 2 False
  cosL = sumDim (Dim 2) RemoveDim Float (o `mul` r) `T.div` s
  out = m `mul` cosL
    & sumDim (Dim 1) RemoveDim Float
    & divScalar (toDouble $ sumAll m)


data ClusterSpec = ClusterSpec{
  clRadius :: Double
  , ptThreshold :: Int
}

data Cluster = Cluster{
}

cluster :: Cluster -> Tensor -> Tensor
cluster cl inp = undefined where
  -- TODO write this, perhaps without CUDA first

-- TODO Voxelize, MLP => Sigmoid
-- TODO Non-Maximum Suppression
