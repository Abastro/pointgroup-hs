module PointNet where

import GHC.Generics ( Generic )
import Data.Foldable ( foldl' )
import Data.Function ((&))
import Torch as T
import Util

data PNSpec = PNSpec Int Int -- ^ PNSpec nDim nFeature

-- PointNet per-point feature implementation
data PointNet = PointNet{
  mkFeat :: Linear, mkFeatBn :: BatchNorm
  , glmlp :: MLP
  , ptmlp :: MLP
} deriving Generic
instance Parameterized PointNet
instance Randomizable PNSpec PointNet where
  sample (PNSpec nDim nFeature) = PointNet
    <$> sample (LinearSpec nDim 64) <*> sample (BatchNormSpec 64)
    <*> sample (MLPSpec [64, 128, 1024])
    <*> sample (MLPSpec [1088, 512, 256, 128, nFeature])

-- TODO implement irregular batch-wise.
--      relevant part: "global feature max pooling" & "pos/feature transform"
-- | PointNet. [batch, nP, dim] -> [batch, nP, nFeature]
pointNet :: TrainSet -> PointNet -> Tensor -> Tensor
pointNet ts PointNet{..} inp = outp where
  -- Feature generation, [b, n, dim] -> [b, n, 64]
  inFeat = inp
    -- TODO pos transform here
    & relu . batchNormDim ts (Dim 1) mkFeatBn . linear mkFeat
    -- TODO feature transform here
  -- Convolutions & Maxpool, [b, n, 64] -> [b, 1, 1024]
  (pooled, _) = inFeat
    & applyMLP ts glmlp
    & maxDim (Dim 1) KeepDim
  -- Per-point feature primer, [b, n, 1088]
  pPoint = cat (Dim 1) [inFeat, T.repeat [1, T.size 1 inFeat, 1] pooled]

  -- Feature summary, [b, n, 1088] -> [b, n, K]
  outp = applyMLP ts ptmlp pPoint

instance Backbone PointNet where
  backbone ts pn = fmap $ pointNet ts pn
instance BackSpec PNSpec where
  modBackSpec _ = undefined -- TODO
