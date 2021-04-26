module UNet where

import Data.Function ((&))
import Torch as T
import Util

data PNSpec = PNSpec{
  nDim :: Int
  , nFeature :: Int
}

-- PointNet global feature implementation (Need more edit)
data PointNet = PointNet{
  conv1 :: Conv1d, conv1bn :: BatchNorm
  , conv2 :: Conv1d, conv2bn :: BatchNorm
  , conv3 :: Conv1d, conv3bn :: BatchNorm
  , fc1 :: Linear, fc1bn :: BatchNorm
  , fc2 :: Linear, fc2bn :: BatchNorm
  , fc3 :: Linear
}

instance Randomizable PNSpec PointNet where
  sample PNSpec{..} = PointNet
    <$> sample (Conv1dSpec nDim 64 1)
    <*> sample (BatchNormSpec 64)
    <*> sample (Conv1dSpec 64 128 1)
    <*> sample (BatchNormSpec 128)
    <*> sample (Conv1dSpec 128 1024 1)
    <*> sample (BatchNormSpec 1024)
    <*> sample (LinearSpec 1024 512)
    <*> sample (BatchNormSpec 512)
    <*> sample (LinearSpec 512 256)
    <*> sample (BatchNormSpec 256)
    <*> sample (LinearSpec 256 nFeature)

-- | PointNet. [batch-size, nPoint, nDim] -> [batch-size, nFeature]
pointNet :: TrainSet -> PointNet -> Tensor -> Tensor
pointNet ts PointNet{..} inp = outp where
  -- Convolutions, [b, n, dim] -> [b, 1024, n]
  conved = inp
    & T.transpose (Dim 2) (Dim 1)
    & conv1dDef conv1 & batchNormOn ts conv1bn & relu
    & conv1dDef conv2 & batchNormOn ts conv2bn & relu
    & conv1dDef conv3 & batchNormOn ts conv3bn & relu
  -- Max pooling, [b, 1024, n] -> [b, 1024]
  pooled = pooled
    & fst . T.maxDim (Dim 2) KeepDim
    & view [-1, 1024]
  -- Feature summary, b * 1024 -> b * K
  outp = pooled
    & linear fc1 & batchNormOn ts fc1bn & relu
    & linear fc2 & batchNormOn ts fc2bn & relu
    & linear fc3

