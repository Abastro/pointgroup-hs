module Util where

import Data.Foldable ( foldl' )
import Torch as T
import Torch.Initializers

data TrainSet = TrainSet{
  train :: Bool
  , momentum :: Double
  , eps :: Double
}

-- | Batch normalization, done on the 1st dimension (right after batch)
batchNormOn :: TrainSet -> BatchNorm -> Tensor -> Tensor
batchNormOn TrainSet{..} layer =
  batchNormForward layer train momentum eps

-- | Batch normalization done on certain dimension
batchNormDim :: TrainSet -> Dim -> BatchNorm -> Tensor -> Tensor
batchNormDim ts dim bn = trans . batchNormOn ts bn . trans
  where trans = T.transpose dim (Dim 1)

{-
data Conv1dSpec = Conv1dSpec{
  inputSize :: Int
  , outputSize :: Int
  , kernelSize :: Int
}
-- |Convolution 1d
data Conv1d = Conv1d{
  conv1dWeight :: Parameter
  , conv1dBias :: Parameter
}
instance Randomizable Conv1dSpec Conv1d where
  sample Conv1dSpec{..} = do
    let sizes = [outputSize, inputSize, kernelSize]
    w <- makeIndependent
      =<< kaimingUniform FanIn (LeakyRelu $ Prelude.sqrt (5.0 :: Float)) sizes
    init <- randIO' [outputSize]
    let bound = 1.0 / Prelude.sqrt (
          fromIntegral (getter FanIn $ calculateFan sizes) :: Float )
    b <- makeIndependent =<< pure (subScalar bound $ mulScalar (bound * 2.0) init)
    return $ Conv1d w b

conv1dForward :: Conv1d -> Int -> Int -> Tensor -> Tensor
conv1dForward layer = conv1d' w b where
  w = toDependent (conv1dWeight layer)
  b = toDependent (conv1dBias layer)

conv1dDef :: Conv1d -> Tensor -> Tensor
conv1dDef layer = conv1dForward layer 1 0
-}

-- | Convolution 2d with default index, stride
conv2dDef :: Conv2d -> Tensor -> Tensor
conv2dDef layer = conv2dForward layer (1, 1) (0, 0)

newtype MLPSpec = MLPSpec [Int] -- Should have length >= 2
data MLP = MLP{ -- MLP layers with batch normalization & ReLu
  mlpLayers :: [(Linear, BatchNorm)]
  , mlpLast :: Linear
}
instance Randomizable MLPSpec MLP where
  sample (MLPSpec [i, o]) = MLP [] <$> sample (LinearSpec i o)
  sample (MLPSpec (i : o : l)) = do
    layer <- (,) <$> sample (LinearSpec i o) <*> sample (BatchNormSpec o)
    next <- sample $ MLPSpec (o : l)
    return $ next{ mlpLayers = layer : mlpLayers next }
  sample _ = error "MLP list length is less than 2"

applyMLP :: TrainSet -> MLP -> Tensor -> Tensor
applyMLP ts MLP{..} inp = linear mlpLast $ foldl' (flip ($!)) inp (apply <$> mlpLayers)
  -- [batch, P, I] -> [batch, P, O]
  where apply (conv, bn) = relu . batchNormDim ts (Dim 2) bn . linear conv

-- | Class for generalizing over backbone network
class Backbone n where
  -- | Backbone to extract the neighbor-aware per-point feature.
  --
  -- point[batch, nPoint, nDim] -> ptFeat[batch, nP, nFeature]
  backbone :: TrainSet -> n -> Tensor -> Tensor
