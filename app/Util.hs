module Util where

import Torch
import Torch.Initializers

data TrainSet = TrainSet{
  train :: Bool
  , momentum :: Double
  , eps :: Double
}

batchNormOn :: TrainSet -> BatchNorm -> Tensor -> Tensor
batchNormOn TrainSet{..} layer =
  batchNormForward layer train momentum eps

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

conv2dDef :: Conv2d -> Tensor -> Tensor
conv2dDef layer = conv2dForward layer (1, 1) (0, 0)
