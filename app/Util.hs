{-# LANGUAGE TypeOperators #-}
module Util where

import Data.Foldable ( foldl' )
import GHC.Generics (Generic)
import Torch as T
import qualified Torch.Functional.Internal as TI
import Data.Array.Accelerate (
  Acc, Array, Scalar, Vector, Matrix, Segments, Exp, Z(..), (:.)(..)
  )
import qualified Data.Array.Accelerate as A
import Data.Int

-- | Train settings
data TrainSet = TrainSet{
  train :: Bool
  , bnMomentum :: Double -- ^ For batch nomalization
  , bnEps :: Double -- ^ For batch normalization
  , learningRate :: LearningRate
}

-- | Batch normalization, done on the 1st dimension (right after batch)
batchNormOn :: TrainSet -> BatchNorm -> Tensor -> Tensor
batchNormOn TrainSet{..} layer =
  batchNormForward layer train bnMomentum bnEps

-- | Batch normalization done on certain dimension <- Now redundant
batchNormDim :: TrainSet -> Dim -> BatchNorm -> Tensor -> Tensor
batchNormDim ts dim bn = trans . batchNormOn ts bn . trans
  where trans = T.transpose dim (Dim 1)

instance Parameterized BatchNorm where -- Somehow missing
  flattenParameters BatchNorm{..} = [batchNormWeight, batchNormBias]
  _replaceParameters bn = do
    weight <- nextParameter; bias <- nextParameter;
    return $ bn{ batchNormWeight = weight, batchNormBias = bias }

-- | Calculates p-norm over a dimension.
normDim :: Float -> Dim -> KeepDim -> Tensor -> Tensor
normDim p (Dim d) kd t = TI.normDim t p d (kd == KeepDim)

-- | Convolution 2d with default index, stride
conv2dDef :: Conv2d -> Tensor -> Tensor
conv2dDef layer = conv2dForward layer (1, 1) (0, 0)

newtype MLPSpec = MLPSpec [Int] -- Should have length >= 2
data MLP = MLP{ -- MLP layers with batch normalization & ReLu
  mlpLayers :: [(Linear, BatchNorm)]
  , mlpLast :: Linear
} deriving Generic
instance Parameterized MLP
instance Randomizable MLPSpec MLP where
  sample (MLPSpec [i, o]) = MLP [] <$> sample (LinearSpec i o)
  sample (MLPSpec (i : o : l)) = do
    layer <- (,) <$> sample (LinearSpec i o) <*> sample (BatchNormSpec o)
    next <- sample $ MLPSpec (o : l)
    return $ next{ mlpLayers = layer : mlpLayers next }
  sample _ = error "MLP list length is less than 2"

applyMLP :: TrainSet -> MLP -> Tensor -> Tensor
applyMLP ts MLP{..} inp = linear mlpLast $ foldl' (flip ($!)) inp (apply <$> mlpLayers)
  where apply (conv, bn) = relu . batchNormOn ts bn . linear conv

ignoreLabel :: Int32
ignoreLabel = -1

-- | Denotes irregular batches
data Irreg a = Irreg {
  batches :: Tensor -- ^ Denotes offset, [nBatch + 1]. Could be used several times
  , irregData :: a  -- ^ Denotes the data
} deriving Functor

-- How to support with Tensor?
-- Raw Torch. Max until .. does not work
-- TODO Batch-aware maximum, eliminating irregularity. (values, args). Act 1st dimension
irregMax :: Irreg Tensor -> (Tensor, Tensor)
irregMax = undefined where


-- | Base of backbone spec
data BackSpecBase = BackSpecBase Int Int -- ^ BackSpecBase nDim nBFeat

class BackSpec s where
  -- | Insert base backbone spec
  modBackSpec :: BackSpecBase -> s -> s

-- | Class for generalizing over backbone network
class Backbone n where
  -- | Backbone to extract the neighbor-aware per-point feature.
  --
  -- Irreg point[N, nDim] -> Irreg ptFeat[N, nFeature]
  backbone :: TrainSet -> n -> Irreg Tensor -> Irreg Tensor


-- Accelerate

-- Indices to segments
indToSeg :: Acc (Vector Int) -> Acc (Segments Int)
indToSeg = undefined

-- | Segmented matrix multiplication
segMultMat :: Acc (Vector Int)
  -> Acc (Array A.DIM3 Float) -> Acc (Matrix Float) -> Acc (Matrix Float)
segMultMat = undefined

-- TODO Do argmax?
-- | Segmented maximum, S[k] -> M[N, d] -> M[k, d]
segMax :: Acc (Segments Int) -> Acc (Matrix Float) -> Acc (Matrix Float)
segMax seg inp = A.transpose $ A.fold1Seg A.max (A.transpose inp) seg

-- | Segmented minimum, S[k] -> M[N, d] -> M[k, d]
segMin :: Acc (Segments Int) -> Acc (Matrix Float) -> Acc (Matrix Float)
segMin seg inp = A.transpose $ A.fold1Seg A.min (A.transpose inp) seg

-- | Segmented average, S[k] -> M[N, d] -> M[k, d]
segAvg :: Acc (Segments Int) -> Acc (Matrix Float) -> Acc (Matrix Float)
segAvg seg inp = A.transpose $ A.zipWith (/) ssum dupSeg where
  ssum = A.fold1Seg (+) (A.transpose inp) seg
  Z :. len :. _ = A.unlift (A.shape ssum) :: Z :. Exp Int :. Exp Int
  dupSeg = A.map fromIntegral $ A.replicate (A.lift $ Z :. len :. A.All) seg
-- TODO Segment differentials here. Differentials are of the same dimension (dL/dz)

clusterAcc = undefined where
  -- Ball query
  -- First partition with semantic labels
  -- Then, list all the stuff in radius in segmented vector
  -- When remaining, keep checking

