module Preprocess where

import Data.String ( IsString(..) )
import GHC.Exts ( IsList(..) )
import Data.List ( isSuffixOf )
import Data.Maybe ( fromMaybe )
import Data.Foldable ( fold, for_, traverse_ )
import qualified Data.Sequence as Seq
import qualified Data.IntSet as IS
import Data.IntMap ( IntMap )
import qualified Data.IntMap.Strict as IM
import qualified Data.Vector.Generic as GV
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import Data.Vector.Storable.ByteString
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as BL
import System.Random
import Control.Concurrent.Async.Pool ( mapConcurrently, withTaskGroup )
import Control.Exception
import Control.Monad ( when, unless )
import Control.Applicative ( Applicative(..) )
import GHC.Generics ( Generic )
import Data.Data ( Typeable )
import qualified Torch as T

import Text.Printf
import System.FilePath ( (</>) )
import System.Directory
import Foreign.Storable.Generic
import PLY
import PLY.Types
import Data.Aeson
import Util
import Models
import PointGroup

-- | Vector of dimension 3
data Vec3 a = Vec3{
  vx :: !a, vy :: !a, vz :: !a
} deriving (Show, Eq, Functor, Foldable, Traversable, Generic)
instance Storable a => GStorable (Vec3 a)
instance Applicative Vec3 where
  pure t = Vec3 t t t
  liftA2 = zipWithV

vecToV3 :: V.Vector a -> Vec3 a
vecToV3 vec = Vec3 (vec V.! 0) (vec V.! 1) (vec V.! 2)
v3ToList :: Vec3 a -> [a]
v3ToList (Vec3 x y z) = [x, y, z]

zipWithV :: (a -> b -> c) -> Vec3 a -> Vec3 b -> Vec3 c
zipWithV fn (Vec3 x y z) (Vec3 x' y' z') = Vec3 (x `fn` x') (y `fn` y') (z `fn` z')

flipX :: Num a => Vec3 a -> Vec3 a
flipX (Vec3 x y z) = Vec3 (-x) y z

rotateZ :: Floating a => a -> Vec3 a -> Vec3 a
rotateZ angle (Vec3 x y z) = Vec3 (x*c - y*s) (x*s + y*c) z where
  c = cos angle; s = sin angle


data Vertex = Vertex{
  vCoord :: !(Vec3 Float) -- ^ Coordinates, centroid ~ 0
  , vColor :: !(Vec3 Float) -- ^ RGB Color in [-1, 1]
  , vSemLabel :: !(Maybe Int32) -- ^ Semantic labels
} deriving Show
newtype Segments = Segments (V.Vector Int32) deriving Show
instance FromJSON Segments where
  parseJSON = withObject "Segments" $ \v -> Segments <$> v .: fromString "segIndices"
data AggGroup = AggGroup{
  id :: !Int
  , objectId :: !Int32
  , segments :: !(SV.Vector Int)
  , label :: !String
} deriving (Generic, Show)
instance FromJSON AggGroup
newtype Aggregation = Aggregation (V.Vector AggGroup)
instance FromJSON Aggregation where
  parseJSON = withObject "Aggregation" $ \v -> Aggregation <$> v .: fromString "segGroups"

-- | Processed Vertex Data
data ProcVert = ProcVert{
  pCoord :: !(Vec3 Float)   -- ^ Coordinates, centroid ~ 0
  , pColor :: !(Vec3 Float) -- ^ RGB Color in [-1, 1]
  , pSemLabel :: !Int32       -- ^ Semantic labels, missing = -1
  , pInsLabel :: !Int32       -- ^ Instance labels, missing = -1
} deriving (Show, Generic)
instance GStorable ProcVert
type ProcVerts = SV.Vector ProcVert

writePVToFile :: FilePath -> ProcVerts -> IO ()
writePVToFile p = B.writeFile p . vectorToByteString
readPVFromFile :: FilePath -> IO ProcVerts
readPVFromFile p = byteStringToVector <$> B.readFile p


-- |Data Contract Exception
data DataContractException a =
  DataNotEqual String a a
  | DataNotUnique String [a]
  deriving (Show, Typeable)
instance (Show a, Typeable a) => Exception (DataContractException a)

sceneList :: FilePath -> String -> IO [String]
sceneList srcP specName = lines <$> readFile (srcP </> specName)

-- | Links the scannet dataset for use
--
-- [@srcP@]: parent path of the dataset
-- [@specName@]: name of file which specifies the scenes
-- [@scanName@]: name of scan directory
-- [@tarP@]: name of target directory to link
-- [@split@]: name of the split directory in the target
linkScannet :: FilePath -> String -> String -> FilePath -> String -> IO ()
linkScannet path specName scanName tarP split = do
  printf "Symlinking in %s for split %s..\n" tarP split
  scenes <- sceneList path specName
  traverse_ linkScene scenes
  where
    linkScene scName = do
      scSrc <- canonicalizePath $ path </> scanName </> scName
      let scTar = tarP </> split
      lists <- listDirectory scSrc
      let link fn = do
            removeFile (scTar </> fn)
            createFileLink (scSrc </> fn) (scTar </> fn)
      traverse_ link lists

-- | Process the scannet dataset
--
-- [@path@]: parent path of the dataset
-- [@test@]: `True` for test
-- [@specName@]: name of file which specifies the scenes
-- [@scanName@]: name of scan directory
-- [@tarName@]: name of target directory to write to
handleScannet :: FilePath -> Bool -> String -> String -> String -> IO ()
handleScannet path test specName scanName tarName = do
  scenes <- sceneList path specName
  let parse = if test then parseTest else parseTrain
  let handleScene scene = parse scene >>= writeToFile scene
  withTaskGroup 16 $ \g -> () <$ mapConcurrently g handleScene scenes
  where
    writeToFile scName pv = do
      createDirectoryIfMissing True (path </> tarName)
      let pvPath = path </> tarName </> scName <> ".dat"
      writePVToFile pvPath pv
      printf "[%s] Processed\n" scName

    parseTest scName = do
      let scDir = path </> scanName </> scName
      printf "[%s] Reading\n" scName
      verts <- readPlyVerts False (scDir </> scName <> filePt)
      let asPV Vertex{..} = ProcVert{
        pCoord = vCoord
        , pColor = vColor
        , pSemLabel = ignoreLabel
        , pInsLabel = ignoreLabel }
      (return $! SV.convert $ asPV <$> verts) <* printf "[%s] Parsed\n" scName

    parseTrain scName = do
      let scDir = path </> scanName </> scName
      printf "[%s] Reading\n" scName
      verts <- readPlyVerts True (scDir </> scName <> fileLbPt)
      Right (Segments segs) <-
        eitherDecode <$> BL.readFile (scDir </> scName <> fileSeg)
      Right (Aggregation aggs) <-
        eitherDecode <$> BL.readFile (scDir </> scName <> fileAgg)
      let fixedAggs = if scName == "scene0217_00"
          then V.take (V.length aggs `div` 2) aggs else aggs

      let numV = V.length verts
      let numS = V.length segs
      when (numV /= numS) $ throw
        $ DataNotEqual (lenErr scName) numV numS

      -- TODO Label check To discard unneeded object
      let segToInst AggGroup{..} = IM.fromSet (const objectId) . IS.fromList $ SV.toList segments
      let seg_Insts = IM.unionsWith (dupInsErr scName) $ segToInst <$> fixedAggs
      -- No instance for the segment: Denoted by -1 (Perhaps -100?)
      let getInst seg = fromMaybe ignoreLabel $ seg_Insts IM.!? seg

      let semLabel = fromMaybe intErr . vSemLabel <$> verts
      let insLabel = getInst . fromIntegral <$> segs

      -- TODO Perform this faster
      -- Check if points of each instance attains the same semantic label
      let insToSems = V.zipWith IM.singleton
            (fromIntegral <$> insLabel) (IS.singleton . fromIntegral <$> semLabel)
      let insToSemDup = IM.filter ((> 1) . IS.size) $ IM.unionsWith (<>) insToSems
      for_ (IM.lookupMin insToSemDup)
        $ \(l, d) -> throw $ DataNotUnique (nonUniErr scName l) (IS.toAscList d)

      let asPV v sem ins = ProcVert{
        pCoord = vCoord v
        , pColor = vColor v
        , pSemLabel = sem
        , pInsLabel = ins }

      (return $! SV.convert $ V.zipWith3 asPV verts semLabel insLabel)
        <* printf "[%s] Parsed, length %d\n" scName numV

    intErr = error "Internal error"
    lenErr scene = printf "[%s] Expected seg-length == #vert" scene
    nonUniErr scene lab = printf "[%s] Non-unique semantic labels for instance %s" scene lab
    dupInsErr scene i j = error $ printf "[%s] Duplicate instance %d vs %d" scene i j

    filePt = "_vh_clean_2.ply"
    fileLbPt = "_vh_clean_2.labels.ply"
    fileSeg = "_vh_clean_2.0.010000.segs.json"
    fileAgg = ".aggregation.json"

readPlyVerts :: Bool -> FilePath -> IO (V.Vector Vertex)
readPlyVerts withLabel path = do
  Right lbPt <- loadElements (fromString "vertex") path
  let verts = mkVert <$> lbPt
  let len = fromIntegral $ V.length verts
  let sumCoord = V.foldl1' (zipWithV (+)) $ vCoord <$> verts
  -- TODO label has to go through remapper (WHY?)
  return $ fitVert ((/ len) <$> sumCoord) <$> verts
  where
    fitVert avgCoord v = v{
        vCoord = zipWithV (-) (vCoord v) avgCoord
        , vColor = subtract 1 . ( / 127.5) <$> vColor v
      }
    mkVert = Vertex <$> vecToV3 . coordOf
      <*> vecToV3 . fmap fromIntegral . colorOf
      <*> if withLabel then Just . fromIntegral . labelOf else const Nothing
    coordOf vert = unsafeUnwrap @Float <$> V.slice 0 3 vert
    colorOf vert = unsafeUnwrap @Word8 <$> V.slice 3 3 vert
    labelOf vert = case vert V.! 7 of Sushort label -> label

-- | Get data paths in the data directory
getDataPaths :: FilePath -> String -> IO (V.Vector FilePath)
getDataPaths path tarName = do
  paths <- listDirectory (path </> tarName)
  return . V.fromList $ filter (".dat" `isSuffixOf`) paths

-- | Denotes config for the dataset
data PGDataSet = PGDataSet{
  nBatch :: Int     -- ^ Number of batches
  , dataPath :: V.Vector FilePath -- ^ Data path
  , withGT :: Bool  -- ^ Does the data contain GT?
  , maxNum :: Int   -- ^ Max number allowed for a scene
  , posScale :: Float -- ^ Scale of each position
  , defScale :: Float -- ^ Default scale to go for when cropping, after scaling
}
instance T.Dataset IO PGDataSet Int (Irreg PGInput, Maybe PGGroundTruth) where
  -- | Keys for the dataset
  keys PGDataSet{..} = fromList [0 .. (length dataPath - 1) `div` nBatch]
  -- | Gets item, merging the entries
  getItem PGDataSet{..} index = do
    let toFetch = toList $ V.take nBatch . V.drop (index * nBatch) $ dataPath
    pr <- traverse process toFetch
    let irreg = T.asTensor @[Int32] . cumsum $ fromIntegral . fst <$> pr
    let inp = catInputs $ fst . snd <$> pr
    let gt = catGTs irreg $ snd . snd <$> pr
    pure (Irreg irreg inp, if withGT then Just gt else Nothing)
    where
      process path = do
        verts <- readPVFromFile path
        appNoise <- applyNoise
        -- MAYBE also perform elastic distortion?
        -- MAYBE jitter?
        let noised = SV.map (applyScale posScale . appNoise) verts
        let moved = SV.map (subOffset $ minVec noised) noised
        sels <- cropRegion maxNum defScale moved
        let entry i = moved SV.! i
        let insts = groupInstance moved sels
        let instList = IM.elems insts
        let instLens = fromIntegral . length <$> instList
        let selList = SV.toList sels
        let pgInp = PGInput{  -- MAYBE This is too slow
          pgPos = T.asTensor $ v3ToList . pCoord . entry <$> selList
        , pgColor = T.asTensor $ v3ToList . pColor . entry <$> selList
        }
        let pgIns = Instances{
          insPts = T.asTensor @[Int32] . toList $ fromIntegral <$> mconcat instList
        , insBegin = Irreg undefined . T.asTensor @[Int32] $ instLens -- gives length here
        } -- instance of single item, so undefined
        let pgGT = PGGroundTruth{
          gtSem = T.asTensor $ pSemLabel . entry <$> selList
        , gtIL = T.asTensor $ packedLabels insts moved
        , gtIns = pgIns
        }
        pure (SV.length sels, (pgInp, pgGT))

      catBatch = T.cat (T.Dim 0)
      catInputs inps = PGInput{
        pgPos = catBatch $ pgPos <$> inps
      , pgColor = catBatch $ pgColor <$> inps
      }
      catGTs irreg gts = PGGroundTruth{
        gtSem = catBatch $ gtSem <$> gts
      , gtIL = catBatch $ gtIL <$> gts
      , gtIns = catInsts irreg $ gtIns <$> gts
      }
      catInsts irreg insts = Instances{
        insPts = catBatch $ insPts <$> insts
      , insBegin = Irreg (T.asTensor @[Int32] . cumsum $ bLens) begins
      } where
        begins = T.cumsum 0 T.Int32 . catBatch $ 0 : (irregData . insBegin <$> insts)
        bLens = fromIntegral . T.size 0 . irregData . insBegin <$> insts

      cumsum = scanl (+) 0 -- Lightweight cumulative sum
      minVec :: ProcVerts -> Vec3 Float
      minVec vs = minimum <$> sequenceA pos where pos = pCoord <$> SV.toList vs
      maxVec :: ProcVerts -> Vec3 Float
      maxVec vs = maximum <$> sequenceA pos where pos = pCoord <$> SV.toList vs

      -- |Apply noise to the vertex
      applyNoise :: IO (ProcVert -> ProcVert)
      applyNoise = do
        rf <- randomIO
        let refl = if rf then flipX else Prelude.id
        ang <- randomRIO (0.0, 2 * pi)
        pure $ \pv -> pv{ pCoord = refl . rotateZ ang $ pCoord pv }

      subOffset :: Vec3 Float -> ProcVert -> ProcVert
      subOffset off pv = pv{ pCoord = zipWithV subtract off $ pCoord pv }
      applyScale :: Float -> ProcVert -> ProcVert
      applyScale sc pv = pv{ pCoord = (sc *) <$> pCoord pv }

      -- | Selects appropriate region and gives the indices to fit in the number
      cropRegion :: Int -> Float -> ProcVerts -> IO (SV.Vector Int)
      cropRegion maxNum defScale verts =
        if SV.length verts <= maxNum then pure indices else eval defScale
        where
          sceneSize = maxVec verts
          indices = SV.enumFromN 0 (SV.length verts)
          inRange low high x = low <= x && x < high
          allInR ls hs xs = and $ inRange <$> ls <*> hs <*> xs
          eval scale = do
            let randUpto t = randomRIO (0.0, max 0.001 (t - scale))
            ls <- traverse randUpto (maxVec verts)
            let hs = (scale +) <$> ls
            let cond = allInR ls hs . pCoord . (verts SV.!)
            let inds = SV.filter cond indices
            if SV.length inds <= maxNum then pure inds else eval (scale - 32)

      -- | Groups vertex indices of each instance, label |-> point_indices
      groupInstance :: ProcVerts -> SV.Vector Int -> IM.IntMap (Seq.Seq Int)
      groupInstance vs = IM.unionsWith (<>) . map single . SV.toList where
        single ind
          | lab == ignoreLabel = IM.empty
          | otherwise = IM.singleton (fromIntegral lab) (pure ind)
          where lab = pInsLabel $ vs SV.! ind

      -- | Infers packed labels from grouped instances (..)
      packedLabels :: IM.IntMap a -> ProcVerts -> [Int32]
      packedLabels insts vs = pack . pInsLabel <$> SV.toList vs where
        packMap = IM.unions $ zipWith IM.singleton (IM.keys insts) [0..]
        pack prev = fromMaybe ignoreLabel $ packMap IM.!? fromIntegral prev
