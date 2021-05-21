import torch
import spconv
#from lib.pointgroup_ops.functions import pointgroup_ops

torch.jit.script(spconv.SparseConv3d(32, 64, 3))
#torch.jit.script(pointgroup_ops.BFSCluster())
