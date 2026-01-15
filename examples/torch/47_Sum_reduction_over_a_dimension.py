import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import match, split_handle, apply_patterns, cse, canonicalize, Bufferization, GpuKernelOutlining, Phase, Pipeline, TilingAndFusion, VectorToXegpu, Vectorization, XeGpu, XegpuLayout

from typing import override
import torch
import torch.nn as nn
from mlir import ir
from mlir.dialects import arith, linalg, math, tensor
from mlir.dialects.transform import AnyOpType, ApplyCanonicalizationPatternsOp, PrintOp
from mlir.dialects.transform.structured import ApplyFoldUnitExtentDimsViaReshapesPatternsOp, FuseIntoContainingOp, TileUsingForallOp, TileReductionUsingForOp
from mlir.dialects.transform.xegpu import SetDescLayoutOp, SetOpLayoutAttrOp
from mlir.dialects.transform.loop import LoopFuseSiblingOp


class Model(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return torch.sum(x, dim=self.dim, keepdim=True)


device = "xpu"
batch_size = 128
dim1 = 4096
dim2 = 4096
reduce_dim = 1


def get_inputs():
    # x = torch.rand(dim1, dim2)
    x = torch.rand(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    return [reduce_dim]


class Layout(Phase):

    @override
    def apply(self, ctx: Phase.Context):
        gpu_func = ctx.gpu_func
        nd_tdesc = split_handle(match(gpu_func, "xegpu.create_nd_tdesc"), 2)
        # collapse_shape = split_handle(match(gpu_func, "memref.extract_aligned_pointer_as_index"), 3)
        SetDescLayoutOp(nd_tdesc[0], (1, 1), (4, 64), inst_data=(4, 16))
        SetDescLayoutOp(nd_tdesc[1], (1,), (64,), inst_data=(16,))
        SetOpLayoutAttrOp(match(gpu_func, "xegpu.load"), (1,), (64,), inst_data=(16,), index=1)
        SetOpLayoutAttrOp(match(gpu_func, "vector.multi_reduction"), (1, 1), (4, 64), inst_data=(1, 16), index=0, slice_dims=(0,), result=True)
        # SetOpLayoutAttrOp(match(gpu_func, "xegpu.store_nd"), (1, 64), (4, 64), inst_data=(1, 16), index=0, slice_dims=(0,))
        # SetOpLayoutAttrOp(match(gpu_func, "xegpu.store_nd"), (1,), (64,), inst_data=(16,), slice_dims=(1,), index=0)
        # SetOpLayoutAttrOp(collapse_shape[0], (1, 1), (4, 64), inst_data=(4, 16), index=0, result=True)
        # SetOpLayoutAttrOp(collapse_shape[1], (1,), (64,), inst_data=(16,), index=0, result=True)

        # for nd_tdesc in split_handle(match(gpu_func, ("xegpu.create_nd_tdesc")), 4):
        #     SetDescLayoutOp(nd_tdesc, (1, 1), (1, 256), inst_data=(1, 16))
        # PrintOp(target=gpu_func, name="After setting layouts")
        # cse(gpu_func)
        # canonicalize(gpu_func)
        # for op in split_handle(match(gpu_func, ("scf.yield")), 2):
        #     SetOpLayoutAttrOp(op, (1, 1), (1, 256), inst_data=(1, 16), index=0, result=True)
        # for op in split_handle(match(gpu_func, ("xegpu.load_nd", )), 3):
        #     SetOpLayoutAttrOp(op, (1, 1), (1, 256), inst_data=(1, 16), index=0, slice_dims=(1,), result=True)
        # for op in split_handle(match(gpu_func, ("xegpu.store_nd",)), 1):
        #     SetOpLayoutAttrOp(op, (1, 1), (1, 256), inst_data=(1, 16), index=0, slice_dims=(1,))


class SumReductionPipeline(Pipeline):
    tile = TilingAndFusion().tile((1, 4, 64), "linalg.generic")#.tile((0, 4, 0), "linalg.generic", loop="forall", reduction=True)
    vectorize = Vectorization()
    bufferize = Bufferization()
    outline = GpuKernelOutlining(1024)
    vecToXegpu = VectorToXegpu()
    layout = Layout()
    xegpu = XeGpu()


if __name__ == "__main__":
    model = Model(reduce_dim)
    inputs = get_inputs()
    output = torch.zeros_like(inputs[0])
    # SumReductionPipeline.tile(model, *inputs, dump=True)
    SumReductionPipeline.apply(model, *inputs, dump=True)
