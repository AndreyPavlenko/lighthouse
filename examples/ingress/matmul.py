import torch
import torch.nn as nn
from torch import Tensor

from mlir import ir
from lighthouse.ingress.torch.pipeline import Bufferization, DpasLayout, GpuKernelOutlining, Pipeline, TilingAndFusion, VectorToXegpu, Vectorization, XeGpu

DEV="xpu"
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


class MatmulModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tensor:
        return torch.matmul(A, B)



def get_inputs():
    A = torch.rand(M, K, device=DEV)
    B = torch.rand(N, K, device=DEV)
    return [A, B]


class MatmulPipeline(Pipeline):
    tile = TilingAndFusion("linalg.*").outer((256, 256)).inner((0, 0, 32), no_tile_no_fuse="linalg.fill")
    vectorize = Vectorization()
    bufferize = Bufferization()
    outline = GpuKernelOutlining(512)
    vecToXegpu = VectorToXegpu()
    dpas = DpasLayout()
    xegpu = XeGpu()


if __name__ == "__main__":
    ctx = ir.Context()
    mod = MatmulModule()
    a = torch.tensor([[2.0]*4096]*4096, dtype=torch.float32, device="xpu")
    b = torch.tensor([[2.0]*4096]*4096, dtype=torch.float32, device="xpu")
    c = torch.zeros(4096, 4096, dtype=torch.float32, device="xpu")
    MatmulPipeline.exec(mod, a, b, c, dump=True)
    print(c)