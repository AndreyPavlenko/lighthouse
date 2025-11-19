import torch
import torch.nn as nn

from lighthouse.ingress.torch.pipeline import Bufferization, DpasLayout, GpuKernelOutlining, Pipeline, TilingAndFusion, VectorToXegpu, Vectorization, XeGpu


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B.T)


DEV = "xpu"
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2


def get_inputs():
    A = torch.rand(M, K, device=DEV)
    B = torch.rand(N, K, device=DEV)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed


class MyPipeline(Pipeline):
    tile = TilingAndFusion("linalg.*").outer((256, 256)).inner((0, 0, 32), no_tile_no_fuse="linalg.fill")
    vectorize = Vectorization()
    bufferize = Bufferization()
    outline = GpuKernelOutlining(512)
    vecToXegpu = VectorToXegpu()
    dpas = DpasLayout()
    xegpu = XeGpu()


if __name__ == "__main__":
    model = Model()
    inputs = get_inputs()
    MyPipeline.apply(model, *inputs, dump=True)
