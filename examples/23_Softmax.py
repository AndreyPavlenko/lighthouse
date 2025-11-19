import torch
import torch.nn as nn

from lighthouse.ingress.torch.pipeline import Bufferization, Pipeline, TilingAndFusion, Vectorization

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)

device="xpu"
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device=device)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed

class MyPipeline(Pipeline):
    tile = TilingAndFusion("linalg.*").outer((256, 256)).inner((0, 0, 32), no_tile_no_fuse="linalg.fill")
    # vectorize = Vectorization()
    # bufferize = Bufferization()

if __name__ == "__main__":
    model = Model()
    inputs = get_inputs()
    MyPipeline.apply(model, *inputs, dump=True)