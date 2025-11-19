import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import match, split_handle, apply_patterns, cse, Bufferization, GpuKernelOutlining, Phase, Pipeline, TilingAndFusion, VectorToXegpu, Vectorization, XeGpu, XegpuLayout


from typing import override
import torch
import torch.nn as nn
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import linalg
from mlir.dialects.transform.xegpu import SetDescLayoutOp
from mlir.dialects.transform import ApplyCanonicalizationPatternsOp
from mlir.dialects.transform.structured import ApplyFoldUnitExtentDimsViaReshapesPatternsOp


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


device = "xpu"
batch_size = 4096
dim = 256
# dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device=device)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed


class Layout(XegpuLayout):

    @override
    def apply(self, ctx: Phase.Context):
        for nd_tdesc in split_handle(match(ctx.gpu_func, ("xegpu.create_nd_tdesc")), 2):
            SetDescLayoutOp(nd_tdesc, (1, 1), (1, 256), inst_data=(1, 16))


class SoftmaxPipeline(Pipeline):
    tile = TilingAndFusion((1, ), "linalg.*")  #.tile((0, 256), "linalg.generic#0#3", loop="for", reduction=True)
    vectorize = Vectorization()
    bufferize = Bufferization()
    outline = GpuKernelOutlining(sg_size=16)
    vecToXegpu = VectorToXegpu()
    layout = Layout()
    xegpu = XeGpu()

    @classmethod
    def to_mlir_module(cls, target, *args, ir_context=None, **kwargs):
        mlir = super().to_mlir_module(target, *args, ir_context=ir_context, **kwargs)
        print(mlir)

        # Rewrite the first generic and remove the second output (keep max computation only)
        op = next(op for op in mlir.body.operations[0].regions[0].blocks[0].operations if op.name == "linalg.generic")
        with ir.InsertionPoint(op), op.location:
            f32_type = ir.F32Type.get()
            max = linalg.GenericOp(
                (op.operands[1].type, ),
                inputs=op.operands[:1],
                outputs=op.operands[1:2],
                indexing_maps=ir.ArrayAttr.get(list(op.attributes["indexing_maps"])[:2]),
                iterator_types=op.attributes["iterator_types"],
            )
            body = max.regions[0].blocks.append(f32_type, f32_type)
            with ir.InsertionPoint(body):
                linalg.YieldOp((arith.MaximumFOp(body.arguments[0], body.arguments[1]).result, ))
        op.results[0].replace_all_uses_with(max.results[0])

        with cls.transform_module(mlir) as mod:
            func = match(mod, "func.func")
            with apply_patterns(func):
                # Remove tensor.expand_shape
                ApplyFoldUnitExtentDimsViaReshapesPatternsOp()
                ApplyCanonicalizationPatternsOp()
            cse(func)
        return mlir


if __name__ == "__main__":
    model = Model()
    inputs = get_inputs()
    output = torch.zeros_like(inputs[0])
    # SoftmaxPipeline.apply(model, *inputs, dump=True)
    SoftmaxPipeline.exec(model, *inputs, output, dump=True)
    print(f"output: {output}")
    expected = model(*inputs)
    print(f"expected: {expected}")
    torch.testing.assert_close(output, expected)
