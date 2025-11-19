import torch
from lighthouse.ingress.torch.pipeline import Bufferization, DpasLayout, GpuKernelOutlining, Pipeline, TilingAndFusion, VectorToXegpu, Vectorization, XeGpu

mlir = """
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x4096xf32> to tensor<4096x4096xf32>
    %3 = linalg.matmul ins(%0, %1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>) outs(%2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %4 = tensor.empty() : tensor<4096x4096xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %6 = linalg.max ins(%3, %5 : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    bufferization.materialize_in_destination %6 in restrict writable %arg2 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    return
  }
"""


class MatmulPipeline(Pipeline):
    tile = TilingAndFusion("linalg.*").outer((256, 256)).inner((0, 0, 32), tile_only="linalg.matmul")
    vectorize = Vectorization()
    bufferize = Bufferization()
    outline = GpuKernelOutlining(512)
    vecToXegpu = VectorToXegpu()
    dpas = DpasLayout()
    xegpu = XeGpu()


if __name__ == "__main__":
    a = torch.tensor([[2.0]*4096]*4096, dtype=torch.float16, device="xpu")
    b = torch.tensor([[2.0]*4096]*4096, dtype=torch.float16, device="xpu")
    c = torch.zeros(4096, 4096, dtype=torch.float32, device="xpu")
    MatmulPipeline.exec(mlir, a, b, c, dump=True)
    print(c)