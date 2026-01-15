#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @main(%arg0: tensor<4096x393216xf32>) -> tensor<4096x393216xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<4096xi64>) -> tensor<4096xi64>
    %2 = tensor.empty() : tensor<4096xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4096xf32>) -> tensor<4096xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4096x393216xf32>) outs(%3, %1 : tensor<4096xf32>, tensor<4096xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %12 = linalg.index 1 : index
      %13 = arith.index_cast %12 : index to i64
      %14 = arith.maximumf %in, %out : f32
      %15 = arith.cmpf ogt, %in, %out : f32
      %16 = arith.select %15, %13, %out_1 : i64
      linalg.yield %14, %16 : f32, i64
    } -> (tensor<4096xf32>, tensor<4096xi64>)
    %expanded = tensor.expand_shape %4#0 [[0, 1]] output_shape [4096, 1] : tensor<4096xf32> into tensor<4096x1xf32>
    %5 = tensor.empty() : tensor<4096x393216xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %expanded : tensor<4096x393216xf32>, tensor<4096x1xf32>) outs(%5 : tensor<4096x393216xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.subf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<4096x393216xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<4096x393216xf32>) outs(%5 : tensor<4096x393216xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = math.exp %in : f32
      linalg.yield %12 : f32
    } -> tensor<4096x393216xf32>
    %8 = tensor.empty() : tensor<4096x1xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<4096x1xf32>) -> tensor<4096x1xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<4096x393216xf32>) outs(%9 : tensor<4096x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.addf %in, %out : f32
      linalg.yield %12 : f32
    } -> tensor<4096x1xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %10 : tensor<4096x393216xf32>, tensor<4096x1xf32>) outs(%5 : tensor<4096x393216xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.divf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<4096x393216xf32>
    return %11 : tensor<4096x393216xf32>
  }
}

[[[ IR printer: Initial ]]]
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%arg0: tensor<4096x393216xf32>) -> tensor<4096x393216xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4096x393216xf32>) outs(%1 : tensor<4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.maximumf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<4096xf32>
    %3 = tensor.empty() : tensor<4096x393216xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<4096x393216xf32>, tensor<4096xf32>) outs(%3 : tensor<4096x393216xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.subf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<4096x393216xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<4096x393216xf32>) outs(%3 : tensor<4096x393216xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = math.exp %in : f32
      linalg.yield %9 : f32
    } -> tensor<4096x393216xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<4096x393216xf32>) outs(%6 : tensor<4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<4096xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %7 : tensor<4096x393216xf32>, tensor<4096xf32>) outs(%3 : tensor<4096x393216xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.divf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<4096x393216xf32>
    return %8 : tensor<4096x393216xf32>
  }
}

[[[ IR printer: tile ]]]
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%arg0: tensor<4096x393216xf32>) -> tensor<4096x393216xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = tensor.empty() : tensor<4096x393216xf32>
    %2 = scf.forall (%arg1) in (4096) shared_outs(%arg2 = %1) -> (tensor<4096x393216xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [1, 393216] [1, 1] : tensor<4096x393216xf32> to tensor<1x393216xf32>
      %extracted_slice_1 = tensor.extract_slice %0[%arg1] [1] [1] : tensor<4096xf32> to tensor<1xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%extracted_slice_1 : tensor<1xf32>) -> tensor<1xf32>
      %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice : tensor<1x393216xf32>) outs(%3 : tensor<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %10 = arith.maximumf %in, %out : f32
        linalg.yield %10 : f32
      } -> tensor<1xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%arg1, 0] [1, 393216] [1, 1] : tensor<4096x393216xf32> to tensor<1x393216xf32>
      %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %4 : tensor<1x393216xf32>, tensor<1xf32>) outs(%extracted_slice_2 : tensor<1x393216xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %10 = arith.subf %in, %in_3 : f32
        linalg.yield %10 : f32
      } -> tensor<1x393216xf32>
      %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x393216xf32>) outs(%extracted_slice_2 : tensor<1x393216xf32>) {
      ^bb0(%in: f32, %out: f32):
        %10 = math.exp %in : f32
        linalg.yield %10 : f32
      } -> tensor<1x393216xf32>
      %7 = linalg.fill ins(%cst_0 : f32) outs(%extracted_slice_1 : tensor<1xf32>) -> tensor<1xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<1x393216xf32>) outs(%7 : tensor<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %10 = arith.addf %in, %out : f32
        linalg.yield %10 : f32
      } -> tensor<1xf32>
      %9 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %8 : tensor<1x393216xf32>, tensor<1xf32>) outs(%extracted_slice_2 : tensor<1x393216xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %10 = arith.divf %in, %in_3 : f32
        linalg.yield %10 : f32
      } -> tensor<1x393216xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %9 into %arg2[%arg1, 0] [1, 393216] [1, 1] : tensor<1x393216xf32> into tensor<4096x393216xf32>
      }
    }
    return %2 : tensor<4096x393216xf32>
  }
}

[[[ IR printer: vectorize ]]]
module {
  func.func @main(%arg0: tensor<4096x393216xf32>) -> tensor<4096x393216xf32> {
    %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    %0 = ub.poison : f32
    %cst_0 = arith.constant dense<0xFF800000> : vector<1xf32>
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<4096x393216xf32>
    %2 = scf.forall (%arg1) in (4096) shared_outs(%arg2 = %1) -> (tensor<4096x393216xf32>) {
      %3 = vector.transfer_read %arg0[%arg1, %c0], %0 {in_bounds = [true, true]} : tensor<4096x393216xf32>, vector<1x393216xf32>
      %4 = vector.multi_reduction <maximumf>, %3, %cst_0 [1] : vector<1x393216xf32> to vector<1xf32>
      %extracted_slice = tensor.extract_slice %arg2[%arg1, 0] [1, 393216] [1, 1] : tensor<4096x393216xf32> to tensor<1x393216xf32>
      %5 = vector.broadcast %4 : vector<1xf32> to vector<1x393216xf32>
      %6 = arith.subf %3, %5 : vector<1x393216xf32>
      %7 = math.exp %6 : vector<1x393216xf32>
      %8 = vector.multi_reduction <add>, %7, %cst [1] : vector<1x393216xf32> to vector<1xf32>
      %9 = vector.broadcast %8 : vector<1xf32> to vector<1x393216xf32>
      %10 = arith.divf %7, %9 : vector<1x393216xf32>
      %11 = vector.transfer_write %10, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x393216xf32>, tensor<1x393216xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %11 into %arg2[%arg1, 0] [1, 393216] [1, 1] : tensor<1x393216xf32> into tensor<4096x393216xf32>
      }
    }
    return %2 : tensor<4096x393216xf32>
  }
}

[[[ IR printer: bufferize ]]]
module {
  func.func @main(%arg0: memref<4096x393216xf32>, %arg1: memref<4096x393216xf32>) {
    %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    %0 = ub.poison : f32
    %cst_0 = arith.constant dense<0xFF800000> : vector<1xf32>
    %c0 = arith.constant 0 : index
    scf.forall (%arg2) in (4096) {
      %1 = vector.transfer_read %arg0[%arg2, %c0], %0 {in_bounds = [true, true]} : memref<4096x393216xf32>, vector<1x393216xf32>
      %2 = vector.multi_reduction <maximumf>, %1, %cst_0 [1] : vector<1x393216xf32> to vector<1xf32>
      %3 = vector.broadcast %2 : vector<1xf32> to vector<1x393216xf32>
      %4 = arith.subf %1, %3 : vector<1x393216xf32>
      %5 = math.exp %4 : vector<1x393216xf32>
      %6 = vector.multi_reduction <add>, %5, %cst [1] : vector<1x393216xf32> to vector<1xf32>
      %7 = vector.broadcast %6 : vector<1xf32> to vector<1x393216xf32>
      %8 = arith.divf %5, %7 : vector<1x393216xf32>
      vector.transfer_write %8, %arg1[%arg2, %c0] {in_bounds = [true, true]} : vector<1x393216xf32>, memref<4096x393216xf32>
    }
    return
  }
}

[[[ IR printer: outline ]]]
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<4096x393216xf32>, %arg1: memref<4096x393216xf32>) {
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c4096, %c1, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4096x393216xf32>, %arg1 : memref<4096x393216xf32>)
    return
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<4096x393216xf32>, %arg1: memref<4096x393216xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4096, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %c0 = arith.constant 0 : index
      %0 = ub.poison : f32
      %cst = arith.constant dense<0xFF800000> : vector<1xf32>
      %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
      %1 = vector.transfer_read %arg0[%block_id_x, %c0], %0 {in_bounds = [true, true]} : memref<4096x393216xf32>, vector<1x393216xf32>
      %2 = vector.multi_reduction <maximumf>, %1, %cst [1] : vector<1x393216xf32> to vector<1xf32>
      %3 = vector.broadcast %2 : vector<1xf32> to vector<1x393216xf32>
      %4 = arith.subf %1, %3 : vector<1x393216xf32>
      %5 = math.exp %4 : vector<1x393216xf32>
      %6 = vector.multi_reduction <add>, %5, %cst_0 [1] : vector<1x393216xf32> to vector<1xf32>
      %7 = vector.broadcast %6 : vector<1xf32> to vector<1x393216xf32>
      %8 = arith.divf %5, %7 : vector<1x393216xf32>
      vector.transfer_write %8, %arg1[%block_id_x, %c0] {in_bounds = [true, true]} : vector<1x393216xf32>, memref<4096x393216xf32>
      gpu.return
    }
  }
}

[[[ IR printer: vecToXegpu ]]]
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<4096x393216xf32>, %arg1: memref<4096x393216xf32>) {
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c4096, %c1, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4096x393216xf32>, %arg1 : memref<4096x393216xf32>)
    return
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<4096x393216xf32>, %arg1: memref<4096x393216xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4096, 1, 1>} {
      %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<1xf32>
      %block_id_x = gpu.block_id  x
      %0 = xegpu.create_nd_tdesc %arg0 : memref<4096x393216xf32> -> !xegpu.tensor_desc<1x393216xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %1 = xegpu.load_nd %0[%block_id_x, 0]  : !xegpu.tensor_desc<1x393216xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x393216xf32>
      %2 = vector.multi_reduction <maximumf>, %1, %cst_0 [1] : vector<1x393216xf32> to vector<1xf32>
      %3 = vector.broadcast %2 : vector<1xf32> to vector<1x393216xf32>
      %4 = arith.subf %1, %3 : vector<1x393216xf32>
      %5 = math.exp %4 : vector<1x393216xf32>
      %6 = vector.multi_reduction <add>, %5, %cst [1] : vector<1x393216xf32> to vector<1xf32>
      %7 = vector.broadcast %6 : vector<1xf32> to vector<1x393216xf32>
      %8 = arith.divf %5, %7 : vector<1x393216xf32>
      %9 = xegpu.create_nd_tdesc %arg1 : memref<4096x393216xf32> -> !xegpu.tensor_desc<1x393216xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %8, %9[%block_id_x, 0]  : vector<1x393216xf32>, !xegpu.tensor_desc<1x393216xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}







[[[ IR printer: Initial ]]]
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%arg0: tensor<4096x393216xf32>) -> tensor<4096x393216xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4096x393216xf32>) outs(%1 : tensor<4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.maximumf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<4096xf32>
    %3 = tensor.empty() : tensor<4096x393216xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<4096x393216xf32>, tensor<4096xf32>) outs(%3 : tensor<4096x393216xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.subf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<4096x393216xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<4096x393216xf32>) outs(%3 : tensor<4096x393216xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = math.exp %in : f32
      linalg.yield %9 : f32
    } -> tensor<4096x393216xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<4096x393216xf32>) outs(%6 : tensor<4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<4096xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %7 : tensor<4096x393216xf32>, tensor<4096xf32>) outs(%3 : tensor<4096x393216xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %9 = arith.divf %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<4096x393216xf32>
    return %8 : tensor<4096x393216xf32>
  }
}

[[[ IR printer: tile ]]]
#map = affine_map<()[s0] -> (s0 floordiv 2048)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%arg0: tensor<4096x393216xf32>) -> tensor<4096x393216xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = tensor.empty() : tensor<4096x393216xf32>
    %2 = scf.forall (%arg1) in (4096) shared_outs(%arg2 = %1) -> (tensor<4096x393216xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [1, 393216] [1, 1] : tensor<4096x393216xf32> to tensor<1x393216xf32>
      %extracted_slice_1 = tensor.extract_slice %0[%arg1] [1] [1] : tensor<4096xf32> to tensor<1xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%extracted_slice_1 : tensor<1xf32>) -> tensor<1xf32>
      %4 = tensor.empty() : tensor<1x192xf32>
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x192xf32>) -> tensor<1x192xf32>
      %6 = scf.forall (%arg3) = (0) to (393216) step (2048) shared_outs(%arg4 = %5) -> (tensor<1x192xf32>) {
        %13 = affine.apply #map()[%arg3]
        %extracted_slice_4 = tensor.extract_slice %extracted_slice[0, %arg3] [1, 2048] [1, 1] : tensor<1x393216xf32> to tensor<1x2048xf32>
        %extracted_slice_5 = tensor.extract_slice %arg4[0, %13] [1, 1] [1, 1] : tensor<1x192xf32> to tensor<1xf32>
        %14 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_4 : tensor<1x2048xf32>) outs(%extracted_slice_5 : tensor<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.maximumf %in, %out : f32
          linalg.yield %15 : f32
        } -> tensor<1xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %14 into %arg4[0, %13] [1, 1] [1, 1] : tensor<1xf32> into tensor<1x192xf32>
        }
      }
      %reduced = linalg.reduce ins(%6 : tensor<1x192xf32>) outs(%3 : tensor<1xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %13 = arith.maximumf %in, %init : f32
          linalg.yield %13 : f32
        }
      %extracted_slice_2 = tensor.extract_slice %arg2[%arg1, 0] [1, 393216] [1, 1] : tensor<4096x393216xf32> to tensor<1x393216xf32>
      %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %reduced : tensor<1x393216xf32>, tensor<1xf32>) outs(%extracted_slice_2 : tensor<1x393216xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %13 = arith.subf %in, %in_4 : f32
        linalg.yield %13 : f32
      } -> tensor<1x393216xf32>
      %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<1x393216xf32>) outs(%extracted_slice_2 : tensor<1x393216xf32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = math.exp %in : f32
        linalg.yield %13 : f32
      } -> tensor<1x393216xf32>
      %9 = linalg.fill ins(%cst_0 : f32) outs(%extracted_slice_1 : tensor<1xf32>) -> tensor<1xf32>
      %10 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1x192xf32>) -> tensor<1x192xf32>
      %11 = scf.forall (%arg3) = (0) to (393216) step (2048) shared_outs(%arg4 = %10) -> (tensor<1x192xf32>) {
        %13 = affine.apply #map()[%arg3]
        %extracted_slice_4 = tensor.extract_slice %8[0, %arg3] [1, 2048] [1, 1] : tensor<1x393216xf32> to tensor<1x2048xf32>
        %extracted_slice_5 = tensor.extract_slice %arg4[0, %13] [1, 1] [1, 1] : tensor<1x192xf32> to tensor<1xf32>
        %14 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_4 : tensor<1x2048xf32>) outs(%extracted_slice_5 : tensor<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.addf %in, %out : f32
          linalg.yield %15 : f32
        } -> tensor<1xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %14 into %arg4[0, %13] [1, 1] [1, 1] : tensor<1xf32> into tensor<1x192xf32>
        }
      }
      %reduced_3 = linalg.reduce ins(%11 : tensor<1x192xf32>) outs(%9 : tensor<1xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %13 = arith.addf %in, %init : f32
          linalg.yield %13 : f32
        }
      %12 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%8, %reduced_3 : tensor<1x393216xf32>, tensor<1xf32>) outs(%extracted_slice_2 : tensor<1x393216xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %13 = arith.divf %in, %in_4 : f32
        linalg.yield %13 : f32
      } -> tensor<1x393216xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %12 into %arg2[%arg1, 0] [1, 393216] [1, 1] : tensor<1x393216xf32> into tensor<4096x393216xf32>
      }
    }
    return %2 : tensor<4096x393216xf32>
  }
}

[[[ IR printer: vectorize ]]]
#map = affine_map<()[s0] -> (s0 floordiv 2048)>
module {
  func.func @main(%arg0: tensor<4096x393216xf32>) -> tensor<4096x393216xf32> {
    %cst = arith.constant dense<0.000000e+00> : vector<1x192xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
    %0 = ub.poison : f32
    %cst_1 = arith.constant dense<0xFF800000> : vector<1x192xf32>
    %cst_2 = arith.constant dense<0xFF800000> : vector<1xf32>
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<4096x393216xf32>
    %2 = scf.forall (%arg1) in (4096) shared_outs(%arg2 = %1) -> (tensor<4096x393216xf32>) {
      %3 = tensor.empty() : tensor<1x192xf32>
      %4 = vector.transfer_write %cst_1, %3[%c0, %c0] {in_bounds = [true, true]} : vector<1x192xf32>, tensor<1x192xf32>
      %5 = scf.forall (%arg3) = (0) to (393216) step (2048) shared_outs(%arg4 = %4) -> (tensor<1x192xf32>) {
        %20 = affine.apply #map()[%arg3]
        %extracted_slice_3 = tensor.extract_slice %arg4[0, %20] [1, 1] [1, 1] : tensor<1x192xf32> to tensor<1xf32>
        %21 = vector.transfer_read %arg0[%arg1, %arg3], %0 {in_bounds = [true, true]} : tensor<4096x393216xf32>, vector<1x2048xf32>
        %22 = vector.transfer_read %arg4[%c0, %20], %0 {in_bounds = [true]} : tensor<1x192xf32>, vector<1xf32>
        %23 = vector.multi_reduction <maximumf>, %21, %22 [1] : vector<1x2048xf32> to vector<1xf32>
        %24 = vector.transfer_write %23, %extracted_slice_3[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %24 into %arg4[0, %20] [1, 1] [1, 1] : tensor<1xf32> into tensor<1x192xf32>
        }
      }
      %6 = vector.transfer_read %5[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x192xf32>, vector<1x192xf32>
      %7 = vector.multi_reduction <maximumf>, %6, %cst_2 [1] : vector<1x192xf32> to vector<1xf32>
      %extracted_slice = tensor.extract_slice %arg2[%arg1, 0] [1, 393216] [1, 1] : tensor<4096x393216xf32> to tensor<1x393216xf32>
      %8 = vector.transfer_read %arg0[%arg1, %c0], %0 {in_bounds = [true, true]} : tensor<4096x393216xf32>, vector<1x393216xf32>
      %9 = vector.broadcast %7 : vector<1xf32> to vector<1x393216xf32>
      %10 = arith.subf %8, %9 : vector<1x393216xf32>
      %11 = math.exp %10 : vector<1x393216xf32>
      %12 = vector.transfer_write %11, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x393216xf32>, tensor<1x393216xf32>
      %13 = vector.transfer_write %cst, %3[%c0, %c0] {in_bounds = [true, true]} : vector<1x192xf32>, tensor<1x192xf32>
      %14 = scf.forall (%arg3) = (0) to (393216) step (2048) shared_outs(%arg4 = %13) -> (tensor<1x192xf32>) {
        %20 = affine.apply #map()[%arg3]
        %extracted_slice_3 = tensor.extract_slice %arg4[0, %20] [1, 1] [1, 1] : tensor<1x192xf32> to tensor<1xf32>
        %21 = vector.transfer_read %12[%c0, %arg3], %0 {in_bounds = [true, true]} : tensor<1x393216xf32>, vector<1x2048xf32>
        %22 = vector.transfer_read %arg4[%c0, %20], %0 {in_bounds = [true]} : tensor<1x192xf32>, vector<1xf32>
        %23 = vector.multi_reduction <add>, %21, %22 [1] : vector<1x2048xf32> to vector<1xf32>
        %24 = vector.transfer_write %23, %extracted_slice_3[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %24 into %arg4[0, %20] [1, 1] [1, 1] : tensor<1xf32> into tensor<1x192xf32>
        }
      }
      %15 = vector.transfer_read %14[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x192xf32>, vector<1x192xf32>
      %16 = vector.multi_reduction <add>, %15, %cst_0 [1] : vector<1x192xf32> to vector<1xf32>
      %17 = vector.broadcast %16 : vector<1xf32> to vector<1x393216xf32>
      %18 = arith.divf %11, %17 : vector<1x393216xf32>
      %19 = vector.transfer_write %18, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x393216xf32>, tensor<1x393216xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %19 into %arg2[%arg1, 0] [1, 393216] [1, 1] : tensor<1x393216xf32> into tensor<4096x393216xf32>
      }
    }
    return %2 : tensor<4096x393216xf32>
  }
}

[[[ IR printer: bufferize ]]]
#map = affine_map<()[s0] -> (s0 floordiv 2048)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%arg0: memref<4096x393216xf32>) -> memref<4096x393216xf32> {
    %cst = arith.constant dense<0.000000e+00> : vector<1x192xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
    %0 = ub.poison : f32
    %cst_1 = arith.constant dense<0xFF800000> : vector<1x192xf32>
    %cst_2 = arith.constant dense<0xFF800000> : vector<1xf32>
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096x393216xf32>
    scf.forall (%arg1) in (4096) {
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x192xf32>
      vector.transfer_write %cst_1, %alloc_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x192xf32>, memref<1x192xf32>
      scf.forall (%arg2) = (0) to (393216) step (2048) {
        %11 = affine.apply #map()[%arg2]
        %12 = vector.transfer_read %arg0[%arg1, %arg2], %0 {in_bounds = [true, true]} : memref<4096x393216xf32>, vector<1x2048xf32>
        %13 = vector.transfer_read %alloc_3[%c0, %11], %0 {in_bounds = [true]} : memref<1x192xf32>, vector<1xf32>
        %14 = vector.multi_reduction <maximumf>, %12, %13 [1] : vector<1x2048xf32> to vector<1xf32>
        vector.transfer_write %14, %alloc_3[%c0, %11] {in_bounds = [true], permutation_map = #map1} : vector<1xf32>, memref<1x192xf32>
      }
      %1 = vector.transfer_read %alloc_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x192xf32>, vector<1x192xf32>
      %2 = vector.multi_reduction <maximumf>, %1, %cst_2 [1] : vector<1x192xf32> to vector<1xf32>
      %3 = vector.transfer_read %arg0[%arg1, %c0], %0 {in_bounds = [true, true]} : memref<4096x393216xf32>, vector<1x393216xf32>
      %4 = vector.broadcast %2 : vector<1xf32> to vector<1x393216xf32>
      %5 = arith.subf %3, %4 : vector<1x393216xf32>
      %6 = math.exp %5 : vector<1x393216xf32>
      vector.transfer_write %6, %alloc[%arg1, %c0] {in_bounds = [true, true]} : vector<1x393216xf32>, memref<4096x393216xf32>
      vector.transfer_write %cst, %alloc_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x192xf32>, memref<1x192xf32>
      scf.forall (%arg2) = (0) to (393216) step (2048) {
        %11 = affine.apply #map()[%arg2]
        %12 = vector.transfer_read %alloc[%arg1, %arg2], %0 {in_bounds = [true, true]} : memref<4096x393216xf32>, vector<1x2048xf32>
        %13 = vector.transfer_read %alloc_3[%c0, %11], %0 {in_bounds = [true]} : memref<1x192xf32>, vector<1xf32>
        %14 = vector.multi_reduction <add>, %12, %13 [1] : vector<1x2048xf32> to vector<1xf32>
        vector.transfer_write %14, %alloc_3[%c0, %11] {in_bounds = [true], permutation_map = #map1} : vector<1xf32>, memref<1x192xf32>
      }
      %7 = vector.transfer_read %alloc_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x192xf32>, vector<1x192xf32>
      %8 = vector.multi_reduction <add>, %7, %cst_0 [1] : vector<1x192xf32> to vector<1xf32>
      %9 = vector.broadcast %8 : vector<1xf32> to vector<1x393216xf32>
      %10 = arith.divf %6, %9 : vector<1x393216xf32>
      vector.transfer_write %10, %alloc[%arg1, %c0] {in_bounds = [true, true]} : vector<1x393216xf32>, memref<4096x393216xf32>
    }
    return %alloc : memref<4096x393216xf32>
  }
}

[[[ IR printer: outline ]]]
#map = affine_map<(d0, d1) -> (d0)>
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<4096x393216xf32>) -> memref<4096x393216xf32> {
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096x393216xf32>
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c4096, %c1, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4096x393216xf32>, %alloc : memref<4096x393216xf32>)
    return %alloc : memref<4096x393216xf32>
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<4096x393216xf32>, %arg1: memref<4096x393216xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4096, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %cst = arith.constant dense<0xFF800000> : vector<1x192xf32>
      %c0 = arith.constant 0 : index
      %c-1 = arith.constant -1 : index
      %c2048 = arith.constant 2048 : index
      %0 = ub.poison : f32
      %cst_0 = arith.constant dense<0xFF800000> : vector<1xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<1x192xf32>
      %cst_2 = arith.constant dense<0.000000e+00> : vector<1xf32>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x192xf32>
      vector.transfer_write %cst, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<1x192xf32>, memref<1x192xf32>
      scf.forall (%arg2) = (0) to (393216) step (2048) {
        %11 = arith.cmpi slt, %arg2, %c0 : index
        %12 = arith.subi %c-1, %arg2 : index
        %13 = arith.select %11, %12, %arg2 : index
        %14 = arith.divsi %13, %c2048 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = arith.select %11, %15, %14 : index
        %17 = vector.transfer_read %arg0[%block_id_x, %arg2], %0 {in_bounds = [true, true]} : memref<4096x393216xf32>, vector<1x2048xf32>
        %18 = vector.transfer_read %alloc[%c0, %16], %0 {in_bounds = [true]} : memref<1x192xf32>, vector<1xf32>
        %19 = vector.multi_reduction <maximumf>, %17, %18 [1] : vector<1x2048xf32> to vector<1xf32>
        vector.transfer_write %19, %alloc[%c0, %16] {in_bounds = [true], permutation_map = #map} : vector<1xf32>, memref<1x192xf32>
      }
      %1 = vector.transfer_read %alloc[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x192xf32>, vector<1x192xf32>
      %2 = vector.multi_reduction <maximumf>, %1, %cst_0 [1] : vector<1x192xf32> to vector<1xf32>
      %3 = vector.transfer_read %arg0[%block_id_x, %c0], %0 {in_bounds = [true, true]} : memref<4096x393216xf32>, vector<1x393216xf32>
      %4 = vector.broadcast %2 : vector<1xf32> to vector<1x393216xf32>
      %5 = arith.subf %3, %4 : vector<1x393216xf32>
      %6 = math.exp %5 : vector<1x393216xf32>
      vector.transfer_write %6, %arg1[%block_id_x, %c0] {in_bounds = [true, true]} : vector<1x393216xf32>, memref<4096x393216xf32>
      vector.transfer_write %cst_1, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<1x192xf32>, memref<1x192xf32>
      scf.forall (%arg2) = (0) to (393216) step (2048) {
        %11 = arith.cmpi slt, %arg2, %c0 : index
        %12 = arith.subi %c-1, %arg2 : index
        %13 = arith.select %11, %12, %arg2 : index
        %14 = arith.divsi %13, %c2048 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = arith.select %11, %15, %14 : index
        %17 = vector.transfer_read %arg1[%block_id_x, %arg2], %0 {in_bounds = [true, true]} : memref<4096x393216xf32>, vector<1x2048xf32>
        %18 = vector.transfer_read %alloc[%c0, %16], %0 {in_bounds = [true]} : memref<1x192xf32>, vector<1xf32>
        %19 = vector.multi_reduction <add>, %17, %18 [1] : vector<1x2048xf32> to vector<1xf32>
        vector.transfer_write %19, %alloc[%c0, %16] {in_bounds = [true], permutation_map = #map} : vector<1xf32>, memref<1x192xf32>
      }
      %7 = vector.transfer_read %alloc[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x192xf32>, vector<1x192xf32>
      %8 = vector.multi_reduction <add>, %7, %cst_2 [1] : vector<1x192xf32> to vector<1xf32>
      %9 = vector.broadcast %8 : vector<1xf32> to vector<1x393216xf32>
      %10 = arith.divf %6, %9 : vector<1x393216xf32>
      vector.transfer_write %10, %arg1[%block_id_x, %c0] {in_bounds = [true, true]} : vector<1x393216xf32>, memref<4096x393216xf32>
      gpu.return
    }
  }
}

[[[ IR printer: vecToXegpu ]]]
#map = affine_map<(d0, d1) -> (d0)>
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<4096x393216xf32>) -> memref<4096x393216xf32> {
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096x393216xf32>
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c4096, %c1, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4096x393216xf32>, %alloc : memref<4096x393216xf32>)
    return %alloc : memref<4096x393216xf32>
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<4096x393216xf32>, %arg1: memref<4096x393216xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 4096, 1, 1>} {
      %cst = arith.constant dense<true> : vector<1xi1>
      %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<1x192xf32>
      %cst_2 = arith.constant dense<0xFF800000> : vector<1xf32>
      %c2048 = arith.constant 2048 : index
      %c-1 = arith.constant -1 : index
      %c0 = arith.constant 0 : index
      %cst_3 = arith.constant dense<0xFF800000> : vector<1x192xf32>
      %block_id_x = gpu.block_id  x
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x192xf32>
      %0 = xegpu.create_nd_tdesc %alloc : memref<1x192xf32> -> !xegpu.tensor_desc<1x192xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %cst_3, %0[0, 0]  : vector<1x192xf32>, !xegpu.tensor_desc<1x192xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      scf.forall (%arg2) = (0) to (393216) step (2048) {
        %13 = arith.cmpi slt, %arg2, %c0 : index
        %14 = arith.subi %c-1, %arg2 : index
        %15 = arith.select %13, %14, %arg2 : index
        %16 = arith.divsi %15, %c2048 : index
        %17 = arith.subi %c-1, %16 : index
        %18 = arith.select %13, %17, %16 : index
        %19 = xegpu.create_nd_tdesc %arg0 : memref<4096x393216xf32> -> !xegpu.tensor_desc<1x2048xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %20 = xegpu.load_nd %19[%block_id_x, %arg2]  : !xegpu.tensor_desc<1x2048xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x2048xf32>
        %21 = vector.step : vector<1xindex>
        %22 = vector.broadcast %18 : index to vector<1xindex>
        %23 = arith.addi %22, %21 : vector<1xindex>
        %intptr = memref.extract_aligned_pointer_as_index %alloc : memref<1x192xf32> -> index
        %24 = arith.index_cast %intptr : index to i64
        %25 = xegpu.load %24[%23], %cst  : i64, vector<1xindex>, vector<1xi1> -> vector<1xf32>
        %26 = vector.multi_reduction <maximumf>, %20, %25 [1] : vector<1x2048xf32> to vector<1xf32>
        vector.transfer_write %26, %alloc[%c0, %18] {in_bounds = [true], permutation_map = #map} : vector<1xf32>, memref<1x192xf32>
      }
      %1 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<1x192xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x192xf32>
      %2 = vector.multi_reduction <maximumf>, %1, %cst_2 [1] : vector<1x192xf32> to vector<1xf32>
      %3 = xegpu.create_nd_tdesc %arg0 : memref<4096x393216xf32> -> !xegpu.tensor_desc<1x393216xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %4 = xegpu.load_nd %3[%block_id_x, 0]  : !xegpu.tensor_desc<1x393216xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x393216xf32>
      %5 = vector.broadcast %2 : vector<1xf32> to vector<1x393216xf32>
      %6 = arith.subf %4, %5 : vector<1x393216xf32>
      %7 = math.exp %6 : vector<1x393216xf32>
      %8 = xegpu.create_nd_tdesc %arg1 : memref<4096x393216xf32> -> !xegpu.tensor_desc<1x393216xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %7, %8[%block_id_x, 0]  : vector<1x393216xf32>, !xegpu.tensor_desc<1x393216xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %cst_1, %0[0, 0]  : vector<1x192xf32>, !xegpu.tensor_desc<1x192xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      scf.forall (%arg2) = (0) to (393216) step (2048) {
        %13 = arith.cmpi slt, %arg2, %c0 : index
        %14 = arith.subi %c-1, %arg2 : index
        %15 = arith.select %13, %14, %arg2 : index
        %16 = arith.divsi %15, %c2048 : index
        %17 = arith.subi %c-1, %16 : index
        %18 = arith.select %13, %17, %16 : index
        %19 = xegpu.create_nd_tdesc %arg1 : memref<4096x393216xf32> -> !xegpu.tensor_desc<1x2048xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %20 = xegpu.load_nd %19[%block_id_x, %arg2]  : !xegpu.tensor_desc<1x2048xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x2048xf32>
        %21 = vector.step : vector<1xindex>
        %22 = vector.broadcast %18 : index to vector<1xindex>
        %23 = arith.addi %22, %21 : vector<1xindex>
        %intptr = memref.extract_aligned_pointer_as_index %alloc : memref<1x192xf32> -> index
        %24 = arith.index_cast %intptr : index to i64
        %25 = xegpu.load %24[%23], %cst  : i64, vector<1xindex>, vector<1xi1> -> vector<1xf32>
        %26 = vector.multi_reduction <add>, %20, %25 [1] : vector<1x2048xf32> to vector<1xf32>
        vector.transfer_write %26, %alloc[%c0, %18] {in_bounds = [true], permutation_map = #map} : vector<1xf32>, memref<1x192xf32>
      }
      %9 = xegpu.load_nd %0[0, 0]  : !xegpu.tensor_desc<1x192xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x192xf32>
      %10 = vector.multi_reduction <add>, %9, %cst_0 [1] : vector<1x192xf32> to vector<1xf32>
      %11 = vector.broadcast %10 : vector<1xf32> to vector<1x393216xf32>
      %12 = arith.divf %7, %11 : vector<1x393216xf32>
      xegpu.store_nd %12, %8[%block_id_x, 0]  : vector<1x393216xf32>, !xegpu.tensor_desc<1x393216xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}