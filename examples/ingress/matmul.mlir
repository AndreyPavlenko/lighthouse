// -----
// [[[ IR printer: Initial ]]]
module {
  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%1 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    return %2 : tensor<4096x4096xf32>
  }
}

// -----
// [[[ IR printer: tile ]]]
#map = affine_map<(d0) -> (d0 * 256)>
module {
  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = scf.forall (%arg2, %arg3) in (16, 16) shared_outs(%arg4 = %0) -> (tensor<4096x4096xf32>) {
      %2 = affine.apply #map(%arg2)
      %3 = affine.apply #map(%arg3)
      %extracted_slice = tensor.extract_slice %arg0[%2, 0] [256, 4096] [1, 1] : tensor<4096x4096xf32> to tensor<256x4096xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %3] [4096, 256] [1, 1] : tensor<4096x4096xf32> to tensor<4096x256xf32>
      %extracted_slice_1 = tensor.extract_slice %arg4[%2, %3] [256, 256] [1, 1] : tensor<4096x4096xf32> to tensor<256x256xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%extracted_slice_1 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %5 = scf.for %arg5 = %c0 to %c4096 step %c32 iter_args(%arg6 = %4) -> (tensor<256x256xf32>) {
        %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg5] [256, 32] [1, 1] : tensor<256x4096xf32> to tensor<256x32xf32>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg5, 0] [32, 256] [1, 1] : tensor<4096x256xf32> to tensor<32x256xf32>
        %6 = linalg.matmul ins(%extracted_slice_2, %extracted_slice_3 : tensor<256x32xf32>, tensor<32x256xf32>) outs(%arg6 : tensor<256x256xf32>) -> tensor<256x256xf32>
        scf.yield %6 : tensor<256x256xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg4[%2, %3] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<4096x4096xf32>
      }
    }
    return %1 : tensor<4096x4096xf32>
  }
}

// -----
// [[[ IR printer: vectorize ]]]
#map = affine_map<(d0) -> (d0 * 256)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = ub.poison : f32
    %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<4096x4096xf32>
    %2 = scf.forall (%arg2, %arg3) in (16, 16) shared_outs(%arg4 = %1) -> (tensor<4096x4096xf32>) {
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map(%arg3)
      %extracted_slice = tensor.extract_slice %arg4[%3, %4] [256, 256] [1, 1] : tensor<4096x4096xf32> to tensor<256x256xf32>
      %5 = scf.for %arg5 = %c0 to %c4096 step %c32 iter_args(%arg6 = %cst) -> (vector<256x256xf32>) {
        %7 = vector.transfer_read %arg0[%3, %arg5], %0 {in_bounds = [true, true]} : tensor<4096x4096xf32>, vector<256x32xf32>
        %8 = vector.transfer_read %arg1[%arg5, %4], %0 {in_bounds = [true, true]} : tensor<4096x4096xf32>, vector<32x256xf32>
        %9 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %8, %arg6 : vector<256x32xf32>, vector<32x256xf32> into vector<256x256xf32>
        scf.yield %9 : vector<256x256xf32>
      }
      %6 = vector.transfer_write %5, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<256x256xf32>, tensor<256x256xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %6 into %arg4[%3, %4] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<4096x4096xf32>
      }
    }
    return %2 : tensor<4096x4096xf32>
  }
}

// -----
// [[[ IR printer: bufferize ]]]
#map = affine_map<(d0) -> (d0 * 256)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %0 = ub.poison : f32
    %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg3, %arg4) in (16, 16) {
      %1 = affine.apply #map(%arg3)
      %2 = affine.apply #map(%arg4)
      %3 = scf.for %arg5 = %c0 to %c4096 step %c32 iter_args(%arg6 = %cst) -> (vector<256x256xf32>) {
        %4 = vector.transfer_read %arg0[%1, %arg5], %0 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<256x32xf32>
        %5 = vector.transfer_read %arg1[%arg5, %2], %0 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<32x256xf32>
        %6 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg6 : vector<256x32xf32>, vector<32x256xf32> into vector<256x256xf32>
        scf.yield %6 : vector<256x256xf32>
      }
      vector.transfer_write %3, %arg2[%1, %2] {in_bounds = [true, true]} : vector<256x256xf32>, memref<4096x4096xf32>
    }
    return
  }
}

// -----
// [[[ IR printer: outline ]]]
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1)  args(%arg0 : memref<4096x4096xf32>, %arg1 : memref<4096x4096xf32>, %arg2 : memref<4096x4096xf32>)
    return
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %c256 = arith.constant 256 : index
      %0 = ub.poison : f32
      %c0 = arith.constant 0 : index
      %c4096 = arith.constant 4096 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
      %1 = arith.muli %block_id_x, %c256 overflow<nsw> : index
      %2 = arith.muli %block_id_y, %c256 overflow<nsw> : index
      %3 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %cst) -> (vector<256x256xf32>) {
        %4 = vector.transfer_read %arg0[%1, %arg3], %0 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<256x32xf32>
        %5 = vector.transfer_read %arg1[%arg3, %2], %0 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<32x256xf32>
        %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg4 : vector<256x32xf32>, vector<32x256xf32> into vector<256x256xf32>
        scf.yield %6 : vector<256x256xf32>
      }
      vector.transfer_write %3, %arg2[%1, %2] {in_bounds = [true, true]} : vector<256x256xf32>, memref<4096x4096xf32>
      gpu.return
    }
  }
}

// -----
// [[[ IR printer: vecToXegpu ]]]
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1)  args(%arg0 : memref<4096x4096xf32>, %arg1 : memref<4096x4096xf32>, %arg2 : memref<4096x4096xf32>)
    return
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
      %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c256 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c256 overflow<nsw> : index
      %2 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %cst) -> (vector<256x256xf32>) {
        %4 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x32xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %5 = xegpu.load_nd %4[%0, %arg3]  : !xegpu.tensor_desc<256x32xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<256x32xf32>
        %6 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf32> -> !xegpu.tensor_desc<32x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %7 = xegpu.load_nd %6[%arg3, %1]  : !xegpu.tensor_desc<32x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x256xf32>
        %8 = xegpu.dpas %5, %7, %arg4 : vector<256x32xf32>, vector<32x256xf32>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %8 : vector<256x256xf32>
      }
      %3 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %2, %3[%0, %1]  : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}
