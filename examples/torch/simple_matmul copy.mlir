// -----
// [[[ IR printer: Initial ]]]
module {
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
}

// -----
// [[[ IR printer: tile ]]]
#map = affine_map<(d0) -> (d0 * 256)>
module {
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x4096xf32> to tensor<4096x4096xf32>
    %3 = tensor.empty() : tensor<4096x4096xf32>
    %4 = scf.forall (%arg3, %arg4) in (16, 16) shared_outs(%arg5 = %2) -> (tensor<4096x4096xf32>) {
      %5 = affine.apply #map(%arg3)
      %6 = affine.apply #map(%arg4)
      %extracted_slice = tensor.extract_slice %0[%5, 0] [256, 4096] [1, 1] : tensor<4096x4096xf16> to tensor<256x4096xf16>
      %extracted_slice_0 = tensor.extract_slice %1[0, %6] [4096, 256] [1, 1] : tensor<4096x4096xf16> to tensor<4096x256xf16>
      %extracted_slice_1 = tensor.extract_slice %arg5[%5, %6] [256, 256] [1, 1] : tensor<4096x4096xf32> to tensor<256x256xf32>
      %7 = scf.for %arg6 = %c0 to %c4096 step %c32 iter_args(%arg7 = %extracted_slice_1) -> (tensor<256x256xf32>) {
        %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, %arg6] [256, 32] [1, 1] : tensor<256x4096xf16> to tensor<256x32xf16>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[%arg6, 0] [32, 256] [1, 1] : tensor<4096x256xf16> to tensor<32x256xf16>
        %10 = linalg.matmul ins(%extracted_slice_3, %extracted_slice_4 : tensor<256x32xf16>, tensor<32x256xf16>) outs(%arg7 : tensor<256x256xf32>) -> tensor<256x256xf32>
        scf.yield %10 : tensor<256x256xf32>
      }
      %extracted_slice_2 = tensor.extract_slice %3[%5, %6] [256, 256] [1, 1] : tensor<4096x4096xf32> to tensor<256x256xf32>
      %8 = linalg.fill ins(%cst : f32) outs(%extracted_slice_2 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %9 = linalg.max ins(%7, %8 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%extracted_slice_1 : tensor<256x256xf32>) -> tensor<256x256xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %9 into %arg5[%5, %6] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<4096x4096xf32>
      }
    }
    bufferization.materialize_in_destination %4 in restrict writable %arg2 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    return
  }
}

// -----
// [[[ IR printer: vectorize ]]]
#map = affine_map<(d0) -> (d0 * 256)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
    %0 = ub.poison : f32
    %1 = ub.poison : f16
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %2 = bufferization.to_tensor %arg0 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %3 = bufferization.to_tensor %arg1 restrict : memref<4096x4096xf16> to tensor<4096x4096xf16>
    %4 = bufferization.to_tensor %arg2 restrict writable : memref<4096x4096xf32> to tensor<4096x4096xf32>
    %5 = scf.forall (%arg3, %arg4) in (16, 16) shared_outs(%arg5 = %4) -> (tensor<4096x4096xf32>) {
      %6 = affine.apply #map(%arg3)
      %7 = affine.apply #map(%arg4)
      %extracted_slice = tensor.extract_slice %arg5[%6, %7] [256, 256] [1, 1] : tensor<4096x4096xf32> to tensor<256x256xf32>
      %8 = vector.transfer_read %extracted_slice[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<256x256xf32>, vector<256x256xf32>
      %9 = scf.for %arg6 = %c0 to %c4096 step %c32 iter_args(%arg7 = %8) -> (vector<256x256xf32>) {
        %12 = vector.transfer_read %2[%6, %arg6], %1 {in_bounds = [true, true]} : tensor<4096x4096xf16>, vector<256x32xf16>
        %13 = vector.transfer_read %3[%arg6, %7], %1 {in_bounds = [true, true]} : tensor<4096x4096xf16>, vector<32x256xf16>
        %14 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %12, %13, %arg7 : vector<256x32xf16>, vector<32x256xf16> into vector<256x256xf32>
        scf.yield %14 : vector<256x256xf32>
      }
      %10 = arith.maximumf %9, %cst : vector<256x256xf32>
      %11 = vector.transfer_write %10, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<256x256xf32>, tensor<256x256xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %11 into %arg5[%6, %7] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<4096x4096xf32>
      }
    }
    bufferization.materialize_in_destination %5 in restrict writable %arg2 : (tensor<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    return
  }
}

// -----
// [[[ IR printer: bufferize ]]]
#map = affine_map<(d0) -> (d0 * 256)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
    %0 = ub.poison : f32
    %1 = ub.poison : f16
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg3, %arg4) in (16, 16) {
      %2 = affine.apply #map(%arg3)
      %3 = affine.apply #map(%arg4)
      %4 = vector.transfer_read %arg2[%2, %3], %0 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<256x256xf32>
      %5 = scf.for %arg5 = %c0 to %c4096 step %c32 iter_args(%arg6 = %4) -> (vector<256x256xf32>) {
        %7 = vector.transfer_read %arg0[%2, %arg5], %1 {in_bounds = [true, true]} : memref<4096x4096xf16>, vector<256x32xf16>
        %8 = vector.transfer_read %arg1[%arg5, %3], %1 {in_bounds = [true, true]} : memref<4096x4096xf16>, vector<32x256xf16>
        %9 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %8, %arg6 : vector<256x32xf16>, vector<32x256xf16> into vector<256x256xf32>
        scf.yield %9 : vector<256x256xf32>
      }
      %6 = arith.maximumf %5, %cst : vector<256x256xf32>
      vector.transfer_write %6, %arg2[%2, %3] {in_bounds = [true, true]} : vector<256x256xf32>, memref<4096x4096xf32>
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
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    gpu.launch_func  @payload_kernel::@payload_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1)  args(%arg2 : memref<4096x4096xf32>, %arg0 : memref<4096x4096xf16>, %arg1 : memref<4096x4096xf16>)
    return
  }
  gpu.module @payload_kernel [#xevm.target<O = 3>] {
    gpu.func @payload_kernel(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %c256 = arith.constant 256 : index
      %0 = ub.poison : f32
      %1 = ub.poison : f16
      %c0 = arith.constant 0 : index
      %c4096 = arith.constant 4096 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
      %2 = arith.muli %block_id_x, %c256 overflow<nsw> : index
      %3 = arith.muli %block_id_y, %c256 overflow<nsw> : index
      %4 = vector.transfer_read %arg0[%2, %3], %0 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<256x256xf32>
      %5 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %4) -> (vector<256x256xf32>) {
        %7 = vector.transfer_read %arg1[%2, %arg3], %1 {in_bounds = [true, true]} : memref<4096x4096xf16>, vector<256x32xf16>
        %8 = vector.transfer_read %arg2[%arg3, %3], %1 {in_bounds = [true, true]} : memref<4096x4096xf16>, vector<32x256xf16>
        %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %8, %arg4 : vector<256x32xf16>, vector<32x256xf16> into vector<256x256xf32>
        scf.yield %9 : vector<256x256xf32>
      }
      %6 = arith.maximumf %5, %cst : vector<256x256xf32>
      vector.transfer_write %6, %arg0[%2, %3] {in_bounds = [true, true]} : vector<256x256xf32>, memref<4096x4096xf32>
      gpu.return
    }
  }
}

// -----
// [[[ IR printer: vecToXegpu ]]]
module attributes {gpu.container_module} {
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    gpu.launch_func  @payload_kernel::@payload_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1)  args(%arg2 : memref<4096x4096xf32>, %arg0 : memref<4096x4096xf16>, %arg1 : memref<4096x4096xf16>)
    return
  }
  gpu.module @payload_kernel [#xevm.target<O = 3>] {
    gpu.func @payload_kernel(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
      %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c256 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c256 overflow<nsw> : index
      %2 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %3 = xegpu.load_nd %2[%0, %1]  : !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<256x256xf32>
      %4 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %3) -> (vector<256x256xf32>) {
        %6 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %7 = xegpu.load_nd %6[%0, %arg3]  : !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<256x32xf16>
        %8 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %9 = xegpu.load_nd %8[%arg3, %1]  : !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x256xf16>
        %10 = xegpu.dpas %7, %9, %arg4 : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %10 : vector<256x256xf32>
      }
      %5 = arith.maximumf %4, %cst : vector<256x256xf32>
      xegpu.store_nd %5, %2[%0, %1]  : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}




[[[ IR printer: dpas ]]]
module attributes {gpu.container_module} {
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    gpu.launch_func  @payload_kernel::@payload_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1)  args(%arg2 : memref<4096x4096xf32>, %arg0 : memref<4096x4096xf16>, %arg1 : memref<4096x4096xf16>)
    return
  }
  gpu.module @payload_kernel [#xevm.target<O = 3>] {
    gpu.func @payload_kernel(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
      %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>} dense<0.000000e+00> : vector<256x256xf32>
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c256 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c256 overflow<nsw> : index
      %2 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>>
      %3 = xegpu.load_nd %2[%0, %1]  : !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>> -> vector<256x256xf32>
      %4 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>>
      %5 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>>
      %6 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %3) -> (vector<256x256xf32>) {
        %8 = xegpu.load_nd %4[%0, %arg3]  : !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>> -> vector<256x32xf16>
        %9 = xegpu.load_nd %5[%arg3, %1]  : !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>> -> vector<32x256xf16>
        %10 = xegpu.dpas %8, %9, %arg4 {layout_operand_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>, layout_operand_1 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>, layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>} : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %10 : vector<256x256xf32>
      }
      %7 = arith.maximumf %6, %cst {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>} : vector<256x256xf32>
      xegpu.store_nd %7, %2[%0, %1]  : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>>
      gpu.return
    }
  }
}


[[[ IR printer: dpas ]]]
module attributes {gpu.container_module} {
  func.func @payload(%arg0: memref<4096x4096xf16>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    gpu.launch_func  @payload_kernel::@payload_kernel blocks in (%c16, %c16, %c1) threads in (%c512, %c1, %c1)  args(%arg2 : memref<4096x4096xf32>, %arg0 : memref<4096x4096xf16>, %arg1 : memref<4096x4096xf16>)
    return
  }
  gpu.module @payload_kernel [#xevm.target<O = 3>] {
    gpu.func @payload_kernel(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf16>, %arg2: memref<4096x4096xf16>) kernel attributes {known_block_size = array<i32: 512, 1, 1>, known_grid_size = array<i32: 16, 16, 1>} {
      %cst = arith.constant {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>} dense<0.000000e+00> : vector<256x256xf32>
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c256 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c256 overflow<nsw> : index
      %2 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %3 = xegpu.load_nd %2[%0, %1]  : !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<256x256xf32>
      %4 = xegpu.create_nd_tdesc %arg1 : memref<4096x4096xf16> -> !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>>
      %5 = xegpu.create_nd_tdesc %arg2 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>>
      %6 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %3) -> (vector<256x256xf32>) {
        %8 = xegpu.load_nd %4[%0, %arg3]  : !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>> -> vector<256x32xf16>
        %9 = xegpu.load_nd %5[%arg3, %1]  : !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>> -> vector<32x256xf16>
        %10 = xegpu.dpas %8, %9, %arg4 {layout_operand_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32], inst_data = [8, 16]>, layout_operand_1 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [16, 16]>, layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>} : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        scf.yield %10 : vector<256x256xf32>
      }
      %7 = arith.maximumf %6, %cst {layout_result_0 = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 64], inst_data = [8, 16]>} : vector<256x256xf32>
      xegpu.store_nd %7, %2[%0, %1]  : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}
