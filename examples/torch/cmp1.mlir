#map = affine_map<()[s0] -> (s0 * 256)>
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<4096x256xf32>, %arg1: memref<4096x256xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    %c16 = arith.constant 16 : index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c4096, %c1, %c1) threads in (%c16, %c1, %c1)  args(%arg0 : memref<4096x256xf32>, %arg1 : memref<4096x256xf32>)
    return
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<4096x256xf32>, %arg1: memref<4096x256xf32>) kernel attributes {intel_reqd_sub_group_size = 16 : i32, known_block_size = array<i32: 16, 1, 1>, known_grid_size = array<i32: 4096, 1, 1>} {
      %c4 = arith.constant 4 : index
      %cst = arith.constant dense<true> : vector<256xi1>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_1 = arith.constant 0xFF800000 : f32
      %block_id_x = gpu.block_id  x
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<4096x256xf32> -> memref<f32>, index, index, index, index, index
      %0 = affine.apply #map()[%block_id_x]
      %1 = vector.step {layout_result_0 = #xegpu.layout<sg_layout = [1], sg_data = [256], inst_data = [16]>} : vector<256xindex>
      %2 = vector.broadcast %0 {layout_result_0 = #xegpu.layout<sg_layout = [1], sg_data = [256], inst_data = [16]>} : index to vector<256xindex>
      %3 = arith.addi %2, %1 : vector<256xindex>
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
      %4 = arith.index_cast %intptr : index to i64
      %5 = xegpu.load %4[%3], %cst <{layout = #xegpu.layout<sg_layout = [1], sg_data = [256], inst_data = [16]>}> : i64, vector<256xindex>, vector<256xi1> -> vector<256xf32>
      %6 = vector.multi_reduction <maximumf>, %5, %cst_1 [0] : vector<256xf32> to f32
      %7 = vector.broadcast %6 {layout_result_0 = #xegpu.layout<sg_layout = [1], sg_data = [256], inst_data = [16]>} : f32 to vector<256xf32>
      %8 = arith.subf %5, %7 : vector<256xf32>
      %9 = math.exp %8 : vector<256xf32>
      %10 = vector.multi_reduction <add>, %9, %cst_0 [0] : vector<256xf32> to f32
      %11 = vector.broadcast %10 {layout_result_0 = #xegpu.layout<sg_layout = [1], sg_data = [256], inst_data = [16]>} : f32 to vector<256xf32>
      %12 = arith.divf %9, %11 : vector<256xf32>
      %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg1 : memref<4096x256xf32> -> memref<f32>, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %13 = arith.muli %0, %c4 : index
      %14 = arith.addi %intptr_6, %13 : index
      %15 = arith.index_cast %14 : index to i64
      %16 = xegpu.create_nd_tdesc %15, shape : [256], strides : [1] : i64 -> !xegpu.tensor_desc<256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %12, %16[0]  : vector<256xf32>, !xegpu.tensor_desc<256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}