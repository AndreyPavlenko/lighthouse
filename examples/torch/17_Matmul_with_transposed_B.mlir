module {
  func.func @main(%arg0: tensor<2048x8192xf32>, %arg1: tensor<4096x8192xf32>) -> tensor<2048x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8192x4096xf32>
    %transposed = linalg.transpose ins(%arg1 : tensor<4096x8192xf32>) outs(%0 : tensor<8192x4096xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<2048x4096xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2048x4096xf32>) -> tensor<2048x4096xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<2048x8192xf32>, tensor<8192x4096xf32>) outs(%2 : tensor<2048x4096xf32>) -> tensor<2048x4096xf32>
    return %3 : tensor<2048x4096xf32>
  }
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.transpose", "linalg.matmul", "linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1:3 = transform.split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op, %forall_op = transform.structured.tile_using_forall %1#2 tile_sizes [256, 256] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1#1 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %1#0 into %new_containing_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %tiled_op tile_sizes [0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_2, %new_containing_op_3 = transform.structured.fuse_into_containing_op %fused_op_0 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %arg0 : !transform.any_op
    transform.yield 
  }
}
module {
  func.func @main(%arg0: tensor<2048x8192xf32>, %arg1: tensor<4096x8192xf32>) -> tensor<2048x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8192x4096xf32>
    %1 = tensor.empty() : tensor<2048x4096xf32>
    %2 = scf.forall (%arg2, %arg3) in (8, 16) shared_outs(%arg4 = %1) -> (tensor<2048x4096xf32>) {
      %3 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg2)
      %4 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg3)
      %extracted_slice = tensor.extract_slice %arg0[%3, 0] [256, 8192] [1, 1] : tensor<2048x8192xf32> to tensor<256x8192xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%4, 0] [256, 8192] [1, 1] : tensor<4096x8192xf32> to tensor<256x8192xf32>
      %extracted_slice_1 = tensor.extract_slice %0[0, %4] [8192, 256] [1, 1] : tensor<8192x4096xf32> to tensor<8192x256xf32>
      %extracted_slice_2 = tensor.extract_slice %arg4[%3, %4] [256, 256] [1, 1] : tensor<2048x4096xf32> to tensor<256x256xf32>
      %5 = linalg.fill ins(%cst : f32) outs(%extracted_slice_2 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %c0 = arith.constant 0 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %6 = scf.for %arg5 = %c0 to %c8192 step %c32 iter_args(%arg6 = %5) -> (tensor<256x256xf32>) {
        %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, %arg5] [256, 32] [1, 1] : tensor<256x8192xf32> to tensor<256x32xf32>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[0, %arg5] [256, 32] [1, 1] : tensor<256x8192xf32> to tensor<256x32xf32>
        %extracted_slice_5 = tensor.extract_slice %extracted_slice_1[%arg5, 0] [32, 256] [1, 1] : tensor<8192x256xf32> to tensor<32x256xf32>
        %transposed = linalg.transpose ins(%extracted_slice_4 : tensor<256x32xf32>) outs(%extracted_slice_5 : tensor<32x256xf32>) permutation = [1, 0] 
        %extracted_slice_6 = tensor.extract_slice %arg6[0, 0] [256, 256] [1, 1] : tensor<256x256xf32> to tensor<256x256xf32>
        %7 = linalg.matmul ins(%extracted_slice_3, %transposed : tensor<256x32xf32>, tensor<32x256xf32>) outs(%extracted_slice_6 : tensor<256x256xf32>) -> tensor<256x256xf32>
        %inserted_slice = tensor.insert_slice %7 into %arg6[0, 0] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<256x256xf32>
        scf.yield %inserted_slice : tensor<256x256xf32>
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %6 into %arg4[%3, %4] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<2048x4096xf32>
      }
    }
    return %2 : tensor<2048x4096xf32>
  }
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill", "linalg.matmul", "linalg.transpose"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1:3 = transform.split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op, %forall_op = transform.structured.tile_using_forall %1#2 tile_sizes [256, 256] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1#1 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %1#0 into %new_containing_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %tiled_op tile_sizes [0, 0, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_2, %new_containing_op_3 = transform.structured.fuse_into_containing_op %fused_op_0 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %arg0 : !transform.any_op
    %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.vectorize_children_and_apply_patterns %2 {fold_type_extensions_into_contract} : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.match ops{["scf.for"]} in %3 : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %4 : !transform.any_op
    transform.apply_cse to %3 : !transform.any_op
    transform.apply_patterns to %3 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield 
  }
}
module {
  func.func @main(%arg0: tensor<2048x8192xf32>, %arg1: tensor<4096x8192xf32>) -> tensor<2048x4096xf32> {
    %0 = ub.poison : f32
    %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
    %c32 = arith.constant 32 : index
    %c8192 = arith.constant 8192 : index
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<2048x4096xf32>
    %2 = scf.forall (%arg2, %arg3) in (8, 16) shared_outs(%arg4 = %1) -> (tensor<2048x4096xf32>) {
      %3 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg2)
      %4 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg3)
      %extracted_slice = tensor.extract_slice %arg4[%3, %4] [256, 256] [1, 1] : tensor<2048x4096xf32> to tensor<256x256xf32>
      %5 = scf.for %arg5 = %c0 to %c8192 step %c32 iter_args(%arg6 = %cst) -> (vector<256x256xf32>) {
        %7 = vector.transfer_read %arg1[%4, %arg5], %0 {in_bounds = [true, true]} : tensor<4096x8192xf32>, vector<256x32xf32>
        %8 = vector.transfer_read %arg0[%3, %arg5], %0 {in_bounds = [true, true]} : tensor<2048x8192xf32>, vector<256x32xf32>
        %9 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %7, %arg6 : vector<256x32xf32>, vector<256x32xf32> into vector<256x256xf32>
        scf.yield %9 : vector<256x256xf32>
      }
      %6 = vector.transfer_write %5, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<256x256xf32>, tensor<256x256xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %6 into %arg4[%3, %4] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<2048x4096xf32>
      }
    }
    return %2 : tensor<2048x4096xf32>
  }
}