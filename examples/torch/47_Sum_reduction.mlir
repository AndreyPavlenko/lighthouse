module {
  func.func @main(%arg0: tensor<128x4096x4095xf32>) -> tensor<128x1x4095xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x1x4095xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x1x4095xf32>) -> tensor<128x1x4095xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0 : tensor<128x4096x4095xf32>) outs(%1 : tensor<128x1x4095xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<128x1x4095xf32>
    return %2 : tensor<128x1x4095xf32>
  }
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    
    // Tile for GPU: d0=4 batches, d2=128 output elements per block
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 
      tile_sizes [4, 0, 128] : 
      (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield 
  }
}


#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 128)>
#map2 = affine_map<(d0) -> (-d0 + 4095, 128)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
module {
  module {
    func.func @main(%arg0: tensor<128x4096x4095xf32>) -> tensor<128x1x4095xf32> {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<128x1x4095xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x1x4095xf32>) -> tensor<128x1x4095xf32>
      %2 = scf.forall (%arg1, %arg2) in (32, 32) shared_outs(%arg3 = %1) -> (tensor<128x1x4095xf32>) {
        %3 = affine.apply #map(%arg1)
        %4 = affine.apply #map1(%arg2)
        %5 = affine.min #map2(%4)
        %extracted_slice = tensor.extract_slice %arg0[%3, 0, %4] [4, 4096, %5] [1, 1, 1] : tensor<128x4096x4095xf32> to tensor<4x4096x?xf32>
        %extracted_slice_0 = tensor.extract_slice %arg3[%3, 0, %4] [4, 1, %5] [1, 1, 1] : tensor<128x1x4095xf32> to tensor<4x1x?xf32>
        %6 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction", "parallel"]} ins(%extracted_slice : tensor<4x4096x?xf32>) outs(%extracted_slice_0 : tensor<4x1x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.addf %in, %out : f32
          linalg.yield %7 : f32
        } -> tensor<4x1x?xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %6 into %arg3[%3, 0, %4] [4, 1, %5] [1, 1, 1] : tensor<4x1x?xf32> into tensor<128x1x4095xf32>
        }
      }
      return %2 : tensor<128x1x4095xf32>
    }
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %tiled_op, %forall_op = transform.structured.tile_using_forall %0 tile_sizes [4, 0, 128] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.apply_cse to %1 : !transform.any_op
      transform.yield 
    }
  }
}