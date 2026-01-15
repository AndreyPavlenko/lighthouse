"builtin.module"() ({
  "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
  ^bb0(%arg0: !transform.any_op):
    "transform.print"(%arg0) <{name = "Initial"}> : (!transform.any_op) -> ()
    %0 = "transform.structured.match"(%arg0) <{ops = ["linalg.generic"]}> : (!transform.any_op) -> !transform.any_op
    %1:2 = "transform.structured.tile_using_forall"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, static_num_threads = array<i64>, static_tile_sizes = array<i64: 1, 4, 64>}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    "transform.annotate"(%1#1) <{name = "gpu_loop"}> : (!transform.any_op) -> ()
    "transform.apply_cse"(%arg0) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%arg0) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes"() : () -> ()
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.print"(%arg0) <{name = "tile"}> : (!transform.any_op) -> ()
    %2 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %3 = "transform.structured.vectorize_children_and_apply_patterns"(%2) <{fold_type_extensions_into_contract}> : (!transform.any_op) -> !transform.any_op
    %4 = "transform.structured.match"(%3) <{ops = ["scf.for", "scf.forall"]}> : (!transform.any_op) -> !transform.any_op
    "transform.loop.hoist_loop_invariant_subsets"(%4) : (!transform.any_op) -> ()
    "transform.apply_cse"(%3) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%3) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.print"(%arg0) <{name = "vectorize"}> : (!transform.any_op) -> ()
    %5 = "transform.bufferization.one_shot_bufferize"(%arg0) <{allow_return_allocs_from_loops = true, allow_unknown_ops = true, bufferize_function_boundaries = true, check_parallel_regions = true, dump_alias_sets = false, function_boundary_type_conversion = 1 : i32, memcpy_op = "memref.copy", print_conflicts = false, test_analysis_only = false}> : (!transform.any_op) -> !transform.any_op
    %6 = "transform.bufferization.one_shot_bufferize"(%5) <{allow_return_allocs_from_loops = false, allow_unknown_ops = true, bufferize_function_boundaries = true, check_parallel_regions = true, dump_alias_sets = false, function_boundary_type_conversion = 1 : i32, memcpy_op = "memref.copy", print_conflicts = false, test_analysis_only = false}> : (!transform.any_op) -> !transform.any_op
    %7 = "transform.apply_registered_pass"(%6) <{options = {}, pass_name = "fold-memref-alias-ops"}> : (!transform.any_op) -> !transform.any_op
    %8 = "transform.apply_registered_pass"(%7) <{options = {}, pass_name = "drop-equivalent-buffer-results"}> : (!transform.any_op) -> !transform.any_op
    %9 = "transform.apply_registered_pass"(%8) <{options = {"add-result-attr" = true, "hoist-dynamic-allocs" = true, "hoist-static-allocs" = true, "modify-public-functions" = true}, pass_name = "buffer-results-to-out-params"}> : (!transform.any_op) -> !transform.any_op
    %10 = "transform.apply_registered_pass"(%9) <{options = {}, pass_name = "buffer-deallocation-simplification"}> : (!transform.any_op) -> !transform.any_op
    %11 = "transform.apply_registered_pass"(%10) <{options = {}, pass_name = "memref-expand"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%11) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%11) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.memref.erase_dead_alloc_and_stores"(%11) : (!transform.any_op) -> ()
    %12 = "transform.structured.match"(%11) <{ops = ["scf.for", "scf.forall"]}> : (!transform.any_op) -> !transform.any_op
    "transform.loop.hoist_loop_invariant_subsets"(%12) : (!transform.any_op) -> ()
    "transform.apply_cse"(%11) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%11) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.print"(%11) <{name = "bufferize"}> : (!transform.any_op) -> ()
    %13 = "transform.structured.match"(%11) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %14 = "transform.structured.match"(%13) <{op_attrs = {gpu_loop}}> : (!transform.any_op) -> !transform.any_op
    %15 = "transform.split_handle"(%14) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> !transform.any_op
    %16 = "transform.loop.forall_to_parallel"(%15) : (!transform.any_op) -> !transform.any_op
    %17 = "transform.apply_registered_pass"(%13) <{options = {}, pass_name = "gpu-map-parallel-loops"}> : (!transform.any_op) -> !transform.any_op
    %18 = "transform.apply_registered_pass"(%17) <{options = {}, pass_name = "convert-parallel-loops-to-gpu"}> : (!transform.any_op) -> !transform.any_op
    %19 = "transform.apply_registered_pass"(%18) <{options = {}, pass_name = "lower-affine"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%19) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%19) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    %20 = "transform.structured.match"(%19) <{ops = ["gpu.launch"]}> : (!transform.any_op) -> !transform.any_op
    %21 = "transform.split_handle"(%20) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> !transform.any_op
    "transform.xegpu.set_gpu_launch_threads"(%21) <{static_threads = array<i64: 1024, 1, 1>}> : (!transform.any_op) -> ()
    %22 = "transform.apply_registered_pass"(%19) <{options = {}, pass_name = "lower-affine"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_patterns"(%22) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    %23 = "transform.apply_registered_pass"(%22) <{options = {}, pass_name = "gpu-launch-sink-index-computations"}> : (!transform.any_op) -> !transform.any_op
    %24 = "transform.apply_registered_pass"(%11) <{options = {}, pass_name = "gpu-kernel-outlining"}> : (!transform.any_op) -> !transform.any_op
    %25 = "transform.apply_registered_pass"(%24) <{options = {O = "3", chip = "bmg"}, pass_name = "xevm-attach-target"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%25) : (!transform.any_op) -> ()
    "transform.print"(%25) <{name = "outline"}> : (!transform.any_op) -> ()
    %26 = "transform.structured.match"(%25) <{ops = ["gpu.module"]}> : (!transform.any_op) -> !transform.any_op
    %27 = "transform.structured.match"(%26) <{ops = ["gpu.func"]}> : (!transform.any_op) -> !transform.any_op
    %28 = "transform.apply_registered_pass"(%27) <{options = {}, pass_name = "convert-vector-to-xegpu"}> : (!transform.any_op) -> !transform.any_op
    %29 = "transform.apply_registered_pass"(%28) <{options = {}, pass_name = "expand-strided-metadata"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%29) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%29) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.print"(%25) <{name = "vecToXegpu"}> : (!transform.any_op) -> ()
    %30 = "transform.structured.match"(%29) <{ops = ["xegpu.create_nd_tdesc"]}> : (!transform.any_op) -> !transform.any_op
    %31:2 = "transform.split_handle"(%30) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %32 = "transform.xegpu.set_desc_layout"(%31#0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_inst_data = array<i64: 4, 16>, static_sg_data = array<i64: 4, 64>, static_sg_layout = array<i64: 1, 1>}> : (!transform.any_op) -> !transform.any_op
    %33 = "transform.xegpu.set_desc_layout"(%31#1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_inst_data = array<i64: 16>, static_sg_data = array<i64: 64>, static_sg_layout = array<i64: 1>}> : (!transform.any_op) -> !transform.any_op
    %34 = "transform.structured.match"(%29) <{ops = ["xegpu.load"]}> : (!transform.any_op) -> !transform.any_op
    "transform.xegpu.set_op_layout_attr"(%34) <{index = 1 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_inst_data = array<i64: 16>, static_sg_data = array<i64: 64>, static_sg_layout = array<i64: 1>}> : (!transform.any_op) -> ()
    %35 = "transform.structured.match"(%29) <{ops = ["vector.multi_reduction"]}> : (!transform.any_op) -> !transform.any_op
    "transform.xegpu.set_op_layout_attr"(%35) <{index = 0 : i64, operandSegmentSizes = array<i32: 1, 0, 0, 0>, result, slice_dims = array<i64: 0>, static_inst_data = array<i64: 1, 16>, static_sg_data = array<i64: 4, 64>, static_sg_layout = array<i64: 1, 1>}> : (!transform.any_op) -> ()
    "transform.print"(%25) <{name = "layout"}> : (!transform.any_op) -> ()
    %36 = "transform.apply_registered_pass"(%29) <{options = {}, pass_name = "xegpu-wg-to-sg-distribute"}> : (!transform.any_op) -> !transform.any_op
    %37 = "transform.apply_registered_pass"(%36) <{options = {"layout-kind" = "inst"}, pass_name = "xegpu-propagate-layout"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%37) : (!transform.any_op) -> ()
    %38 = "transform.apply_registered_pass"(%37) <{options = {}, pass_name = "lower-affine"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%38) : (!transform.any_op) -> ()
    %39 = "transform.apply_registered_pass"(%38) <{options = {}, pass_name = "xegpu-blocking"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_patterns"(%39) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.apply_cse"(%39) : (!transform.any_op) -> ()
    %40 = "transform.apply_registered_pass"(%26) <{options = {}, pass_name = "xegpu-subgroup-distribute"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_patterns"(%40) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.apply_cse"(%40) : (!transform.any_op) -> ()
    %41 = "transform.apply_registered_pass"(%40) <{options = {}, pass_name = "loop-invariant-code-motion"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%41) : (!transform.any_op) -> ()
    %42 = "transform.apply_registered_pass"(%41) <{options = {}, pass_name = "xegpu-vector-linearize"}> : (!transform.any_op) -> !transform.any_op
    %43 = "transform.apply_registered_pass"(%42) <{options = {}, pass_name = "convert-xegpu-to-xevm"}> : (!transform.any_op) -> !transform.any_op
    %44 = "transform.apply_registered_pass"(%43) <{options = {"use-64bit-index" = "true"}, pass_name = "convert-gpu-to-llvm-spv"}> : (!transform.any_op) -> !transform.any_op
    %45 = "transform.apply_registered_pass"(%44) <{options = {}, pass_name = "convert-xevm-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%45) : (!transform.any_op) -> ()
    %46 = "transform.structured.match"(%45) <{ops = ["gpu.func"]}> : (!transform.any_op) -> !transform.any_op
    %47 = "transform.apply_registered_pass"(%46) <{options = {}, pass_name = "gpu-async-region"}> : (!transform.any_op) -> !transform.any_op
    %48 = "transform.apply_registered_pass"(%25) <{options = {}, pass_name = "reconcile-unrealized-casts"}> : (!transform.any_op) -> !transform.any_op
    %49 = "transform.apply_registered_pass"(%48) <{options = {}, pass_name = "convert-vector-to-scf"}> : (!transform.any_op) -> !transform.any_op
    %50 = "transform.apply_registered_pass"(%49) <{options = {}, pass_name = "convert-scf-to-cf"}> : (!transform.any_op) -> !transform.any_op
    %51 = "transform.apply_registered_pass"(%50) <{options = {}, pass_name = "expand-strided-metadata"}> : (!transform.any_op) -> !transform.any_op
    %52 = "transform.apply_registered_pass"(%51) <{options = {}, pass_name = "finalize-memref-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %53 = "transform.apply_registered_pass"(%52) <{options = {}, pass_name = "convert-cf-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %54 = "transform.apply_registered_pass"(%53) <{options = {}, pass_name = "convert-vector-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %55 = "transform.apply_registered_pass"(%54) <{options = {}, pass_name = "convert-arith-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %56 = "transform.apply_registered_pass"(%55) <{options = {}, pass_name = "convert-index-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %57 = "transform.apply_registered_pass"(%56) <{options = {}, pass_name = "convert-func-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %58 = "transform.apply_registered_pass"(%57) <{options = {}, pass_name = "convert-math-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %59 = "transform.apply_registered_pass"(%58) <{options = {}, pass_name = "gpu-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %60 = "transform.apply_registered_pass"(%59) <{options = {}, pass_name = "lower-affine"}> : (!transform.any_op) -> !transform.any_op
    %61 = "transform.apply_registered_pass"(%60) <{options = {}, pass_name = "reconcile-unrealized-casts"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%61) : (!transform.any_op) -> ()
    %62 = "transform.apply_registered_pass"(%61) <{options = {}, pass_name = "gpu-module-to-binary"}> : (!transform.any_op) -> !transform.any_op
    "transform.print"(%62) <{name = "xegpu"}> : (!transform.any_op) -> ()
    "transform.yield"() : () -> ()
  }) : () -> ()
}) {transform.with_named_sequence} : () -> ()

// -----

[[[ IR printer: Initial ]]]
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
module {
  func.func @main(%arg0: tensor<128x4096x4096xf32>) -> tensor<128x1x4096xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x1x4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x1x4096xf32>) -> tensor<128x1x4096xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0 : tensor<128x4096x4096xf32>) outs(%1 : tensor<128x1x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<128x1x4096xf32>
    return %2 : tensor<128x1x4096xf32>
  }
}

[[[ IR printer: tile ]]]
#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 64)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<128x4096x4096xf32>) -> tensor<128x1x4096xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x1x4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x1x4096xf32>) -> tensor<128x1x4096xf32>
    %2 = scf.forall (%arg1, %arg2, %arg3) in (128, 1024, 64) shared_outs(%arg4 = %1) -> (tensor<128x1x4096xf32>) {
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map1(%arg3)
      %extracted_slice = tensor.extract_slice %arg0[%arg1, %3, %4] [1, 4, 64] [1, 1, 1] : tensor<128x4096x4096xf32> to tensor<1x4x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg4[%arg1, 0, %4] [1, 1, 64] [1, 1, 1] : tensor<128x1x4096xf32> to tensor<1x1x64xf32>
      %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x4x64xf32> into tensor<4x64xf32>
      %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [[0, 1, 2]] : tensor<1x1x64xf32> into tensor<64xf32>
      %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel"]} ins(%collapsed : tensor<4x64xf32>) outs(%collapsed_1 : tensor<64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<64xf32>
      %expanded = tensor.expand_shape %5 [[0, 1, 2]] output_shape [1, 1, 64] : tensor<64xf32> into tensor<1x1x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %expanded into %arg4[%arg1, 0, %4] [1, 1, 64] [1, 1, 1] : tensor<1x1x64xf32> into tensor<128x1x4096xf32>
      }
    } {gpu_loop}
    return %2 : tensor<128x1x4096xf32>
  }
}

[[[ IR printer: vectorize ]]]
#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @main(%arg0: tensor<128x4096x4096xf32>) -> tensor<128x1x4096xf32> attributes {llvm.emit_c_interface} {
    %0 = ub.poison : f32
    %cst = arith.constant dense<0.000000e+00> : vector<128x1x4096xf32>
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<128x1x4096xf32>
    %2 = vector.transfer_write %cst, %1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<128x1x4096xf32>, tensor<128x1x4096xf32>
    %3 = scf.forall (%arg1, %arg2, %arg3) in (128, 1024, 64) shared_outs(%arg4 = %2) -> (tensor<128x1x4096xf32>) {
      %4 = affine.apply #map(%arg2)
      %5 = affine.apply #map1(%arg3)
      %extracted_slice = tensor.extract_slice %arg0[%arg1, %4, %5] [1, 4, 64] [1, 1, 1] : tensor<128x4096x4096xf32> to tensor<1x4x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg4[%arg1, 0, %5] [1, 1, 64] [1, 1, 1] : tensor<128x1x4096xf32> to tensor<1x1x64xf32>
      %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x4x64xf32> into tensor<4x64xf32>
      %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [[0, 1, 2]] : tensor<1x1x64xf32> into tensor<64xf32>
      %6 = vector.transfer_read %collapsed[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
      %7 = vector.transfer_read %collapsed_1[%c0], %0 {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
      %8 = vector.multi_reduction <add>, %6, %7 [0] : vector<4x64xf32> to vector<64xf32>
      %9 = vector.transfer_write %8, %collapsed_1[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
      %expanded = tensor.expand_shape %9 [[0, 1, 2]] output_shape [1, 1, 64] : tensor<64xf32> into tensor<1x1x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %expanded into %arg4[%arg1, 0, %5] [1, 1, 64] [1, 1, 1] : tensor<1x1x64xf32> into tensor<128x1x4096xf32>
      }
    } {gpu_loop}
    return %3 : tensor<128x1x4096xf32>
  }
}

[[[ IR printer: bufferize ]]]
#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @main(%arg0: memref<128x4096x4096xf32>, %arg1: memref<128x1x4096xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %0 = ub.poison : f32
    %cst = arith.constant dense<0.000000e+00> : vector<128x1x4096xf32>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst, %arg1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<128x1x4096xf32>, memref<128x1x4096xf32>
    scf.forall (%arg2, %arg3, %arg4) in (128, 1024, 64) {
      %1 = affine.apply #map(%arg3)
      %2 = affine.apply #map1(%arg4)
      %subview = memref.subview %arg0[%arg2, %1, %2] [1, 4, 64] [1, 1, 1] : memref<128x4096x4096xf32> to memref<1x4x64xf32, strided<[16777216, 4096, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg2, 0, %2] [1, 1, 64] [1, 1, 1] : memref<128x1x4096xf32> to memref<1x1x64xf32, strided<[4096, 4096, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x4x64xf32, strided<[16777216, 4096, 1], offset: ?>> into memref<4x64xf32, strided<[4096, 1], offset: ?>>
      %collapse_shape_1 = memref.collapse_shape %subview_0 [[0, 1, 2]] : memref<1x1x64xf32, strided<[4096, 4096, 1], offset: ?>> into memref<64xf32, strided<[1], offset: ?>>
      %3 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<4x64xf32, strided<[4096, 1], offset: ?>>, vector<4x64xf32>
      %4 = vector.transfer_read %collapse_shape_1[%c0], %0 {in_bounds = [true]} : memref<64xf32, strided<[1], offset: ?>>, vector<64xf32>
      %5 = vector.multi_reduction <add>, %3, %4 [0] : vector<4x64xf32> to vector<64xf32>
      vector.transfer_write %5, %collapse_shape_1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, strided<[1], offset: ?>>
    } {gpu_loop}
    return
  }
}

[[[ IR printer: outline ]]]
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<128x4096x4096xf32>, %arg1: memref<128x1x4096xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0.000000e+00> : vector<128x1x4096xf32>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst, %arg1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<128x1x4096xf32>, memref<128x1x4096xf32>
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c128, %c1024, %c64) threads in (%c1024, %c1, %c1)  args(%arg0 : memref<128x4096x4096xf32>, %arg1 : memref<128x1x4096xf32>)
    return
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<128x4096x4096xf32>, %arg1: memref<128x1x4096xf32>) kernel attributes {known_block_size = array<i32: 1024, 1, 1>, known_grid_size = array<i32: 128, 1024, 64>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %0 = ub.poison : f32
      %1 = arith.muli %block_id_y, %c4 overflow<nsw> : index
      %2 = arith.muli %block_id_z, %c64 overflow<nsw> : index
      %subview = memref.subview %arg0[%block_id_x, %1, %2] [1, 4, 64] [1, 1, 1] : memref<128x4096x4096xf32> to memref<1x4x64xf32, strided<[16777216, 4096, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%block_id_x, 0, %2] [1, 1, 64] [1, 1, 1] : memref<128x1x4096xf32> to memref<1x1x64xf32, strided<[4096, 4096, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x4x64xf32, strided<[16777216, 4096, 1], offset: ?>> into memref<4x64xf32, strided<[4096, 1], offset: ?>>
      %collapse_shape_1 = memref.collapse_shape %subview_0 [[0, 1, 2]] : memref<1x1x64xf32, strided<[4096, 4096, 1], offset: ?>> into memref<64xf32, strided<[1], offset: ?>>
      %3 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<4x64xf32, strided<[4096, 1], offset: ?>>, vector<4x64xf32>
      %4 = vector.transfer_read %collapse_shape_1[%c0], %0 {in_bounds = [true]} : memref<64xf32, strided<[1], offset: ?>>, vector<64xf32>
      %5 = vector.multi_reduction <add>, %3, %4 [0] : vector<4x64xf32> to vector<64xf32>
      vector.transfer_write %5, %collapse_shape_1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, strided<[1], offset: ?>>
      gpu.return
    }
  }
}

[[[ IR printer: vecToXegpu ]]]
#map = affine_map<()[s0, s1, s2] -> (s0 * 16777216 + s1 * 4096 + s2)>
#map1 = affine_map<()[s0, s1] -> (s0 * 4096 + s1)>
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<128x4096x4096xf32>, %arg1: memref<128x1x4096xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0.000000e+00> : vector<128x1x4096xf32>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst, %arg1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<128x1x4096xf32>, memref<128x1x4096xf32>
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c128, %c1024, %c64) threads in (%c1024, %c1, %c1)  args(%arg0 : memref<128x4096x4096xf32>, %arg1 : memref<128x1x4096xf32>)
    return
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<128x4096x4096xf32>, %arg1: memref<128x1x4096xf32>) kernel attributes {known_block_size = array<i32: 1024, 1, 1>, known_grid_size = array<i32: 128, 1024, 64>} {
      %cst = arith.constant dense<true> : vector<64xi1>
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = arith.muli %block_id_y, %c4 overflow<nsw> : index
      %1 = arith.muli %block_id_z, %c64 overflow<nsw> : index
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg1 : memref<128x1x4096xf32> -> memref<f32>, index, index, index, index, index, index, index
      %base_buffer_0, %offset_1, %sizes_2:3, %strides_3:3 = memref.extract_strided_metadata %arg0 : memref<128x4096x4096xf32> -> memref<f32>, index, index, index, index, index, index, index
      %2 = affine.apply #map()[%block_id_x, %0, %1]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer_0 : memref<f32> -> index
      %3 = arith.muli %2, %c4 : index
      %4 = arith.addi %intptr, %3 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = xegpu.create_nd_tdesc %5, shape : [4, 64], strides : [4096, 1] : i64 -> !xegpu.tensor_desc<4x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      %7 = xegpu.load_nd %6[0, 0]  : !xegpu.tensor_desc<4x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<4x64xf32>
      %8 = affine.apply #map1()[%block_id_x, %1]
      %9 = vector.step : vector<64xindex>
      %10 = vector.broadcast %8 : index to vector<64xindex>
      %11 = arith.addi %10, %9 : vector<64xindex>
      %intptr_4 = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
      %12 = arith.index_cast %intptr_4 : index to i64
      %13 = xegpu.load %12[%11], %cst  : i64, vector<64xindex>, vector<64xi1> -> vector<64xf32>
      %14 = vector.multi_reduction <add>, %7, %13 [0] : vector<4x64xf32> to vector<64xf32>
      %15 = arith.muli %8, %c4 : index
      %16 = arith.addi %intptr_4, %15 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = xegpu.create_nd_tdesc %17, shape : [64], strides : [1] : i64 -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %14, %18[0]  : vector<64xf32>, !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}

[[[ IR printer: layout ]]]
#map = affine_map<()[s0, s1, s2] -> (s0 * 16777216 + s1 * 4096 + s2)>
#map1 = affine_map<()[s0, s1] -> (s0 * 4096 + s1)>
module attributes {gpu.container_module} {
  func.func @main(%arg0: memref<128x4096x4096xf32>, %arg1: memref<128x1x4096xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0.000000e+00> : vector<128x1x4096xf32>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst, %arg1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<128x1x4096xf32>, memref<128x1x4096xf32>
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c128, %c1024, %c64) threads in (%c1024, %c1, %c1)  args(%arg0 : memref<128x4096x4096xf32>, %arg1 : memref<128x1x4096xf32>)
    return
  }
  gpu.module @main_kernel [#xevm.target<O = 3>] {
    gpu.func @main_kernel(%arg0: memref<128x4096x4096xf32>, %arg1: memref<128x1x4096xf32>) kernel attributes {known_block_size = array<i32: 1024, 1, 1>, known_grid_size = array<i32: 128, 1024, 64>} {
      %cst = arith.constant dense<true> : vector<64xi1>
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %0 = arith.muli %block_id_y, %c4 overflow<nsw> : index
      %1 = arith.muli %block_id_z, %c64 overflow<nsw> : index
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg1 : memref<128x1x4096xf32> -> memref<f32>, index, index, index, index, index, index, index
      %base_buffer_0, %offset_1, %sizes_2:3, %strides_3:3 = memref.extract_strided_metadata %arg0 : memref<128x4096x4096xf32> -> memref<f32>, index, index, index, index, index, index, index
      %2 = affine.apply #map()[%block_id_x, %0, %1]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer_0 : memref<f32> -> index
      %3 = arith.muli %2, %c4 : index
      %4 = arith.addi %intptr, %3 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = xegpu.create_nd_tdesc %5, shape : [4, 64], strides : [4096, 1] : i64 -> !xegpu.tensor_desc<4x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [1, 1], sg_data = [4, 64], inst_data = [4, 16]>>
      %7 = xegpu.load_nd %6[0, 0]  : !xegpu.tensor_desc<4x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [1, 1], sg_data = [4, 64], inst_data = [4, 16]>> -> vector<4x64xf32>
      %8 = affine.apply #map1()[%block_id_x, %1]
      %9 = vector.step : vector<64xindex>
      %10 = vector.broadcast %8 : index to vector<64xindex>
      %11 = arith.addi %10, %9 : vector<64xindex>
      %intptr_4 = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
      %12 = arith.index_cast %intptr_4 : index to i64
      %13 = xegpu.load %12[%11], %cst  {layout_operand_1 = #xegpu.layout<sg_layout = [1], sg_data = [64], inst_data = [16]>} : i64, vector<64xindex>, vector<64xi1> -> vector<64xf32>
      %14 = vector.multi_reduction <add>, %7, %13 {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [1, 1], sg_data = [4, 64], inst_data = [1, 16]>, dims = [0]>} [0] : vector<4x64xf32> to vector<64xf32>
      %15 = arith.muli %8, %c4 : index
      %16 = arith.addi %intptr_4, %15 : index
      %17 = arith.index_cast %16 : index to i64
      %18 = xegpu.create_nd_tdesc %17, shape : [64], strides : [1] : i64 -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [1], sg_data = [64], inst_data = [16]>>
      xegpu.store_nd %14, %18[0]  : vector<64xf32>, !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<sg_layout = [1], sg_data = [64], inst_data = [16]>>
      gpu.return
    }
  }
}
