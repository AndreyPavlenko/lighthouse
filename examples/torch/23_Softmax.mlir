#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @main(%arg0: tensor<4096x256xf32>) -> tensor<4096x256xf32> attributes {llvm.emit_c_interface} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<4096xi64>) -> tensor<4096xi64>
    %2 = tensor.empty() : tensor<4096xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4096xf32>) -> tensor<4096xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4096x256xf32>) outs(%3, %1 : tensor<4096xf32>, tensor<4096xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %12 = linalg.index 1 : index
      %13 = arith.index_cast %12 : index to i64
      %14 = arith.maximumf %in, %out : f32
      %15 = arith.cmpf ogt, %in, %out : f32
      %16 = arith.select %15, %13, %out_1 : i64
      linalg.yield %14, %16 : f32, i64
    } -> (tensor<4096xf32>, tensor<4096xi64>)
    %expanded = tensor.expand_shape %4#0 [[0, 1]] output_shape [4096, 1] : tensor<4096xf32> into tensor<4096x1xf32>
    %5 = tensor.empty() : tensor<4096x256xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %expanded : tensor<4096x256xf32>, tensor<4096x1xf32>) outs(%5 : tensor<4096x256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.subf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<4096x256xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<4096x256xf32>) outs(%5 : tensor<4096x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = math.exp %in : f32
      linalg.yield %12 : f32
    } -> tensor<4096x256xf32>
    %8 = tensor.empty() : tensor<4096x1xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<4096x1xf32>) -> tensor<4096x1xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<4096x256xf32>) outs(%9 : tensor<4096x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.addf %in, %out : f32
      linalg.yield %12 : f32
    } -> tensor<4096x1xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %10 : tensor<4096x256xf32>, tensor<4096x1xf32>) outs(%5 : tensor<4096x256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.divf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<4096x256xf32>
    return %11 : tensor<4096x256xf32>
  }
}

"builtin.module"() ({
  "transform.named_sequence"() <{arg_attrs = [{transform.readonly}], function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
  ^bb0(%arg0: !transform.any_op):
    "transform.print"(%arg0) <{name = "Initial"}> : (!transform.any_op) -> ()
    %0 = "transform.structured.match"(%arg0) <{ops = ["linalg.fill", "linalg.generic"]}> : (!transform.any_op) -> !transform.any_op
    %1:7 = "transform.split_handle"(%0) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %2:2 = "transform.structured.tile_using_forall"(%1#6) <{operandSegmentSizes = array<i32: 1, 0, 0, 0, 0>, static_num_threads = array<i64>, static_tile_sizes = array<i64: 1>}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    "transform.annotate"(%2#1) <{name = "gpu_loop"}> : (!transform.any_op) -> ()
    %3:2 = "transform.structured.fuse_into_containing_op"(%1#5, %2#1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %4:2 = "transform.structured.fuse_into_containing_op"(%1#4, %3#1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %5:2 = "transform.structured.fuse_into_containing_op"(%1#3, %4#1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %6:2 = "transform.structured.fuse_into_containing_op"(%1#2, %5#1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %7:2 = "transform.structured.fuse_into_containing_op"(%1#1, %6#1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %8:2 = "transform.structured.fuse_into_containing_op"(%1#0, %7#1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    "transform.apply_cse"(%arg0) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%arg0) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes"() : () -> ()
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.print"(%arg0) <{name = "tile"}> : (!transform.any_op) -> ()
    %9 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %10 = "transform.structured.vectorize_children_and_apply_patterns"(%9) <{fold_type_extensions_into_contract}> : (!transform.any_op) -> !transform.any_op
    %11 = "transform.structured.match"(%10) <{ops = ["scf.for", "scf.forall"]}> : (!transform.any_op) -> !transform.any_op
    "transform.loop.hoist_loop_invariant_subsets"(%11) : (!transform.any_op) -> ()
    "transform.apply_cse"(%10) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%10) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.print"(%arg0) <{name = "vectorize"}> : (!transform.any_op) -> ()
    %12 = "transform.bufferization.one_shot_bufferize"(%arg0) <{allow_return_allocs_from_loops = true, allow_unknown_ops = true, bufferize_function_boundaries = true, check_parallel_regions = true, dump_alias_sets = false, function_boundary_type_conversion = 1 : i32, memcpy_op = "memref.copy", print_conflicts = false, test_analysis_only = false}> : (!transform.any_op) -> !transform.any_op
    %13 = "transform.bufferization.one_shot_bufferize"(%12) <{allow_return_allocs_from_loops = false, allow_unknown_ops = true, bufferize_function_boundaries = true, check_parallel_regions = true, dump_alias_sets = false, function_boundary_type_conversion = 1 : i32, memcpy_op = "memref.copy", print_conflicts = false, test_analysis_only = false}> : (!transform.any_op) -> !transform.any_op
    %14 = "transform.apply_registered_pass"(%13) <{options = {}, pass_name = "fold-memref-alias-ops"}> : (!transform.any_op) -> !transform.any_op
    %15 = "transform.apply_registered_pass"(%14) <{options = {}, pass_name = "drop-equivalent-buffer-results"}> : (!transform.any_op) -> !transform.any_op
    %16 = "transform.apply_registered_pass"(%15) <{options = {"add-result-attr" = true, "hoist-dynamic-allocs" = true, "hoist-static-allocs" = true, "modify-public-functions" = true}, pass_name = "buffer-results-to-out-params"}> : (!transform.any_op) -> !transform.any_op
    %17 = "transform.apply_registered_pass"(%16) <{options = {}, pass_name = "buffer-deallocation-simplification"}> : (!transform.any_op) -> !transform.any_op
    %18 = "transform.apply_registered_pass"(%17) <{options = {}, pass_name = "memref-expand"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%18) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%18) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.memref.erase_dead_alloc_and_stores"(%18) : (!transform.any_op) -> ()
    %19 = "transform.structured.match"(%18) <{ops = ["scf.for", "scf.forall"]}> : (!transform.any_op) -> !transform.any_op
    "transform.loop.hoist_loop_invariant_subsets"(%19) : (!transform.any_op) -> ()
    "transform.apply_cse"(%18) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%18) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.print"(%18) <{name = "bufferize"}> : (!transform.any_op) -> ()
    %20 = "transform.structured.match"(%18) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %21 = "transform.structured.match"(%20) <{op_attrs = {gpu_loop}}> : (!transform.any_op) -> !transform.any_op
    %22 = "transform.split_handle"(%21) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> !transform.any_op
    %23 = "transform.loop.forall_to_parallel"(%22) : (!transform.any_op) -> !transform.any_op
    %24 = "transform.apply_registered_pass"(%20) <{options = {}, pass_name = "gpu-map-parallel-loops"}> : (!transform.any_op) -> !transform.any_op
    %25 = "transform.apply_registered_pass"(%24) <{options = {}, pass_name = "convert-parallel-loops-to-gpu"}> : (!transform.any_op) -> !transform.any_op
    %26 = "transform.apply_registered_pass"(%25) <{options = {}, pass_name = "lower-affine"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%26) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%26) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    %27 = "transform.structured.match"(%26) <{ops = ["gpu.launch"]}> : (!transform.any_op) -> !transform.any_op
    %28 = "transform.split_handle"(%27) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> !transform.any_op
    "transform.xegpu.set_gpu_launch_threads"(%28) <{static_threads = array<i64: 16, 1, 1>}> : (!transform.any_op) -> ()
    %29 = "transform.apply_registered_pass"(%26) <{options = {}, pass_name = "lower-affine"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_patterns"(%29) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    %30 = "transform.apply_registered_pass"(%29) <{options = {}, pass_name = "gpu-launch-sink-index-computations"}> : (!transform.any_op) -> !transform.any_op
    %31 = "transform.apply_registered_pass"(%18) <{options = {}, pass_name = "gpu-kernel-outlining"}> : (!transform.any_op) -> !transform.any_op
    %32 = "transform.apply_registered_pass"(%31) <{options = {O = "3", chip = "bmg"}, pass_name = "xevm-attach-target"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%32) : (!transform.any_op) -> ()
    %33 = "transform.structured.match"(%32) <{ops = ["gpu.func"]}> : (!transform.any_op) -> !transform.any_op
    %34 = "transform.split_handle"(%33) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> !transform.any_op
    %35 = "transform.param.constant"() <{value = 16 : i32}> : () -> !transform.param<i32>
    "transform.annotate"(%34, %35) <{name = "intel_reqd_sub_group_size"}> : (!transform.any_op, !transform.param<i32>) -> ()
    "transform.print"(%32) <{name = "outline"}> : (!transform.any_op) -> ()
    %36 = "transform.structured.match"(%32) <{ops = ["gpu.module"]}> : (!transform.any_op) -> !transform.any_op
    %37 = "transform.structured.match"(%36) <{ops = ["gpu.func"]}> : (!transform.any_op) -> !transform.any_op
    %38 = "transform.apply_registered_pass"(%37) <{options = {}, pass_name = "convert-vector-to-xegpu"}> : (!transform.any_op) -> !transform.any_op
    %39 = "transform.apply_registered_pass"(%38) <{options = {}, pass_name = "expand-strided-metadata"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%39) : (!transform.any_op) -> ()
    "transform.apply_patterns"(%39) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.print"(%32) <{name = "vecToXegpu"}> : (!transform.any_op) -> ()
    %40 = "transform.structured.match"(%39) <{ops = ["xegpu.create_nd_tdesc"]}> : (!transform.any_op) -> !transform.any_op
    %41:2 = "transform.split_handle"(%40) <{fail_on_payload_too_small = true, pass_through_empty_handle = true}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %42 = "transform.xegpu.set_desc_layout"(%41#0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_inst_data = array<i64: 1, 16>, static_sg_data = array<i64: 1, 256>, static_sg_layout = array<i64: 1, 1>}> : (!transform.any_op) -> !transform.any_op
    %43 = "transform.xegpu.set_desc_layout"(%41#1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_inst_data = array<i64: 1, 16>, static_sg_data = array<i64: 1, 256>, static_sg_layout = array<i64: 1, 1>}> : (!transform.any_op) -> !transform.any_op
    "transform.print"(%32) <{name = "layout"}> : (!transform.any_op) -> ()
    %44 = "transform.apply_registered_pass"(%39) <{options = {}, pass_name = "xegpu-wg-to-sg-distribute"}> : (!transform.any_op) -> !transform.any_op
    %45 = "transform.apply_registered_pass"(%44) <{options = {"layout-kind" = "inst"}, pass_name = "xegpu-propagate-layout"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%45) : (!transform.any_op) -> ()
    %46 = "transform.apply_registered_pass"(%45) <{options = {}, pass_name = "lower-affine"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%46) : (!transform.any_op) -> ()
    %47 = "transform.apply_registered_pass"(%46) <{options = {}, pass_name = "xegpu-blocking"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_patterns"(%47) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.apply_cse"(%47) : (!transform.any_op) -> ()
    %48 = "transform.apply_registered_pass"(%36) <{options = {}, pass_name = "xegpu-subgroup-distribute"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_patterns"(%48) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
      "transform.apply_patterns.canonicalization"() : () -> ()
    }) : (!transform.any_op) -> ()
    "transform.apply_cse"(%48) : (!transform.any_op) -> ()
    %49 = "transform.apply_registered_pass"(%48) <{options = {}, pass_name = "loop-invariant-code-motion"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%49) : (!transform.any_op) -> ()
    %50 = "transform.apply_registered_pass"(%49) <{options = {}, pass_name = "xegpu-vector-linearize"}> : (!transform.any_op) -> !transform.any_op
    %51 = "transform.apply_registered_pass"(%50) <{options = {}, pass_name = "convert-xegpu-to-xevm"}> : (!transform.any_op) -> !transform.any_op
    %52 = "transform.apply_registered_pass"(%51) <{options = {"use-64bit-index" = "true"}, pass_name = "convert-gpu-to-llvm-spv"}> : (!transform.any_op) -> !transform.any_op
    %53 = "transform.apply_registered_pass"(%52) <{options = {}, pass_name = "convert-xevm-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%53) : (!transform.any_op) -> ()
    %54 = "transform.structured.match"(%53) <{ops = ["gpu.func"]}> : (!transform.any_op) -> !transform.any_op
    %55 = "transform.apply_registered_pass"(%54) <{options = {}, pass_name = "gpu-async-region"}> : (!transform.any_op) -> !transform.any_op
    %56 = "transform.apply_registered_pass"(%32) <{options = {}, pass_name = "reconcile-unrealized-casts"}> : (!transform.any_op) -> !transform.any_op
    %57 = "transform.apply_registered_pass"(%56) <{options = {}, pass_name = "convert-vector-to-scf"}> : (!transform.any_op) -> !transform.any_op
    %58 = "transform.apply_registered_pass"(%57) <{options = {}, pass_name = "convert-scf-to-cf"}> : (!transform.any_op) -> !transform.any_op
    %59 = "transform.apply_registered_pass"(%58) <{options = {}, pass_name = "expand-strided-metadata"}> : (!transform.any_op) -> !transform.any_op
    %60 = "transform.apply_registered_pass"(%59) <{options = {}, pass_name = "finalize-memref-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %61 = "transform.apply_registered_pass"(%60) <{options = {}, pass_name = "convert-cf-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %62 = "transform.apply_registered_pass"(%61) <{options = {}, pass_name = "convert-vector-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %63 = "transform.apply_registered_pass"(%62) <{options = {}, pass_name = "convert-arith-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %64 = "transform.apply_registered_pass"(%63) <{options = {}, pass_name = "convert-index-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %65 = "transform.apply_registered_pass"(%64) <{options = {}, pass_name = "convert-func-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %66 = "transform.apply_registered_pass"(%65) <{options = {}, pass_name = "convert-math-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %67 = "transform.apply_registered_pass"(%66) <{options = {}, pass_name = "gpu-to-llvm"}> : (!transform.any_op) -> !transform.any_op
    %68 = "transform.apply_registered_pass"(%67) <{options = {}, pass_name = "lower-affine"}> : (!transform.any_op) -> !transform.any_op
    %69 = "transform.apply_registered_pass"(%68) <{options = {}, pass_name = "reconcile-unrealized-casts"}> : (!transform.any_op) -> !transform.any_op
    "transform.apply_cse"(%69) : (!transform.any_op) -> ()
    %70 = "transform.apply_registered_pass"(%69) <{options = {}, pass_name = "gpu-module-to-binary"}> : (!transform.any_op) -> !transform.any_op
    "transform.print"(%70) <{name = "xegpu"}> : (!transform.any_op) -> ()
    "transform.yield"() : () -> ()
  }) : () -> ()
}) {transform.with_named_sequence} : () -> ()

// -----

[[[ IR printer: Initial ]]]
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%arg0: tensor<4096x256xf32>) -> tensor<4096x256xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4096x256xf32>) outs(%1 : tensor<4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.maximumf %in, %out : f32
      linalg.yield %10 : f32
    } -> tensor<4096xf32>
    %3 = tensor.empty() : tensor<4096x256xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<4096x256xf32>, tensor<4096xf32>) outs(%3 : tensor<4096x256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %10 = arith.subf %in, %in_1 : f32
      linalg.yield %10 : f32
    } -> tensor<4096x256xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<4096x256xf32>) outs(%3 : tensor<4096x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = math.exp %in : f32
      linalg.yield %10 : f32
    } -> tensor<4096x256xf32>
    %6 = tensor.empty() : tensor<4096x1xf32>
    %collapsed = tensor.collapse_shape %6 [[0, 1]] : tensor<4096x1xf32> into tensor<4096xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%collapsed : tensor<4096xf32>) -> tensor<4096xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<4096x256xf32>) outs(%7 : tensor<4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.addf %in, %out : f32
      linalg.yield %10 : f32
    } -> tensor<4096xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %8 : tensor<4096x256xf32>, tensor<4096xf32>) outs(%3 : tensor<4096x256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %10 = arith.divf %in, %in_1 : f32
      linalg.yield %10 : f32
    } -> tensor<4096x256xf32>
    return %9 : tensor<4096x256xf32>
  }
}

[[[ IR printer: tile ]]]
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @main(%arg0: tensor<4096x256xf32>) -> tensor<4096x256xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = tensor.empty() : tensor<4096x256xf32>
    %2 = tensor.empty() : tensor<4096x1xf32>
    %collapsed = tensor.collapse_shape %2 [[0, 1]] : tensor<4096x1xf32> into tensor<4096xf32>
    %3 = scf.forall (%arg1) in (4096) shared_outs(%arg2 = %1) -> (tensor<4096x256xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [1, 256] [1, 1] : tensor<4096x256xf32> to tensor<1x256xf32>
      %extracted_slice_1 = tensor.extract_slice %0[%arg1] [1] [1] : tensor<4096xf32> to tensor<1xf32>
      %collapsed_2 = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x256xf32> into tensor<256xf32>
      %collapsed_3 = tensor.collapse_shape %extracted_slice_1 [] : tensor<1xf32> into tensor<f32>
      %4 = linalg.fill ins(%cst : f32) outs(%collapsed_3 : tensor<f32>) -> tensor<f32>
      %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%collapsed_2 : tensor<256xf32>) outs(%4 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %11 = arith.maximumf %in, %out : f32
        linalg.yield %11 : f32
      } -> tensor<f32>
      %extracted_slice_4 = tensor.extract_slice %arg2[%arg1, 0] [1, 256] [1, 1] : tensor<4096x256xf32> to tensor<1x256xf32>
      %collapsed_5 = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x256xf32> into tensor<256xf32>
      %collapsed_6 = tensor.collapse_shape %extracted_slice_4 [[0, 1]] : tensor<1x256xf32> into tensor<256xf32>
      %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%collapsed_5, %5 : tensor<256xf32>, tensor<f32>) outs(%collapsed_6 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %11 = arith.subf %in, %in_11 : f32
        linalg.yield %11 : f32
      } -> tensor<256xf32>
      %collapsed_7 = tensor.collapse_shape %extracted_slice_4 [[0, 1]] : tensor<1x256xf32> into tensor<256xf32>
      %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%6 : tensor<256xf32>) outs(%collapsed_7 : tensor<256xf32>) {
      ^bb0(%in: f32, %out: f32):
        %11 = math.exp %in : f32
        linalg.yield %11 : f32
      } -> tensor<256xf32>
      %extracted_slice_8 = tensor.extract_slice %collapsed[%arg1] [1] [1] : tensor<4096xf32> to tensor<1xf32>
      %collapsed_9 = tensor.collapse_shape %extracted_slice_8 [] : tensor<1xf32> into tensor<f32>
      %8 = linalg.fill ins(%cst_0 : f32) outs(%collapsed_9 : tensor<f32>) -> tensor<f32>
      %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%7 : tensor<256xf32>) outs(%8 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %11 = arith.addf %in, %out : f32
        linalg.yield %11 : f32
      } -> tensor<f32>
      %collapsed_10 = tensor.collapse_shape %extracted_slice_4 [[0, 1]] : tensor<1x256xf32> into tensor<256xf32>
      %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%7, %9 : tensor<256xf32>, tensor<f32>) outs(%collapsed_10 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %11 = arith.divf %in, %in_11 : f32
        linalg.yield %11 : f32
      } -> tensor<256xf32>
      %expanded = tensor.expand_shape %10 [[0, 1]] output_shape [1, 256] : tensor<256xf32> into tensor<1x256xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %expanded into %arg2[%arg1, 0] [1, 256] [1, 1] : tensor<1x256xf32> into tensor<4096x256xf32>
      }
    } {gpu_loop}
    return %3 : tensor<4096x256xf32>
  }
}

[[[ IR printer: vectorize ]]]
module {
  func.func @main(%arg0: tensor<4096x256xf32>) -> tensor<4096x256xf32> attributes {llvm.emit_c_interface} {
    %0 = ub.poison : f32
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<4096x256xf32>
    %2 = scf.forall (%arg1) in (4096) shared_outs(%arg2 = %1) -> (tensor<4096x256xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [1, 256] [1, 1] : tensor<4096x256xf32> to tensor<1x256xf32>
      %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x256xf32> into tensor<256xf32>
      %3 = vector.transfer_read %collapsed[%c0], %0 {in_bounds = [true]} : tensor<256xf32>, vector<256xf32>
      %4 = vector.multi_reduction <maximumf>, %3, %cst [0] : vector<256xf32> to f32
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg1, 0] [1, 256] [1, 1] : tensor<4096x256xf32> to tensor<1x256xf32>
      %5 = vector.broadcast %4 : f32 to vector<256xf32>
      %6 = arith.subf %3, %5 : vector<256xf32>
      %7 = math.exp %6 : vector<256xf32>
      %8 = vector.multi_reduction <add>, %7, %cst_0 [0] : vector<256xf32> to f32
      %collapsed_2 = tensor.collapse_shape %extracted_slice_1 [[0, 1]] : tensor<1x256xf32> into tensor<256xf32>
      %9 = vector.broadcast %8 : f32 to vector<256xf32>
      %10 = arith.divf %7, %9 : vector<256xf32>
      %11 = vector.transfer_write %10, %collapsed_2[%c0] {in_bounds = [true]} : vector<256xf32>, tensor<256xf32>
      %expanded = tensor.expand_shape %11 [[0, 1]] output_shape [1, 256] : tensor<256xf32> into tensor<1x256xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %expanded into %arg2[%arg1, 0] [1, 256] [1, 1] : tensor<1x256xf32> into tensor<4096x256xf32>
      }
    } {gpu_loop}
    return %2 : tensor<4096x256xf32>
  }
}

[[[ IR printer: bufferize ]]]
module {
  func.func @main(%arg0: memref<4096x256xf32>, %arg1: memref<4096x256xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %0 = ub.poison : f32
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    scf.forall (%arg2) in (4096) {
      %subview = memref.subview %arg0[%arg2, 0] [1, 256] [1, 1] : memref<4096x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<1x256xf32, strided<[256, 1], offset: ?>> into memref<256xf32, strided<[1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0], %0 {in_bounds = [true]} : memref<256xf32, strided<[1], offset: ?>>, vector<256xf32>
      %2 = vector.multi_reduction <maximumf>, %1, %cst [0] : vector<256xf32> to f32
      %subview_1 = memref.subview %arg1[%arg2, 0] [1, 256] [1, 1] : memref<4096x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
      %3 = vector.broadcast %2 : f32 to vector<256xf32>
      %4 = arith.subf %1, %3 : vector<256xf32>
      %5 = math.exp %4 : vector<256xf32>
      %6 = vector.multi_reduction <add>, %5, %cst_0 [0] : vector<256xf32> to f32
      %collapse_shape_2 = memref.collapse_shape %subview_1 [[0, 1]] : memref<1x256xf32, strided<[256, 1], offset: ?>> into memref<256xf32, strided<[1], offset: ?>>
      %7 = vector.broadcast %6 : f32 to vector<256xf32>
      %8 = arith.divf %5, %7 : vector<256xf32>
      vector.transfer_write %8, %collapse_shape_2[%c0] {in_bounds = [true]} : vector<256xf32>, memref<256xf32, strided<[1], offset: ?>>
    } {gpu_loop}
    return
  }
}

[[[ IR printer: outline ]]]
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
      %block_id_x = gpu.block_id  x
      %c0 = arith.constant 0 : index
      %0 = ub.poison : f32
      %cst = arith.constant 0xFF800000 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %subview = memref.subview %arg0[%block_id_x, 0] [1, 256] [1, 1] : memref<4096x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<1x256xf32, strided<[256, 1], offset: ?>> into memref<256xf32, strided<[1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0], %0 {in_bounds = [true]} : memref<256xf32, strided<[1], offset: ?>>, vector<256xf32>
      %2 = vector.multi_reduction <maximumf>, %1, %cst [0] : vector<256xf32> to f32
      %subview_1 = memref.subview %arg1[%block_id_x, 0] [1, 256] [1, 1] : memref<4096x256xf32> to memref<1x256xf32, strided<[256, 1], offset: ?>>
      %3 = vector.broadcast %2 : f32 to vector<256xf32>
      %4 = arith.subf %1, %3 : vector<256xf32>
      %5 = math.exp %4 : vector<256xf32>
      %6 = vector.multi_reduction <add>, %5, %cst_0 [0] : vector<256xf32> to f32
      %collapse_shape_2 = memref.collapse_shape %subview_1 [[0, 1]] : memref<1x256xf32, strided<[256, 1], offset: ?>> into memref<256xf32, strided<[1], offset: ?>>
      %7 = vector.broadcast %6 : f32 to vector<256xf32>
      %8 = arith.divf %5, %7 : vector<256xf32>
      vector.transfer_write %8, %collapse_shape_2[%c0] {in_bounds = [true]} : vector<256xf32>, memref<256xf32, strided<[1], offset: ?>>
      gpu.return
    }
  }
}

[[[ IR printer: vecToXegpu ]]]
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
      %1 = vector.step : vector<256xindex>
      %2 = vector.broadcast %0 : index to vector<256xindex>
      %3 = arith.addi %2, %1 : vector<256xindex>
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
      %4 = arith.index_cast %intptr : index to i64
      %5 = xegpu.load %4[%3], %cst  : i64, vector<256xindex>, vector<256xi1> -> vector<256xf32>
      %6 = vector.multi_reduction <maximumf>, %5, %cst_1 [0] : vector<256xf32> to f32
      %7 = vector.broadcast %6 : f32 to vector<256xf32>
      %8 = arith.subf %5, %7 : vector<256xf32>
      %9 = math.exp %8 : vector<256xf32>
      %10 = vector.multi_reduction <add>, %9, %cst_0 [0] : vector<256xf32> to f32
      %11 = vector.broadcast %10 : f32 to vector<256xf32>
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

[[[ IR printer: layout ]]]
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
      %1 = vector.step : vector<256xindex>
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
