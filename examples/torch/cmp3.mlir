// Matmul
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
      %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4096x4096xf16> -> index
      %0 = arith.index_castui %intptr : index to i64
      %intptr_0 = memref.extract_aligned_pointer_as_index %arg2 : memref<4096x4096xf16> -> index
      %1 = arith.index_castui %intptr_0 : index to i64
      %intptr_1 = memref.extract_aligned_pointer_as_index %arg0 : memref<4096x4096xf32> -> index
      %2 = arith.index_castui %intptr_1 : index to i64
      %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c4096 = arith.constant 4096 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c48 = arith.constant 48 : index
      %c24 = arith.constant 24 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %3 = gpu.subgroup_id : index
      %4 = arith.divui %3, %c4 : index
      %5 = arith.remui %3, %c4 : index
      %6 = arith.remui %4, %c8 : index
      %7 = arith.muli %5, %c64 : index
      %8 = arith.muli %6, %c32 : index
      %9 = arith.muli %block_id_y, %c256 overflow<nsw> : index
      %10 = arith.remui %7, %c256 : index
      %11 = arith.muli %block_id_x, %c256 overflow<nsw> : index
      %12 = arith.remui %8, %c256 : index
      %13 = arith.addi %10, %9 : index
      %14 = arith.addi %12, %11 : index
      %cst_2 = arith.constant dense<0> : vector<8xi32>
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_3 = arith.constant 0 : i32
      %c4096_i64 = arith.constant 4096 : i64
      %15 = arith.trunci %c4096_i64 : i64 to i32
      %c4096_i64_4 = arith.constant 4096 : i64
      %16 = arith.trunci %c4096_i64_4 : i64 to i32
      %17 = vector.bitcast %cst_2 : vector<8xi32> to vector<4xi64>
      %18 = vector.insert %2, %17 [0] : i64 into vector<4xi64>
      %19 = vector.bitcast %18 : vector<4xi64> to vector<8xi32>
      %20 = vector.insert %15, %19 [2] : i32 into vector<8xi32>
      %21 = vector.insert %16, %20 [3] : i32 into vector<8xi32>
      %22 = vector.insert %c0_i32, %21 [4] : i32 into vector<8xi32>
      %23 = vector.insert %c0_i32_3, %22 [5] : i32 into vector<8xi32>
      %c4_i32 = arith.constant 4 : i32
      %24 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %25 = vector.extract %24[0] : i64 from vector<4xi64>
      %26 = vector.extract %23[2] : i32 from vector<8xi32>
      %27 = vector.extract %23[3] : i32 from vector<8xi32>
      %28 = arith.index_cast %13 : index to i32
      %29 = arith.index_cast %14 : index to i32
      %30 = llvm.inttoptr %25 : i64 to !llvm.ptr<1>
      %31 = arith.muli %26, %c4_i32 : i32
      %32 = xevm.blockload2d %30, %31, %27, %31, %28, %29 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %33 = vector.bitcast %32 : vector<8xi32> to vector<8xf32>
      %34 = vector.shape_cast %33 : vector<8xf32> to vector<8x1xf32>
      %35 = vector.shape_cast %34 : vector<8x1xf32> to vector<8xf32>
      %36 = arith.addi %13, %c16 : index
      %c4_i32_5 = arith.constant 4 : i32
      %37 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %38 = vector.extract %37[0] : i64 from vector<4xi64>
      %39 = vector.extract %23[2] : i32 from vector<8xi32>
      %40 = vector.extract %23[3] : i32 from vector<8xi32>
      %41 = arith.index_cast %36 : index to i32
      %42 = arith.index_cast %14 : index to i32
      %43 = llvm.inttoptr %38 : i64 to !llvm.ptr<1>
      %44 = arith.muli %39, %c4_i32_5 : i32
      %45 = xevm.blockload2d %43, %44, %40, %44, %41, %42 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %46 = vector.bitcast %45 : vector<8xi32> to vector<8xf32>
      %47 = vector.shape_cast %46 : vector<8xf32> to vector<8x1xf32>
      %48 = vector.shape_cast %47 : vector<8x1xf32> to vector<8xf32>
      %49 = arith.addi %13, %c32 : index
      %c4_i32_6 = arith.constant 4 : i32
      %50 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %51 = vector.extract %50[0] : i64 from vector<4xi64>
      %52 = vector.extract %23[2] : i32 from vector<8xi32>
      %53 = vector.extract %23[3] : i32 from vector<8xi32>
      %54 = arith.index_cast %49 : index to i32
      %55 = arith.index_cast %14 : index to i32
      %56 = llvm.inttoptr %51 : i64 to !llvm.ptr<1>
      %57 = arith.muli %52, %c4_i32_6 : i32
      %58 = xevm.blockload2d %56, %57, %53, %57, %54, %55 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %59 = vector.bitcast %58 : vector<8xi32> to vector<8xf32>
      %60 = vector.shape_cast %59 : vector<8xf32> to vector<8x1xf32>
      %61 = vector.shape_cast %60 : vector<8x1xf32> to vector<8xf32>
      %62 = arith.addi %13, %c48 : index
      %c4_i32_7 = arith.constant 4 : i32
      %63 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %64 = vector.extract %63[0] : i64 from vector<4xi64>
      %65 = vector.extract %23[2] : i32 from vector<8xi32>
      %66 = vector.extract %23[3] : i32 from vector<8xi32>
      %67 = arith.index_cast %62 : index to i32
      %68 = arith.index_cast %14 : index to i32
      %69 = llvm.inttoptr %64 : i64 to !llvm.ptr<1>
      %70 = arith.muli %65, %c4_i32_7 : i32
      %71 = xevm.blockload2d %69, %70, %66, %70, %67, %68 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %72 = vector.bitcast %71 : vector<8xi32> to vector<8xf32>
      %73 = vector.shape_cast %72 : vector<8xf32> to vector<8x1xf32>
      %74 = vector.shape_cast %73 : vector<8x1xf32> to vector<8xf32>
      %75 = arith.addi %14, %c8 : index
      %c4_i32_8 = arith.constant 4 : i32
      %76 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %77 = vector.extract %76[0] : i64 from vector<4xi64>
      %78 = vector.extract %23[2] : i32 from vector<8xi32>
      %79 = vector.extract %23[3] : i32 from vector<8xi32>
      %80 = arith.index_cast %13 : index to i32
      %81 = arith.index_cast %75 : index to i32
      %82 = llvm.inttoptr %77 : i64 to !llvm.ptr<1>
      %83 = arith.muli %78, %c4_i32_8 : i32
      %84 = xevm.blockload2d %82, %83, %79, %83, %80, %81 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %85 = vector.bitcast %84 : vector<8xi32> to vector<8xf32>
      %86 = vector.shape_cast %85 : vector<8xf32> to vector<8x1xf32>
      %87 = vector.shape_cast %86 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_9 = arith.constant 4 : i32
      %88 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %89 = vector.extract %88[0] : i64 from vector<4xi64>
      %90 = vector.extract %23[2] : i32 from vector<8xi32>
      %91 = vector.extract %23[3] : i32 from vector<8xi32>
      %92 = arith.index_cast %36 : index to i32
      %93 = arith.index_cast %75 : index to i32
      %94 = llvm.inttoptr %89 : i64 to !llvm.ptr<1>
      %95 = arith.muli %90, %c4_i32_9 : i32
      %96 = xevm.blockload2d %94, %95, %91, %95, %92, %93 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %97 = vector.bitcast %96 : vector<8xi32> to vector<8xf32>
      %98 = vector.shape_cast %97 : vector<8xf32> to vector<8x1xf32>
      %99 = vector.shape_cast %98 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_10 = arith.constant 4 : i32
      %100 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %101 = vector.extract %100[0] : i64 from vector<4xi64>
      %102 = vector.extract %23[2] : i32 from vector<8xi32>
      %103 = vector.extract %23[3] : i32 from vector<8xi32>
      %104 = arith.index_cast %49 : index to i32
      %105 = arith.index_cast %75 : index to i32
      %106 = llvm.inttoptr %101 : i64 to !llvm.ptr<1>
      %107 = arith.muli %102, %c4_i32_10 : i32
      %108 = xevm.blockload2d %106, %107, %103, %107, %104, %105 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %109 = vector.bitcast %108 : vector<8xi32> to vector<8xf32>
      %110 = vector.shape_cast %109 : vector<8xf32> to vector<8x1xf32>
      %111 = vector.shape_cast %110 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_11 = arith.constant 4 : i32
      %112 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %113 = vector.extract %112[0] : i64 from vector<4xi64>
      %114 = vector.extract %23[2] : i32 from vector<8xi32>
      %115 = vector.extract %23[3] : i32 from vector<8xi32>
      %116 = arith.index_cast %62 : index to i32
      %117 = arith.index_cast %75 : index to i32
      %118 = llvm.inttoptr %113 : i64 to !llvm.ptr<1>
      %119 = arith.muli %114, %c4_i32_11 : i32
      %120 = xevm.blockload2d %118, %119, %115, %119, %116, %117 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %121 = vector.bitcast %120 : vector<8xi32> to vector<8xf32>
      %122 = vector.shape_cast %121 : vector<8xf32> to vector<8x1xf32>
      %123 = vector.shape_cast %122 : vector<8x1xf32> to vector<8xf32>
      %124 = arith.addi %14, %c16 : index
      %c4_i32_12 = arith.constant 4 : i32
      %125 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %126 = vector.extract %125[0] : i64 from vector<4xi64>
      %127 = vector.extract %23[2] : i32 from vector<8xi32>
      %128 = vector.extract %23[3] : i32 from vector<8xi32>
      %129 = arith.index_cast %13 : index to i32
      %130 = arith.index_cast %124 : index to i32
      %131 = llvm.inttoptr %126 : i64 to !llvm.ptr<1>
      %132 = arith.muli %127, %c4_i32_12 : i32
      %133 = xevm.blockload2d %131, %132, %128, %132, %129, %130 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %134 = vector.bitcast %133 : vector<8xi32> to vector<8xf32>
      %135 = vector.shape_cast %134 : vector<8xf32> to vector<8x1xf32>
      %136 = vector.shape_cast %135 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_13 = arith.constant 4 : i32
      %137 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %138 = vector.extract %137[0] : i64 from vector<4xi64>
      %139 = vector.extract %23[2] : i32 from vector<8xi32>
      %140 = vector.extract %23[3] : i32 from vector<8xi32>
      %141 = arith.index_cast %36 : index to i32
      %142 = arith.index_cast %124 : index to i32
      %143 = llvm.inttoptr %138 : i64 to !llvm.ptr<1>
      %144 = arith.muli %139, %c4_i32_13 : i32
      %145 = xevm.blockload2d %143, %144, %140, %144, %141, %142 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %146 = vector.bitcast %145 : vector<8xi32> to vector<8xf32>
      %147 = vector.shape_cast %146 : vector<8xf32> to vector<8x1xf32>
      %148 = vector.shape_cast %147 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_14 = arith.constant 4 : i32
      %149 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %150 = vector.extract %149[0] : i64 from vector<4xi64>
      %151 = vector.extract %23[2] : i32 from vector<8xi32>
      %152 = vector.extract %23[3] : i32 from vector<8xi32>
      %153 = arith.index_cast %49 : index to i32
      %154 = arith.index_cast %124 : index to i32
      %155 = llvm.inttoptr %150 : i64 to !llvm.ptr<1>
      %156 = arith.muli %151, %c4_i32_14 : i32
      %157 = xevm.blockload2d %155, %156, %152, %156, %153, %154 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %158 = vector.bitcast %157 : vector<8xi32> to vector<8xf32>
      %159 = vector.shape_cast %158 : vector<8xf32> to vector<8x1xf32>
      %160 = vector.shape_cast %159 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_15 = arith.constant 4 : i32
      %161 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %162 = vector.extract %161[0] : i64 from vector<4xi64>
      %163 = vector.extract %23[2] : i32 from vector<8xi32>
      %164 = vector.extract %23[3] : i32 from vector<8xi32>
      %165 = arith.index_cast %62 : index to i32
      %166 = arith.index_cast %124 : index to i32
      %167 = llvm.inttoptr %162 : i64 to !llvm.ptr<1>
      %168 = arith.muli %163, %c4_i32_15 : i32
      %169 = xevm.blockload2d %167, %168, %164, %168, %165, %166 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %170 = vector.bitcast %169 : vector<8xi32> to vector<8xf32>
      %171 = vector.shape_cast %170 : vector<8xf32> to vector<8x1xf32>
      %172 = vector.shape_cast %171 : vector<8x1xf32> to vector<8xf32>
      %173 = arith.addi %14, %c24 : index
      %c4_i32_16 = arith.constant 4 : i32
      %174 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %175 = vector.extract %174[0] : i64 from vector<4xi64>
      %176 = vector.extract %23[2] : i32 from vector<8xi32>
      %177 = vector.extract %23[3] : i32 from vector<8xi32>
      %178 = arith.index_cast %13 : index to i32
      %179 = arith.index_cast %173 : index to i32
      %180 = llvm.inttoptr %175 : i64 to !llvm.ptr<1>
      %181 = arith.muli %176, %c4_i32_16 : i32
      %182 = xevm.blockload2d %180, %181, %177, %181, %178, %179 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %183 = vector.bitcast %182 : vector<8xi32> to vector<8xf32>
      %184 = vector.shape_cast %183 : vector<8xf32> to vector<8x1xf32>
      %185 = vector.shape_cast %184 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_17 = arith.constant 4 : i32
      %186 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %187 = vector.extract %186[0] : i64 from vector<4xi64>
      %188 = vector.extract %23[2] : i32 from vector<8xi32>
      %189 = vector.extract %23[3] : i32 from vector<8xi32>
      %190 = arith.index_cast %36 : index to i32
      %191 = arith.index_cast %173 : index to i32
      %192 = llvm.inttoptr %187 : i64 to !llvm.ptr<1>
      %193 = arith.muli %188, %c4_i32_17 : i32
      %194 = xevm.blockload2d %192, %193, %189, %193, %190, %191 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %195 = vector.bitcast %194 : vector<8xi32> to vector<8xf32>
      %196 = vector.shape_cast %195 : vector<8xf32> to vector<8x1xf32>
      %197 = vector.shape_cast %196 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_18 = arith.constant 4 : i32
      %198 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %199 = vector.extract %198[0] : i64 from vector<4xi64>
      %200 = vector.extract %23[2] : i32 from vector<8xi32>
      %201 = vector.extract %23[3] : i32 from vector<8xi32>
      %202 = arith.index_cast %49 : index to i32
      %203 = arith.index_cast %173 : index to i32
      %204 = llvm.inttoptr %199 : i64 to !llvm.ptr<1>
      %205 = arith.muli %200, %c4_i32_18 : i32
      %206 = xevm.blockload2d %204, %205, %201, %205, %202, %203 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %207 = vector.bitcast %206 : vector<8xi32> to vector<8xf32>
      %208 = vector.shape_cast %207 : vector<8xf32> to vector<8x1xf32>
      %209 = vector.shape_cast %208 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_19 = arith.constant 4 : i32
      %210 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %211 = vector.extract %210[0] : i64 from vector<4xi64>
      %212 = vector.extract %23[2] : i32 from vector<8xi32>
      %213 = vector.extract %23[3] : i32 from vector<8xi32>
      %214 = arith.index_cast %62 : index to i32
      %215 = arith.index_cast %173 : index to i32
      %216 = llvm.inttoptr %211 : i64 to !llvm.ptr<1>
      %217 = arith.muli %212, %c4_i32_19 : i32
      %218 = xevm.blockload2d %216, %217, %213, %217, %214, %215 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
      %219 = vector.bitcast %218 : vector<8xi32> to vector<8xf32>
      %220 = vector.shape_cast %219 : vector<8xf32> to vector<8x1xf32>
      %221 = vector.shape_cast %220 : vector<8x1xf32> to vector<8xf32>
      %cst_20 = arith.constant dense<0> : vector<8xi32>
      %c0_i32_21 = arith.constant 0 : i32
      %c0_i32_22 = arith.constant 0 : i32
      %c4096_i64_23 = arith.constant 4096 : i64
      %222 = arith.trunci %c4096_i64_23 : i64 to i32
      %c4096_i64_24 = arith.constant 4096 : i64
      %223 = arith.trunci %c4096_i64_24 : i64 to i32
      %224 = vector.bitcast %cst_20 : vector<8xi32> to vector<4xi64>
      %225 = vector.insert %1, %224 [0] : i64 into vector<4xi64>
      %226 = vector.bitcast %225 : vector<4xi64> to vector<8xi32>
      %227 = vector.insert %222, %226 [2] : i32 into vector<8xi32>
      %228 = vector.insert %223, %227 [3] : i32 into vector<8xi32>
      %229 = vector.insert %c0_i32_21, %228 [4] : i32 into vector<8xi32>
      %230 = vector.insert %c0_i32_22, %229 [5] : i32 into vector<8xi32>
      %cst_25 = arith.constant dense<0> : vector<8xi32>
      %c0_i32_26 = arith.constant 0 : i32
      %c0_i32_27 = arith.constant 0 : i32
      %c4096_i64_28 = arith.constant 4096 : i64
      %231 = arith.trunci %c4096_i64_28 : i64 to i32
      %c4096_i64_29 = arith.constant 4096 : i64
      %232 = arith.trunci %c4096_i64_29 : i64 to i32
      %233 = vector.bitcast %cst_25 : vector<8xi32> to vector<4xi64>
      %234 = vector.insert %0, %233 [0] : i64 into vector<4xi64>
      %235 = vector.bitcast %234 : vector<4xi64> to vector<8xi32>
      %236 = vector.insert %231, %235 [2] : i32 into vector<8xi32>
      %237 = vector.insert %232, %236 [3] : i32 into vector<8xi32>
      %238 = vector.insert %c0_i32_26, %237 [4] : i32 into vector<8xi32>
      %239 = vector.insert %c0_i32_27, %238 [5] : i32 into vector<8xi32>
      %240 = arith.muli %5, %c32 : index
      %241 = arith.remui %240, %c32 : index
      %242 = arith.remui %8, %c32 : index
      %243:16 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %35, %arg5 = %48, %arg6 = %61, %arg7 = %74, %arg8 = %87, %arg9 = %99, %arg10 = %111, %arg11 = %123, %arg12 = %136, %arg13 = %148, %arg14 = %160, %arg15 = %172, %arg16 = %185, %arg17 = %197, %arg18 = %209, %arg19 = %221) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {
        %436 = vector.shape_cast %arg19 : vector<8xf32> to vector<8x1xf32>
        %437 = vector.shape_cast %arg18 : vector<8xf32> to vector<8x1xf32>
        %438 = vector.shape_cast %arg17 : vector<8xf32> to vector<8x1xf32>
        %439 = vector.shape_cast %arg16 : vector<8xf32> to vector<8x1xf32>
        %440 = vector.shape_cast %arg15 : vector<8xf32> to vector<8x1xf32>
        %441 = vector.shape_cast %arg14 : vector<8xf32> to vector<8x1xf32>
        %442 = vector.shape_cast %arg13 : vector<8xf32> to vector<8x1xf32>
        %443 = vector.shape_cast %arg12 : vector<8xf32> to vector<8x1xf32>
        %444 = vector.shape_cast %arg11 : vector<8xf32> to vector<8x1xf32>
        %445 = vector.shape_cast %arg10 : vector<8xf32> to vector<8x1xf32>
        %446 = vector.shape_cast %arg9 : vector<8xf32> to vector<8x1xf32>
        %447 = vector.shape_cast %arg8 : vector<8xf32> to vector<8x1xf32>
        %448 = vector.shape_cast %arg7 : vector<8xf32> to vector<8x1xf32>
        %449 = vector.shape_cast %arg6 : vector<8xf32> to vector<8x1xf32>
        %450 = vector.shape_cast %arg5 : vector<8xf32> to vector<8x1xf32>
        %451 = vector.shape_cast %arg4 : vector<8xf32> to vector<8x1xf32>
        %452 = arith.addi %241, %arg3 : index
        %c2_i32 = arith.constant 2 : i32
        %453 = vector.bitcast %239 : vector<8xi32> to vector<4xi64>
        %454 = vector.extract %453[0] : i64 from vector<4xi64>
        %455 = vector.extract %239[2] : i32 from vector<8xi32>
        %456 = vector.extract %239[3] : i32 from vector<8xi32>
        %457 = arith.index_cast %452 : index to i32
        %458 = arith.index_cast %14 : index to i32
        %459 = llvm.inttoptr %454 : i64 to !llvm.ptr<1>
        %460 = arith.muli %455, %c2_i32 : i32
        %461 = xevm.blockload2d %459, %460, %456, %460, %457, %458 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
        %462 = vector.bitcast %461 : vector<8xi16> to vector<8xf16>
        %463 = arith.addi %452, %c16 : index
        %c2_i32_46 = arith.constant 2 : i32
        %464 = vector.bitcast %239 : vector<8xi32> to vector<4xi64>
        %465 = vector.extract %464[0] : i64 from vector<4xi64>
        %466 = vector.extract %239[2] : i32 from vector<8xi32>
        %467 = vector.extract %239[3] : i32 from vector<8xi32>
        %468 = arith.index_cast %463 : index to i32
        %469 = arith.index_cast %14 : index to i32
        %470 = llvm.inttoptr %465 : i64 to !llvm.ptr<1>
        %471 = arith.muli %466, %c2_i32_46 : i32
        %472 = xevm.blockload2d %470, %471, %467, %471, %468, %469 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
        %473 = vector.bitcast %472 : vector<8xi16> to vector<8xf16>
        %c2_i32_47 = arith.constant 2 : i32
        %474 = vector.bitcast %239 : vector<8xi32> to vector<4xi64>
        %475 = vector.extract %474[0] : i64 from vector<4xi64>
        %476 = vector.extract %239[2] : i32 from vector<8xi32>
        %477 = vector.extract %239[3] : i32 from vector<8xi32>
        %478 = arith.index_cast %452 : index to i32
        %479 = arith.index_cast %75 : index to i32
        %480 = llvm.inttoptr %475 : i64 to !llvm.ptr<1>
        %481 = arith.muli %476, %c2_i32_47 : i32
        %482 = xevm.blockload2d %480, %481, %477, %481, %478, %479 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
        %483 = vector.bitcast %482 : vector<8xi16> to vector<8xf16>
        %c2_i32_48 = arith.constant 2 : i32
        %484 = vector.bitcast %239 : vector<8xi32> to vector<4xi64>
        %485 = vector.extract %484[0] : i64 from vector<4xi64>
        %486 = vector.extract %239[2] : i32 from vector<8xi32>
        %487 = vector.extract %239[3] : i32 from vector<8xi32>
        %488 = arith.index_cast %463 : index to i32
        %489 = arith.index_cast %75 : index to i32
        %490 = llvm.inttoptr %485 : i64 to !llvm.ptr<1>
        %491 = arith.muli %486, %c2_i32_48 : i32
        %492 = xevm.blockload2d %490, %491, %487, %491, %488, %489 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
        %493 = vector.bitcast %492 : vector<8xi16> to vector<8xf16>
        %c2_i32_49 = arith.constant 2 : i32
        %494 = vector.bitcast %239 : vector<8xi32> to vector<4xi64>
        %495 = vector.extract %494[0] : i64 from vector<4xi64>
        %496 = vector.extract %239[2] : i32 from vector<8xi32>
        %497 = vector.extract %239[3] : i32 from vector<8xi32>
        %498 = arith.index_cast %452 : index to i32
        %499 = arith.index_cast %124 : index to i32
        %500 = llvm.inttoptr %495 : i64 to !llvm.ptr<1>
        %501 = arith.muli %496, %c2_i32_49 : i32
        %502 = xevm.blockload2d %500, %501, %497, %501, %498, %499 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
        %503 = vector.bitcast %502 : vector<8xi16> to vector<8xf16>
        %c2_i32_50 = arith.constant 2 : i32
        %504 = vector.bitcast %239 : vector<8xi32> to vector<4xi64>
        %505 = vector.extract %504[0] : i64 from vector<4xi64>
        %506 = vector.extract %239[2] : i32 from vector<8xi32>
        %507 = vector.extract %239[3] : i32 from vector<8xi32>
        %508 = arith.index_cast %463 : index to i32
        %509 = arith.index_cast %124 : index to i32
        %510 = llvm.inttoptr %505 : i64 to !llvm.ptr<1>
        %511 = arith.muli %506, %c2_i32_50 : i32
        %512 = xevm.blockload2d %510, %511, %507, %511, %508, %509 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
        %513 = vector.bitcast %512 : vector<8xi16> to vector<8xf16>
        %c2_i32_51 = arith.constant 2 : i32
        %514 = vector.bitcast %239 : vector<8xi32> to vector<4xi64>
        %515 = vector.extract %514[0] : i64 from vector<4xi64>
        %516 = vector.extract %239[2] : i32 from vector<8xi32>
        %517 = vector.extract %239[3] : i32 from vector<8xi32>
        %518 = arith.index_cast %452 : index to i32
        %519 = arith.index_cast %173 : index to i32
        %520 = llvm.inttoptr %515 : i64 to !llvm.ptr<1>
        %521 = arith.muli %516, %c2_i32_51 : i32
        %522 = xevm.blockload2d %520, %521, %517, %521, %518, %519 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
        %523 = vector.bitcast %522 : vector<8xi16> to vector<8xf16>
        %c2_i32_52 = arith.constant 2 : i32
        %524 = vector.bitcast %239 : vector<8xi32> to vector<4xi64>
        %525 = vector.extract %524[0] : i64 from vector<4xi64>
        %526 = vector.extract %239[2] : i32 from vector<8xi32>
        %527 = vector.extract %239[3] : i32 from vector<8xi32>
        %528 = arith.index_cast %463 : index to i32
        %529 = arith.index_cast %173 : index to i32
        %530 = llvm.inttoptr %525 : i64 to !llvm.ptr<1>
        %531 = arith.muli %526, %c2_i32_52 : i32
        %532 = xevm.blockload2d %530, %531, %527, %531, %528, %529 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
        %533 = vector.bitcast %532 : vector<8xi16> to vector<8xf16>
        %534 = arith.addi %242, %arg3 : index
        %c2_i32_53 = arith.constant 2 : i32
        %535 = vector.bitcast %230 : vector<8xi32> to vector<4xi64>
        %536 = vector.extract %535[0] : i64 from vector<4xi64>
        %537 = vector.extract %230[2] : i32 from vector<8xi32>
        %538 = vector.extract %230[3] : i32 from vector<8xi32>
        %539 = arith.index_cast %13 : index to i32
        %540 = arith.index_cast %534 : index to i32
        %541 = llvm.inttoptr %536 : i64 to !llvm.ptr<1>
        %542 = arith.muli %537, %c2_i32_53 : i32
        %543 = xevm.blockload2d %541, %542, %538, %542, %539, %540 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = true, tile_height = 16 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %544 = vector.bitcast %543 : vector<8xi32> to vector<16xf16>
        %c2_i32_54 = arith.constant 2 : i32
        %545 = vector.bitcast %230 : vector<8xi32> to vector<4xi64>
        %546 = vector.extract %545[0] : i64 from vector<4xi64>
        %547 = vector.extract %230[2] : i32 from vector<8xi32>
        %548 = vector.extract %230[3] : i32 from vector<8xi32>
        %549 = arith.index_cast %36 : index to i32
        %550 = arith.index_cast %534 : index to i32
        %551 = llvm.inttoptr %546 : i64 to !llvm.ptr<1>
        %552 = arith.muli %547, %c2_i32_54 : i32
        %553 = xevm.blockload2d %551, %552, %548, %552, %549, %550 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = true, tile_height = 16 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %554 = vector.bitcast %553 : vector<8xi32> to vector<16xf16>
        %c2_i32_55 = arith.constant 2 : i32
        %555 = vector.bitcast %230 : vector<8xi32> to vector<4xi64>
        %556 = vector.extract %555[0] : i64 from vector<4xi64>
        %557 = vector.extract %230[2] : i32 from vector<8xi32>
        %558 = vector.extract %230[3] : i32 from vector<8xi32>
        %559 = arith.index_cast %49 : index to i32
        %560 = arith.index_cast %534 : index to i32
        %561 = llvm.inttoptr %556 : i64 to !llvm.ptr<1>
        %562 = arith.muli %557, %c2_i32_55 : i32
        %563 = xevm.blockload2d %561, %562, %558, %562, %559, %560 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = true, tile_height = 16 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %564 = vector.bitcast %563 : vector<8xi32> to vector<16xf16>
        %c2_i32_56 = arith.constant 2 : i32
        %565 = vector.bitcast %230 : vector<8xi32> to vector<4xi64>
        %566 = vector.extract %565[0] : i64 from vector<4xi64>
        %567 = vector.extract %230[2] : i32 from vector<8xi32>
        %568 = vector.extract %230[3] : i32 from vector<8xi32>
        %569 = arith.index_cast %62 : index to i32
        %570 = arith.index_cast %534 : index to i32
        %571 = llvm.inttoptr %566 : i64 to !llvm.ptr<1>
        %572 = arith.muli %567, %c2_i32_56 : i32
        %573 = xevm.blockload2d %571, %572, %568, %572, %569, %570 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = true, tile_height = 16 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %574 = vector.bitcast %573 : vector<8xi32> to vector<16xf16>
        %575 = arith.addi %534, %c16 : index
        %c2_i32_57 = arith.constant 2 : i32
        %576 = vector.bitcast %230 : vector<8xi32> to vector<4xi64>
        %577 = vector.extract %576[0] : i64 from vector<4xi64>
        %578 = vector.extract %230[2] : i32 from vector<8xi32>
        %579 = vector.extract %230[3] : i32 from vector<8xi32>
        %580 = arith.index_cast %13 : index to i32
        %581 = arith.index_cast %575 : index to i32
        %582 = llvm.inttoptr %577 : i64 to !llvm.ptr<1>
        %583 = arith.muli %578, %c2_i32_57 : i32
        %584 = xevm.blockload2d %582, %583, %579, %583, %580, %581 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = true, tile_height = 16 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %585 = vector.bitcast %584 : vector<8xi32> to vector<16xf16>
        %c2_i32_58 = arith.constant 2 : i32
        %586 = vector.bitcast %230 : vector<8xi32> to vector<4xi64>
        %587 = vector.extract %586[0] : i64 from vector<4xi64>
        %588 = vector.extract %230[2] : i32 from vector<8xi32>
        %589 = vector.extract %230[3] : i32 from vector<8xi32>
        %590 = arith.index_cast %36 : index to i32
        %591 = arith.index_cast %575 : index to i32
        %592 = llvm.inttoptr %587 : i64 to !llvm.ptr<1>
        %593 = arith.muli %588, %c2_i32_58 : i32
        %594 = xevm.blockload2d %592, %593, %589, %593, %590, %591 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = true, tile_height = 16 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %595 = vector.bitcast %594 : vector<8xi32> to vector<16xf16>
        %c2_i32_59 = arith.constant 2 : i32
        %596 = vector.bitcast %230 : vector<8xi32> to vector<4xi64>
        %597 = vector.extract %596[0] : i64 from vector<4xi64>
        %598 = vector.extract %230[2] : i32 from vector<8xi32>
        %599 = vector.extract %230[3] : i32 from vector<8xi32>
        %600 = arith.index_cast %49 : index to i32
        %601 = arith.index_cast %575 : index to i32
        %602 = llvm.inttoptr %597 : i64 to !llvm.ptr<1>
        %603 = arith.muli %598, %c2_i32_59 : i32
        %604 = xevm.blockload2d %602, %603, %599, %603, %600, %601 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = true, tile_height = 16 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %605 = vector.bitcast %604 : vector<8xi32> to vector<16xf16>
        %c2_i32_60 = arith.constant 2 : i32
        %606 = vector.bitcast %230 : vector<8xi32> to vector<4xi64>
        %607 = vector.extract %606[0] : i64 from vector<4xi64>
        %608 = vector.extract %230[2] : i32 from vector<8xi32>
        %609 = vector.extract %230[3] : i32 from vector<8xi32>
        %610 = arith.index_cast %62 : index to i32
        %611 = arith.index_cast %575 : index to i32
        %612 = llvm.inttoptr %607 : i64 to !llvm.ptr<1>
        %613 = arith.muli %608, %c2_i32_60 : i32
        %614 = xevm.blockload2d %612, %613, %609, %613, %610, %611 <{cache_control = #xevm.load_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 16 : i32, pack_register = true, tile_height = 16 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %615 = vector.bitcast %614 : vector<8xi32> to vector<16xf16>
        %616 = vector.shape_cast %436 : vector<8x1xf32> to vector<8xf32>
        %617 = xevm.mma %523, %574, %616 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %618 = vector.shape_cast %437 : vector<8x1xf32> to vector<8xf32>
        %619 = xevm.mma %523, %564, %618 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %620 = vector.shape_cast %438 : vector<8x1xf32> to vector<8xf32>
        %621 = xevm.mma %523, %554, %620 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %622 = vector.shape_cast %439 : vector<8x1xf32> to vector<8xf32>
        %623 = xevm.mma %523, %544, %622 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %624 = vector.shape_cast %440 : vector<8x1xf32> to vector<8xf32>
        %625 = xevm.mma %503, %574, %624 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %626 = vector.shape_cast %441 : vector<8x1xf32> to vector<8xf32>
        %627 = xevm.mma %503, %564, %626 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %628 = vector.shape_cast %442 : vector<8x1xf32> to vector<8xf32>
        %629 = xevm.mma %503, %554, %628 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %630 = vector.shape_cast %443 : vector<8x1xf32> to vector<8xf32>
        %631 = xevm.mma %503, %544, %630 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %632 = vector.shape_cast %444 : vector<8x1xf32> to vector<8xf32>
        %633 = xevm.mma %483, %574, %632 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %634 = vector.shape_cast %445 : vector<8x1xf32> to vector<8xf32>
        %635 = xevm.mma %483, %564, %634 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %636 = vector.shape_cast %446 : vector<8x1xf32> to vector<8xf32>
        %637 = xevm.mma %483, %554, %636 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %638 = vector.shape_cast %447 : vector<8x1xf32> to vector<8xf32>
        %639 = xevm.mma %483, %544, %638 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %640 = vector.shape_cast %448 : vector<8x1xf32> to vector<8xf32>
        %641 = xevm.mma %462, %574, %640 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %642 = vector.shape_cast %449 : vector<8x1xf32> to vector<8xf32>
        %643 = xevm.mma %462, %564, %642 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %644 = vector.shape_cast %450 : vector<8x1xf32> to vector<8xf32>
        %645 = xevm.mma %462, %554, %644 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %646 = vector.shape_cast %451 : vector<8x1xf32> to vector<8xf32>
        %647 = xevm.mma %462, %544, %646 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %648 = xevm.mma %533, %615, %617 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %649 = vector.shape_cast %648 : vector<8xf32> to vector<8x1xf32>
        %650 = vector.shape_cast %649 : vector<8x1xf32> to vector<8xf32>
        %651 = xevm.mma %533, %605, %619 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %652 = vector.shape_cast %651 : vector<8xf32> to vector<8x1xf32>
        %653 = vector.shape_cast %652 : vector<8x1xf32> to vector<8xf32>
        %654 = xevm.mma %533, %595, %621 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %655 = vector.shape_cast %654 : vector<8xf32> to vector<8x1xf32>
        %656 = vector.shape_cast %655 : vector<8x1xf32> to vector<8xf32>
        %657 = xevm.mma %533, %585, %623 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %658 = vector.shape_cast %657 : vector<8xf32> to vector<8x1xf32>
        %659 = vector.shape_cast %658 : vector<8x1xf32> to vector<8xf32>
        %660 = xevm.mma %513, %615, %625 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %661 = vector.shape_cast %660 : vector<8xf32> to vector<8x1xf32>
        %662 = vector.shape_cast %661 : vector<8x1xf32> to vector<8xf32>
        %663 = xevm.mma %513, %605, %627 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %664 = vector.shape_cast %663 : vector<8xf32> to vector<8x1xf32>
        %665 = vector.shape_cast %664 : vector<8x1xf32> to vector<8xf32>
        %666 = xevm.mma %513, %595, %629 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %667 = vector.shape_cast %666 : vector<8xf32> to vector<8x1xf32>
        %668 = vector.shape_cast %667 : vector<8x1xf32> to vector<8xf32>
        %669 = xevm.mma %513, %585, %631 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %670 = vector.shape_cast %669 : vector<8xf32> to vector<8x1xf32>
        %671 = vector.shape_cast %670 : vector<8x1xf32> to vector<8xf32>
        %672 = xevm.mma %493, %615, %633 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %673 = vector.shape_cast %672 : vector<8xf32> to vector<8x1xf32>
        %674 = vector.shape_cast %673 : vector<8x1xf32> to vector<8xf32>
        %675 = xevm.mma %493, %605, %635 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %676 = vector.shape_cast %675 : vector<8xf32> to vector<8x1xf32>
        %677 = vector.shape_cast %676 : vector<8x1xf32> to vector<8xf32>
        %678 = xevm.mma %493, %595, %637 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %679 = vector.shape_cast %678 : vector<8xf32> to vector<8x1xf32>
        %680 = vector.shape_cast %679 : vector<8x1xf32> to vector<8xf32>
        %681 = xevm.mma %493, %585, %639 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %682 = vector.shape_cast %681 : vector<8xf32> to vector<8x1xf32>
        %683 = vector.shape_cast %682 : vector<8x1xf32> to vector<8xf32>
        %684 = xevm.mma %473, %615, %641 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %685 = vector.shape_cast %684 : vector<8xf32> to vector<8x1xf32>
        %686 = vector.shape_cast %685 : vector<8x1xf32> to vector<8xf32>
        %687 = xevm.mma %473, %605, %643 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %688 = vector.shape_cast %687 : vector<8xf32> to vector<8x1xf32>
        %689 = vector.shape_cast %688 : vector<8x1xf32> to vector<8xf32>
        %690 = xevm.mma %473, %595, %645 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %691 = vector.shape_cast %690 : vector<8xf32> to vector<8x1xf32>
        %692 = vector.shape_cast %691 : vector<8x1xf32> to vector<8xf32>
        %693 = xevm.mma %473, %585, %647 {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>} : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %694 = vector.shape_cast %693 : vector<8xf32> to vector<8x1xf32>
        %695 = vector.shape_cast %694 : vector<8x1xf32> to vector<8xf32>
        scf.yield %695, %692, %689, %686, %683, %680, %677, %674, %671, %668, %665, %662, %659, %656, %653, %650 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
      }
      %244 = arith.maximumf %243#0, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %245 = vector.shape_cast %244 : vector<8xf32> to vector<8x1xf32>
      %246 = arith.maximumf %243#1, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %247 = vector.shape_cast %246 : vector<8xf32> to vector<8x1xf32>
      %248 = arith.maximumf %243#2, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %249 = vector.shape_cast %248 : vector<8xf32> to vector<8x1xf32>
      %250 = arith.maximumf %243#3, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %251 = vector.shape_cast %250 : vector<8xf32> to vector<8x1xf32>
      %252 = arith.maximumf %243#4, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %253 = vector.shape_cast %252 : vector<8xf32> to vector<8x1xf32>
      %254 = arith.maximumf %243#5, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %255 = vector.shape_cast %254 : vector<8xf32> to vector<8x1xf32>
      %256 = arith.maximumf %243#6, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %257 = vector.shape_cast %256 : vector<8xf32> to vector<8x1xf32>
      %258 = arith.maximumf %243#7, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %259 = vector.shape_cast %258 : vector<8xf32> to vector<8x1xf32>
      %260 = arith.maximumf %243#8, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %261 = vector.shape_cast %260 : vector<8xf32> to vector<8x1xf32>
      %262 = arith.maximumf %243#9, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %263 = vector.shape_cast %262 : vector<8xf32> to vector<8x1xf32>
      %264 = arith.maximumf %243#10, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %265 = vector.shape_cast %264 : vector<8xf32> to vector<8x1xf32>
      %266 = arith.maximumf %243#11, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %267 = vector.shape_cast %266 : vector<8xf32> to vector<8x1xf32>
      %268 = arith.maximumf %243#12, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %269 = vector.shape_cast %268 : vector<8xf32> to vector<8x1xf32>
      %270 = arith.maximumf %243#13, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %271 = vector.shape_cast %270 : vector<8xf32> to vector<8x1xf32>
      %272 = arith.maximumf %243#14, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %273 = vector.shape_cast %272 : vector<8xf32> to vector<8x1xf32>
      %274 = arith.maximumf %243#15, %cst {layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8xf32>
      %275 = vector.shape_cast %274 : vector<8xf32> to vector<8x1xf32>
      %276 = vector.shape_cast %245 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_30 = arith.constant 4 : i32
      %277 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %278 = vector.extract %277[0] : i64 from vector<4xi64>
      %279 = vector.extract %23[2] : i32 from vector<8xi32>
      %280 = vector.extract %23[3] : i32 from vector<8xi32>
      %281 = arith.index_cast %13 : index to i32
      %282 = arith.index_cast %14 : index to i32
      %283 = llvm.inttoptr %278 : i64 to !llvm.ptr<1>
      %284 = arith.muli %279, %c4_i32_30 : i32
      %285 = vector.bitcast %276 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %283, %284, %280, %284, %281, %282, %285 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %286 = vector.shape_cast %247 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_31 = arith.constant 4 : i32
      %287 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %288 = vector.extract %287[0] : i64 from vector<4xi64>
      %289 = vector.extract %23[2] : i32 from vector<8xi32>
      %290 = vector.extract %23[3] : i32 from vector<8xi32>
      %291 = arith.index_cast %36 : index to i32
      %292 = arith.index_cast %14 : index to i32
      %293 = llvm.inttoptr %288 : i64 to !llvm.ptr<1>
      %294 = arith.muli %289, %c4_i32_31 : i32
      %295 = vector.bitcast %286 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %293, %294, %290, %294, %291, %292, %295 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %296 = vector.shape_cast %249 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_32 = arith.constant 4 : i32
      %297 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %298 = vector.extract %297[0] : i64 from vector<4xi64>
      %299 = vector.extract %23[2] : i32 from vector<8xi32>
      %300 = vector.extract %23[3] : i32 from vector<8xi32>
      %301 = arith.index_cast %49 : index to i32
      %302 = arith.index_cast %14 : index to i32
      %303 = llvm.inttoptr %298 : i64 to !llvm.ptr<1>
      %304 = arith.muli %299, %c4_i32_32 : i32
      %305 = vector.bitcast %296 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %303, %304, %300, %304, %301, %302, %305 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %306 = vector.shape_cast %251 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_33 = arith.constant 4 : i32
      %307 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %308 = vector.extract %307[0] : i64 from vector<4xi64>
      %309 = vector.extract %23[2] : i32 from vector<8xi32>
      %310 = vector.extract %23[3] : i32 from vector<8xi32>
      %311 = arith.index_cast %62 : index to i32
      %312 = arith.index_cast %14 : index to i32
      %313 = llvm.inttoptr %308 : i64 to !llvm.ptr<1>
      %314 = arith.muli %309, %c4_i32_33 : i32
      %315 = vector.bitcast %306 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %313, %314, %310, %314, %311, %312, %315 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %316 = vector.shape_cast %253 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_34 = arith.constant 4 : i32
      %317 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %318 = vector.extract %317[0] : i64 from vector<4xi64>
      %319 = vector.extract %23[2] : i32 from vector<8xi32>
      %320 = vector.extract %23[3] : i32 from vector<8xi32>
      %321 = arith.index_cast %13 : index to i32
      %322 = arith.index_cast %75 : index to i32
      %323 = llvm.inttoptr %318 : i64 to !llvm.ptr<1>
      %324 = arith.muli %319, %c4_i32_34 : i32
      %325 = vector.bitcast %316 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %323, %324, %320, %324, %321, %322, %325 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %326 = vector.shape_cast %255 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_35 = arith.constant 4 : i32
      %327 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %328 = vector.extract %327[0] : i64 from vector<4xi64>
      %329 = vector.extract %23[2] : i32 from vector<8xi32>
      %330 = vector.extract %23[3] : i32 from vector<8xi32>
      %331 = arith.index_cast %36 : index to i32
      %332 = arith.index_cast %75 : index to i32
      %333 = llvm.inttoptr %328 : i64 to !llvm.ptr<1>
      %334 = arith.muli %329, %c4_i32_35 : i32
      %335 = vector.bitcast %326 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %333, %334, %330, %334, %331, %332, %335 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %336 = vector.shape_cast %257 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_36 = arith.constant 4 : i32
      %337 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %338 = vector.extract %337[0] : i64 from vector<4xi64>
      %339 = vector.extract %23[2] : i32 from vector<8xi32>
      %340 = vector.extract %23[3] : i32 from vector<8xi32>
      %341 = arith.index_cast %49 : index to i32
      %342 = arith.index_cast %75 : index to i32
      %343 = llvm.inttoptr %338 : i64 to !llvm.ptr<1>
      %344 = arith.muli %339, %c4_i32_36 : i32
      %345 = vector.bitcast %336 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %343, %344, %340, %344, %341, %342, %345 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %346 = vector.shape_cast %259 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_37 = arith.constant 4 : i32
      %347 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %348 = vector.extract %347[0] : i64 from vector<4xi64>
      %349 = vector.extract %23[2] : i32 from vector<8xi32>
      %350 = vector.extract %23[3] : i32 from vector<8xi32>
      %351 = arith.index_cast %62 : index to i32
      %352 = arith.index_cast %75 : index to i32
      %353 = llvm.inttoptr %348 : i64 to !llvm.ptr<1>
      %354 = arith.muli %349, %c4_i32_37 : i32
      %355 = vector.bitcast %346 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %353, %354, %350, %354, %351, %352, %355 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %356 = vector.shape_cast %261 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_38 = arith.constant 4 : i32
      %357 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %358 = vector.extract %357[0] : i64 from vector<4xi64>
      %359 = vector.extract %23[2] : i32 from vector<8xi32>
      %360 = vector.extract %23[3] : i32 from vector<8xi32>
      %361 = arith.index_cast %13 : index to i32
      %362 = arith.index_cast %124 : index to i32
      %363 = llvm.inttoptr %358 : i64 to !llvm.ptr<1>
      %364 = arith.muli %359, %c4_i32_38 : i32
      %365 = vector.bitcast %356 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %363, %364, %360, %364, %361, %362, %365 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %366 = vector.shape_cast %263 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_39 = arith.constant 4 : i32
      %367 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %368 = vector.extract %367[0] : i64 from vector<4xi64>
      %369 = vector.extract %23[2] : i32 from vector<8xi32>
      %370 = vector.extract %23[3] : i32 from vector<8xi32>
      %371 = arith.index_cast %36 : index to i32
      %372 = arith.index_cast %124 : index to i32
      %373 = llvm.inttoptr %368 : i64 to !llvm.ptr<1>
      %374 = arith.muli %369, %c4_i32_39 : i32
      %375 = vector.bitcast %366 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %373, %374, %370, %374, %371, %372, %375 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %376 = vector.shape_cast %265 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_40 = arith.constant 4 : i32
      %377 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %378 = vector.extract %377[0] : i64 from vector<4xi64>
      %379 = vector.extract %23[2] : i32 from vector<8xi32>
      %380 = vector.extract %23[3] : i32 from vector<8xi32>
      %381 = arith.index_cast %49 : index to i32
      %382 = arith.index_cast %124 : index to i32
      %383 = llvm.inttoptr %378 : i64 to !llvm.ptr<1>
      %384 = arith.muli %379, %c4_i32_40 : i32
      %385 = vector.bitcast %376 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %383, %384, %380, %384, %381, %382, %385 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %386 = vector.shape_cast %267 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_41 = arith.constant 4 : i32
      %387 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %388 = vector.extract %387[0] : i64 from vector<4xi64>
      %389 = vector.extract %23[2] : i32 from vector<8xi32>
      %390 = vector.extract %23[3] : i32 from vector<8xi32>
      %391 = arith.index_cast %62 : index to i32
      %392 = arith.index_cast %124 : index to i32
      %393 = llvm.inttoptr %388 : i64 to !llvm.ptr<1>
      %394 = arith.muli %389, %c4_i32_41 : i32
      %395 = vector.bitcast %386 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %393, %394, %390, %394, %391, %392, %395 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %396 = vector.shape_cast %269 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_42 = arith.constant 4 : i32
      %397 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %398 = vector.extract %397[0] : i64 from vector<4xi64>
      %399 = vector.extract %23[2] : i32 from vector<8xi32>
      %400 = vector.extract %23[3] : i32 from vector<8xi32>
      %401 = arith.index_cast %13 : index to i32
      %402 = arith.index_cast %173 : index to i32
      %403 = llvm.inttoptr %398 : i64 to !llvm.ptr<1>
      %404 = arith.muli %399, %c4_i32_42 : i32
      %405 = vector.bitcast %396 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %403, %404, %400, %404, %401, %402, %405 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %406 = vector.shape_cast %271 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_43 = arith.constant 4 : i32
      %407 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %408 = vector.extract %407[0] : i64 from vector<4xi64>
      %409 = vector.extract %23[2] : i32 from vector<8xi32>
      %410 = vector.extract %23[3] : i32 from vector<8xi32>
      %411 = arith.index_cast %36 : index to i32
      %412 = arith.index_cast %173 : index to i32
      %413 = llvm.inttoptr %408 : i64 to !llvm.ptr<1>
      %414 = arith.muli %409, %c4_i32_43 : i32
      %415 = vector.bitcast %406 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %413, %414, %410, %414, %411, %412, %415 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %416 = vector.shape_cast %273 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_44 = arith.constant 4 : i32
      %417 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %418 = vector.extract %417[0] : i64 from vector<4xi64>
      %419 = vector.extract %23[2] : i32 from vector<8xi32>
      %420 = vector.extract %23[3] : i32 from vector<8xi32>
      %421 = arith.index_cast %49 : index to i32
      %422 = arith.index_cast %173 : index to i32
      %423 = llvm.inttoptr %418 : i64 to !llvm.ptr<1>
      %424 = arith.muli %419, %c4_i32_44 : i32
      %425 = vector.bitcast %416 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %423, %424, %420, %424, %421, %422, %425 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      %426 = vector.shape_cast %275 : vector<8x1xf32> to vector<8xf32>
      %c4_i32_45 = arith.constant 4 : i32
      %427 = vector.bitcast %23 : vector<8xi32> to vector<4xi64>
      %428 = vector.extract %427[0] : i64 from vector<4xi64>
      %429 = vector.extract %23[2] : i32 from vector<8xi32>
      %430 = vector.extract %23[3] : i32 from vector<8xi32>
      %431 = arith.index_cast %62 : index to i32
      %432 = arith.index_cast %173 : index to i32
      %433 = llvm.inttoptr %428 : i64 to !llvm.ptr<1>
      %434 = arith.muli %429, %c4_i32_45 : i32
      %435 = vector.bitcast %426 : vector<8xf32> to vector<8xi32>
      xevm.blockstore2d %433, %434, %430, %434, %431, %432, %435 <{cache_control = #xevm.store_cache_control<L1uc_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
      gpu.return
    }
  }
}