[[[ IR printer: bufferize ]]]
module {
  func.func @main(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    %0 = ub.poison : f32
    %cst_0 = arith.constant dense<0xFF800000> : vector<1x256xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<1xf32>
    %c256 = arith.constant 256 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg2) in (4096) {
      %1 = scf.for %arg3 = %c0 to %c4096 step %c256 iter_args(%arg4 = %cst_0) -> (vector<1x256xf32>) {
        %4 = vector.transfer_read %arg0[%arg2, %arg3], %0 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<1x256xf32>
        %5 = arith.maximumf %4, %arg4 : vector<1x256xf32>
        scf.yield %5 : vector<1x256xf32>
      }
      %2 = vector.multi_reduction <maximumf>, %1, %cst_1 [1] : vector<1x256xf32> to vector<1xf32>
      %subview = memref.subview %arg1[%arg2, 0] [1, 4096] [1, 1] : memref<4096x4096xf32> to memref<1x4096xf32, strided<[4096, 1], offset: ?>>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4096xf32>
      memref.copy %subview, %alloc : memref<1x4096xf32, strided<[4096, 1], offset: ?>> to memref<1x4096xf32>
      %3 = scf.for %arg3 = %c0 to %c4096 step %c256 iter_args(%arg4 = %cst) -> (vector<1xf32>) {
        %4 = vector.transfer_read %arg0[%arg2, %arg3], %0 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<1x256xf32>
        %5 = vector.broadcast %2 : vector<1xf32> to vector<1x256xf32>
        %6 = arith.subf %4, %5 : vector<1x256xf32>
        %7 = math.exp %6 : vector<1x256xf32>
        %8 = vector.multi_reduction <add>, %7, %arg4 [1] : vector<1x256xf32> to vector<1xf32>
        vector.transfer_write %7, %alloc[%c0, %arg3] {in_bounds = [true, true]} : vector<1x256xf32>, memref<1x4096xf32>
        scf.yield %8 : vector<1xf32>
      }
      scf.for %arg3 = %c0 to %c4096 step %c256 {
        %4 = vector.transfer_read %alloc[%c0, %arg3], %0 {in_bounds = [true, true]} : memref<1x4096xf32>, vector<1x256xf32>
        %5 = vector.broadcast %3 : vector<1xf32> to vector<1x256xf32>
        %6 = arith.divf %4, %5 : vector<1x256xf32>
        vector.transfer_write %6, %subview[%c0, %arg3] {in_bounds = [true, true]} : vector<1x256xf32>, memref<1x4096xf32, strided<[4096, 1], offset: ?>>
      }
    }
    return
  }
}