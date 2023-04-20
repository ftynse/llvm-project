#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module {
  func.func @resnet50v1(%arg0: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
    %cst = arith.constant dense<1.0> : tensor<2x32xf32>
    %c0_f32 = arith.constant 0.0 : f32
    %cst_0 = arith.constant dense<0.0> : tensor<1x2x56x56x32xf32>
    %cst_1 = arith.constant dense<0.0> : tensor<2x2x1x1x32x32xf32>
    %0 = tensor.empty() : tensor<1x56x56x64xf32>
    %1 = tensor.empty() : tensor<1x2x56x56x32xf32>
    %pack = tensor.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %1 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
    %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
    %3 = linalg.fill ins(%c0_f32 : f32) outs(%2 : tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
    //nhwc_hwcf
//  %2 = linalg.generic {                    N   H   W   F   KH  KW  C
//    indexing_maps = [#map,  // affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
//                     #map1, // affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
//                     #map2],// affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
    %4 = linalg.generic {   //               N   Fb  H   W   F'  Cb  KH  KW  C'
      indexing_maps = [#map,  // affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
                       #map1, // affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
                       #map2],// affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
//    iterator_types = ["parallel", "parallel", "parallel", "parallel",             "reduction", "reduction", "reduction"]} 
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]}
//                             N   H  W  C                   H W C  F                         N   H  W  F
//  ins(%arg0, %cst_1 : tensor<1x  56x56x64xf32>, tensor<    1x1x64x64xf32>) outs(%1 : tensor<1x  56x56x64xf32>) {      
    ins(%pack, %cst_1 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%3 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %11 = arith.mulf %in, %in_2 : f32
      %12 = arith.addf %out, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<1x2x56x56x32xf32>
    %5 = tensor.empty() : tensor<1x2x56x56x32xf32>
    %6 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%cst : tensor<2x32xf32>) outs(%5 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2x56x56x32xf32>
    %7 = tensor.empty() : tensor<1x2x56x56x32xf32>
    %8 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%4, %6 : tensor<1x2x56x56x32xf32>, tensor<1x2x56x56x32xf32>) outs(%7 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %11 = arith.addf %in, %in_2 : f32
      linalg.yield %11 : f32
    } -> tensor<1x2x56x56x32xf32>
    %9 = tensor.empty() : tensor<1x2x56x56x32xf32>
    %10 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%8, %cst_0 : tensor<1x2x56x56x32xf32>, tensor<1x2x56x56x32xf32>) outs(%9 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %11 = arith.maxf %in, %in_2 : f32
      linalg.yield %11 : f32
    } -> tensor<1x2x56x56x32xf32>
    %unpack = tensor.unpack %10 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
    return %unpack : tensor<1x56x56x64xf32>
  } 
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @do_nothing(%arg0: !transform.any_op {transform.readonly}, %rag1: !transform.param<i64>{transform.readonly}) {
    transform.yield
  }
  transform.named_sequence @packed_reduction(%arg0: !transform.any_op{transform.readonly}) -> (!transform.any_op, !transform.param<i64>) {
    %tile_sizes = transform.match.structured %arg0 : (!transform.any_op) -> !transform.param<i64> {
    ^bb0(%arg1: !transform.any_op):
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      
      // TODO: generalize to arbitrary rank?
      %c9p = transform.param.constant 9 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c9p : !transform.param<i64>
      transform.match.structured.dim %arg1[0,1,2,3,4] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-1,-2,-3,-4] { reduction } : !transform.any_op
      %sizes = transform.match.structured.dim %arg1[0,1,2] : (!transform.any_op) -> !transform.param<i64>

      // TODO: rewrite to fma so we can:
      // transform.match.structured.body %arg1 { elementwise = "math.fma" } : !transform.any_op

      // Capture which dimension is which by analyzing the access patterns of input#0, separate
      //   - parallel dimensions that appear in addition (H, W)
      //   - reduction dimensions that appear in addition (KH, KW)
      //   - other parallel dimensions (N, Fb, F')
      //   - other reduction dimensions (Cb, C')
      // We can further differentiate N and Fb,F' as N doesn't appear in input#1
      // But we cannot differentiate Cb' from C and Fb from F' because they always appear together.
      // If we need this, it will have to be propagated from packing, or guessed.

      // TODO: make this generalizable to "seq" and/or lists.
      %c0p = transform.param.constant 0 : i64 -> !transform.param<i64>
      %c1p = transform.param.constant 1 : i64 -> !transform.param<i64>
      %c2p = transform.param.constant 2 : i64 -> !transform.param<i64>
      %c3p = transform.param.constant 3 : i64 -> !transform.param<i64>
      %c4p = transform.param.constant 4 : i64 -> !transform.param<i64>
      %c5p = transform.param.constant 5 : i64 -> !transform.param<i64>
      %c6p = transform.param.constant 6 : i64 -> !transform.param<i64>
      %c7p = transform.param.constant 7 : i64 -> !transform.param<i64>
      %c8p = transform.param.constant 8 : i64 -> !transform.param<i64>
      
      %ps0 = transform.param.set_union %c0p, %c1p : !transform.param<i64>
      %ps1 = transform.param.set_union %ps0, %c2p : !transform.param<i64>
      %ps2 = transform.param.set_union %ps1, %c3p : !transform.param<i64>
      %ps  = transform.param.set_union %ps2, %c4p : !transform.param<i64>
      %rs0 = transform.param.set_union %c5p, %c6p : !transform.param<i64>
      %rs1 = transform.param.set_union %rs0, %c7p : !transform.param<i64>
      %rs =  transform.param.set_union %rs1, %c8p : !transform.param<i64>

      %ids = transform.match.structured.input %arg1[0] { identity_dims } : (!transform.any_op) -> !transform.param<i64>
      %add = transform.match.structured.input %arg1[0] { pairwise_add_dims } : (!transform.any_op) -> !transform.param<i64>
      %fs = transform.match.structured.input %arg1[1] { identity_dims } : (!transform.any_op) -> !transform.param<i64>
      
      %hw = transform.param.set_intersect %ps, %add : !transform.param<i64>
      %khw = transform.param.set_intersect %rs, %add : !transform.param<i64>
      %nf = transform.param.set_difference %ps, %hw : !transform.param<i64>
      %c = transform.param.set_difference %rs, %khw : !transform.param<i64>
      %f = transform.param.set_intersect %nf, %fs : !transform.param<i64>
      %n = transform.param.set_difference %nf, %f : !transform.param<i64>

      %num_hw = transform.param.payload_size %hw : (!transform.param<i64>) -> !transform.param<i64>
      %num_khw = transform.param.payload_size %khw : (!transform.param<i64>) -> !transform.param<i64>
      %num_c = transform.param.payload_size %c : (!transform.param<i64>) -> !transform.param<i64>
      %num_n = transform.param.payload_size %n : (!transform.param<i64>) -> !transform.param<i64>
      %num_f = transform.param.payload_size %f : (!transform.param<i64>) -> !transform.param<i64>
      transform.match.param.cmpi eq %num_hw, %c2p : !transform.param<i64>
      transform.match.param.cmpi eq %num_khw, %c2p : !transform.param<i64>
      transform.match.param.cmpi eq %num_c, %c2p : !transform.param<i64>
      transform.match.param.cmpi eq %num_n, %c1p : !transform.param<i64>
      transform.match.param.cmpi eq %num_f, %c2p : !transform.param<i64>

      transform.test_print_param %hw, "hw" : !transform.param<i64>
      transform.test_print_param %khw, "khw" : !transform.param<i64>
      transform.test_print_param %c, "c" : !transform.param<i64>
      transform.test_print_param %n, "n" : !transform.param<i64>
      transform.test_print_param %f, "f" : !transform.param<i64>

      transform.match.structured.yield %sizes : !transform.param<i64>
    }
    transform.yield %arg0, %tile_sizes : !transform.any_op, !transform.param<i64>
  }

  transform.named_sequence @resnet_stage(%relu: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_op) {
    %conv3, %add2, %bcast3 = transform.match.structured %relu
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op) {
    ^bb0(%arg1: !transform.any_op):
      transform.match.structured.body %arg1 { elementwise = "arith.maxf" } : !transform.any_op
      %add = transform.match.structured.input %arg1[0] : (!transform.any_op) -> !transform.any_op
      %conv2, %bcast2 = transform.match.structured %add 
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op) {
      ^bb1(%arg2: !transform.any_op):
        transform.match.structured.body %arg2 { elementwise = "arith.addf" } : !transform.any_op
        %conv = transform.match.structured.input %arg2[0] : (!transform.any_op) -> !transform.any_op
        %bcast = transform.match.structured.input %arg2[1] : (!transform.any_op) -> !transform.any_op
        transform.include @packed_reduction failures(propagate) (%conv) : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
        transform.match.structured %bcast : !transform.any_op {
        ^bb2(%arg3: !transform.any_op):
          transform.match.structured.body %arg3 { passthrough } : !transform.any_op
          transform.match.structured.yield
        }
        transform.match.structured.yield %conv, %bcast : !transform.any_op, !transform.any_op
      }
      transform.match.structured.yield %conv2, %add, %bcast2 : !transform.any_op, !transform.any_op, !transform.any_op
    }
    %trailing = transform.merge_handles %relu, %add2, %bcast3 : !transform.any_op
    transform.yield %conv3, %trailing : !transform.any_op, !transform.any_op
  }
  
  // TODO: warning emission should not stop matching...
  transform.named_sequence @fuse_conv(
      %conv: !pdl.operation {transform.consumed},
      %trailing: !transform.any_op {transform.readonly}) {
    // transform.test_print_remark_at_operand %conv, "conv" : !transform.any_op
    // transform.test_print_remark_at_operand %trailing, "trailing" : !transform.any_op
    %1:3 = transform.split_handles %trailing in [3]
      : (!transform.any_op) -> (!pdl.operation, !pdl.operation, !pdl.operation)
    %forall, %tiled = transform.structured.tile_to_forall_op %1#0 tile_sizes [1, 1, 1]
    transform.structured.fuse_into_containing_op %1#1 into %forall
    transform.structured.fuse_into_containing_op %1#2 into %forall
    transform.structured.fuse_into_containing_op %conv into %forall

    %func = transform.get_closest_isolated_parent %forall : (!pdl.operation) -> !pdl.operation
    %func2 = transform.structured.fold_unit_extent_dims %func

    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.foreach_match in %arg0
      @resnet_stage -> @fuse_conv
      // @packed_reduction -> @do_nothing
       : (!transform.any_op) -> !transform.any_op
  }
}
