// RUN: mlir-opt %s --test-transform-dialect-interpreter --verify-diagnostics

module attributes { transform.with_named_sequence } {
  transform.named_sequence @blocked_conv(%entry: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_op) {
    %conv_operand2, %add_operand = transform.match.structured %entry 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op) {
    ^bb0(%structured: !transform.any_op):
      transform.match.structured.body %structured { elementwise = "arith.maxf" } : !transform.any_op

      %num_inputs = transform.match.structured.num_inputs %structured : (!transform.any_op) -> !transform.param<i64>
      %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %num_inputs, %c2 : !transform.param<i64>

      %add = transform.match.structured.input %structured[0] : (!transform.any_op) -> !transform.any_op
      %conv_operand = transform.match.structured %add : (!transform.any_op) -> !transform.any_op {
      ^bb1(%add_structured: !transform.any_op):
        transform.match.structured.body %add_structured { elementwise = "arith.addf" } : !transform.any_op
        
        %num_inputs_add = transform.match.structured.num_inputs %add_structured : (!transform.any_op) -> !transform.param<i64>
        transform.match.param.cmpi eq %num_inputs, %c2 : !transform.param<i64>

        %conv = transform.match.structured.input %add_structured[0] : (!transform.any_op) -> !transform.any_op
        transform.match.structured %conv : !transform.any_op {
        ^bb2(%conv_structured: !transform.any_op):
          // The structured op is a specific convolution.
          // TODO: relax to any convolution.
          transform.match.operation_name %conv_structured["linalg.conv_2d_nhwc_hwcf"] : !transform.any_op

          // With 1x1 filter.
          // TODO: capture which dimensions of the convolution correspond to the
          // filter and use them here instead of hardcoding.
          %d1 = transform.match.structured.dim %conv_structured[4] : (!transform.any_op) -> !transform.param<i64>
          %d2 = transform.match.structured.dim %conv_structured[5] : (!transform.any_op) -> !transform.param<i64>
          %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
          transform.match.param.cmpi eq %d1, %c1 : !transform.param<i64>
          transform.match.param.cmpi eq %d2, %c1 : !transform.param<i64>
          transform.match.structured.yield
        }

        %broadcast = transform.match.structured.input %add_structured[1] : (!transform.any_op) -> !transform.any_op
        transform.match.structured %broadcast : !transform.any_op {
        ^bb3(%broadcast_structured: !transform.any_op):
          transform.match.structured.body %broadcast_structured { passthrough } : !transform.any_op
          transform.match.structured.yield
        }

        transform.match.structured.yield %conv : !transform.any_op
      }
      transform.match.structured.yield %conv_operand, %add : !transform.any_op, !transform.any_op
    }

    transform.yield %add_operand, %conv_operand2 : !transform.any_op, !transform.any_op
  }

  transform.named_sequence @print_blocked_conv(
      %add: !transform.any_op {transform.readonly},
      %conv: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %conv, "convolution" : !transform.any_op
    transform.test_print_remark_at_operand %add, "trailing elementwise add" : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    transform.foreach_match in %root
      @blocked_conv -> @print_blocked_conv
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @main(%arg0: tensor<1x56x56x64xf32>) -> tensor<1x58x58x64xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x56x56x64xf32>
  %cst_0 = arith.constant dense<0.142857149> : tensor<1x1x64x64xf32>
  %cst_1 = arith.constant dense<1.250000e-01> : tensor<64xf32>
  %cst_2 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  // expected-remark @below {{convolution}}
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, metadata = "expect_to_map2", strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst_0 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_1 : tensor<64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  // expected-remark @below {{trailing elementwise add}}
  %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %6 = arith.addf %in, %in_3 : f32
      linalg.yield %6 : f32
  } -> tensor<1x56x56x64xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %cst : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %6 = arith.maxf %in, %in_3 : f32
      linalg.yield %6 : f32
  } -> tensor<1x56x56x64xf32>
  %padded = tensor.pad %5 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_2 : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
  return %padded : tensor<1x58x58x64xf32>
}
