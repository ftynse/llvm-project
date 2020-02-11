
#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmul_trait = {
  args_in = 2,
  args_out = 1,
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = #matmul_accesses
}

func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  linalg.generic #matmul_trait %A, %B, %C {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %0 = mulf %a, %b : f32
    %1 = addf %c, %0 : f32
    linalg.yield %1 : f32
  } {mdh} : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>

  return
}
