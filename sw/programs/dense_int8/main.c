// One int8 dense layer with ReLU, the basic block of a quantized network.
// main() returns 0 when every output matches the reference computed in
// Python (scripts/gen_dense_int8.py), or 1 + the index of the first mismatch.
//
// The helpers are kept out of line so each one shows up as its own frame in
// the trace's flame graph.
#include <stdint.h>

#include "dense_int8_data.h"

static int8_t y[BATCH * OUT];

__attribute__((noinline)) static int32_t dot_int8(const int8_t* a, const int8_t* b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; ++i) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

__attribute__((noinline)) static int8_t requantize_relu(int32_t acc) {
  int32_t q = (int32_t)(((int64_t)acc * MULT) >> SHIFT);
  if (q > 127) q = 127;
  if (q < 0) q = 0;  // ReLU, which also covers the lower clamp at -128
  return (int8_t)q;
}

__attribute__((noinline)) static void dense_int8(const int8_t* x, int8_t* out) {
  for (int o = 0; o < OUT; ++o) out[o] = requantize_relu(dot_int8(&W[o * IN], x, IN) + B[o]);
}

__attribute__((noinline)) static int check(void) {
  for (int i = 0; i < BATCH * OUT; ++i)
    if (y[i] != Y_EXPECTED[i]) return i + 1;
  return 0;
}

int main(void) {
  for (int n = 0; n < BATCH; ++n) dense_int8(&X[n * IN], &y[n * OUT]);
  return check();
}
