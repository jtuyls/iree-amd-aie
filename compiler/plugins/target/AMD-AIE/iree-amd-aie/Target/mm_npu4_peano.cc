// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

R"peano(

template<int M, int N, int r>
void scale_shift_trunc_vectorized(v32int32 *__restrict in, int64_t offsetIn, int64_t scale, int64_t shift, 
                                  v32int8 *__restrict out, int64_t offsetOut) {
  set_sat();
  const v32uint16 mulLow = broadcast_u16((uint16)(scale & 0xFFFF));
  const v32uint16 mulHigh = broadcast_u16((uint16)(scale >> 16));
  for (unsigned i = 0; i < M * N / r; i++) {
    // out[offsetOut + i] = ssrs((v32acc32)in[offsetIn + i], 7, 0);
    v32int32 vecIn = in[offsetIn + i];
    v16int32 lo = extract_v16int32(vecIn, 0);
    v16int32 hi = extract_v16int32(vecIn, 1);
    // v16uint32 signLo = (v16uint32)lo & broadcast_u32(0x80000000);
    // v16uint32 signHi = (v16uint32)hi & broadcast_u32(0x80000000);
    // v16int16 signLoMask = lsrs((v16acc32)min(lo, broadcast_s32(-1)), 0, 0); // lsrs((v16acc32)min(max(lo, broadcast_s32(-1)), broadcast_s32(1)), 0, 0);
    // v16int16 signHiMask = lsrs((v16acc32)min(max(hi, broadcast_s32(-1)), broadcast_s32(1)), 0, 0);
    unsigned signLoMask = lt(lo, broadcast_s32(0));
    unsigned signHiMask = lt(hi, broadcast_s32(0));
    v16int32 t1 = sel(broadcast_s32(1), broadcast_s32(-1), signLoMask);
    v16int32 t2 = sel(broadcast_s32(1), broadcast_s32(-1), signHiMask);
    v16uint32 ulo = abs(lo);
    v16uint32 uhi = abs(hi);
    v32uint16 vecInLow = (v32uint16)shuffle((v32uint16)ulo, (v32uint16)uhi, 2);
    v32uint16 vecInHigh= (v32uint16)shuffle((v32uint16)ulo, (v32uint16)uhi, 3);

    v32acc32 acc = (v32acc32)concat(t2, t2);
    out[offsetOut + i] = ssrs(acc, 0, 0);

    // // v32acc64 test = ups_to_v32acc64(vecInHigh, 0);
    // // v16uint32 ilMlLo = ulsrs(extract_v16acc64(test, 0), 0, 0);
    // // v16uint32 ilMlHi = ulsrs(extract_v16acc64(test, 1), 0, 0);
    // // v32acc32 acc = (v32acc32)concat(mulLow, mulHigh);
    // // out[offsetOut + i] = ssrs(acc, 7, 0);
    

    // // v32acc64 ilMl = mul_elem_32(mulLow, vecInLow);
    // // v16uint32 ilMlLo = ulsrs(extract_v16acc64(ilMl, 0), 0, 0);
    // // v16uint32 ilMlHi = ulsrs(extract_v16acc64(ilMl, 1), 0, 0);
    // // v32acc32 acc = (v32acc32)concat(ilMlLo, ilMlHi);
    // // out[offsetOut + i] = ssrs(acc, 7, 0);

    // v32acc64 ilMl = mul_elem_32(vecInLow, mulLow);
    // v32acc64 ilMh = mul_elem_32(vecInLow, mulHigh);
    // v32acc64 ihMl = mul_elem_32(vecInHigh, mulLow);
    // v32acc64 ihMh = mul_elem_32(vecInHigh, mulHigh);

    // v16uint32 ilMlLo = ulsrs(extract_v16acc64(ilMl, 0), 0, 0);
    // v16uint32 ilMhLo = ulsrs(extract_v16acc64(ilMh, 0), 0, 0);
    // v16uint32 ihMlLo = ulsrs(extract_v16acc64(ihMl, 0), 0, 0);
    // v16uint32 ihMhLo = ulsrs(extract_v16acc64(ihMh, 0), 0, 0);

    // v16acc64 accLo = ups_to_v16acc64((v16uint32)ilMlLo, 0);
    // accLo = add(accLo, ups_to_v16acc64((v16uint32)ilMhLo, 16));
    // accLo = add(accLo, ups_to_v16acc64((v16uint32)ihMlLo, 16));
    // accLo = add(accLo, ups_to_v16acc64((v16uint32)ihMhLo, 32));
    // v16int16 outLo = (v16int16)ssrs(accLo, shift);

    // v16uint32 ilMlHi = ulsrs(extract_v16acc64(ilMl, 1), 0, 0);
    // v16uint32 ilMhHi = ulsrs(extract_v16acc64(ilMh, 1), 0, 0);
    // v16uint32 ihMlHi = ulsrs(extract_v16acc64(ihMl, 1), 0, 0);
    // v16uint32 ihMhHi = ulsrs(extract_v16acc64(ihMh, 1), 0, 0);

    // v16acc64 accHi = ups_to_v16acc64((v16uint32)ilMlHi, 0);
    // accHi = add(accHi, ups_to_v16acc64((v16uint32)ilMhHi, 16));
    // accHi = add(accHi, ups_to_v16acc64((v16uint32)ihMlHi, 16));
    // accHi = add(accHi, ups_to_v16acc64((v16uint32)ihMhHi, 32));
    // v16int16 outHi = (v16int16)ssrs(accHi, shift);

    // v32int16 acc = (v32int16)concat(outLo, outHi);
    // // v32int16 accSign = (v32int16)concat(signLoMask, signHiMask);
    // // v32acc64 acc64 = mul_elem_32(acc, accSign);
    // // v32int16 out16 = ssrs(acc64, 0, 0);
    // out[offsetOut + i] = ssrs(ups_to_v32acc32(acc, 0), 0, 0);
  }
  clr_sat();
}

template<int M, int N, int r>
void zero_vectorized(v16int32 *__restrict pC, unsigned offsetC)
{
  v16int32 zeros = broadcast_zero_to_v16int32();
  for (unsigned i = offsetC / r; i < offsetC / r + M * N / r; i++) {
    pC[i] = zeros;
  }
}

template<unsigned rowA, unsigned colA, unsigned colB, unsigned L0_M, unsigned L0_K, unsigned L0_N>
void matmul_vectorized_i8_i32(const int8 * __restrict pA, unsigned offsetA, const int8 * __restrict pB, unsigned offsetB, int32 * __restrict pC, unsigned offsetC)
{
  const unsigned size_A = L0_M * L0_K;
  const unsigned size_B = L0_K * L0_N;
  const unsigned size_C = L0_M * L0_N;

  v64int8 A0;
  v64int8 A1;
  v64int8 B0;
  v64int8 B1;
  v64acc32 acc_C00;
  v64acc32 acc_C01;
  v64acc32 acc_C10;
  v64acc32 acc_C11;

  for (unsigned z = 0; z < rowA; z += 2) {
    v64acc32 *__restrict pC0 = (v64acc32 *)(pC + offsetC + (z)*size_C);
    v64acc32 *__restrict pC1 = (v64acc32 *)(pC + offsetC + ((z + 1)) * size_C);

    for (unsigned j = 0; j < colB; j += 2) {
      const v64int8 *__restrict pA0 = (v64int8 *)(pA + offsetA + (z)*size_A);
      const v64int8 *__restrict pA1 = (v64int8 *)(pA + offsetA + ((z + 1)) * size_A);

      const v64int8 *__restrict pB0 = (v64int8 *)(pB + offsetB + (j)*colA*size_B);
      const v64int8 *__restrict pB1 = (v64int8 *)(pB + offsetB + ((j + 1))*colA * size_B);

      A0 = *pA0;
      pA0 += rowA;
      A1 = *pA1;
      pA1 += rowA;

      B0 = *pB0++;
      B1 = *pB1++;

      acc_C00 = *pC0;
      acc_C01 = *(pC0 + rowA);

      acc_C10 = *pC1;
      acc_C11 = *(pC1 + rowA);

      acc_C00 = mac_8x8_8x8(A0, B0, acc_C00);
      acc_C01 = mac_8x8_8x8(A0, B1, acc_C01);
      acc_C10 = mac_8x8_8x8(A1, B0, acc_C10);
      acc_C11 = mac_8x8_8x8(A1, B1, acc_C11);

      for (unsigned i = 1; i < colA; ++i) {
        A0 = *pA0;
        pA0 += rowA;
        A1 = *pA1;
        pA1 += rowA;

        B0 = *pB0++;
        B1 = *pB1++;

        acc_C00 = mac_8x8_8x8(A0, B0, acc_C00);
        acc_C01 = mac_8x8_8x8(A0, B1, acc_C01);
        acc_C10 = mac_8x8_8x8(A1, B0, acc_C10);
        acc_C11 = mac_8x8_8x8(A1, B1, acc_C11);
      }

      // -----

      v64acc32 * __restrict pOut00 = pC0;
      *pOut00 = acc_C00;
      pC0 += rowA;

      v64acc32 * __restrict pOut01 = pC0;
      *pOut01 = acc_C01;
      pC0 += rowA;

      // -----

      v64acc32 * __restrict pOut10 = pC1;
      *pOut10 = acc_C10;
      pC1 += rowA;

      v64acc32 * __restrict pOut11 = pC1;
      *pOut11 = acc_C11;
      pC1 += rowA;
    }
  }
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_8x8x8_i8_i8_i32(const int8 *__restrict pA,
                                      unsigned offsetA,
                                      const int8 *__restrict pB,
                                      unsigned offsetB, int32 *__restrict pC,
                                      unsigned offsetC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(m / r > 0);
  static_assert(k / s > 0);
  static_assert(n / t > 0);
  return matmul_vectorized_i8_i32<m / r, k / s, n / t, r, s, t>
      (pA, offsetA, pB, offsetB, pC, offsetC);
}

v64bfp16ebs8 load_v64bf16_as_bfp16(const bfloat16 *__restrict p) {
  v32bfloat16 v0 = *(v32bfloat16 *)(p);
  v32bfloat16 v1 = *(v32bfloat16 *)(p + 32);
  v32accfloat accum0 = ups(v0);
  v32accfloat accum1 = ups(v1);
  v64accfloat accum = concat(accum0, accum1);
  return to_v64bfp16ebs8(accum);
}

v64bfp16ebs8 load_v64bf16_as_bfp16_T(const bfloat16 *__restrict p) {
  v32bfloat16 v0 = *(v32bfloat16 *)(p);
  v32bfloat16 v1 = *(v32bfloat16 *)(p + 32);
  v32bfloat16 v0_shuffed = shuffle(v0, 29);
  v32bfloat16 v1_shuffed = shuffle(v1, 29);
  v32bfloat16 v_shuffed_lo = shuffle(v0_shuffed, v1_shuffed, 14);
  v32bfloat16 v_shuffed_hi = shuffle(v0_shuffed, v1_shuffed, 15);
  v32accfloat accum0 = ups(v_shuffed_lo);
  v32accfloat accum1 = ups(v_shuffed_hi);
  v64accfloat accum = concat(accum0, accum1);
  return to_v64bfp16ebs8(accum);
}

template<int M, int N, int r>
void zero_vectorized(v16float *__restrict pC, unsigned offsetC)
{
  v16float zeros = broadcast_zero_to_v16float();
  for (unsigned i = offsetC / r; i < offsetC / r + M * N / r; i++) {
    pC[i] = zeros;
  }
}


template<unsigned rowA, unsigned colA, unsigned colB, unsigned L0_M, unsigned L0_K, unsigned L0_N>
void matmul_vectorized_bf16_f32(const bfloat16 * __restrict pA, unsigned offsetA, const bfloat16 * __restrict pB,
                                unsigned offsetB, float * __restrict pC, unsigned offsetC)
{
  const unsigned size_A = L0_M * L0_K;
  const unsigned size_B = L0_K * L0_N;
  const unsigned size_C = L0_M * L0_N;

  v64bfp16ebs8 A0;
  v64bfp16ebs8 A1;
  v64bfp16ebs8 B0;
  v64bfp16ebs8 B1;
  v64accfloat acc_C00;
  v64accfloat acc_C01;
  v64accfloat acc_C10;
  v64accfloat acc_C11;

  for (unsigned z = 0; z < rowA; z += 2) {
    v64accfloat *__restrict pC0 = (v64accfloat *)(pC + offsetC + (z)*size_C);
    v64accfloat *__restrict pC1 = (v64accfloat *)(pC + offsetC + ((z + 1)) * size_C);

    for (unsigned j = 0; j < colB; j += 2) {
      const bfloat16 *__restrict pA0 = (bfloat16 *)(pA + offsetA + (z)*size_A);
      const bfloat16 *__restrict pA1 = (bfloat16 *)(pA + offsetA + ((z + 1)) * size_A);

      const bfloat16 *__restrict pB0 = (bfloat16 *)(pB + offsetB + (j)*colA*size_B);
      const bfloat16 *__restrict pB1 = (bfloat16 *)(pB + offsetB + ((j + 1))*colA * size_B);

      A0 = load_v64bf16_as_bfp16(pA0);
      pA0 += rowA * size_A;
      A1 = load_v64bf16_as_bfp16(pA1);
      pA1 += rowA * size_A;

      B0 = load_v64bf16_as_bfp16_T(pB0);
      pB0 += size_B;
      B1 = load_v64bf16_as_bfp16_T(pB1);
      pB1 += size_B;

      acc_C00 = *pC0;
      acc_C01 = *(pC0 + rowA);

      acc_C10 = *pC1;
      acc_C11 = *(pC1 + rowA);

      acc_C00 = mac_8x8_8x8T( A0, B0, acc_C00);
      acc_C01 = mac_8x8_8x8T( A0, B1, acc_C01);
      acc_C10 = mac_8x8_8x8T( A1, B0, acc_C10);
      acc_C11 = mac_8x8_8x8T( A1, B1, acc_C11);


      for (unsigned i = 1; i < colA; ++i) {
        A0 = load_v64bf16_as_bfp16(pA0);
        pA0 += rowA * size_A;
        A1 = load_v64bf16_as_bfp16(pA1);
        pA1 += rowA * size_A;

        B0 = load_v64bf16_as_bfp16_T(pB0);
        pB0 += size_B;
        B1 = load_v64bf16_as_bfp16_T(pB1);
        pB1 += size_B;

        acc_C00 = mac_8x8_8x8T( A0, B0, acc_C00);
        acc_C01 = mac_8x8_8x8T( A0, B1, acc_C01);
        acc_C10 = mac_8x8_8x8T( A1, B0, acc_C10);
        acc_C11 = mac_8x8_8x8T( A1, B1, acc_C11);
      }

      // -----

      v64accfloat * __restrict pOut00 = pC0;
      *pOut00 = acc_C00;
      pC0 += rowA;

      v64accfloat * __restrict pOut01 = pC0;
      *pOut01 = acc_C01;
      pC0 += rowA;

      // -----

      v64accfloat * __restrict pOut10 = pC1;
      *pOut10 = acc_C10;
      pC1 += rowA;

      v64accfloat * __restrict pOut11 = pC1;
      *pOut11 = acc_C11;
      pC1 += rowA;
    }
  }
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_8x8x8_bf16_bf16_f32(const bfloat16 *__restrict pA,
                                      unsigned offsetA,
                                      const bfloat16 *__restrict pB,
                                      unsigned offsetB, float *__restrict pC,
                                      unsigned offsetC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(m / r > 0);
  static_assert(k / s > 0);
  static_assert(n / t > 0);
  return matmul_vectorized_bf16_f32<m / r, k / s, n / t, r, s, t>
      (pA, offsetA, pB, offsetB, pC, offsetC);
}

extern "C" {

#define matmul_combos_i8(X, M, N, K)                                  \
  X(int8, i8, int8, i8, int32, i32, M, N, K, 8, 8, 8)

#define zero_fill_combos_i32(X, M, N)  \
  X(v16int32, i32, M, N, 16)

#define scale_shift_trunc_combos_i32_i8(X, M, N)  \
  X(v32int32, i32, v32int8, i8, M, N, 16)

#define matmul_combos_bfp16(X, M, N, K)                                     \
  X(bfloat16, bf16, bfloat16, bf16, float, f32, M, N, K, 8, 8, 8)

#define zero_fill_combos_f32(X, M, N)  \
  X(v16float, f32, M, N, 16)

#define matmul_vectorized_c_func(lhs_ctype_in, lhs_mlir_type_in,                                                 \
                                 rhs_ctype_in, rhs_mlir_type_in,                                                 \
                                 acc_ctype_out, acc_mlir_type_out, M, N, K, r, s, t)                             \
  void matmul_##lhs_mlir_type_in##_##rhs_mlir_type_in##_##acc_mlir_type_out##_##M##x##N##x##K##_##r##x##s##x##t( \
      lhs_ctype_in *a_in, unsigned offsetA, rhs_ctype_in *b_in, unsigned offsetB,                                \
      acc_ctype_out *c_out, unsigned offsetC) {                                                                  \
    matmul_vectorized_##r##x##s##x##t##_##lhs_mlir_type_in##_##rhs_mlir_type_in##_##acc_mlir_type_out<           \
        M, K, N>(a_in, offsetA, b_in, offsetB, c_out, offsetC);                                                  \
  }

#define zero_vectorized_c_func(ctype_out, mlir_type_out, M, N, r)             \
  void zero_##mlir_type_out##_##M##x##N(ctype_out *c_out, unsigned offsetC) { \
    zero_vectorized<M, N, r>(c_out, offsetC);                      \
  }

#define scale_shift_trunc_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, M, N, r)   \
  void scale_shift_trunc_##mlir_type_in##_##mlir_type_out##_##M##x##N(                        \
      ctype_in *in, int64_t offsetIn, int64_t scale, int64_t shift,                           \
      ctype_out *out, int64_t offsetOut) {                                                    \
    scale_shift_trunc_vectorized<M, N, r>(in, offsetIn, scale, shift, out, offsetOut);                                                 \
  }

matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 64)
matmul_combos_i8(matmul_vectorized_c_func, 32, 32, 128)
matmul_combos_i8(matmul_vectorized_c_func, 64, 64, 128)

zero_fill_combos_i32(zero_vectorized_c_func, 32, 32)
zero_fill_combos_i32(zero_vectorized_c_func, 64, 64)

scale_shift_trunc_combos_i32_i8(scale_shift_trunc_c_func, 32, 32)
scale_shift_trunc_combos_i32_i8(scale_shift_trunc_c_func, 64, 64)

matmul_combos_bfp16(matmul_vectorized_c_func, 16, 8, 32)
matmul_combos_bfp16(matmul_vectorized_c_func, 16, 8, 64)
matmul_combos_bfp16(matmul_vectorized_c_func, 16, 16, 32)
matmul_combos_bfp16(matmul_vectorized_c_func, 32, 32, 32)
matmul_combos_bfp16(matmul_vectorized_c_func, 32, 32, 64)

zero_fill_combos_f32(zero_vectorized_c_func, 16, 8)
zero_fill_combos_f32(zero_vectorized_c_func, 16, 16)
zero_fill_combos_f32(zero_vectorized_c_func, 32, 32)

}  // extern "C"
)peano"
