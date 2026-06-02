// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bring-up ukernels for SmolLM2 decode attention. This intentionally starts as
// scalar Peano C++ so the first milestone is end-to-end execution without
// depending on libm or the high-level AIE API.

#include <aie2pintrin.h>
#include <stdint.h>

constexpr int D = 64;
constexpr int SKV_T = 64;
constexpr float SCALE = 0.18033688011112042f;  // log2(e) / sqrt(64)
constexpr float LN2 = 0.6931471805599453f;
constexpr float NEG_INF = -3.0e38f;
constexpr float TAIL_MARKER_BASE = 128.0f;

static inline bfloat16 exp2_scalar_bf16(float x) {
  // Coarse branch-only exp2 approximation in log2 domain. This keeps Peano
  // away from libm/aie_api/native-exp2 while the shifted vector exp path is
  // isolated. Good enough for single-tile dataflow probes, not final softmax.
  if (x < -8.0f) return (bfloat16)0.00390625f;
  if (x < -7.0f) return (bfloat16)0.0078125f;
  if (x < -6.0f) return (bfloat16)0.015625f;
  if (x < -5.0f) return (bfloat16)0.03125f;
  if (x < -4.0f) return (bfloat16)0.0625f;
  if (x < -3.0f) return (bfloat16)0.125f;
  if (x < -2.0f) return (bfloat16)0.25f;
  if (x < -1.0f) return (bfloat16)0.5f;
  if (x < -0.5f) return (bfloat16)0.7071067811865476f;
  if (x < -0.25f) return (bfloat16)0.8408964152537145f;
  if (x < -0.125f) return (bfloat16)0.9170040432046712f;
  if (x < -0.0625f) return (bfloat16)0.9576032806985737f;
  return (bfloat16)1.0f;
}

static inline uint16_t f32_to_bf16_bits(float x) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = x;
  uint32_t lsb = (v.u >> 16) & 1u;
  v.u += 0x7fffu + lsb;
  return (uint16_t)(v.u >> 16);
}

static inline float bf16_bits_to_f32(uint16_t x) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = (uint32_t)x << 16;
  return v.f;
}

static inline float bf16_to_f32(bfloat16 x) {
  return bf16_bits_to_f32(*(uint16_t *)&x);
}

static inline float pow2_int_f32(int n) {
  if (n < -126) return 0.0f;
  if (n > 127) n = 127;
  union {
    uint32_t u;
    float f;
  } v;
  v.u = (uint32_t)(n + 127) << 23;
  return v.f;
}

static inline float exp_approx_f32(float x) {
  constexpr float LOG2E = 1.4426950408889634f;
  constexpr float LN2 = 0.6931471805599453f;

  // Range reduce into approximately [-ln(2)/2, ln(2)/2] without libm. The
  // polynomial is only used for SwiGLU bring-up; the vector exp2 path remains
  // the performance target once the full fused path is numerically
  // characterized.
  float nf = x * LOG2E;
  int n = (int)(nf + (nf >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)n * LN2;
  float r2 = r * r;
  float r3 = r2 * r;
  float r4 = r2 * r2;
  float r5 = r4 * r;
  float e = 1.0f + r + 0.5f * r2 + 0.16666666666666666f * r3 +
            0.041666666666666664f * r4 + 0.008333333333333333f * r5;
  return e * pow2_int_f32(n);
}

static inline float exp_nonpos_approx_f32(float x) {
  if (x != x) return 0.0f;
  if (x >= 0.0f) return 1.0f;
  if (x < -20.0f) return 0.0f;
  return exp_approx_f32(x);
}

static inline void store_f32_as_bf16(float x, bfloat16 *out) {
  *(uint16_t *)out = f32_to_bf16_bits(x);
}

static inline void accumulate_pv_bf16(bfloat16 p, const bfloat16 *__restrict vp,
                                      float *__restrict o) {
  v32bfloat16 pvec = broadcast_to_v32bfloat16(p);

  v32accfloat acc0 = *(v32accfloat *)(o);
  v32bfloat16 v0 = *(const v32bfloat16 *)(vp);
  acc0 = mac_elem_32(pvec, v0, acc0);
  *(v32accfloat *)(o) = acc0;

  v32accfloat acc1 = *(v32accfloat *)(o + 32);
  v32bfloat16 v1 = *(const v32bfloat16 *)(vp + 32);
  acc1 = mac_elem_32(pvec, v1, acc1);
  *(v32accfloat *)(o + 32) = acc1;
}

static inline void exp2_shifted_block_bf16(const bfloat16 *__restrict s,
                                           bfloat16 m, bfloat16 *__restrict p) {
  v32bfloat16 sv = *(const v32bfloat16 *)(s);
  v32bfloat16 mv = broadcast_to_v32bfloat16(m);
  v32accfloat shifted = sub(ups(sv), ups(mv));
  v32bfloat16 pv = exp2(shifted);
  *(v32bfloat16 *)(p) = pv;
}

extern "C" {

void flash_init_bf16(float *m, unsigned offsetM, float *l, unsigned offsetL,
                     float *o, unsigned offsetO) {
  m += offsetM;
  l += offsetL;
  o += offsetO;
  m[0] = NEG_INF;
  l[0] = 0.0f;
  for (int d = 0; d < D; ++d) {
    o[d] = 0.0f;
  }
}

void copy_q_bf16(const bfloat16 *__restrict staged, unsigned offsetStaged,
                 bfloat16 *__restrict q, unsigned offsetQ) {
  staged += offsetStaged;
  q += offsetQ;
  for (int d = 0; d < D; ++d) {
    q[d] = staged[d];
  }
}

void copy_vec_bf16(const bfloat16 *__restrict in, unsigned offsetIn,
                   bfloat16 *__restrict out, unsigned offsetOut) {
  in += offsetIn;
  out += offsetOut;
  for (int d = 0; d < D; ++d) {
    out[d] = in[d];
  }
}

void copy_vec_offset_bf16(int len, int xOffset,
                          const bfloat16 *__restrict in, unsigned offsetIn,
                          bfloat16 *__restrict out, unsigned offsetOut) {
  in += offsetIn;
  out += offsetOut + xOffset;
  for (int i = 0; i < len; ++i) {
    out[i] = in[i];
  }
}

void add_rmsnorm_prepare_bf16(int len, const bfloat16 *__restrict x,
                              unsigned offsetX,
                              const bfloat16 *__restrict residual,
                              unsigned offsetResidual, float *__restrict tmp,
                              unsigned offsetTmp, float *__restrict inv,
                              unsigned offsetInv) {
  x += offsetX;
  residual += offsetResidual;
  tmp += offsetTmp;
  inv += offsetInv;

  float sumSq = 0.0f;
  for (int i = 0; i < len; ++i) {
    float v = bf16_to_f32(x[i]) + bf16_to_f32(residual[i]);
    tmp[i] = v;
    sumSq += v * v;
  }
  inv[0] = invsqrt(sumSq / (float)len + 1.0e-5f);
}

void rmsnorm_copy_tmp_bf16(int len, const bfloat16 *__restrict x,
                           unsigned offsetX, float *__restrict tmp,
                           unsigned offsetTmp) {
  x += offsetX;
  tmp += offsetTmp;

  for (int i = 0; i < len; ++i) {
    tmp[i] = bf16_to_f32(x[i]);
  }
}

void rmsnorm_sum_init_bf16(float *__restrict inv, unsigned offsetInv) {
  inv += offsetInv;
  inv[0] = 0.0f;
}

void rmsnorm_copy_tmp_offset_bf16(int len, int xOffset,
                                  const bfloat16 *__restrict x,
                                  unsigned offsetX,
                                  float *__restrict tmp,
                                  unsigned offsetTmp) {
  x += offsetX;
  tmp += offsetTmp + xOffset;

  for (int i = 0; i < len; ++i) {
    tmp[i] = bf16_to_f32(x[i]);
  }
}

void rmsnorm_add_residual_accum_bf16(
    int len, int xOffset, const bfloat16 *__restrict residual,
    unsigned offsetResidual, float *__restrict tmp, unsigned offsetTmp,
    float *__restrict inv, unsigned offsetInv) {
  residual += offsetResidual;
  tmp += offsetTmp + xOffset;
  inv += offsetInv;

  float sumSq = inv[0];
  for (int i = 0; i < len; ++i) {
    float v = tmp[i] + bf16_to_f32(residual[i]);
    tmp[i] = v;
    sumSq += v * v;
  }
  inv[0] = sumSq;
}

void rmsnorm_finalize_inv_bf16(int len, float *__restrict inv,
                               unsigned offsetInv) {
  inv += offsetInv;
  inv[0] = invsqrt(inv[0] / (float)len + 1.0e-5f);
}

void rmsnorm_add_residual_prepare_bf16(int len,
                                       const bfloat16 *__restrict residual,
                                       unsigned offsetResidual,
                                       float *__restrict tmp,
                                       unsigned offsetTmp,
                                       float *__restrict inv,
                                       unsigned offsetInv) {
  residual += offsetResidual;
  tmp += offsetTmp;
  inv += offsetInv;

  float sumSq = 0.0f;
  for (int i = 0; i < len; ++i) {
    float v = tmp[i] + bf16_to_f32(residual[i]);
    tmp[i] = v;
    sumSq += v * v;
  }
  inv[0] = invsqrt(sumSq / (float)len + 1.0e-5f);
}

void add_rmsnorm_prepare_residual_bf16(
    int len, const bfloat16 *__restrict x, unsigned offsetX,
    const bfloat16 *__restrict residual, unsigned offsetResidual,
    float *__restrict tmp, unsigned offsetTmp, float *__restrict inv,
    unsigned offsetInv, bfloat16 *__restrict hidden, unsigned offsetHidden) {
  x += offsetX;
  residual += offsetResidual;
  tmp += offsetTmp;
  inv += offsetInv;
  hidden += offsetHidden;

  float sumSq = 0.0f;
  for (int i = 0; i < len; ++i) {
    float v = bf16_to_f32(x[i]) + bf16_to_f32(residual[i]);
    tmp[i] = v;
    store_f32_as_bf16(v, hidden + i);
    sumSq += v * v;
  }
  inv[0] = invsqrt(sumSq / (float)len + 1.0e-5f);
}

void add_residual_bf16(int len, const bfloat16 *__restrict x, unsigned offsetX,
                       const bfloat16 *__restrict residual,
                       unsigned offsetResidual, bfloat16 *__restrict hidden,
                       unsigned offsetHidden) {
  x += offsetX;
  residual += offsetResidual;
  hidden += offsetHidden;

  for (int i = 0; i < len; ++i) {
    store_f32_as_bf16(bf16_to_f32(x[i]) + bf16_to_f32(residual[i]),
                      hidden + i);
  }
}

void rmsnorm_scale_bf16(int len, const float *__restrict tmp,
                        unsigned offsetTmp, const bfloat16 *__restrict weight,
                        unsigned offsetWeight, const float *__restrict inv,
                        unsigned offsetInv, bfloat16 *__restrict out,
                        unsigned offsetOut) {
  tmp += offsetTmp;
  weight += offsetWeight;
  inv += offsetInv;
  out += offsetOut;

  const float s = inv[0];
  for (int i = 0; i < len; ++i) {
    store_f32_as_bf16(tmp[i] * s * bf16_to_f32(weight[i]), out + i);
  }
}

void matvec_accum_bf16(int k, int n, const bfloat16 *__restrict x,
                       unsigned offsetX, const bfloat16 *__restrict w,
                       unsigned offsetW, float *__restrict acc,
                       unsigned offsetAcc) {
  x += offsetX;
  w += offsetW;
  acc += offsetAcc;

  for (int kk = 0; kk < k; ++kk) {
    const float xv = bf16_to_f32(x[kk]);
    const bfloat16 *__restrict wp = w + kk * n;
    for (int j = 0; j < n; ++j) {
      acc[j] += xv * bf16_to_f32(wp[j]);
    }
  }
}

void rmsnorm_matvec_accum_bf16(
    int k, int n, int xOffset, const float *__restrict tmp,
    unsigned offsetTmp, const bfloat16 *__restrict rmsWeight,
    unsigned offsetRmsWeight, const float *__restrict inv, unsigned offsetInv,
    const bfloat16 *__restrict w, unsigned offsetW, float *__restrict acc,
    unsigned offsetAcc) {
  tmp += offsetTmp + xOffset;
  rmsWeight += offsetRmsWeight + xOffset;
  inv += offsetInv;
  w += offsetW;
  acc += offsetAcc;

  const float s = inv[0];
  for (int kk = 0; kk < k; ++kk) {
    const float xv = bf16_bits_to_f32(
        f32_to_bf16_bits(tmp[kk] * s * bf16_to_f32(rmsWeight[kk])));
    const bfloat16 *__restrict wp = w + kk * n;
    for (int j = 0; j < n; ++j) {
      acc[j] += xv * bf16_to_f32(wp[j]);
    }
  }
}

void rmsnorm_matvec_accum_chunk_bf16(
    int k, int n, int xOffset, const float *__restrict tmp,
    unsigned offsetTmp, const bfloat16 *__restrict rmsWeight,
    unsigned offsetRmsWeight, const float *__restrict inv, unsigned offsetInv,
    const bfloat16 *__restrict w, unsigned offsetW, float *__restrict acc,
    unsigned offsetAcc) {
  tmp += offsetTmp + xOffset;
  rmsWeight += offsetRmsWeight;
  inv += offsetInv;
  w += offsetW;
  acc += offsetAcc;

  const float s = inv[0];
  for (int kk = 0; kk < k; ++kk) {
    const float xv = bf16_bits_to_f32(
        f32_to_bf16_bits(tmp[kk] * s * bf16_to_f32(rmsWeight[kk])));
    const bfloat16 *__restrict wp = w + kk * n;
    for (int j = 0; j < n; ++j) {
      acc[j] += xv * bf16_to_f32(wp[j]);
    }
  }
}

void swiglu_matvec_accum_bf16(int k, int n, const bfloat16 *__restrict gate,
                              unsigned offsetGate,
                              const bfloat16 *__restrict up, unsigned offsetUp,
                              const bfloat16 *__restrict w, unsigned offsetW,
                              float *__restrict acc, unsigned offsetAcc) {
  gate += offsetGate;
  up += offsetUp;
  w += offsetW;
  acc += offsetAcc;

  for (int kk = 0; kk < k; ++kk) {
    const float g = bf16_to_f32(gate[kk]);
    const float expNeg = exp_approx_f32(-g);
    const float sigmoid = 1.0f / (1.0f + expNeg);
    const float silu = bf16_bits_to_f32(f32_to_bf16_bits(g * sigmoid));
    const float xv =
        bf16_bits_to_f32(f32_to_bf16_bits(silu * bf16_to_f32(up[kk])));
    const bfloat16 *__restrict wp = w + kk * n;
    for (int j = 0; j < n; ++j) {
      acc[j] += xv * bf16_to_f32(wp[j]);
    }
  }
}

void matvec_store_bf16(int n, const float *__restrict acc, unsigned offsetAcc,
                       bfloat16 *__restrict out, unsigned offsetOut) {
  acc += offsetAcc;
  out += offsetOut;
  for (int j = 0; j < n; ++j) {
    store_f32_as_bf16(acc[j], out + j);
  }
}

void matvec_store_add_bf16(int n, const float *__restrict acc,
                           unsigned offsetAcc,
                           const bfloat16 *__restrict residual,
                           unsigned offsetResidual, bfloat16 *__restrict out,
                           unsigned offsetOut) {
  acc += offsetAcc;
  residual += offsetResidual;
  out += offsetOut;
  for (int j = 0; j < n; ++j) {
    const float proj = bf16_bits_to_f32(f32_to_bf16_bits(acc[j]));
    store_f32_as_bf16(proj + bf16_to_f32(residual[j]), out + j);
  }
}

void matvec_store_pack_add_bf16(int n, const float *__restrict acc,
                                unsigned offsetAcc,
                                const bfloat16 *__restrict residual,
                                unsigned offsetResidual,
                                bfloat16 *__restrict packed,
                                unsigned offsetPacked) {
  acc += offsetAcc;
  residual += offsetResidual;
  packed += offsetPacked;
  for (int j = 0; j < n; ++j) {
    const float proj = bf16_bits_to_f32(f32_to_bf16_bits(acc[j]));
    store_f32_as_bf16(proj, packed + j);
    store_f32_as_bf16(proj + bf16_to_f32(residual[j]), packed + n + j);
  }
}

void matvec_store_add_if_row_bf16(int n, int activeRow, int packetRow,
                                  const float *__restrict acc,
                                  unsigned offsetAcc,
                                  const bfloat16 *__restrict residual,
                                  unsigned offsetResidual,
                                  bfloat16 *__restrict out,
                                  unsigned offsetOut) {
  if (activeRow != packetRow)
    return;
  matvec_store_add_bf16(n, acc, offsetAcc, residual, offsetResidual, out,
                        offsetOut);
}

void swiglu_bf16(int len, const bfloat16 *__restrict gate, unsigned offsetGate,
                 const bfloat16 *__restrict up, unsigned offsetUp,
                 bfloat16 *__restrict out, unsigned offsetOut) {
  gate += offsetGate;
  up += offsetUp;
  out += offsetOut;

  for (int i = 0; i < len; ++i) {
    float g = bf16_to_f32(gate[i]);
    float expNeg = exp_approx_f32(-g);
    float sigmoid = 1.0f / (1.0f + expNeg);
    float silu = bf16_bits_to_f32(f32_to_bf16_bits(g * sigmoid));
    store_f32_as_bf16(silu * bf16_to_f32(up[i]), out + i);
  }
}

void rope_rotate_half_bf16(int len, const bfloat16 *__restrict x,
                           unsigned offsetX, const bfloat16 *__restrict lut,
                           unsigned offsetLut, bfloat16 *__restrict out,
                           unsigned offsetOut) {
  x += offsetX;
  lut += offsetLut;
  out += offsetOut;

  const int half = len >> 1;
  const bfloat16 *__restrict cosv = lut;
  const bfloat16 *__restrict sinv = lut + len;
  for (int i = 0; i < half; ++i) {
    const int j = i + half;
    const float x0 = bf16_to_f32(x[i]);
    const float x1 = bf16_to_f32(x[j]);
    store_f32_as_bf16(x0 * bf16_to_f32(cosv[i]) - x1 * bf16_to_f32(sinv[i]),
                      out + i);
    store_f32_as_bf16(x1 * bf16_to_f32(cosv[j]) + x0 * bf16_to_f32(sinv[j]),
                      out + j);
  }
}

void exp2_probe_bf16(const bfloat16 *__restrict s, unsigned offsetS,
                     const bfloat16 *__restrict m, unsigned offsetM,
                     bfloat16 *__restrict p, unsigned offsetP) {
  s += offsetS;
  m += offsetM;
  p += offsetP;

  exp2_shifted_block_bf16(s, m[0], p);
  exp2_shifted_block_bf16(s + 32, m[0], p + 32);

  float sum = 0.0f;
  for (int j = 0; j < SKV_T; ++j) {
    sum += bf16_to_f32(p[j]);
  }
  ((uint16_t *)p)[SKV_T] = f32_to_bf16_bits(sum);
}

void attn_qk_bf16(const bfloat16 *__restrict q, unsigned offsetQ,
                  const bfloat16 *__restrict k, unsigned offsetK,
                  bfloat16 *__restrict s, unsigned offsetS,
                  float *__restrict prod, unsigned offsetProd) {
  q += offsetQ;
  k += offsetK;
  s += offsetS;
  prod += offsetProd;
  const v32bfloat16 q0 = *(const v32bfloat16 *)(q);
  const v32bfloat16 q1 = *(const v32bfloat16 *)(q + 32);
  uint16_t *__restrict sRaw = (uint16_t *)s;

  for (int j = 0; j < SKV_T; ++j) {
    const bfloat16 *__restrict kp = k + j * D;
    float acc = 0.0f;
    *(v32accfloat *)prod = mul_elem_32(q0, *(const v32bfloat16 *)(kp));
    for (int d = 0; d < 32; ++d) {
      acc += prod[d];
    }
    *(v32accfloat *)prod = mul_elem_32(q1, *(const v32bfloat16 *)(kp + 32));
    for (int d = 0; d < 32; ++d) {
      acc += prod[d];
    }
    sRaw[j] = f32_to_bf16_bits(SCALE * acc);
  }
}

void attn_qk_seq_bf16(int seqLen, int tileStart,
                      const bfloat16 *__restrict q, unsigned offsetQ,
                      const bfloat16 *__restrict k, unsigned offsetK,
                      bfloat16 *__restrict s, unsigned offsetS,
                      float *__restrict prod, unsigned offsetProd) {
  if (seqLen <= tileStart) return;
  attn_qk_bf16(q, offsetQ, k, offsetK, s, offsetS, prod, offsetProd);
}

static inline void flash_update_lanes_bf16(
    int lanes, const bfloat16 *__restrict s, unsigned offsetS,
    const bfloat16 *__restrict v, unsigned offsetV, bfloat16 *__restrict pbuf,
    unsigned offsetP, float *__restrict m, unsigned offsetM,
    float *__restrict l, unsigned offsetL, float *__restrict o,
    unsigned offsetO) {
  s += offsetS;
  v += offsetV;
  pbuf += offsetP;
  m += offsetM;
  l += offsetL;
  o += offsetO;

  if (lanes < 0) lanes = 0;
  if (lanes > SKV_T) lanes = SKV_T;

  float tmax = NEG_INF;
  for (int j = 0; j < lanes; ++j) {
    float sj = bf16_to_f32(s[j]);
    if (sj == sj) tmax = tmax > sj ? tmax : sj;
  }

  float m_old = m[0];
  if (m_old != m_old) m_old = NEG_INF;
  float m_new = m_old > tmax ? m_old : tmax;
  float corr = 1.0f;
  if (m_old == NEG_INF)
    corr = 0.0f;
  else
    corr = exp_nonpos_approx_f32((m_old - m_new) * LN2);

  for (int d = 0; d < D; ++d) {
    o[d] *= corr;
  }

  float lsum = l[0] * corr;
  for (int j = 0; j < lanes; ++j) {
    float p = exp_nonpos_approx_f32((bf16_to_f32(s[j]) - m_new) * LN2);
    bfloat16 pbf;
    uint16_t pBits = f32_to_bf16_bits(p);
    *(uint16_t *)&pbf = pBits;
    *(uint16_t *)(pbuf + j) = pBits;
    lsum += bf16_to_f32(pbf);
    const bfloat16 *__restrict vp = v + j * D;
    accumulate_pv_bf16(pbf, vp, o);
  }

  m[0] = m_new;
  l[0] = lsum;
}

void flash_update_bf16(const bfloat16 *__restrict s, unsigned offsetS,
                       const bfloat16 *__restrict v, unsigned offsetV,
                       bfloat16 *__restrict pbuf, unsigned offsetP,
                       float *__restrict m, unsigned offsetM,
                       float *__restrict l, unsigned offsetL,
                       float *__restrict o, unsigned offsetO) {
  flash_update_lanes_bf16(SKV_T, s, offsetS, v, offsetV, pbuf, offsetP, m,
                          offsetM, l, offsetL, o, offsetO);
}

void flash_update_tail_bf16(int lanes, const bfloat16 *__restrict s,
                            unsigned offsetS, const bfloat16 *__restrict v,
                            unsigned offsetV, bfloat16 *__restrict pbuf,
                            unsigned offsetP, float *__restrict m,
                            unsigned offsetM, float *__restrict l,
                            unsigned offsetL, float *__restrict o,
                            unsigned offsetO) {
  flash_update_lanes_bf16(lanes, s, offsetS, v, offsetV, pbuf, offsetP, m,
                          offsetM, l, offsetL, o, offsetO);
}

void flash_update_seq_bf16(int seqLen, int tileStart,
                           const bfloat16 *__restrict s, unsigned offsetS,
                           const bfloat16 *__restrict v, unsigned offsetV,
                           bfloat16 *__restrict pbuf, unsigned offsetP,
                           float *__restrict m, unsigned offsetM,
                           float *__restrict l, unsigned offsetL,
                           float *__restrict o, unsigned offsetO) {
  flash_update_lanes_bf16(seqLen - tileStart, s, offsetS, v, offsetV, pbuf,
                          offsetP, m, offsetM, l, offsetL, o, offsetO);
}

void flash_update_tail_marker_bf16(const bfloat16 *__restrict s,
                                   unsigned offsetS,
                                   const bfloat16 *__restrict v,
                                   unsigned offsetV, bfloat16 *__restrict pbuf,
                                   unsigned offsetP, float *__restrict m,
                                   unsigned offsetM, float *__restrict l,
                                   unsigned offsetL, float *__restrict o,
                                   unsigned offsetO) {
  int lanes = SKV_T;
  float marker = bf16_to_f32(v[offsetV + (SKV_T - 1) * D]);
  int encoded = (int)(marker - TAIL_MARKER_BASE);
  if (encoded >= 0 && encoded < SKV_T) lanes = encoded;
  flash_update_lanes_bf16(lanes, s, offsetS, v, offsetV, pbuf, offsetP, m,
                          offsetM, l, offsetL, o, offsetO);
}

void flash_update_mean_bf16(const bfloat16 *__restrict s, unsigned offsetS,
                            const bfloat16 *__restrict v, unsigned offsetV,
                            bfloat16 *__restrict pbuf, unsigned offsetP,
                            float *__restrict m, unsigned offsetM,
                            float *__restrict l, unsigned offsetL,
                            float *__restrict o, unsigned offsetO) {
  (void)s;
  (void)offsetS;
  (void)pbuf;
  (void)offsetP;
  v += offsetV;
  m += offsetM;
  l += offsetL;
  o += offsetO;

  const bfloat16 one = (bfloat16)1.0f;
  for (int j = 0; j < SKV_T; ++j) {
    const bfloat16 *__restrict vp = v + j * D;
    accumulate_pv_bf16(one, vp, o);
  }
  l[0] += (float)SKV_T;
  m[0] = 0.0f;
}

void flash_finalize_bf16(const float *__restrict o, unsigned offsetO,
                         const float *__restrict l, unsigned offsetL,
                         bfloat16 *__restrict out, unsigned offsetOut) {
  o += offsetO;
  l += offsetL;
  out += offsetOut;

  float inv = l[0] == 0.0f ? 0.0f : (1.0f / l[0]);
  uint16_t *__restrict rawOut = (uint16_t *)out;
  for (int d = 0; d < D; ++d) {
    rawOut[d] = f32_to_bf16_bits(o[d] * inv);
  }
}

void rtp_fill_bf16(int value, bfloat16 *__restrict out, unsigned offsetOut) {
  out += offsetOut;
  uint16_t *__restrict rawOut = (uint16_t *)out;
  uint16_t v = f32_to_bf16_bits((float)value);
  for (int i = 0; i < 64; ++i) rawOut[i] = v;
}

void fill7_bf16(bfloat16 *__restrict out, unsigned offsetOut) {
  out += offsetOut;
  uint16_t *__restrict rawOut = (uint16_t *)out;
  uint16_t v = 0x40e0;
  for (int i = 0; i < 64; ++i) rawOut[i] = v;
}

}  // extern "C"
