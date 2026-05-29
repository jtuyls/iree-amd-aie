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
constexpr float SCALE = 0.18033688011112042f; // log2(e) / sqrt(64)
constexpr float NEG_INF = -3.0e38f;

static inline bfloat16 exp2_scalar_bf16(float x) {
  // Coarse branch-only exp2 approximation in log2 domain. This keeps Peano
  // away from libm/aie_api/native-exp2 while the shifted vector exp path is
  // isolated. Good enough for single-tile dataflow probes, not final softmax.
  if (x < -8.0f)
    return (bfloat16)0.00390625f;
  if (x < -7.0f)
    return (bfloat16)0.0078125f;
  if (x < -6.0f)
    return (bfloat16)0.015625f;
  if (x < -5.0f)
    return (bfloat16)0.03125f;
  if (x < -4.0f)
    return (bfloat16)0.0625f;
  if (x < -3.0f)
    return (bfloat16)0.125f;
  if (x < -2.0f)
    return (bfloat16)0.25f;
  if (x < -1.0f)
    return (bfloat16)0.5f;
  if (x < -0.5f)
    return (bfloat16)0.7071067811865476f;
  if (x < -0.25f)
    return (bfloat16)0.8408964152537145f;
  if (x < -0.125f)
    return (bfloat16)0.9170040432046712f;
  if (x < -0.0625f)
    return (bfloat16)0.9576032806985737f;
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
                                           bfloat16 m,
                                           bfloat16 *__restrict p) {
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

void flash_update_bf16(const bfloat16 *__restrict s, unsigned offsetS,
                       const bfloat16 *__restrict v, unsigned offsetV,
                       bfloat16 *__restrict pbuf, unsigned offsetP,
                       float *__restrict m, unsigned offsetM,
                       float *__restrict l, unsigned offsetL,
                       float *__restrict o, unsigned offsetO) {
  s += offsetS;
  v += offsetV;
  pbuf += offsetP;
  m += offsetM;
  l += offsetL;
  o += offsetO;

  float tmax = NEG_INF;
  for (int j = 0; j < SKV_T; ++j) {
    tmax = tmax > bf16_to_f32(s[j]) ? tmax : bf16_to_f32(s[j]);
  }

  float m_old = m[0];
  float m_new = m_old > tmax ? m_old : tmax;
  float corr = 1.0f;
  if (m_old == NEG_INF)
    corr = 0.0f;
  else
    corr = bf16_to_f32(exp2_scalar_bf16(m_old - m_new));

  for (int d = 0; d < D; ++d) {
    o[d] *= corr;
  }

  bfloat16 m_new_bf16 = (bfloat16)m_new;
  exp2_shifted_block_bf16(s, m_new_bf16, pbuf);
  exp2_shifted_block_bf16(s + 32, m_new_bf16, pbuf + 32);

  float lsum = l[0] * corr;
  for (int j = 0; j < SKV_T; ++j) {
    bfloat16 p = pbuf[j];
    lsum += bf16_to_f32(p);
    const bfloat16 *__restrict vp = v + j * D;
    accumulate_pv_bf16(p, vp, o);
  }

  m[0] = m_new;
  l[0] = lsum;
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

}  // extern "C"
