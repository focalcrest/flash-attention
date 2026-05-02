/******************************************************************************
 * TurboQuant KV cache dequantization helpers for SM70 Flash Attention.
 *
 * TQ k8v4 slot layout (196 bytes per head per token, head_dim=128):
 *   [key_FP8 (128B) | value_4bit_packed (64B) | scale_fp16 (2B) | zero_fp16 (2B)]
 *
 * FP8 key format: fp8e4b15 (1 sign, 4 exp with bias=15, 3 mantissa)
 *   - Same exponent bias as FP16 (15), so conversion is a simple mantissa widen.
 *
 * 4-bit value packing: 2 values per byte, little-endian
 *   - Byte d//2 holds values for dimensions d and d+1
 *   - Low nibble (bits 0-3) = dimension d
 *   - High nibble (bits 4-7) = dimension d+1
 *   - Dequantize: value = v_idx * scale + zero
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <string.h>

namespace FLASH_NAMESPACE {

// Convert a single fp8e4b15 value to FP16 (branchless).
// fp8e4b15: SEEEEMMM (sign=1, exp=4 bias=15, mant=3)
// fp16:     SEEEEMMMMMMMMMM (sign=1, exp=5 bias=15, mant=10)
// Same bias → sign<<8 shifts to bit 15; exp+mant together shift <<7.
// Zeros (exp=0, mant=0) map correctly. Subnormals map to tiny FP16 subnormals
// (negligible for TQ quantized data which never produces FP8 subnormals).
__forceinline__ __device__ half fp8e4b15_to_half(uint8_t v) {
    uint32_t fp16_bits = ((uint32_t)(v & 0x80) << 8) | ((uint32_t)(v & 0x7F) << 7);
    half result;
    memcpy(&result, &fp16_bits, sizeof(half));
    return result;
}

// Vectorized FP8→FP16 conversion for 8 consecutive elements.
// Each thread processes 8 key elements from a single KV position.
__forceinline__ __device__ void fp8e4b15_to_fp16x8(const uint8_t* src, half* dst) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[i] = fp8e4b15_to_half(src[i]);
    }
}

// Dequantize 4-bit packed values for a single KV position.
// packed: 64 bytes of 4-bit packed values (128 elements at 4 bits each)
// scale:  FP16 scale factor for this position
// zero:   FP16 zero point for this position
// dst:    output FP16 array [head_dim] in shared memory
// valid_dim: number of valid dimensions (usually head_dim=128)
__forceinline__ __device__ void dequant_4bit_values(
    const uint8_t* packed,
    half scale,
    half zero,
    half* dst,
    int head_dim) {

    const float scale_f = __half2float(scale);
    const float zero_f  = __half2float(zero);

    #pragma unroll
    for (int d = 0; d < head_dim; d += 2) {
        uint8_t byte_val = packed[d >> 1];
        // Low nibble = dimension d
        float v0 = static_cast<float>(byte_val & 0xF) * scale_f + zero_f;
        dst[d] = __float2half(v0);
        // High nibble = dimension d+1
        float v1 = static_cast<float>((byte_val >> 4) & 0xF) * scale_f + zero_f;
        dst[d + 1] = __float2half(v1);
    }
}

// Read scale and zero_point from a TQ slot's value section.
// val_section points to the start of the value portion (after key bytes).
// Layout: [packed_values (val_data_bytes) | scale_fp16 (2B) | zero_fp16 (2B)]
__forceinline__ __device__ void tq_read_scale_zero(
    const uint8_t* val_section,
    int val_data_bytes,
    half& scale,
    half& zero) {

    // Scale is stored as 2 bytes (FP16) at offset val_data_bytes
    uint16_t sc_bits;
    memcpy(&sc_bits, val_section + val_data_bytes, sizeof(uint16_t));
    memcpy(&scale, &sc_bits, sizeof(half));

    // Zero is stored as 2 bytes (FP16) at offset val_data_bytes + 2
    uint16_t zr_bits;
    memcpy(&zr_bits, val_section + val_data_bytes + 2, sizeof(uint16_t));
    memcpy(&zero, &zr_bits, sizeof(half));
}

} // namespace FLASH_NAMESPACE
