/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <float.h>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_brgemm_weights_decompression_kernel.hpp"

#define GET_OFF(field) offsetof(weights_decompression_runtime_params_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;
using namespace std::placeholders;

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::init_decomp_params(std::function<Vmm(int, int)> vmm_params, Xbyak::Reg64 reg_params, bool broadcast_values) {
    auto ic_iters = jcp_.decomp_buffer_dt == data_type::bf16 ? jcp_.ic_internal_size : 1;
    size_t oc_blocks_num = div_up(jcp_.oc_size, vec_size);
    for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
        if (broadcast_values) {
            for (size_t ic = 0; ic < ic_iters; ic++) {
                uni_vbroadcastss(vmm_params(ocb, ic), ptr[reg_params]);
            }
        } else {
            if (ic_iters > 1) {
                uni_vmovups(vmm_tmp(0), ptr[reg_params + ocb * vec_size * sizeof(float)]);
                for (size_t ic = 0; ic < ic_iters; ic++) {
                    vpermd(vmm_params(ocb, ic), vmm_mask(ic), vmm_tmp(0));
                }
            } else {
                uni_vmovups(vmm_params(ocb, 0), ptr[reg_params + ocb * vec_size * sizeof(float)]);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::generate() {
    preamble();

    mov(reg_weights, ptr[param1 + GET_OFF(weights_ptr)]);
    mov(reg_decomp_buffer, ptr[param1 + GET_OFF(decomp_buffer_ptr)]);
    if (jcp_.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(scales_ptr)]);
    }
    if (jcp_.with_zero_points) {
        mov(reg_zero_points, ptr[param1 + GET_OFF(zero_points_ptr)]);
    }
    mov(reg_ic_size, ptr[param1 + GET_OFF(ic_size)]);

    if (jcp_.ic_internal_size > 1 && jcp_.decomp_buffer_dt == data_type::bf16) {
        static const int32_t mask_low[16] = {
            0, 0, 1, 1, 2, 2, 3, 3,
            4, 4, 5, 5, 6, 6, 7, 7
        };
        static const int32_t mask_high[16] = {
            8, 8, 9, 9, 10, 10, 11, 11,
            12, 12, 13, 13, 14, 14, 15, 15
        };

        mov(reg_tmp, (size_t)mask_low);
        uni_vmovups(vmm_mask(0), ptr[reg_tmp]);
        mov(reg_tmp, (size_t)mask_high);
        uni_vmovups(vmm_mask(1), ptr[reg_tmp]);
    }

    auto vmm_lookup = vmm_tmp(0);
    auto vmm_lookup_low = vmm_tmp(0);
    auto vmm_lookup_high = vmm_tmp(1);
    auto vmm_mask8 = vmm_tmp(2);
    auto vmm_mask7 = vmm_tmp(3);
    if (jcp_.weights_dt == data_type::nf4) {
        static const float lookup[16] = {
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0
        };

        static const int32_t mask8[16] = {
            8, 8, 8, 8, 8, 8, 8, 8
        };
        static const int32_t mask7[16] = {
            7, 7, 7, 7, 7, 7, 7, 7
        };

        if (isa == avx2) {
            mov(reg_tmp, (size_t)lookup);
            uni_vmovups(vmm_lookup_low, ptr[reg_tmp]);
            uni_vmovups(vmm_lookup_high, ptr[reg_tmp + 8 * sizeof(float)]);
            mov(reg_tmp, (size_t)mask8);
            uni_vmovups(vmm_mask8, ptr[reg_tmp]);
            mov(reg_tmp, (size_t)mask7);
            uni_vmovups(vmm_mask7, ptr[reg_tmp]);
        } else {
            mov(reg_tmp, (size_t)lookup);
            uni_vmovups(vmm_lookup, ptr[reg_tmp]);
        }
    }

    if (jcp_.with_scales)
        init_decomp_params(std::bind(&jit_brgemm_weights_decompression_kernel_t::vmm_scales, this, _1, _2), reg_scales, jcp_.broadcast_scales);

    if (jcp_.with_zero_points)
        init_decomp_params(std::bind(&jit_brgemm_weights_decompression_kernel_t::vmm_zero_points, this, _1, _2), reg_zero_points, jcp_.broadcast_zero_points);

    size_t oc_blocks_num = div_up(jcp_.oc_size, vec_size);

    Xbyak::Label ic_loop_label;
    Xbyak::Label ic_end_label;

    size_t weights_dt_size = types::data_type_size(jcp_.weights_dt);
    size_t typesize_scale = one_of(jcp_.weights_dt, data_type::nf4, data_type::s4, data_type::u4) ? 2 : 1;
    size_t decomp_buf_dt_size = types::data_type_size(jcp_.decomp_buffer_dt);

    L(ic_loop_label);
    {
        cmp(reg_ic_size, 1);
        jl(ic_end_label, T_NEAR);

        for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
            for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                size_t weights_offset;
                if (jcp_.decomp_buffer_dt == data_type::bf16) {
                    weights_offset = (ocb * jcp_.ic_internal_size + ic) * vec_size * weights_dt_size / typesize_scale;
                } else {
                    weights_offset = ocb * jcp_.ic_internal_size * vec_size * weights_dt_size / typesize_scale;
                }
                const auto weights_addr = ptr[reg_weights + weights_offset];
                switch (jcp_.weights_dt) {
                    case data_type::u8: {
                        uni_vpmovzxbd(vmm_weights(0), weights_addr);
                        uni_vcvtdq2ps(vmm_weights(0), vmm_weights(0));
                        break;
                    }
                    case data_type::u4: {
                        uni_vpmovzxbd(vmm_weights(0), weights_addr);
                        if (ic % 2 == 0) {
                            uni_vpsrld(vmm_weights(0), vmm_weights(0), 4);
                        } else {
                            uni_vpslld(vmm_weights(0), vmm_weights(0), 28);
                            uni_vpsrld(vmm_weights(0), vmm_weights(0), 28);
                        }
                        uni_vcvtdq2ps(vmm_weights(0), vmm_weights(0));
                        break;
                    }
                    case data_type::s4: {
                        uni_vpmovsxbd(vmm_weights(0), weights_addr);
                        if (ic % 2 == 0) {
                            vpsrad(vmm_weights(0), vmm_weights(0), 4);
                        } else {
                            uni_vpslld(vmm_weights(0), vmm_weights(0), 28);
                            vpsrad(vmm_weights(0), vmm_weights(0), 28);
                        }
                        uni_vcvtdq2ps(vmm_weights(0), vmm_weights(0));
                        break;
                    }
                    case data_type::nf4: {
                        uni_vpmovzxbd(vmm_weights(0), weights_addr);
                        if (ic % 2 == 0) {
                            uni_vpsrld(vmm_weights(0), vmm_weights(0), 4);
                        } else {
                            uni_vpslld(vmm_weights(0), vmm_weights(0), 28);
                            uni_vpsrld(vmm_weights(0), vmm_weights(0), 28);
                        }

                        if (isa == avx2) {
                            auto res = vmm_weights(1);
                            auto mask = vmm_weights(2);
                            vpcmpgtd(mask, vmm_weights(0), vmm_mask7);
                            vpermd(res, vmm_weights(0), vmm_lookup_low);
                            vpsubd(vmm_weights(0), vmm_weights(0), vmm_mask8);
                            vpermd(vmm_weights(0), vmm_weights(0), vmm_lookup_high);
                            vblendvps(vmm_weights(0), res, vmm_weights(0), mask);
                        } else {
                            vpermd(vmm_weights(0), vmm_weights(0), vmm_lookup);
                        }
                        break;
                    }
                    default: assert(!"unsupported data type");
                }

                if (jcp_.with_zero_points)
                    uni_vsubps(vmm_weights(0), vmm_weights(0), vmm_zero_points(ocb, ic));
                if (jcp_.with_scales)
                    uni_vmulps(vmm_weights(0), vmm_weights(0), vmm_scales(ocb, ic));

                size_t decomp_buffer_offset;
                if (jcp_.decomp_buffer_dt == data_type::bf16) {
                    decomp_buffer_offset = (ocb * jcp_.ic_internal_size + ic) * vec_size * decomp_buf_dt_size;
                } else {
                    decomp_buffer_offset = (ic * jcp_.oc_size + ocb * vec_size) * decomp_buf_dt_size;
                }
                const auto decomp_buffer_addr = ptr[reg_decomp_buffer + decomp_buffer_offset];
                switch (jcp_.decomp_buffer_dt) {
                    case data_type::f32: {
                        uni_vmovups(decomp_buffer_addr, vmm_weights(0));
                        break;
                    }
                    case data_type::bf16: {
                        Ymm ymm_weights = Ymm(vmm_weights(0).getIdx());
                        vcvtneps2bf16(ymm_weights, vmm_weights(0));
                        vmovdqu16(decomp_buffer_addr, ymm_weights);
                        break;
                    }
                    default: assert(!"unsupported data type");
                }
            }
        }

        dec(reg_ic_size);
        add(reg_weights, weights_dt_size * jcp_.oc_size * jcp_.ic_internal_size / typesize_scale);
        add(reg_decomp_buffer, decomp_buf_dt_size * jcp_.oc_size * jcp_.ic_internal_size);

        jmp(ic_loop_label, T_NEAR);
    }
    L(ic_end_label);

    postamble();
}

template struct jit_brgemm_weights_decompression_kernel_t<avx512_core>;
template struct jit_brgemm_weights_decompression_kernel_t<avx2>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl