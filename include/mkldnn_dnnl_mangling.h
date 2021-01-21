/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

// Mangle mkldnn entities to dnnl ones to preserve source-code level backwards
// compatibility. The compatibility will be dropped in oneDNN v2.0.
// Please switch to the new names as soon as possible.

#ifndef MKLDNN_DNNL_MANGLING_H
#define MKLDNN_DNNL_MANGLING_H

#define MKLDNN_API DNNL_API
#define MKLDNN_ARG_BIAS DNNL_ARG_BIAS
#define MKLDNN_ARG_DIFF_BIAS DNNL_ARG_DIFF_BIAS
#define MKLDNN_ARG_DIFF_DST DNNL_ARG_DIFF_DST
#define MKLDNN_ARG_DIFF_DST_0 DNNL_ARG_DIFF_DST_0
#define MKLDNN_ARG_DIFF_DST_1 DNNL_ARG_DIFF_DST_1
#define MKLDNN_ARG_DIFF_DST_2 DNNL_ARG_DIFF_DST_2
#define MKLDNN_ARG_DIFF_DST_ITER DNNL_ARG_DIFF_DST_ITER
#define MKLDNN_ARG_DIFF_DST_ITER_C DNNL_ARG_DIFF_DST_ITER_C
#define MKLDNN_ARG_DIFF_DST_LAYER DNNL_ARG_DIFF_DST_LAYER
#define MKLDNN_ARG_DIFF_SCALE_SHIFT DNNL_ARG_DIFF_SCALE_SHIFT
#define MKLDNN_ARG_DIFF_SRC DNNL_ARG_DIFF_SRC
#define MKLDNN_ARG_DIFF_SRC_0 DNNL_ARG_DIFF_SRC_0
#define MKLDNN_ARG_DIFF_SRC_1 DNNL_ARG_DIFF_SRC_1
#define MKLDNN_ARG_DIFF_SRC_2 DNNL_ARG_DIFF_SRC_2
#define MKLDNN_ARG_DIFF_SRC_ITER DNNL_ARG_DIFF_SRC_ITER
#define MKLDNN_ARG_DIFF_SRC_ITER_C DNNL_ARG_DIFF_SRC_ITER_C
#define MKLDNN_ARG_DIFF_SRC_LAYER DNNL_ARG_DIFF_SRC_LAYER
#define MKLDNN_ARG_DIFF_WEIGHTS DNNL_ARG_DIFF_WEIGHTS
#define MKLDNN_ARG_DIFF_WEIGHTS_0 DNNL_ARG_DIFF_WEIGHTS_0
#define MKLDNN_ARG_DIFF_WEIGHTS_1 DNNL_ARG_DIFF_WEIGHTS_1
#define MKLDNN_ARG_DIFF_WEIGHTS_ITER DNNL_ARG_DIFF_WEIGHTS_ITER
#define MKLDNN_ARG_DIFF_WEIGHTS_LAYER DNNL_ARG_DIFF_WEIGHTS_LAYER
#define MKLDNN_ARG_DST DNNL_ARG_DST
#define MKLDNN_ARG_DST_0 DNNL_ARG_DST_0
#define MKLDNN_ARG_DST_1 DNNL_ARG_DST_1
#define MKLDNN_ARG_DST_2 DNNL_ARG_DST_2
#define MKLDNN_ARG_DST_ITER DNNL_ARG_DST_ITER
#define MKLDNN_ARG_DST_ITER_C DNNL_ARG_DST_ITER_C
#define MKLDNN_ARG_DST_LAYER DNNL_ARG_DST_LAYER
#define MKLDNN_ARG_FROM DNNL_ARG_FROM
#define MKLDNN_ARG_MEAN DNNL_ARG_MEAN
#define MKLDNN_ARG_MULTIPLE_DST DNNL_ARG_MULTIPLE_DST
#define MKLDNN_ARG_MULTIPLE_SRC DNNL_ARG_MULTIPLE_SRC
#define MKLDNN_ARG_SCALE_SHIFT DNNL_ARG_SCALE_SHIFT
#define MKLDNN_ARG_SCRATCHPAD DNNL_ARG_SCRATCHPAD
#define MKLDNN_ARG_SRC DNNL_ARG_SRC
#define MKLDNN_ARG_SRC_0 DNNL_ARG_SRC_0
#define MKLDNN_ARG_SRC_1 DNNL_ARG_SRC_1
#define MKLDNN_ARG_SRC_2 DNNL_ARG_SRC_2
#define MKLDNN_ARG_SRC_ITER DNNL_ARG_SRC_ITER
#define MKLDNN_ARG_SRC_ITER_C DNNL_ARG_SRC_ITER_C
#define MKLDNN_ARG_SRC_LAYER DNNL_ARG_SRC_LAYER
#define MKLDNN_ARG_TO DNNL_ARG_TO
#define MKLDNN_ARG_VARIANCE DNNL_ARG_VARIANCE
#define MKLDNN_ARG_WEIGHTS DNNL_ARG_WEIGHTS
#define MKLDNN_ARG_WEIGHTS_0 DNNL_ARG_WEIGHTS_0
#define MKLDNN_ARG_WEIGHTS_1 DNNL_ARG_WEIGHTS_1
#define MKLDNN_ARG_WEIGHTS_ITER DNNL_ARG_WEIGHTS_ITER
#define MKLDNN_ARG_WEIGHTS_LAYER DNNL_ARG_WEIGHTS_LAYER
#define MKLDNN_ARG_WORKSPACE DNNL_ARG_WORKSPACE
#define MKLDNN_CPU_RUNTIME DNNL_CPU_RUNTIME
#define MKLDNN_DEFINE_BITMASK_OPS DNNL_DEFINE_BITMASK_OPS
#define MKLDNN_GPU_RUNTIME DNNL_GPU_RUNTIME
#define MKLDNN_JIT_DUMP DNNL_JIT_DUMP
#define MKLDNN_MAX_NDIMS DNNL_MAX_NDIMS
#define MKLDNN_MEMORY_ALLOCATE DNNL_MEMORY_ALLOCATE
#define MKLDNN_MEMORY_NONE DNNL_MEMORY_NONE
#define MKLDNN_RNN_MAX_N_PARTS DNNL_RNN_MAX_N_PARTS
#define MKLDNN_RUNTIME_NONE DNNL_RUNTIME_NONE
#define MKLDNN_RUNTIME_OCL DNNL_RUNTIME_OCL
#define MKLDNN_RUNTIME_OMP DNNL_RUNTIME_OMP
#define MKLDNN_RUNTIME_SEQ DNNL_RUNTIME_SEQ
#define MKLDNN_RUNTIME_TBB DNNL_RUNTIME_TBB
#define MKLDNN_RUNTIME_SYCL DNNL_RUNTIME_SYCL
#define MKLDNN_WITH_SYCL DNNL_WITH_SYCL
#define MKLDNN_VERBOSE DNNL_VERBOSE
#define MKLDNN_VERSION_HASH DNNL_VERSION_HASH
#define MKLDNN_VERSION_MAJOR DNNL_VERSION_MAJOR
#define MKLDNN_VERSION_MINOR DNNL_VERSION_MINOR
#define MKLDNN_VERSION_PATCH DNNL_VERSION_PATCH
#define const_mkldnn_engine_t const_dnnl_engine_t
#define const_mkldnn_memory_t const_dnnl_memory_t
#define const_mkldnn_op_desc_t const_dnnl_op_desc_t
#define const_mkldnn_post_ops_t const_dnnl_post_ops_t
#define const_mkldnn_primitive_attr_t const_dnnl_primitive_attr_t
#define const_mkldnn_primitive_desc_iterator_t \
    const_dnnl_primitive_desc_iterator_t
#define const_mkldnn_primitive_desc_t const_dnnl_primitive_desc_t
#define const_mkldnn_primitive_t const_dnnl_primitive_t
#define const_mkldnn_stream_t const_dnnl_stream_t
#define mkldnn dnnl
#define mkldnn_ dnnl_
#define mkldnn_ABc16a16b dnnl_ABc16a16b
#define mkldnn_ABc4a4b dnnl_ABc4a4b
#define mkldnn_ABc16b16a dnnl_ABc16b16a
#define mkldnn_ABc4b16a4b dnnl_ABc4b16a4b
#define mkldnn_ABc4b4a dnnl_ABc4b4a
#define mkldnn_ABc8a16b2a dnnl_ABc8a16b2a
#define mkldnn_ABc8a8b dnnl_ABc8a8b
#define mkldnn_ABc8b16a2b dnnl_ABc8b16a2b
#define mkldnn_ABc8b8a dnnl_ABc8b8a
#define mkldnn_ABcd16a16b dnnl_ABcd16a16b
#define mkldnn_ABcd16b16a dnnl_ABcd16b16a
#define mkldnn_ABcd2a8b8a2b dnnl_ABcd2a8b8a2b
#define mkldnn_ABcd32a32b dnnl_ABcd32a32b
#define mkldnn_ABcd4a8b8a4b dnnl_ABcd4a8b8a4b
#define mkldnn_ABcd4b16a4b dnnl_ABcd4b16a4b
#define mkldnn_OIhw16i16o4i dnnl_ABcd16b16a4b
#define mkldnn_OIhw16i16o2i dnnl_ABcd16b16a2b
#define mkldnn_ABcd4b4a dnnl_ABcd4b4a
#define mkldnn_ABcd4a4b dnnl_ABcd4a4b
#define mkldnn_ABcd8a16b2a dnnl_ABcd8a16b2a
#define mkldnn_ABcd8a8b dnnl_ABcd8a8b
#define mkldnn_ABcd8a32b dnnl_ABcd8a32b
#define mkldnn_ABcd16a32b dnnl_ABcd16a32b
#define mkldnn_ABcd8b16a2b dnnl_ABcd8b16a2b
#define mkldnn_ABcd8b8a dnnl_ABcd8b8a
#define mkldnn_ABcde16a16b dnnl_ABcde16a16b
#define mkldnn_ABcde16b16a dnnl_ABcde16b16a
#define mkldnn_ABcde4b4a dnnl_ABcde4b4a
#define mkldnn_ABcde4a4b dnnl_ABcde4a4b
#define mkldnn_ABcde8a16b2a dnnl_ABcde8a16b2a
#define mkldnn_ABcde8a8b dnnl_ABcde8a8b
#define mkldnn_ABcde8b16a2b dnnl_ABcde8b16a2b
#define mkldnn_ABcde4b16a4b dnnl_ABcde4b16a4b
#define mkldnn_ABcde8b8a dnnl_ABcde8b8a
#define mkldnn_Abc16a dnnl_Abc16a
#define mkldnn_Abc4a dnnl_Abc4a
#define mkldnn_Abcd16a dnnl_Abcd16a
#define mkldnn_Abcd4a dnnl_Abcd4a
#define mkldnn_Abcde16a dnnl_Abcde16a
#define mkldnn_Abcde4a dnnl_Abcde4a
#define mkldnn_Abcde8a dnnl_Abcde8a
#define mkldnn_Abcdef4a dnnl_Abcdef4a
#define mkldnn_Abcdef8a dnnl_Abcdef8a
#define mkldnn_Abcdef16a dnnl_Abcdef16a
#define mkldnn_Acb16a dnnl_Acb16a
#define mkldnn_Acb4a dnnl_Acb4a
#define mkldnn_Acb8a dnnl_Acb8a
#define mkldnn_Acdb16a dnnl_Acdb16a
#define mkldnn_Acdb32a dnnl_Acdb32a
#define mkldnn_Acdb4a dnnl_Acdb4a
#define mkldnn_Acdb8a dnnl_Acdb8a
#define mkldnn_Acdeb16a dnnl_Acdeb16a
#define mkldnn_Acdeb4a dnnl_Acdeb4a
#define mkldnn_Acdeb8a dnnl_Acdeb8a
#define mkldnn_BAc16a16b dnnl_BAc16a16b
#define mkldnn_BAc16b16a dnnl_BAc16b16a
#define mkldnn_BAc8a16b2a dnnl_BAc8a16b2a
#define mkldnn_BAcd16a16b dnnl_BAcd16a16b
#define mkldnn_BAcd16b16a dnnl_BAcd16b16a
#define mkldnn_BAcd8a16b2a dnnl_BAcd8a16b2a
#define mkldnn_BAcde16b16a dnnl_BAcde16b16a
#define mkldnn_BAcde16a16b dnnl_BAcde16a16b
#define mkldnn_BAcde8a16b2a dnnl_BAcde8a16b2a
#define mkldnn_Goidhw4g dnnl_Goidhw4g
#define mkldnn_Goidhw8g dnnl_Goidhw8g
#define mkldnn_Goidhw16g dnnl_Goidhw16g
#define mkldnn_Goihw16g dnnl_Goihw16g
#define mkldnn_Goihw8g dnnl_Goihw8g
#define mkldnn_Goiw16g dnnl_Goiw16g
#define mkldnn_IOdhw16i16o dnnl_IOdhw16i16o
#define mkldnn_IOdhw16o16i dnnl_IOdhw16o16i
#define mkldnn_IOdhw8o16i2o dnnl_IOdhw8o16i2o
#define mkldnn_IOhw16i16o dnnl_IOhw16i16o
#define mkldnn_IOhw16o16i dnnl_IOhw16o16i
#define mkldnn_IOhw8o16i2o dnnl_IOhw8o16i2o
#define mkldnn_IOw16i16o dnnl_IOw16i16o
#define mkldnn_IOw16o16i dnnl_IOw16o16i
#define mkldnn_IOw8o16i2o dnnl_IOw8o16i2o
#define mkldnn_NCdhw16n16c dnnl_NCdhw16n16c
#define mkldnn_NChw16n16c dnnl_NChw16n16c
#define mkldnn_NChw32n32c dnnl_NChw32n32c
#define mkldnn_NCw16n16c dnnl_NCw16n16c
#define mkldnn_OIdhw16i16o dnnl_OIdhw16i16o
#define mkldnn_OIdhw16o16i dnnl_OIdhw16o16i
#define mkldnn_OIdhw4i4o dnnl_OIdhw4i4o
#define mkldnn_OIdhw4o4i dnnl_OIdhw4o4i
#define mkldnn_OIdhw8i16o2i dnnl_OIdhw8i16o2i
#define mkldnn_OIdhw4i16o4i dnnl_OIdhw4i16o4i
#define mkldnn_OIdhw8i8o dnnl_OIdhw8i8o
#define mkldnn_OIdhw8o16i2o dnnl_OIdhw8o16i2o
#define mkldnn_OIdhw8o8i dnnl_OIdhw8o8i
#define mkldnn_OIhw16i16o dnnl_OIhw16i16o
#define mkldnn_OIhw16o16i dnnl_OIhw16o16i
#define mkldnn_OIhw2o8i8o2i dnnl_OIhw2o8i8o2i
#define mkldnn_OIhw4i16o4i dnnl_OIhw4i16o4i
#define mkldnn_OIhw4i4o dnnl_OIhw4i4o
#define mkldnn_OIhw4o4i dnnl_OIhw4o4i
#define mkldnn_OIhw4o8i8o4i dnnl_OIhw4o8i8o4i
#define mkldnn_OIhw8i16o2i dnnl_OIhw8i16o2i
#define mkldnn_OIhw8i8o dnnl_OIhw8i8o
#define mkldnn_OIhw8o16i2o dnnl_OIhw8o16i2o
#define mkldnn_OIhw8o8i dnnl_OIhw8o8i
#define mkldnn_OIhw8o32i dnnl_OIhw8o32i
#define mkldnn_OIhw16o32i dnnl_OIhw16o32i
#define mkldnn_OIw16i16o dnnl_OIw16i16o
#define mkldnn_OIw16o16i dnnl_OIw16o16i
#define mkldnn_OIw4i16o4i dnnl_OIw4i16o4i
#define mkldnn_OIw4i4o dnnl_OIw4i4o
#define mkldnn_OIw4o4i dnnl_OIw4o4i
#define mkldnn_OIw8i16o2i dnnl_OIw8i16o2i
#define mkldnn_OIw8i8o dnnl_OIw8i8o
#define mkldnn_OIw8o16i2o dnnl_OIw8o16i2o
#define mkldnn_OIw8o8i dnnl_OIw8o8i
#define mkldnn_Odhwi16o dnnl_Odhwi16o
#define mkldnn_Odhwi4o dnnl_Odhwi4o
#define mkldnn_Odhwi8o dnnl_Odhwi8o
#define mkldnn_Ohwi16o dnnl_Ohwi16o
#define mkldnn_Ohwi32o dnnl_Ohwi32o
#define mkldnn_Ohwi4o dnnl_Ohwi4o
#define mkldnn_Ohwi8o dnnl_Ohwi8o
#define mkldnn_Oidhw16o dnnl_Oidhw16o
#define mkldnn_Oidhw4o dnnl_Oidhw4o
#define mkldnn_Oihw16o dnnl_Oihw16o
#define mkldnn_Oihw4o dnnl_Oihw4o
#define mkldnn_Oiw16o dnnl_Oiw16o
#define mkldnn_Oiw4o dnnl_Oiw4o
#define mkldnn_Owi16o dnnl_Owi16o
#define mkldnn_Owi4o dnnl_Owi4o
#define mkldnn_Owi8o dnnl_Owi8o
#define mkldnn_a dnnl_a
#define mkldnn_aBCd16b16c dnnl_aBCd16b16c
#define mkldnn_aBCd16c16b dnnl_aBCd16c16b
#define mkldnn_aBCd4c16b4c dnnl_aBCd4c16b4c
#define mkldnn_aBCd4c4b dnnl_aBCd4c4b
#define mkldnn_aBCd4b4c dnnl_aBCd4b4c
#define mkldnn_aBCd8b16c2b dnnl_aBCd8b16c2b
#define mkldnn_aBCd8b8c dnnl_aBCd8b8c
#define mkldnn_aBCd8c16b2c dnnl_aBCd8c16b2c
#define mkldnn_aBCd8c8b dnnl_aBCd8c8b
#define mkldnn_aBCde16b16c dnnl_aBCde16b16c
#define mkldnn_aBCde16c16b dnnl_aBCde16c16b
#define mkldnn_aBCde2b8c8b2c dnnl_aBCde2b8c8b2c
#define mkldnn_aBCde2c8b4c dnnl_aBCde2c8b4c
#define mkldnn_gOIhw16i16o4i = dnnl_aBCde16c16b4c
#define mkldnn_gOIhw16i16o2i = dnnl_aBCde16c16b2c
#define mkldnn_aBCde4b4c dnnl_aBCde4b4c
#define mkldnn_aBCde4b8c8b4c dnnl_aBCde4b8c8b4c
#define mkldnn_aBCde4c16b4c dnnl_aBCde4c16b4c
#define mkldnn_aBCde4c4b dnnl_aBCde4c4b
#define mkldnn_aBCde8b16c2b dnnl_aBCde8b16c2b
#define mkldnn_aBCde8b8c dnnl_aBCde8b8c
#define mkldnn_aBCde8c16b2c dnnl_aBCde8c16b2c
#define mkldnn_aBCde8c8b dnnl_aBCde8c8b
#define mkldnn_aBCdef16b16c dnnl_aBCdef16b16c
#define mkldnn_aBCdef16c16b dnnl_aBCdef16c16b
#define mkldnn_aBCdef4c4b dnnl_aBCdef4c4b
#define mkldnn_aBCdef4b4c dnnl_aBCdef4b4c
#define mkldnn_aBCdef8b16c2b dnnl_aBCdef8b16c2b
#define mkldnn_aBCdef8b8c dnnl_aBCdef8b8c
#define mkldnn_aBCdef8c16b2c dnnl_aBCdef8c16b2c
#define mkldnn_aBCdef4c16b4c dnnl_aBCdef4c16b4c
#define mkldnn_aBCdef8c8b dnnl_aBCdef8c8b
#define mkldnn_aBc16b dnnl_aBc16b
#define mkldnn_aBc4b dnnl_aBc4b
#define mkldnn_aBc8b dnnl_aBc8b
#define mkldnn_aBcd16b dnnl_aBcd16b
#define mkldnn_aBcd4b dnnl_aBcd4b
#define mkldnn_aBcd8b dnnl_aBcd8b
#define mkldnn_aBcde16b dnnl_aBcde16b
#define mkldnn_aBcde4b dnnl_aBcde4b
#define mkldnn_aBcde8b dnnl_aBcde8b
#define mkldnn_aBcdef16b dnnl_aBcdef16b
#define mkldnn_aBcdef4b dnnl_aBcdef4b
#define mkldnn_aBdc16b dnnl_aBdc16b
#define mkldnn_aBdc4b dnnl_aBdc4b
#define mkldnn_aBdc8b dnnl_aBdc8b
#define mkldnn_aBdec16b dnnl_aBdec16b
#define mkldnn_aBdec32b dnnl_aBdec32b
#define mkldnn_aBdec4b dnnl_aBdec4b
#define mkldnn_aBdec8b dnnl_aBdec8b
#define mkldnn_aBdefc16b dnnl_aBdefc16b
#define mkldnn_aBdefc4b dnnl_aBdefc4b
#define mkldnn_aBdefc8b dnnl_aBdefc8b
#define mkldnn_aCBd16b16c dnnl_aCBd16b16c
#define mkldnn_aCBd16c16b dnnl_aCBd16c16b
#define mkldnn_aCBd8b16c2b dnnl_aCBd8b16c2b
#define mkldnn_aCBde16b16c dnnl_aCBde16b16c
#define mkldnn_aCBde16c16b dnnl_aCBde16c16b
#define mkldnn_aCBde8b16c2b dnnl_aCBde8b16c2b
#define mkldnn_aCBdef16c16b dnnl_aCBdef16c16b
#define mkldnn_aCBdef16b16c dnnl_aCBdef16b16c
#define mkldnn_aCBdef8b16c2b dnnl_aCBdef8b16c2b
#define mkldnn_ab dnnl_ab
#define mkldnn_abc dnnl_abc
#define mkldnn_abcd dnnl_abcd
#define mkldnn_abcde dnnl_abcde
#define mkldnn_abcdef dnnl_abcdef
#define mkldnn_abdec dnnl_abdec
#define mkldnn_acb dnnl_acb
#define mkldnn_acbde dnnl_acbde
#define mkldnn_acdb dnnl_acdb
#define mkldnn_acdeb dnnl_acdeb
#define mkldnn_alg_kind2str dnnl_alg_kind2str
#define mkldnn_alg_kind_t dnnl_alg_kind_t
#define mkldnn_alg_kind_undef dnnl_alg_kind_undef
#define mkldnn_any_engine dnnl_any_engine
#define mkldnn_ba dnnl_ba
#define mkldnn_bac dnnl_bac
#define mkldnn_bacd dnnl_bacd
#define mkldnn_backward dnnl_backward
#define mkldnn_backward_bias dnnl_backward_bias
#define mkldnn_backward_data dnnl_backward_data
#define mkldnn_backward_weights dnnl_backward_weights
#define mkldnn_batch_normalization dnnl_batch_normalization
#define mkldnn_batch_normalization_backward_desc_init \
    dnnl_batch_normalization_backward_desc_init
#define mkldnn_batch_normalization_desc_t dnnl_batch_normalization_desc_t
#define mkldnn_batch_normalization_forward_desc_init \
    dnnl_batch_normalization_forward_desc_init
#define mkldnn_bca dnnl_bca
#define mkldnn_bcda dnnl_bcda
#define mkldnn_bcdea dnnl_bcdea
#define mkldnn_bf16 dnnl_bf16
#define mkldnn_bidirectional_concat dnnl_bidirectional_concat
#define mkldnn_bidirectional_sum dnnl_bidirectional_sum
#define mkldnn_blocked dnnl_blocked
#define mkldnn_blocking_desc_t dnnl_blocking_desc_t
#define mkldnn_cba dnnl_cba
#define mkldnn_cdba dnnl_cdba
#define mkldnn_cdeba dnnl_cdeba
#define mkldnn_chwn dnnl_chwn
#define mkldnn_cn dnnl_cn
#define mkldnn_concat dnnl_concat
#define mkldnn_concat_primitive_desc_create dnnl_concat_primitive_desc_create
#define mkldnn_config dnnl_config
#define mkldnn_convolution dnnl_convolution
#define mkldnn_convolution_auto dnnl_convolution_auto
#define mkldnn_convolution_backward_data_desc_init \
    dnnl_convolution_backward_data_desc_init
#define mkldnn_convolution_backward_weights_desc_init \
    dnnl_convolution_backward_weights_desc_init
#define mkldnn_convolution_desc_t dnnl_convolution_desc_t
#define mkldnn_convolution_direct dnnl_convolution_direct
#define mkldnn_convolution_compress dnnl_convolution_compress
#define mkldnn_convolution_forward_desc_init dnnl_convolution_forward_desc_init
#define mkldnn_convolution_winograd dnnl_convolution_winograd
#define mkldnn_cpu dnnl_cpu
#define mkldnn_data_type_t dnnl_data_type_t
#define mkldnn_data_type_undef dnnl_data_type_undef
#define mkldnn_decab dnnl_decab
#define mkldnn_deconvolution dnnl_deconvolution
#define mkldnn_deconvolution_backward_data_desc_init \
    dnnl_deconvolution_backward_data_desc_init
#define mkldnn_deconvolution_backward_weights_desc_init \
    dnnl_deconvolution_backward_weights_desc_init
#define mkldnn_deconvolution_desc_t dnnl_deconvolution_desc_t
#define mkldnn_deconvolution_direct dnnl_deconvolution_direct
#define mkldnn_deconvolution_forward_desc_init \
    dnnl_deconvolution_forward_desc_init
#define mkldnn_deconvolution_winograd dnnl_deconvolution_winograd
#define mkldnn_dhwio dnnl_dhwio
#define mkldnn_dilated_convolution_backward_data_desc_init \
    dnnl_dilated_convolution_backward_data_desc_init
#define mkldnn_dilated_convolution_backward_weights_desc_init \
    dnnl_dilated_convolution_backward_weights_desc_init
#define mkldnn_dilated_convolution_forward_desc_init \
    dnnl_dilated_convolution_forward_desc_init
#define mkldnn_dilated_deconvolution_backward_data_desc_init \
    dnnl_dilated_deconvolution_backward_data_desc_init
#define mkldnn_dilated_deconvolution_backward_weights_desc_init \
    dnnl_dilated_deconvolution_backward_weights_desc_init
#define mkldnn_dilated_deconvolution_forward_desc_init \
    dnnl_dilated_deconvolution_forward_desc_init
#define mkldnn_dim_t dnnl_dim_t
#define mkldnn_dims_t dnnl_dims_t
#define mkldnn_dt2str dnnl_dt2str
#define mkldnn_eltwise dnnl_eltwise
#define mkldnn_eltwise_abs dnnl_eltwise_abs
#define mkldnn_eltwise_backward_desc_init dnnl_eltwise_backward_desc_init
#define mkldnn_eltwise_bounded_relu dnnl_eltwise_bounded_relu
#define mkldnn_eltwise_desc_t dnnl_eltwise_desc_t
#define mkldnn_eltwise_elu dnnl_eltwise_elu
#define mkldnn_eltwise_exp dnnl_eltwise_exp
#define mkldnn_eltwise_forward_desc_init dnnl_eltwise_forward_desc_init
#define mkldnn_eltwise_gelu dnnl_eltwise_gelu
#define mkldnn_eltwise_linear dnnl_eltwise_linear
#define mkldnn_eltwise_logistic dnnl_eltwise_logistic
#define mkldnn_eltwise_relu dnnl_eltwise_relu
#define mkldnn_eltwise_soft_relu dnnl_eltwise_soft_relu
#define mkldnn_eltwise_sqrt dnnl_eltwise_sqrt
#define mkldnn_eltwise_square dnnl_eltwise_square
#define mkldnn_eltwise_swish dnnl_eltwise_swish
#define mkldnn_eltwise_tanh dnnl_eltwise_tanh
#define mkldnn_engine dnnl_engine
#define mkldnn_engine_create dnnl_engine_create
#define mkldnn_engine_create_ocl dnnl_ocl_interop_engine_create
#define mkldnn_engine_destroy dnnl_engine_destroy
#define mkldnn_engine_get_count dnnl_engine_get_count
#define mkldnn_engine_get_kind dnnl_engine_get_kind
#define mkldnn_engine_get_ocl_context dnnl_ocl_interop_engine_get_context
#define mkldnn_engine_get_ocl_device dnnl_ocl_interop_get_device
#define mkldnn_engine_kind2str dnnl_engine_kind2str
#define mkldnn_engine_kind_t dnnl_engine_kind_t
#define mkldnn_engine_t dnnl_engine_t
#define mkldnn_exec_arg_t dnnl_exec_arg_t
#define mkldnn_f16 dnnl_f16
#define mkldnn_f32 dnnl_f32
#define mkldnn_fmt_kind2str dnnl_fmt_kind2str
#define mkldnn_fmt_tag2str dnnl_fmt_tag2str
#define mkldnn_format_kind_any dnnl_format_kind_any
#define mkldnn_format_kind_rnn_packed dnnl_format_kind_rnn_packed
#define mkldnn_format_kind_t dnnl_format_kind_t
#define mkldnn_format_kind_undef dnnl_format_kind_undef
#define mkldnn_format_kind_wino dnnl_format_kind_wino
#define mkldnn_format_tag_any dnnl_format_tag_any
#define mkldnn_format_tag_last dnnl_format_tag_last
#define mkldnn_format_tag_t dnnl_format_tag_t
#define mkldnn_format_tag_undef dnnl_format_tag_undef
#define mkldnn_forward dnnl_forward
#define mkldnn_forward_inference dnnl_forward_inference
#define mkldnn_forward_scoring dnnl_forward_scoring
#define mkldnn_forward_training dnnl_forward_training
#define mkldnn_fuse_norm_relu dnnl_fuse_norm_relu
#define mkldnn_gIOdhw16i16o dnnl_gIOdhw16i16o
#define mkldnn_gIOdhw16o16i dnnl_gIOdhw16o16i
#define mkldnn_gIOdhw8o16i2o dnnl_gIOdhw8o16i2o
#define mkldnn_gIOhw16i16o dnnl_gIOhw16i16o
#define mkldnn_gIOhw16o16i dnnl_gIOhw16o16i
#define mkldnn_gIOhw8o16i2o dnnl_gIOhw8o16i2o
#define mkldnn_gIOw16i16o dnnl_gIOw16i16o
#define mkldnn_gIOw16o16i dnnl_gIOw16o16i
#define mkldnn_gIOw8o16i2o dnnl_gIOw8o16i2o
#define mkldnn_gOIdhw16i16o dnnl_gOIdhw16i16o
#define mkldnn_gOIdhw16o16i dnnl_gOIdhw16o16i
#define mkldnn_gOIdhw4i4o dnnl_gOIdhw4i4o
#define mkldnn_gOIdhw4o4i dnnl_gOIdhw4o4i
#define mkldnn_gOIdhw8i16o2i dnnl_gOIdhw8i16o2i
#define mkldnn_gOIdhw4i16o4i dnnl_gOIdhw4i16o4i
#define mkldnn_gOIdhw8i8o dnnl_gOIdhw8i8o
#define mkldnn_gOIdhw8o16i2o dnnl_gOIdhw8o16i2o
#define mkldnn_gOIdhw8o8i dnnl_gOIdhw8o8i
#define mkldnn_gOIhw16i16o dnnl_gOIhw16i16o
#define mkldnn_gOIhw16o16i dnnl_gOIhw16o16i
#define mkldnn_gOIhw2i8o4i dnnl_gOIhw2i8o4i
#define mkldnn_gOIhw2o8i8o2i dnnl_gOIhw2o8i8o2i
#define mkldnn_gOIhw4i16o4i dnnl_gOIhw4i16o4i
#define mkldnn_gOIhw4i4o dnnl_gOIhw4i4o
#define mkldnn_gOIhw4o4i dnnl_gOIhw4o4i
#define mkldnn_gOIhw4o8i8o4i dnnl_gOIhw4o8i8o4i
#define mkldnn_gOIhw8i16o2i dnnl_gOIhw8i16o2i
#define mkldnn_gOIhw8i8o dnnl_gOIhw8i8o
#define mkldnn_gOIhw8o16i2o dnnl_gOIhw8o16i2o
#define mkldnn_gOIhw8o8i dnnl_gOIhw8o8i
#define mkldnn_gOIw16i16o dnnl_gOIw16i16o
#define mkldnn_gOIw16o16i dnnl_gOIw16o16i
#define mkldnn_gOIw4i16o4i dnnl_gOIw4i16o4i
#define mkldnn_gOIw4i4o dnnl_gOIw4i4o
#define mkldnn_gOIw4o4i dnnl_gOIw4o4i
#define mkldnn_gOIw8i16o2i dnnl_gOIw8i16o2i
#define mkldnn_gOIw8i8o dnnl_gOIw8i8o
#define mkldnn_gOIw8o16i2o dnnl_gOIw8o16i2o
#define mkldnn_gOIw8o8i dnnl_gOIw8o8i
#define mkldnn_gOdhwi16o dnnl_gOdhwi16o
#define mkldnn_gOdhwi4o dnnl_gOdhwi4o
#define mkldnn_gOdhwi8o dnnl_gOdhwi8o
#define mkldnn_gOhwi16o dnnl_gOhwi16o
#define mkldnn_gOhwi32o dnnl_gOhwi32o
#define mkldnn_gOhwi4o dnnl_gOhwi4o
#define mkldnn_gOhwi8o dnnl_gOhwi8o
#define mkldnn_gOidhw16o dnnl_gOidhw16o
#define mkldnn_gOidhw4o dnnl_gOidhw4o
#define mkldnn_gOihw16o dnnl_gOihw16o
#define mkldnn_gOihw4o dnnl_gOihw4o
#define mkldnn_gOiw16o dnnl_gOiw16o
#define mkldnn_gOiw4o dnnl_gOiw4o
#define mkldnn_gOwi16o dnnl_gOwi16o
#define mkldnn_gOwi4o dnnl_gOwi4o
#define mkldnn_gOwi8o dnnl_gOwi8o
#define mkldnn_gemm dnnl_gemm
#define mkldnn_gemm_s8s8s32 dnnl_gemm_s8s8s32
#define mkldnn_gemm_u8s8s32 dnnl_gemm_u8s8s32
#define mkldnn_giohw dnnl_giohw
#define mkldnn_goidhw dnnl_goidhw
#define mkldnn_goihw dnnl_goihw
#define mkldnn_goiw dnnl_goiw
#define mkldnn_gpu dnnl_gpu
#define mkldnn_gru_backward_desc_init dnnl_gru_backward_desc_init
#define mkldnn_gru_forward_desc_init dnnl_gru_forward_desc_init
#define mkldnn_hwigo dnnl_hwigo
#define mkldnn_hwio dnnl_hwio
#define mkldnn_idhwo dnnl_idhwo
#define mkldnn_ihwo dnnl_ihwo
#define mkldnn_inner_product dnnl_inner_product
#define mkldnn_inner_product_backward_data_desc_init \
    dnnl_inner_product_backward_data_desc_init
#define mkldnn_inner_product_backward_weights_desc_init \
    dnnl_inner_product_backward_weights_desc_init
#define mkldnn_inner_product_desc_t dnnl_inner_product_desc_t
#define mkldnn_inner_product_forward_desc_init \
    dnnl_inner_product_forward_desc_init
#define mkldnn_invalid_arguments dnnl_invalid_arguments
#define mkldnn_io dnnl_io
#define mkldnn_iohw dnnl_iohw
#define mkldnn_iterator_ends dnnl_iterator_ends
#define mkldnn_iwo dnnl_iwo
#define mkldnn_layer_normalization dnnl_layer_normalization
#define mkldnn_layer_normalization_backward_desc_init \
    dnnl_layer_normalization_backward_desc_init
#define mkldnn_layer_normalization_desc_t dnnl_layer_normalization_desc_t
#define mkldnn_layer_normalization_forward_desc_init \
    dnnl_layer_normalization_forward_desc_init
#define mkldnn_lbr_gru dnnl_lbr_gru
#define mkldnn_lbr_gru_backward_desc_init dnnl_lbr_gru_backward_desc_init
#define mkldnn_lbr_gru_forward_desc_init dnnl_lbr_gru_forward_desc_init
#define mkldnn_ldgo dnnl_ldgo
#define mkldnn_ldgoi dnnl_ldgoi
#define mkldnn_ldgoi_p dnnl_ldgoi_p
#define mkldnn_ldigo dnnl_ldigo
#define mkldnn_ldigo_p dnnl_ldigo_p
#define mkldnn_ldnc dnnl_ldnc
#define mkldnn_lrn dnnl_lrn
#define mkldnn_lrn_across_channels dnnl_lrn_across_channels
#define mkldnn_lrn_backward_desc_init dnnl_lrn_backward_desc_init
#define mkldnn_lrn_desc_t dnnl_lrn_desc_t
#define mkldnn_lrn_forward_desc_init dnnl_lrn_forward_desc_init
#define mkldnn_lrn_within_channel dnnl_lrn_within_channel
#define mkldnn_lstm_backward_desc_init dnnl_lstm_backward_desc_init
#define mkldnn_lstm_forward_desc_init dnnl_lstm_forward_desc_init
#define mkldnn_md2dim_str dnnl_md2dim_str
#define mkldnn_md2fmt_str dnnl_md2fmt_str
#define mkldnn_memory dnnl_memory
#define mkldnn_memory_create dnnl_memory_create
#define mkldnn_memory_desc_equal dnnl_memory_desc_equal
#define mkldnn_memory_desc_get_size dnnl_memory_desc_get_size
#define mkldnn_memory_desc_init_by_strides dnnl_memory_desc_init_by_strides
#define mkldnn_memory_desc_init_by_tag dnnl_memory_desc_init_by_tag
#define mkldnn_memory_desc_init_submemory dnnl_memory_desc_init_submemory
#define mkldnn_memory_desc_t dnnl_memory_desc_t
#define mkldnn_memory_destroy dnnl_memory_destroy
#define mkldnn_memory_extra_desc_t dnnl_memory_extra_desc_t
#define mkldnn_memory_extra_flag_compensation_conv_s8s8 \
    dnnl_memory_extra_flag_compensation_conv_s8s8
#define mkldnn_memory_extra_flag_none dnnl_memory_extra_flag_none
#define mkldnn_memory_extra_flag_scale_adjust \
    dnnl_memory_extra_flag_scale_adjust
#define mkldnn_memory_extra_flags_t dnnl_memory_extra_flags_t
#define mkldnn_memory_get_data_handle dnnl_memory_get_data_handle
#define mkldnn_memory_get_engine dnnl_memory_get_engine
#define mkldnn_memory_get_memory_desc dnnl_memory_get_memory_desc
#define mkldnn_memory_get_ocl_mem_object dnnl_ocl_interop_memory_get_mem_object
#define mkldnn_memory_map_data dnnl_memory_map_data
#define mkldnn_memory_set_data_handle dnnl_memory_set_data_handle
#define mkldnn_memory_set_ocl_mem_object dnnl_ocl_interop_memory_set_mem_object
#define mkldnn_memory_t dnnl_memory_t
#define mkldnn_memory_unmap_data dnnl_memory_unmap_data
#define mkldnn_nCdhw16c dnnl_nCdhw16c
#define mkldnn_nCdhw4c dnnl_nCdhw4c
#define mkldnn_nCdhw8c dnnl_nCdhw8c
#define mkldnn_nChw16c dnnl_nChw16c
#define mkldnn_nChw4c dnnl_nChw4c
#define mkldnn_nChw8c dnnl_nChw8c
#define mkldnn_nCw16c dnnl_nCw16c
#define mkldnn_nCw4c dnnl_nCw4c
#define mkldnn_nCw8c dnnl_nCw8c
#define mkldnn_nc dnnl_nc
#define mkldnn_ncdhw dnnl_ncdhw
#define mkldnn_nchw dnnl_nchw
#define mkldnn_ncw dnnl_ncw
#define mkldnn_ndhwc dnnl_ndhwc
#define mkldnn_nhwc dnnl_nhwc
#define mkldnn_normalization_flags2str dnnl_normalization_flags2str
#define mkldnn_normalization_flags_t dnnl_normalization_flags_t
#define mkldnn_not_required dnnl_not_required
#define mkldnn_nt dnnl_nt
#define mkldnn_ntc dnnl_ntc
#define mkldnn_nwc dnnl_nwc
#define mkldnn_odhwi dnnl_odhwi
#define mkldnn_ohwi dnnl_ohwi
#define mkldnn_oi dnnl_oi
#define mkldnn_oidhw dnnl_oidhw
#define mkldnn_oihw dnnl_oihw
#define mkldnn_oiw dnnl_oiw
#define mkldnn_op_desc_t dnnl_op_desc_t
#define mkldnn_out_of_memory dnnl_out_of_memory
#define mkldnn_owi dnnl_owi
#define mkldnn_packed_format_undef dnnl_packed_format_undef
#define mkldnn_pooling dnnl_pooling
#define mkldnn_pooling_avg dnnl_pooling_avg
#define mkldnn_pooling_avg_exclude_padding dnnl_pooling_avg_exclude_padding
#define mkldnn_pooling_avg_include_padding dnnl_pooling_avg_include_padding
#define mkldnn_pooling_backward_desc_init dnnl_pooling_backward_desc_init
#define mkldnn_pooling_desc_t dnnl_pooling_desc_t
#define mkldnn_pooling_forward_desc_init dnnl_pooling_forward_desc_init
#define mkldnn_pooling_max dnnl_pooling_max
#define mkldnn_post_ops dnnl_post_ops
#define mkldnn_post_ops_append_eltwise dnnl_post_ops_append_eltwise
#define mkldnn_post_ops_append_sum dnnl_post_ops_append_sum
#define mkldnn_post_ops_create dnnl_post_ops_create
#define mkldnn_post_ops_destroy dnnl_post_ops_destroy
#define mkldnn_post_ops_get_kind dnnl_post_ops_get_kind
#define mkldnn_post_ops_get_params_eltwise dnnl_post_ops_get_params_eltwise
#define mkldnn_post_ops_get_params_sum dnnl_post_ops_get_params_sum
#define mkldnn_post_ops_len dnnl_post_ops_len
#define mkldnn_post_ops_t dnnl_post_ops_t
#define mkldnn_prim_kind2str dnnl_prim_kind2str
#define mkldnn_primitive dnnl_primitive
#define mkldnn_primitive_attr dnnl_primitive_attr
#define mkldnn_primitive_attr_clone dnnl_primitive_attr_clone
#define mkldnn_primitive_attr_create dnnl_primitive_attr_create
#define mkldnn_primitive_attr_destroy dnnl_primitive_attr_destroy
#define mkldnn_primitive_attr_get_output_scales \
    dnnl_primitive_attr_get_output_scales
#define mkldnn_primitive_attr_get_post_ops dnnl_primitive_attr_get_post_ops
#define mkldnn_primitive_attr_get_scratchpad_mode \
    dnnl_primitive_attr_get_scratchpad_mode
#define mkldnn_primitive_attr_set_output_scales \
    dnnl_primitive_attr_set_output_scales
#define mkldnn_primitive_attr_set_post_ops dnnl_primitive_attr_set_post_ops
#define mkldnn_primitive_attr_set_rnn_data_qparams \
    dnnl_primitive_attr_set_rnn_data_qparams
#define mkldnn_primitive_attr_set_rnn_weights_qparams \
    dnnl_primitive_attr_set_rnn_weights_qparams
#define mkldnn_primitive_attr_set_scratchpad_mode \
    dnnl_primitive_attr_set_scratchpad_mode
#define mkldnn_primitive_attr_t dnnl_primitive_attr_t
#define mkldnn_primitive_create dnnl_primitive_create
#define mkldnn_primitive_desc dnnl_primitive_desc
#define mkldnn_primitive_desc_clone dnnl_primitive_desc_clone
#define mkldnn_primitive_desc_create dnnl_primitive_desc_create
#define mkldnn_primitive_desc_destroy dnnl_primitive_desc_destroy
#define mkldnn_primitive_desc_get_attr dnnl_primitive_desc_get_attr
#define mkldnn_primitive_desc_iterator dnnl_primitive_desc_iterator
#define mkldnn_primitive_desc_iterator_create \
    dnnl_primitive_desc_iterator_create
#define mkldnn_primitive_desc_iterator_destroy \
    dnnl_primitive_desc_iterator_destroy
#define mkldnn_primitive_desc_iterator_fetch dnnl_primitive_desc_iterator_fetch
#define mkldnn_primitive_desc_iterator_next dnnl_primitive_desc_iterator_next
#define mkldnn_primitive_desc_iterator_t dnnl_primitive_desc_iterator_t
#define mkldnn_primitive_desc_query dnnl_primitive_desc_query
#define mkldnn_primitive_desc_query_md dnnl_primitive_desc_query_md
#define mkldnn_primitive_desc_query_pd dnnl_primitive_desc_query_pd
#define mkldnn_primitive_desc_query_s32 dnnl_primitive_desc_query_s32
#define mkldnn_primitive_desc_t dnnl_primitive_desc_t
#define mkldnn_primitive_destroy dnnl_primitive_destroy
#define mkldnn_primitive_execute dnnl_primitive_execute
#define mkldnn_primitive_get_primitive_desc dnnl_primitive_get_primitive_desc
#define mkldnn_primitive_kind_t dnnl_primitive_kind_t
#define mkldnn_primitive_t dnnl_primitive_t
#define mkldnn_prop_kind2str dnnl_prop_kind2str
#define mkldnn_prop_kind_t dnnl_prop_kind_t
#define mkldnn_prop_kind_undef dnnl_prop_kind_undef
#define mkldnn_query_batch_normalization_d dnnl_query_batch_normalization_d
#define mkldnn_query_convolution_d dnnl_query_convolution_d
#define mkldnn_query_deconvolution_d dnnl_query_deconvolution_d
#define mkldnn_query_diff_dst_md dnnl_query_diff_dst_md
#define mkldnn_query_diff_src_md dnnl_query_diff_src_md
#define mkldnn_query_diff_weights_md dnnl_query_diff_weights_md
#define mkldnn_query_dst_md dnnl_query_dst_md
#define mkldnn_query_eltwise_d dnnl_query_eltwise_d
#define mkldnn_query_engine dnnl_query_engine
#define mkldnn_query_gemm_d dnnl_query_gemm_d
#define mkldnn_query_impl_info_str dnnl_query_impl_info_str
#define mkldnn_query_inner_product_d dnnl_query_inner_product_d
#define mkldnn_query_layer_normalization_d dnnl_query_layer_normalization_d
#define mkldnn_query_lrn_d dnnl_query_lrn_d
#define mkldnn_query_memory_consumption_s64 dnnl_query_memory_consumption_s64
#define mkldnn_query_num_of_inputs_s32 dnnl_query_num_of_inputs_s32
#define mkldnn_query_num_of_outputs_s32 dnnl_query_num_of_outputs_s32
#define mkldnn_query_op_d dnnl_query_op_d
#define mkldnn_query_pooling_d dnnl_query_pooling_d
#define mkldnn_query_primitive_kind dnnl_query_primitive_kind
#define mkldnn_query_rnn_d dnnl_query_rnn_d
#define mkldnn_query_scratchpad_engine dnnl_query_scratchpad_engine
#define mkldnn_query_scratchpad_md dnnl_query_scratchpad_md
#define mkldnn_query_shuffle_d dnnl_query_shuffle_d
#define mkldnn_query_softmax_d dnnl_query_softmax_d
#define mkldnn_query_some_d dnnl_query_some_d
#define mkldnn_query_some_md dnnl_query_some_md
#define mkldnn_query_src_md dnnl_query_src_md
#define mkldnn_query_t dnnl_query_t
#define mkldnn_query_time_estimate_f64 dnnl_query_time_estimate_f64
#define mkldnn_query_undef dnnl_query_undef
#define mkldnn_query_weights_md dnnl_query_weights_md
#define mkldnn_query_workspace_md dnnl_query_workspace_md
#define mkldnn_reorder dnnl_reorder
#define mkldnn_reorder_primitive_desc_create dnnl_reorder_primitive_desc_create
#define mkldnn_rnn dnnl_rnn
#define mkldnn_rnn_desc_t dnnl_rnn_desc_t
#define mkldnn_rnn_direction2str dnnl_rnn_direction2str
#define mkldnn_rnn_direction_t dnnl_rnn_direction_t
#define mkldnn_rnn_flags2str dnnl_rnn_flags2str
#define mkldnn_rnn_flags_t dnnl_rnn_flags_t
#define mkldnn_rnn_flags_undef dnnl_rnn_flags_undef
#define mkldnn_rnn_packed_desc_t dnnl_rnn_packed_desc_t
#define mkldnn_rnn_packed_memory_format_t dnnl_rnn_packed_memory_format_t
#define mkldnn_runtime_error dnnl_runtime_error
#define mkldnn_s32 dnnl_s32
#define mkldnn_s8 dnnl_s8
#define mkldnn_scratchpad_mode2str dnnl_scratchpad_mode2str
#define mkldnn_scratchpad_mode_library dnnl_scratchpad_mode_library
#define mkldnn_scratchpad_mode_t dnnl_scratchpad_mode_t
#define mkldnn_scratchpad_mode_user dnnl_scratchpad_mode_user
#define mkldnn_set_jit_dump dnnl_set_jit_dump
#define mkldnn_set_verbose dnnl_set_verbose
#define mkldnn_sgemm dnnl_sgemm
#define mkldnn_shuffle dnnl_shuffle
#define mkldnn_shuffle_backward_desc_init dnnl_shuffle_backward_desc_init
#define mkldnn_shuffle_desc_t dnnl_shuffle_desc_t
#define mkldnn_shuffle_forward_desc_init dnnl_shuffle_forward_desc_init
#define mkldnn_softmax dnnl_softmax
#define mkldnn_softmax_backward_desc_init dnnl_softmax_backward_desc_init
#define mkldnn_softmax_desc_t dnnl_softmax_desc_t
#define mkldnn_softmax_forward_desc_init dnnl_softmax_forward_desc_init
#define mkldnn_status2str dnnl_status2str
#define mkldnn_status_t dnnl_status_t
#define mkldnn_stream dnnl_stream
#define mkldnn_stream_create dnnl_stream_create
#define mkldnn_stream_create_ocl dnnl_ocl_interop_stream_create
#define mkldnn_stream_default_flags dnnl_stream_default_flags
#define mkldnn_stream_destroy dnnl_stream_destroy
#define mkldnn_stream_flags_t dnnl_stream_flags_t
#define mkldnn_stream_get_ocl_command_queue \
    dnnl_ocl_interop_stream_get_command_queue
#define mkldnn_stream_in_order dnnl_stream_in_order
#define mkldnn_stream_out_of_order dnnl_stream_out_of_order
#define mkldnn_stream_t dnnl_stream_t
#define mkldnn_stream_wait dnnl_stream_wait
#define mkldnn_success dnnl_success
#define mkldnn_sum dnnl_sum
#define mkldnn_sum_primitive_desc_create dnnl_sum_primitive_desc_create
#define mkldnn_tn dnnl_tn
#define mkldnn_tnc dnnl_tnc
#define mkldnn_types dnnl_types
#define mkldnn_u8 dnnl_u8
#define mkldnn_undefined_primitive dnnl_undefined_primitive
#define mkldnn_unidirectional dnnl_unidirectional
#define mkldnn_unidirectional_left2right dnnl_unidirectional_left2right
#define mkldnn_unidirectional_right2left dnnl_unidirectional_right2left
#define mkldnn_unimplemented dnnl_unimplemented
#define mkldnn_use_global_stats dnnl_use_global_stats
#define mkldnn_use_scaleshift dnnl_use_scaleshift
#define mkldnn_vanilla_gru dnnl_vanilla_gru
#define mkldnn_vanilla_lstm dnnl_vanilla_lstm
#define mkldnn_vanilla_rnn dnnl_vanilla_rnn
#define mkldnn_vanilla_rnn_backward_desc_init \
    dnnl_vanilla_rnn_backward_desc_init
#define mkldnn_vanilla_rnn_forward_desc_init dnnl_vanilla_rnn_forward_desc_init
#define mkldnn_version dnnl_version
#define mkldnn_version_t dnnl_version_t
#define mkldnn_wino_desc_t dnnl_wino_desc_t
#define mkldnn_wino_memory_format_t dnnl_wino_memory_format_t
#define mkldnn_wino_undef dnnl_wino_undef
#define mkldnn_wino_wei_OBaaIBOIio dnnl_wino_wei_OBaaIBOIio
#define mkldnn_wino_wei_aaOBiOo dnnl_wino_wei_aaOBiOo
#define mkldnn_wino_wei_aaOIoi dnnl_wino_wei_aaOIoi
#define mkldnn_wino_wei_aaOio dnnl_wino_wei_aaOio
#define mkldnn_wio dnnl_wio
#define mkldnn_x dnnl_x

#endif /* MKLDNN_DNNL_MANGLING_H */
