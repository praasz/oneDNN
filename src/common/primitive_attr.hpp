/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_ATTR_HPP
#define COMMON_PRIMITIVE_ATTR_HPP

#include <map>
#include <initializer_list>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

const primitive_attr_t &default_attr();

struct rnn_data_qparams_t : public c_compatible {
    rnn_data_qparams_t() : scale_(1.), shift_(0.) {}
    bool has_default_values() const { return (scale_ == 1. && shift_ == 0.); }
    bool defined() const {
        return !is_runtime_value(scale_) && !is_runtime_value(shift_);
    }

    status_t set(float scale, float shift) {
        scale_ = scale;
        shift_ = shift;
        return status::success;
    }

    bool operator==(const rnn_data_qparams_t &rhs) const {
        using namespace utils;
        return equal_with_nan(scale_, rhs.scale_)
                && equal_with_nan(shift_, rhs.shift_);
    }

    float scale_;
    float shift_;
};

struct rnn_tparams_t : public c_compatible {
    rnn_tparams_t()
        : test_mode_(false), scales_(nullptr), ngates_(0), cscale_(0.0f) {}

    ~rnn_tparams_t() {
        test_mode_ = false;
        if (scales_ != nullptr) impl::free(scales_);
        scales_ = nullptr;
        ngates_ = 0;
        cscale_ = 0.0f;
    }

    bool operator==(const rnn_tparams_t &rhs) const {
        using namespace utils;

        bool ret = test_mode_ == rhs.test_mode_ && ngates_ == rhs.ngates_
                && equal_with_nan(cscale_, rhs.cscale_);

        if (!ret) return ret;

        if (scales_) {
            if (std::memcmp(scales_, rhs.scales_, sizeof(float) * ngates_))
                return false;
        }
        return true;
    }

    bool has_default_values() const {
        return (test_mode_ == false && scales_ == nullptr && ngates_ == 0
                && cscale_ == 0.0f);
    }

    status_t set(bool mode, dim_t ngates, const float *scales, float cscale) {
        test_mode_ = mode;
        ngates_ = ngates;
        scales_ = nullptr;
        if (scales != nullptr) {
            scales_ = (float *)impl::malloc(ngates_ * sizeof(*scales_), 64);
            if (scales_ == nullptr) return status::out_of_memory;
            utils::array_copy(scales_, scales, ngates_);
        }

        cscale_ = cscale;

        return status::success;
    }

    // copy_from() functions are used for each attribute member instead of
    // operator= in order to return a status.
    // TODO: consider replacing copy_from() functions with copy-constructors and
    // std::move, since there are only a few places in the library that actually
    // use them.
    status_t copy_from(const rnn_tparams_t &other) {
        return set(
                other.test_mode_, other.ngates_, other.scales_, other.cscale_);
    }

    bool test_mode_; /* we could also use scale_ == nullptr as a test to check test_mode*/
    float *scales_;
    dim_t ngates_; /* ngates is equel to the number of scales */
    float cscale_; /* =0.0f if no c state */

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(rnn_tparams_t);
};

// Note: keep for RNN quantization
struct scales_t : public c_compatible {
    scales_t() : count_(1), mask_(0), scales_(scales_buf_) {
        set_single_scale(1.);
    }

    ~scales_t() { cleanup(); }

    bool operator==(const scales_t &rhs) const {
        bool ret = count_ == rhs.count_ && mask_ == rhs.mask_
                && !utils::any_null(scales_, rhs.scales_)
                && defined() == rhs.defined()
                && IMPLICATION(defined(),
                        !std::memcmp(
                                scales_, rhs.scales_, sizeof(float) * count_));
        return ret;
    }

    bool has_default_values() const {
        for (dim_t c = 0; c < count_; ++c) {
            if (scales_[c] != 1.) return false;
        }
        return true;
    }

    bool defined() const { return !is_runtime_value(scales_[0]); }

    void set_single_scale(float single_scale);
    status_t set(dim_t count, int mask, const float *scales);
    status_t set(float single_scale) {
        set_single_scale(single_scale);
        return status::success;
    }

    status_t copy_from(const scales_t &other) {
        return set(other.count_, other.mask_, other.scales_);
    }

    dim_t count_;
    int mask_;
    float *scales_;

private:
    enum { scales_buf_size = 16 };
    float scales_buf_[scales_buf_size];

    void cleanup() {
        if (scales_ != scales_buf_ && scales_ != nullptr) impl::free(scales_);

        count_ = 1;
        mask_ = 0;
        scales_ = scales_buf_;
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(scales_t);
};

template <typename T>
struct shifts_t: public c_compatible {
    shifts_t(): count_(1), mask_(0), shifts_(shifts_buf_)
    { set(0); }

    ~shifts_t() { cleanup(); }

    bool operator==(const shifts_t<T> &rhs) const {
        bool ret = count_ == rhs.count_ && mask_ == rhs.mask_
                   && !utils::any_null(shifts_, rhs.shifts_)
                   && defined() == rhs.defined()
                   && IMPLICATION(defined(),
                                  utils::array_cmp(shifts_, rhs.shifts_, count_));
        return ret;
    }

    bool has_default_values() const {
        for (int c = 0; c < count_; ++c) {
            if(shifts_[c] != 0) return false;
        }
        return true;
    }

    bool defined() const { return !is_runtime_value(shifts_[0]); }

    status_t set(int count, int mask, const T *zero_points);
    status_t set(T single_zero_point) { return this->set(1, 0, &single_zero_point); }

    status_t copy_from(const shifts_t &other) {
        return set(other.count_, other.mask_, other.shifts_);
    }

    dim_t count_;
    int mask_;
    T *shifts_;

private:
    enum { shifts_buf_size = 16 };
    T shifts_buf_[shifts_buf_size];

    void cleanup() {
        if (shifts_ != shifts_buf_ && shifts_ != nullptr)
            impl::free(shifts_);

        count_ = 1;
        mask_ = 0;
        shifts_ = shifts_buf_;
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(shifts_t);
};

struct runtime_scales_t : public c_compatible {
    // Clang-3.8.1 raises an error for a default initialization of a const
    // object. Const runtime_scales_t object is used as default_scales.
    // runtime_scales_t() = default;
    runtime_scales_t() {}

    status_t set(int mask) {
        mask_ = mask;
        is_set_ = true;
        return status::success;
    }

    status_t set(const dims_t dims, int ndims) {
        is_set_ = true;
        ndims_ = ndims;
        mask_ = 1;
        utils::array_copy(dims_, dims, ndims_);
        return status::success;
    }

    bool operator==(const runtime_scales_t &rhs) const {
        return mask_ == rhs.mask_ && is_set_ == rhs.is_set_ &&
               ndims_ == rhs.ndims_ && utils::array_cmp(dims_, rhs.dims_, ndims_);
    }

    bool has_default_values() const { return !is_set_; }

    bool defined() const { return has_default_values(); }

    void reset() {
        mask_ = 0;
        is_set_ = false;
    }

    // TODO: replace with `-1` to remove `is_set_`.
    // Hide `mask_` under `private:` to force interface usage.
    int mask_ = 0;
    bool is_set_ = false;

    int ndims_ = 0;
    dnnl::impl::dims_t dims_;
};

struct arg_scales_t : public c_compatible {
    arg_scales_t() = default;

    const runtime_scales_t &get(int arg) const {
        static const runtime_scales_t default_scales;
        const auto it = scales_.find(arg);
        if (it == scales_.end()) return default_scales;
        return it->second;
    }

    bool operator==(const arg_scales_t &rhs) const {
        return scales_ == rhs.scales_;
    }

    bool has_default_values(const std::vector<int> &skip_args = {}) const {
        for (const auto &s : scales_) {
            if (!s.second.has_default_values()) {
                bool skip = false;
                for (const auto &skip_a : skip_args)
                    if (s.first == skip_a) {
                        skip = true;
                        break;
                    }
                if (skip) continue;
                return false;
            }
        }
        return true;
    }

    status_t set(int arg, int mask) {
        if (!check_arg(arg)) return status::invalid_arguments;
        return scales_[arg].set(mask);
    }
    status_t set(int arg, const dims_t dims, int ndims) {
        if (!check_arg(arg)) return status::invalid_arguments;
        return scales_[arg].set(dims, ndims);
    }

    status_t get(int arg, int *mask, bool *is_set) const {
        if (!check_arg(arg)) return status::invalid_arguments;
        const auto &s = get(arg);
        if (mask) *mask = s.mask_;
        if (is_set) *is_set = s.is_set_;
        return status::success;
    }

    status_t reset(int arg) {
        if (!check_arg(arg)) return status::invalid_arguments;
        const auto it = scales_.find(arg);
        if (it != scales_.end()) scales_.erase(it);
        return status::success;
    }

    bool defined() const { return has_default_values(); }

    status_t copy_from(const arg_scales_t &other) {
        for (auto it = other.scales_.begin(); it != other.scales_.end(); ++it) {
            // Find an entry that can match the arguments without constructing a
            // new object.
            if (scales_.count(it->first) == 1) {
                auto &entry = scales_[it->first];
                bool exists = entry.mask_ == it->second.mask_;
                if (exists) continue;
            }

            CHECK(set(it->first, it->second.mask_));
        }
        return status::success;
    }

    std::map<int, runtime_scales_t> scales_;

private:
    bool check_arg(int arg) const {
        // binary
        for (const auto &sa : {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1}) {
            if (arg == sa) return true;
        }
        // concat
        if (arg & DNNL_ARG_MULTIPLE_SRC) return true;
        // convolution
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == sa) return true;
        }
        // depth-wise convolution post op
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == (DNNL_ARG_ATTR_POST_OP_DW | sa)) return true;
        }
        return false;
    }
};

struct zero_points_t : public c_compatible {
    bool operator==(const zero_points_t &rhs) const {
        return mask_src == rhs.mask_src && mask_wei == rhs.mask_wei
                && mask_dst == rhs.mask_dst && is_set_src == rhs.is_set_src
                && is_set_wei == rhs.is_set_wei && is_set_dst == rhs.is_set_dst
                && IMPLICATION(ndims_wei > 0, ndims_wei == rhs.ndims_wei && utils::array_cmp(dims_wei, rhs.dims_wei, ndims_wei))
                && data_type_wei == rhs.data_type_wei;
    }

    // arg-specific checks
    bool common(int arg) const { return get_mask(arg) == 0; }
    bool defined(int arg) const { return has_default_values(arg); }
    bool has_default_values(int arg) const { return is_set(arg) == false && has_default_data_type(arg); }
    bool has_default_data_type(int arg) const {
        return get_data_type(arg) == data_type::s32;
    }

    // same checks but for all supported arguments at once
    bool common() const { return check_all(&zero_points_t::common); }
    bool defined() const { return has_default_values(); }
    bool has_default_values() const {
        return check_all(&zero_points_t::has_default_values);
    }
    bool has_default_data_type() const {
        return check_all(&zero_points_t::has_default_data_type);
    }

    status_t get(int arg, int *mask) const;
    int get(int arg) const; // Returns 0 if dimension is unset

    status_t set(int arg, int mask);
    status_t set(int arg, const dims_t dims, int ndims, data_type_t data_type);
    status_t set(int arg) { return set(arg, 0); }

    const dims_t & get_dims(int /*arg*/) const {
        return dims_wei;
    }
    int get_ndims(int arg) const {
        switch (arg) {
            case DNNL_ARG_WEIGHTS: return ndims_wei; break;
            default: return 0;
        }
    }

    data_type_t get_data_type(int arg) const {
        if (arg == DNNL_ARG_WEIGHTS) return data_type_wei;
        return data_type::s32;
    }

private:
    bool is_set_src = false, is_set_wei = false, is_set_dst = false;
    int mask_src = 0, mask_wei = 0, mask_dst = 0;
    data_type_t data_type_wei = data_type::s32;

    int ndims_wei = 0;
    dnnl::impl::dims_t dims_wei;

    int get_mask(int arg) const {
        int mask = 0;
        switch (arg) {
            case DNNL_ARG_SRC: mask = mask_src; break;
            case DNNL_ARG_WEIGHTS: mask = mask_wei; break;
            case DNNL_ARG_DST: mask = mask_dst; break;
            default: mask = 0;
        }
        return mask;
    }

    bool is_set(int arg) const {
        bool arg_is_set = false;
        switch (arg) {
            case DNNL_ARG_SRC: arg_is_set = is_set_src; break;
            case DNNL_ARG_WEIGHTS: arg_is_set = is_set_wei; break;
            case DNNL_ARG_DST: arg_is_set = is_set_dst; break;
            default: arg_is_set = 0;
        }
        return arg_is_set;
    }

    bool check_all(bool (zero_points_t::*f)(int) const) const {
        for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST})
            if (!(this->*f)(arg)) return false;
        return true;
    }
};

struct serialization_stream_t;

struct primitive_attr_item_t {
    virtual std::unique_ptr<primitive_attr_item_t> clone() const = 0;
    virtual bool has_default_values() const = 0;
    virtual bool is_equal(const primitive_attr_item_t &other) const = 0;
    virtual size_t get_hash() const = 0;
    virtual void serialize(serialization_stream_t &stream) const = 0;
    virtual ~primitive_attr_item_t() = default;
};

struct legacy_zero_points_t : public c_compatible {
    bool operator==(const legacy_zero_points_t &rhs) const {
        return count_ == rhs.count_ && mask_ == rhs.mask_;
    }

    bool has_default_values() const {
        return count_ == 0 && mask_ == 0;
    }

    status_t set(dim_t count, int mask) {
        count_ = count;
        mask_ = mask;

        return status::success;
    }

    dim_t count_ = 0;
    int mask_ = 0;
};

struct src_dyn_quant_params_t : public c_compatible {
    src_dyn_quant_params_t() : group_size_(0) {}
    bool has_default_values() const {
        return (group_size_ == 0);
    }
    bool defined() const {
        return true;
    }

    status_t set(size_t group_size) {
        group_size_ = group_size;
        return status::success;
    }

    bool operator==(const src_dyn_quant_params_t &rhs) const {
        using namespace utils;
        return group_size_ == rhs.group_size_;
    }

    size_t group_size_;
};

} // namespace impl
} // namespace dnnl

struct dnnl_post_ops : public dnnl::impl::c_compatible {
    struct entry_t {
        entry_t() : kind(dnnl::impl::primitive_kind::undefined) {}
        entry_t(const entry_t &other) { copy_from(other); }

        void copy_from(const entry_t &other) { set(other); }

        // TODO: This operator has to be deleted, and its usage has to be
        // replaced with copy_from() or copy/move constructors in order to
        // extract a status.
        entry_t &operator=(const entry_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            set(other);
            return *this;
        }

        struct eltwise_t {
            dnnl::impl::alg_kind_t alg;
            float scale, alpha, beta;
        };

        struct depthwise_conv_t {
            dnnl::impl::dim_t kernel;
            dnnl::impl::dim_t stride;
            dnnl::impl::dim_t padding;
            dnnl::impl::data_type_t wei_dt;
            dnnl::impl::data_type_t bias_dt;
            dnnl::impl::data_type_t dst_dt;
        };

        struct binary_t {
            dnnl::impl::alg_kind_t alg;
            // This is an unmodifiable user copy of attributes which is used in
            // caching mechanism. Not to be used internally.
            dnnl::impl::memory_desc_t user_src1_desc;
            // This is a modifiable copy of memory desc. It changes format kind
            // and tag of md in case user passed format_kind::any. To be used
            // everywhere internally.
            dnnl::impl::memory_desc_t src1_desc;
        };

        struct prelu_t {
            int mask;
        };

        struct depthwise_t {
            enum depthwise_fields {
                scales,
                shifts,

                fields_count
            };

            dnnl::impl::alg_kind_t alg;
            size_t offset[fields_count];
        };

        struct quantization_t {
            enum quantization_fields {
                crop_low,
                crop_high,
                inp_scale,
                inp_shift,
                output_scale,
                output_shift,

                fields_count
            };

            dnnl::impl::alg_kind_t alg;
            bool per_channel[fields_count];
            bool all_default[fields_count];
            size_t offset[fields_count];
        };

        struct binarization_t {
            dnnl::impl::alg_kind_t alg;
            const float* weights_data;
            const float* output_mask_data;
        };

        struct depthwise_conv_old_t {
            int in_h;
            int in_w;
            int ker_h;
            int ker_w;
            int str_h;
            int str_w;
            dnnl::impl::data_type_t in_dt;
        };

        dnnl::impl::primitive_kind_t kind
                = dnnl::impl::primitive_kind::undefined;
        union {
            struct {
                float scale;
                int32_t zero_point;
                dnnl::impl::data_type_t dt;
            } sum;
            eltwise_t eltwise;
            depthwise_conv_t depthwise_conv;
            depthwise_conv_old_t depthwise_conv_old;
            binary_t binary;
            prelu_t prelu;
            depthwise_t depthwise;
            quantization_t quantization;
            binarization_t binarization;
        };

        bool is_eltwise(bool require_scale_one = false) const {
            using namespace dnnl::impl;
            return kind == primitive_kind::eltwise
                    && IMPLICATION(require_scale_one, eltwise.scale == 1.f);
        }

        bool is_relu(bool require_scale_one = true,
                bool require_nslope_zero = true) const {
            using namespace dnnl::impl;
            return is_eltwise(require_scale_one)
                    && eltwise.alg == alg_kind::eltwise_relu
                    && IMPLICATION(require_nslope_zero, eltwise.alpha == 0.f);
        }

        bool is_sum(bool require_scale_one = true,
                bool require_zp_zero = true) const {
            using namespace dnnl::impl;
            return kind == primitive_kind::sum
                    && IMPLICATION(require_scale_one, sum.scale == 1.f)
                    && IMPLICATION(require_zp_zero, sum.zero_point == 0);
        }

        bool is_convolution() const {
            using namespace dnnl::impl;
            return kind == primitive_kind::convolution;
        }

        bool is_binary() const {
            return kind == dnnl::impl::primitive_kind::binary;
        }

        bool is_prelu() const {
            return kind == dnnl::impl::primitive_kind::prelu;
        }

        bool is_like_binary() const { return is_binary() || is_prelu(); }
        bool is_depthwise() const {
            using namespace dnnl::impl;
            return kind == primitive_kind::depthwise;
        }

        bool is_quantization() const {
            using namespace dnnl::impl;
            return kind == primitive_kind::quantization;
        }

        bool is_binarization() const {
            using namespace dnnl::impl;
            return kind == primitive_kind::binarization;
        }

        dnnl::impl::status_t set_depthwise_scales(const float *scales);

        bool operator==(const entry_t &rhs) const {
            using namespace dnnl::impl;
            using namespace dnnl::impl::utils;
            if (kind != rhs.kind) { return false; }
            bool ret = true;
            switch (kind) {
                case primitive_kind::eltwise:
                    ret = eltwise.alg == rhs.eltwise.alg
                            && equal_with_nan(eltwise.scale, rhs.eltwise.scale)
                            && equal_with_nan(eltwise.alpha, rhs.eltwise.alpha)
                            && equal_with_nan(eltwise.beta, rhs.eltwise.beta);
                    break;
                case primitive_kind::sum:
                    ret = equal_with_nan(sum.scale, rhs.sum.scale)
                            && sum.zero_point == rhs.sum.zero_point
                            && sum.dt == rhs.sum.dt;
                    break;
                case primitive_kind::convolution:
                    // todo: [antonvor] uncomment when new behavior of dw convolution fusing from oneDNN 1.6 will be supported
                    // // Depthwise Only
                    // ret = depthwise_conv.kernel == rhs.depthwise_conv.kernel
                    //         && depthwise_conv.stride
                    //                 == rhs.depthwise_conv.stride
                    //         && depthwise_conv.padding
                    //                 == rhs.depthwise_conv.padding
                    //         && depthwise_conv.wei_dt
                    //                 == rhs.depthwise_conv.wei_dt
                    //         && depthwise_conv.bias_dt
                    //                 == rhs.depthwise_conv.bias_dt
                    //         && depthwise_conv.dst_dt
                    //                 == rhs.depthwise_conv.dst_dt;
                    ret = depthwise_conv_old.in_h == rhs.depthwise_conv_old.in_h
                        && depthwise_conv_old.in_w == rhs.depthwise_conv_old.in_w
                        && depthwise_conv_old.ker_h == rhs.depthwise_conv_old.ker_h
                        && depthwise_conv_old.ker_w == rhs.depthwise_conv_old.ker_w
                        && depthwise_conv_old.str_h == rhs.depthwise_conv_old.str_h
                        && depthwise_conv_old.str_w == rhs.depthwise_conv_old.str_w
                        && depthwise_conv_old.in_dt == rhs.depthwise_conv_old.in_dt;
                    break;
                case primitive_kind::binary:
                    ret = binary.alg == rhs.binary.alg
                            && binary.user_src1_desc
                                    == rhs.binary.user_src1_desc;
                    break;
                case primitive_kind::prelu:
                    ret = prelu.mask == rhs.prelu.mask;
                    break;
                case primitive_kind::depthwise:
                    ret = depthwise.alg == rhs.depthwise.alg
                          && array_cmp(depthwise.offset, rhs.depthwise.offset, depthwise.fields_count);
                    break;
                case primitive_kind::quantization:
                    ret = quantization.alg == rhs.quantization.alg
                          && array_cmp(quantization.per_channel, rhs.quantization.per_channel, quantization.fields_count)
                          && array_cmp(quantization.all_default, rhs.quantization.all_default, quantization.fields_count)
                          && array_cmp(quantization.offset, rhs.quantization.offset, quantization.fields_count);
                    break;
                case primitive_kind::binarization:
                    ret = depthwise.alg == rhs.depthwise.alg
                          && binarization.weights_data == rhs.binarization.weights_data
                          && binarization.output_mask_data == rhs.binarization.output_mask_data;
                    break;
                default: assert(!"unsupported post_op");
            }
            return ret;
        }

        bool operator!=(const entry_t &rhs) const {
            return !this->operator==(rhs);
        }

    private:
        void set(const entry_t &other) {
            // Copying by if (is_convolution()) {} else if(is_sum()) {}
            // else if(is_relu()) {} seems to be unreliable. memcpying for now.
            dnnl::impl::utils::array_copy(
                    (char *)this, (char *)&other, sizeof(*this));
        }
    };

    dnnl_post_ops() : entry_() {}
    ~dnnl_post_ops() = default;

    dnnl_post_ops(const dnnl_post_ops &other) { copy_from(other); }

    dnnl::impl::status_t append_sum(float scale, int32_t zero_point = 0,
            dnnl::impl::data_type_t dt = dnnl_data_type_undef);
    dnnl::impl::status_t append_eltwise(
            float scale, dnnl::impl::alg_kind_t alg, float alpha, float beta);
    dnnl::impl::status_t append_dw(dnnl::impl::data_type_t wei_dt,
            dnnl::impl::data_type_t bias_dt, dnnl::impl::data_type_t dst_dt,
            dnnl::impl::dim_t kernel_size, dnnl::impl::dim_t stride_size,
            dnnl::impl::dim_t padding_l_size);
    dnnl::impl::status_t append_binary(dnnl::impl::alg_kind_t alg,
            const dnnl::impl::memory_desc_t *user_src1_desc);
    dnnl::impl::status_t append_prelu(int mask);
    dnnl::impl::status_t append_depthwise(dnnl::impl::alg_kind_t alg, size_t offset_size, const size_t* offset);
    dnnl::impl::status_t append_quantization(dnnl::impl::alg_kind_t alg,
            size_t per_channel_size, const bool* per_channel,
            size_t all_default_size, const bool* all_default,
            size_t offset_size, const size_t* offset);
    dnnl::impl::status_t append_binarization(dnnl::impl::alg_kind_t alg, const float* weights_data,
            const float* output_mask_data);
    dnnl::impl::status_t append_dw_conv(int in_h, int in_w, int ker_h, int ker_w, int str_h, int str_w,
            dnnl::impl::data_type_t in_dt);

    dnnl::impl::status_t prepend_binary(dnnl::impl::alg_kind_t alg,
            const dnnl::impl::memory_desc_t *user_src1_desc);

    int find(dnnl::impl::primitive_kind_t kind, int start = 0,
            int stop = -1) const {
        if (stop == -1) stop = len();
        stop = dnnl::impl::nstl::min(stop, len());
        for (int idx = start; idx < stop; ++idx)
            if (entry_[idx].kind == kind) return idx;
        return -1;
    }

    dnnl::impl::data_type_t get_sum_dt(
            const dnnl::impl::data_type_t dst_dt, int sum_ind = -1) const {
        if (sum_ind == -1) sum_ind = find(dnnl::impl::primitive_kind::sum);
        if (sum_ind == -1) return dst_dt;
        const auto sum_dt = entry_[sum_ind].sum.dt;
        if (sum_dt != dnnl::impl::data_type::undef) return sum_dt;
        return dst_dt;
    }

    int count(dnnl::impl::primitive_kind_t kind, int start = 0,
              int stop = -1) const {
        if (stop == -1) stop = len();
        stop = dnnl::impl::nstl::min(stop, len());
        int cnt = 0;
        for (int idx = start; idx < stop; ++idx)
            if (entry_[idx].kind == kind) cnt++;
        return cnt;
    }

    bool defined() const;
    int len() const { return (int)entry_.size(); }
    bool has_default_values(
            const std::vector<dnnl::impl::primitive_kind_t> &skip_pk
            = {}) const {
        if (len() == 0) return true;

        for (const auto &e : entry_) {
            bool skip = false;
            for (const auto &pk : skip_pk)
                if (e.kind == pk) {
                    skip = true;
                    break;
                }
            if (skip) continue;
            return false;
        }
        return true;
    }

    dnnl::impl::status_t set_default_formats(
            const dnnl::impl::memory_desc_t *dst_md);

    bool check_sum_consistency(const dnnl::impl::data_type_t dst_dt,
            const bool is_int8,
            const bool diverse_sum_dt_allowed = false) const;

    bool sum_with_default_dt(
            dnnl::impl::data_type_t dst_dt = dnnl_data_type_undef) const {
        int sum_ind = find(dnnl::impl::primitive_kind::sum);
        return sum_ind == -1 || entry_[sum_ind].sum.dt == dnnl_data_type_undef
                || entry_[sum_ind].sum.dt == dst_dt;
    }

    bool contain(dnnl::impl::primitive_kind_t kind, int index) const {
        return find(kind, index, index + 1) == index;
    }

    bool operator==(const dnnl_post_ops &rhs) const {
        bool ret = len() == rhs.len();
        for (int i = 0; i < len(); ++i)
            ret = ret && entry_[i] == rhs.entry_[i];
        return ret;
    }

    void copy_from(const dnnl_post_ops &other) {
        using namespace dnnl::impl;

        for (int idx = 0; idx < other.len(); ++idx) {
            if (len() > idx) {
                if (entry_[idx] == other.entry_[idx]) continue;
            } else {
                entry_.emplace_back();
            }
            entry_[idx].copy_from(other.entry_[idx]);
        }
    }

    bool is_initialized() const { return is_initialized_; }

    std::vector<entry_t> entry_;

    // Since binary post op accepts no more than 32 memory arguments by
    // design, we limit the amount of post-ops to 32.
    static constexpr int post_ops_limit = 32;

private:
    dnnl::impl::status_t validate_binary(dnnl::impl::alg_kind_t alg,
            const dnnl::impl::memory_desc_t *user_src1_desc) const;

    bool check_sum_consistent_dt(const dnnl::impl::data_type_t dst_dt,
            const bool diverse_sum_dt_allowed = false) const;

    bool check_sum_consistent_quantization(
            const dnnl::impl::data_type_t dst_dt, const bool is_int8) const;
};

struct dnnl_primitive_attr : public dnnl::impl::c_compatible {
    dnnl_primitive_attr()
        : scratchpad_mode_(dnnl::impl::scratchpad_mode::library)
        , fpmath_mode_(dnnl::impl::get_fpmath_mode()) {}
    ~dnnl_primitive_attr() = default;

    dnnl_primitive_attr *clone() const {
        return new dnnl_primitive_attr(*this);
    }

    dnnl_primitive_attr(const dnnl_primitive_attr &other) {
        if (copy_from(other) != dnnl::impl::status::success)
            is_initialized_ = false;
    }

    dnnl::impl::status_t copy_from(const dnnl_primitive_attr &other) {
        using namespace dnnl::impl;

        output_scales_ = other.output_scales_;
        scales_ = other.scales_;
        zero_points_ = other.zero_points_;
        scratchpad_mode_ = other.scratchpad_mode_;
        fpmath_mode_ = other.fpmath_mode_;
        post_ops_.copy_from(other.post_ops_);
        rnn_data_qparams_ = other.rnn_data_qparams_;
        CHECK(rnn_weights_qparams_.copy_from(other.rnn_weights_qparams_));
        CHECK(rnn_weights_projection_qparams_.copy_from(
                other.rnn_weights_projection_qparams_));
        CHECK(rnn_tparams_.copy_from(other.rnn_tparams_));
        if (other.gpu_attr_) gpu_attr_ = other.gpu_attr_->clone();
        input_zero_points_ = (other.input_zero_points_);
        weights_zero_points_ = (other.weights_zero_points_);
        output_compensations_ = (other.output_compensations_);
        src_dyn_quant_params_ = other.src_dyn_quant_params_;

        return status::success;
    }

    bool is_initialized() const { return is_initialized_; }

    enum class skip_mask_t : unsigned {
        none = 0,
        oscale = 1u << 0,
        oscale_runtime = 1u << 1,
        scales = 1u << 2,
        scales_runtime = (unsigned)scales | (1u << 3),
        zero_points = 1u << 4,
        zero_points_runtime = (unsigned)zero_points | (1u << 5),
        post_ops = 1u << 6,
        rnn_data_qparams = 1u << 7,
        rnn_weights_qparams = 1u << 8,
        rnn_tparams = 1u << 9,
        sum_dt = 1u << 10,
        rnn_weights_projection_qparams = 1u << 11,
        gpu_attr = 1u << 12,
        input_zero_points = 1 << 13,
        weights_zero_points = 1 << 14,
        output_compensations = 1 << 15,
        src_dyn_quant_params = 1u << 16,
    };

    /** Returns true if the attributes have default values.
     *
     * @note The scratchpad_mode_ is not take into account */
    bool has_default_values(skip_mask_t mask = skip_mask_t::none,
            dnnl::impl::data_type_t dst_dt = dnnl_data_type_undef) const;

    /** Returns true if the attributes are fully defined. */
    bool defined(skip_mask_t mask = skip_mask_t::none) const;

    bool operator==(const dnnl_primitive_attr &rhs) const {
        bool ret = scratchpad_mode_ == rhs.scratchpad_mode_
                && fpmath_mode_ == rhs.fpmath_mode_
                && output_scales_ == rhs.output_scales_
                && scales_ == rhs.scales_ && zero_points_ == rhs.zero_points_
                && post_ops_ == rhs.post_ops_
                && rnn_data_qparams_ == rhs.rnn_data_qparams_
                && rnn_weights_qparams_ == rhs.rnn_weights_qparams_
                && rnn_weights_projection_qparams_
                        == rhs.rnn_weights_projection_qparams_
                && rnn_tparams_ == rhs.rnn_tparams_
                && ((gpu_attr_ && rhs.gpu_attr_
                            && gpu_attr_->is_equal(*rhs.gpu_attr_))
                        || (!gpu_attr_ && !rhs.gpu_attr_))
                && input_zero_points_ == rhs.input_zero_points_
                && weights_zero_points_ == rhs.weights_zero_points_
                && output_compensations_ == rhs.output_compensations_
                && src_dyn_quant_params_ == rhs.src_dyn_quant_params_;
        return ret;
    }

    dnnl::impl::status_t set_fpmath_mode(dnnl::impl::fpmath_mode_t fpmath_mode);
    dnnl::impl::status_t set_scratchpad_mode(
            dnnl::impl::scratchpad_mode_t scratchpad_mode);
    dnnl::impl::status_t set_post_ops(const dnnl::impl::post_ops_t &post_ops);
    dnnl::impl::status_t set_gpu_attr(
            const dnnl::impl::primitive_attr_item_t &gpu_attr);
    dnnl::impl::status_t set_default_formats(
            const dnnl::impl::memory_desc_t *dst_md);

    /* Auxiliary functions */
    bool mayidownconvert(dnnl::impl::data_type_t dt_from,
            dnnl::impl::data_type_t dt_to) const {
        using namespace dnnl::impl;

        bool is_compat = is_fpsubtype(dt_to, dt_from);
        auto can_downconvert = [&]() {
            switch (fpmath_mode_) {
                case fpmath_mode::strict: return dt_from == dt_to;
                case fpmath_mode::any: return true;
                case fpmath_mode::bf16:
                    return is_fpsubtype(data_type::bf16, dt_to);
                case fpmath_mode::f16:
                    return is_fpsubtype(data_type::f16, dt_to);
                case fpmath_mode::tf32:
                    return is_fpsubtype(data_type::tf32, dt_to);
                default: return false;
            }
        };
        return is_compat && can_downconvert();
    }

    // NOTE: make sure that the types below have overloaded comparison operator
    dnnl::impl::runtime_scales_t output_scales_;
    dnnl::impl::arg_scales_t scales_;
    dnnl::impl::zero_points_t zero_points_;
    dnnl::impl::scratchpad_mode_t scratchpad_mode_;
    dnnl::impl::fpmath_mode_t fpmath_mode_;
    dnnl::impl::post_ops_t post_ops_;
    dnnl::impl::rnn_data_qparams_t rnn_data_qparams_;
    dnnl::impl::scales_t rnn_weights_qparams_;
    dnnl::impl::scales_t rnn_weights_projection_qparams_;
    dnnl::impl::rnn_tparams_t rnn_tparams_;
    std::unique_ptr<dnnl::impl::primitive_attr_item_t> gpu_attr_;

    dnnl::impl::legacy_zero_points_t input_zero_points_;
    dnnl::impl::legacy_zero_points_t weights_zero_points_;
    dnnl::impl::legacy_zero_points_t output_compensations_;

    dnnl::impl::src_dyn_quant_params_t src_dyn_quant_params_;

    dnnl_primitive_attr &operator=(const dnnl_primitive_attr &other) = delete;
};

inline dnnl_primitive_attr::skip_mask_t operator|(
        dnnl_primitive_attr::skip_mask_t lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    return static_cast<dnnl_primitive_attr::skip_mask_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}
inline dnnl_primitive_attr::skip_mask_t operator&(
        dnnl_primitive_attr::skip_mask_t lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    return static_cast<dnnl_primitive_attr::skip_mask_t>(
            static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}
inline dnnl_primitive_attr::skip_mask_t &operator|=(
        dnnl_primitive_attr::skip_mask_t &lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    lhs = lhs | rhs;
    return lhs;
}
inline dnnl_primitive_attr::skip_mask_t &operator&=(
        dnnl_primitive_attr::skip_mask_t &lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    lhs = lhs & rhs;
    return lhs;
}
inline bool operator!=(dnnl_primitive_attr::skip_mask_t lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    return (static_cast<unsigned>(lhs) != static_cast<unsigned>(rhs));
}
inline dnnl_primitive_attr::skip_mask_t operator~(
        dnnl_primitive_attr::skip_mask_t rhs) {
    return static_cast<dnnl_primitive_attr::skip_mask_t>(
            ~static_cast<unsigned>(rhs));
}

#endif
