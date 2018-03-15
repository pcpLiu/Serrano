//
//  activation_op.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 6/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>

using namespace metal;

namespace serrano_ios {
    kernel void ReLU(constant float* in_tensor    [[ buffer(0) ]],
                     device float* out_tensor    [[ buffer(1) ]],
                     constant uint& count    [[ buffer(2) ]],
                     constant float& alpha [[buffer(3)]],
                     uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = max(alpha, in_tensor[gid.x]);
    }
    
    kernel void ReLU_grad(constant float* in_tensor   [[ buffer(0) ]],
                          device float* out_tensor    [[ buffer(1) ]],
                          constant uint& count        [[ buffer(2) ]],
                          uint2 gid                   [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] >= 0.0f ? 1.0f : 0.0f);
    }
    
    kernel void Sigmoid(constant float* in_tensor  [[ buffer(0) ]],
                        device float* out_tensor   [[ buffer(1) ]],
                        constant uint& count       [[ buffer(2) ]],
                        uint2 gid                  [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = 1.0f / (1.0f + exp(-in_tensor[gid.x]));
    }
    
    kernel void Sigmoid_grad(constant float* in_tensor    [[ buffer(0) ]],
                             device float* out_tensor     [[ buffer(1) ]],
                             constant uint& count         [[ buffer(2) ]],
                             uint2 gid                    [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = exp(-in_tensor[gid.x]) / pow((1.0f + exp(-in_tensor[gid.x])), 2.0);
    }
    
    kernel void Softplus(constant float* in_tensor    [[ buffer(0) ]],
                        device float* out_tensor    [[ buffer(1) ]],
                        constant uint& count    [[ buffer(2) ]],
                        uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = log(exp(in_tensor[gid.x]) + 1.0f);
    }
    
    kernel void Softplus_grad(constant float* in_tensor    [[ buffer(0) ]],
                             device float* out_tensor     [[ buffer(1) ]],
                             constant uint& count         [[ buffer(2) ]],
                             uint2 gid                    [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = 1.0f / (1.0f + 1.0f / exp(in_tensor[gid.x]));
    }
    
    kernel void Softsign(constant float* in_tensor    [[ buffer(0) ]],
                         device float* out_tensor    [[ buffer(1) ]],
                         constant uint& count    [[ buffer(2) ]],
                         uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = in_tensor[gid.x] / ( 1.0f + abs(in_tensor[gid.x]));
    }
    
    kernel void Softsign_grad(constant float* in_tensor    [[ buffer(0) ]],
                              device float* out_tensor     [[ buffer(1) ]],
                              constant uint& count         [[ buffer(2) ]],
                              uint2 gid                    [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = 1.0f / pow((1.0f + abs(in_tensor[gid.x])), 2.0f);
    }
    
    kernel void Linear(constant float* in_tensor    [[ buffer(0) ]],
                       device float* out_tensor    [[ buffer(1) ]],
                       constant uint& count    [[ buffer(2) ]],
                       uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = in_tensor[gid.x];
    }
    
    kernel void ELU(constant float* in_tensor    [[ buffer(0) ]],
                    device float* out_tensor    [[ buffer(1) ]],
                    constant uint& count    [[ buffer(2) ]],
                    constant float& alpha [[buffer(3)]],
                    uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] >= 0.0f ? in_tensor[gid.x] : alpha*(exp(in_tensor[gid.x]) - 1.0f));
    }
    
    kernel void ELU_grad(constant float* in_tensor    [[ buffer(0) ]],
                         device float* out_tensor     [[ buffer(1) ]],
                         constant uint& count         [[ buffer(2) ]],
                         constant float& alpha        [[ buffer(3) ]],
                         uint2 gid                    [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] >= 0.0f ? 1.0f : alpha*(exp(in_tensor[gid.x])));
    }
    
    kernel void SELU(constant float* in_tensor    [[ buffer(0) ]],
                    device float* out_tensor    [[ buffer(1) ]],
                    constant uint& count    [[ buffer(2) ]],
                    constant float& alpha [[buffer(3)]],
                    constant float& scale [[buffer(4)]],
                    uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] >= 0.0f ? scale * in_tensor[gid.x]  : scale * alpha * (exp(in_tensor[gid.x]) - 1.0f));
    }
    
    kernel void SELU_grad(constant float* in_tensor   [[ buffer(0) ]],
                         device float* out_tensor     [[ buffer(1) ]],
                         constant uint& count         [[ buffer(2) ]],
                         constant float& alpha        [[ buffer(3) ]],
                         constant float& scale        [[ buffer(4) ]],
                         uint2 gid                    [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] >= 0.0f ? scale : scale * alpha * exp(in_tensor[gid.x]));
    }
    
    kernel void LeakyReLU(constant float* in_tensor    [[ buffer(0) ]],
                          device float* out_tensor    [[ buffer(1) ]],
                          constant uint& count    [[ buffer(2) ]],
                          constant float& alpha [[buffer(3)]],
                          uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] >= 0.0f ? in_tensor[gid.x] : alpha * in_tensor[gid.x]);
    }
    
    kernel void LeakyReLU_grad(constant float* in_tensor    [[ buffer(0) ]],
                               device float* out_tensor    [[ buffer(1) ]],
                               constant uint& count    [[ buffer(2) ]],
                               constant float& alpha [[buffer(3)]],
                               uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] >= 0.0f ? 1.0f : alpha);
    }
    
    kernel void ThresholdedReLU(constant float* in_tensor    [[ buffer(0) ]],
                                device float* out_tensor    [[ buffer(1) ]],
                                constant uint& count    [[ buffer(2) ]],
                                constant float& alpha [[buffer(3)]],
                                uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] > alpha ? in_tensor[gid.x] : 0.0f);
    }
    
    kernel void ThresholdedReLU_grad(constant float* in_tensor    [[ buffer(0) ]],
                                     device float* out_tensor    [[ buffer(1) ]],
                                     constant uint& count    [[ buffer(2) ]],
                                     constant float& alpha [[buffer(3)]],
                                     uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] > alpha ? 1.0f : 0.0f);
    }
    
    kernel void PReLU(constant float* in_tensor    [[ buffer(0) ]],
                      device float* out_tensor    [[ buffer(1) ]],
                      constant uint& count    [[ buffer(2) ]],
                      constant float* alpha [[buffer(3)]],
                      uint2 gid             [[ thread_position_in_grid ]]) {
        if (gid.x >= count) return;
        out_tensor[gid.x] = (in_tensor[gid.x] >= 0.0f ? in_tensor[gid.x] : in_tensor[gid.x] * alpha[gid.x]);
    }
}
