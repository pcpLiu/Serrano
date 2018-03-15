//
//  batchnorm_op.metal
//  Serrano
//
//  Created by ZHONGHAO LIU on 11/27/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;
#include "serrano_misc.h"

typedef struct {
    short channelPosition; // 0 --> First, 1 --> Last
    int channels;
    int inputWidth;
    int inputHeight;
    float epsilon;
} BatchNormInfo;

void kernel batchNorm_inference(constant float* input        [[ buffer(0) ]],
                    device float* output         [[ buffer(1) ]],
                    constant float* mean         [[ buffer(2) ]],
                    constant float* var          [[ buffer(3) ]],
                    constant float* scale        [[ buffer(4) ]],
                    constant float* offset       [[ buffer(5) ]],
                    constant BatchNormInfo& info [[ buffer(6) ]],
                    uint3 thread_id              [[ thread_position_in_grid ]]) {
    // boundary check
    if (thread_id.x >= uint(info.inputWidth) || thread_id.y >= uint(info.inputHeight)) {
        return;
    }
    
    ChannelPosition channelPosition = getChannelPosition(info.channelPosition);
    uint inputOutputOffset;
    if (channelPosition == Last) {
        inputOutputOffset = thread_id.y * info.inputWidth * info.channels + thread_id.x * info.channels + thread_id.z;
    } else {
        inputOutputOffset = thread_id.z * info.inputHeight * info.inputWidth + thread_id.y * info.inputWidth + thread_id.x;
    }
    
    output[inputOutputOffset] = scale[thread_id.z] * (input[inputOutputOffset] - mean[thread_id.z]) / sqrt(var[thread_id.z] + info.epsilon) + offset[thread_id.z];
}
