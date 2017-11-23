//
//  stat_op.metal
//  Serrano
//
//  Created by ZHONGHAO LIU on 1/1/18.
//  Copyright © 2018 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

void kernel stat_mean(constant float* input    [[ buffer(0) ]],
                      device float* output     [[ buffer(1) ]],
                      constant int& prev_count [[ buffer(2) ]],
                      constant uint& boundary  [[ buffer(3) ]],
                      uint2 gid                [[ thread_position_in_grid ]]) {
    if (gid.x >= boundary) {
        return;
    }
    output[gid.x] = (output[gid.x] * prev_count + input[gid.x]) / (prev_count + 1);
}
