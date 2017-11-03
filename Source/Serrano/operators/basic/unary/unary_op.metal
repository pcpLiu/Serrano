//
//  unary_op.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 6/6/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

namespace serrano_ios {
    kernel void Sin(device float* in_tensor    [[ buffer(0) ]],
                    device float* out_tensor    [[ buffer(1) ]],
                    constant uint* count    [[ buffer(2) ]],
                    uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = sin(in_tensor[gid.x]);
    }
    
    kernel void Arcsin(device float* in_tensor    [[ buffer(0) ]],
                       device float* out_tensor    [[ buffer(1) ]],
                       constant uint* count    [[ buffer(2) ]],
                       uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = asin(in_tensor[gid.x]);
    }
    
    kernel void Cos(device float* in_tensor    [[ buffer(0) ]],
                    device float* out_tensor    [[ buffer(1) ]],
                    constant uint* count    [[ buffer(2) ]],
                    uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = cos(in_tensor[gid.x]);
    }
    
    kernel void Tan(device float* in_tensor    [[ buffer(0) ]],
                    device float* out_tensor    [[ buffer(1) ]],
                    constant uint* count    [[ buffer(2) ]],
                    uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = tan(in_tensor[gid.x]);
    }
    
    kernel void Arctan(device float* in_tensor    [[ buffer(0) ]],
                       device float* out_tensor    [[ buffer(1) ]],
                       constant uint* count    [[ buffer(2) ]],
                       uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = atan(in_tensor[gid.x]);
    }
    
    kernel void Abs(device float* in_tensor    [[ buffer(0) ]],
                    device float* out_tensor    [[ buffer(1) ]],
                    constant uint* count    [[ buffer(2) ]],
                    uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = abs(in_tensor[gid.x]);
    }
    
    kernel void Degree(device float* in_tensor    [[ buffer(0) ]],
                       device float* out_tensor    [[ buffer(1) ]],
                       constant uint* count    [[ buffer(2) ]],
                       uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = 180 / M_PI_F * in_tensor[gid.x] ;
    }
    
    kernel void Arccos(device float* in_tensor    [[ buffer(0) ]],
                       device float* out_tensor    [[ buffer(1) ]],
                       constant uint* count    [[ buffer(2) ]],
                       uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = acos(in_tensor[gid.x]);
    }
    
    kernel void Radien(device float* in_tensor    [[ buffer(0) ]],
                       device float* out_tensor    [[ buffer(1) ]],
                       constant uint* count    [[ buffer(2) ]],
                       uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = M_PI_F / 180 * in_tensor[gid.x];
    }
    
    kernel void Sinh(device float* in_tensor    [[ buffer(0) ]],
                     device float* out_tensor    [[ buffer(1) ]],
                     constant uint* count    [[ buffer(2) ]],
                     uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = sinh(in_tensor[gid.x]);
    }
    
    kernel void Cosh(device float* in_tensor    [[ buffer(0) ]],
                     device float* out_tensor    [[ buffer(1) ]],
                     constant uint* count    [[ buffer(2) ]],
                     uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = cosh(in_tensor[gid.x]);
    }
    
    kernel void Tanh(device float* in_tensor    [[ buffer(0) ]],
                     device float* out_tensor    [[ buffer(1) ]],
                     constant uint* count    [[ buffer(2) ]],
                     uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = tanh(in_tensor[gid.x]);
    }
    
    kernel void Arctanh(device float* in_tensor    [[ buffer(0) ]],
                        device float* out_tensor    [[ buffer(1) ]],
                        constant uint* count    [[ buffer(2) ]],
                        uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = atanh(in_tensor[gid.x]);
    }
    
    kernel void Arccosh(device float* in_tensor    [[ buffer(0) ]],
                        device float* out_tensor    [[ buffer(1) ]],
                        constant uint* count    [[ buffer(2) ]],
                        uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = acosh(in_tensor[gid.x]);
    }
    
    kernel void Arcsinh(device float* in_tensor    [[ buffer(0) ]],
                        device float* out_tensor    [[ buffer(1) ]],
                        constant uint* count    [[ buffer(2) ]],
                        uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = asinh(in_tensor[gid.x]);
    }
    
    kernel void Floor(device float* in_tensor    [[ buffer(0) ]],
                      device float* out_tensor    [[ buffer(1) ]],
                      constant uint* count    [[ buffer(2) ]],
                      uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = floor(in_tensor[gid.x]);
    }
    
    kernel void Ceil(device float* in_tensor    [[ buffer(0) ]],
                     device float* out_tensor    [[ buffer(1) ]],
                     constant uint* count    [[ buffer(2) ]],
                     uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = ceil(in_tensor[gid.x]);
    }
    
    kernel void Rint(device float* in_tensor    [[ buffer(0) ]],
                     device float* out_tensor    [[ buffer(1) ]],
                     constant uint* count    [[ buffer(2) ]],
                     uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = rint(in_tensor[gid.x]);
    }
    
    kernel void Round(device float* in_tensor    [[ buffer(0) ]],
                      device float* out_tensor    [[ buffer(1) ]],
                      constant uint* count    [[ buffer(2) ]],
                      uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = round(in_tensor[gid.x]);
    }
    
    kernel void Square(device float* in_tensor    [[ buffer(0) ]],
                       device float* out_tensor    [[ buffer(1) ]],
                       constant uint* count    [[ buffer(2) ]],
                       uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] =  in_tensor[gid.x] * in_tensor[gid.x];
    }
    
    kernel void Rsqrt(device float* in_tensor    [[ buffer(0) ]],
                      device float* out_tensor    [[ buffer(1) ]],
                      constant uint* count    [[ buffer(2) ]],
                      uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = rsqrt(in_tensor[gid.x]);
    }
    
    kernel void Sqrt(device float* in_tensor    [[ buffer(0) ]],
                     device float* out_tensor    [[ buffer(1) ]],
                     constant uint* count    [[ buffer(2) ]],
                     uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = sqrt(in_tensor[gid.x]);
    }
    
    kernel void Log1p(device float* in_tensor    [[ buffer(0) ]],
                      device float* out_tensor    [[ buffer(1) ]],
                      constant uint* count    [[ buffer(2) ]],
                      uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = log(in_tensor[gid.x] + 1);
    }
    
    kernel void Log2(device float* in_tensor    [[ buffer(0) ]],
                     device float* out_tensor    [[ buffer(1) ]],
                     constant uint* count    [[ buffer(2) ]],
                     uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = log2(in_tensor[gid.x]);
    }
    
    kernel void Log10(device float* in_tensor    [[ buffer(0) ]],
                      device float* out_tensor    [[ buffer(1) ]],
                      constant uint* count    [[ buffer(2) ]],
                      uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = log10(in_tensor[gid.x]);
    }
    
    kernel void Log(device float* in_tensor    [[ buffer(0) ]],
                    device float* out_tensor    [[ buffer(1) ]],
                    constant uint* count    [[ buffer(2) ]],
                    uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = log(in_tensor[gid.x]);
    }
    
    kernel void Expm1(device float* in_tensor    [[ buffer(0) ]],
                      device float* out_tensor    [[ buffer(1) ]],
                      constant uint* count    [[ buffer(2) ]],
                      uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = exp(in_tensor[gid.x]) - 1;
    }
    
    kernel void Exp(device float* in_tensor    [[ buffer(0) ]],
                    device float* out_tensor    [[ buffer(1) ]],
                    constant uint* count    [[ buffer(2) ]],
                    uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        out_tensor[gid.x] = exp(in_tensor[gid.x]);
    }
}
