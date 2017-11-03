//
//  binary_op.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 6/7/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

namespace serrano_ios {
    kernel void Add(device float* inputA    [[ buffer(0) ]],
                    device float* inputB    [[ buffer(1) ]],
                    device float* outputC   [[ buffer(2) ]],
                    constant uint* count    [[ buffer(3) ]],
                    uint2 gid             [[ thread_position_in_grid ]])
    {
        if (gid.x >= count[0]) return;
        outputC[gid.x] = inputA[gid.x] + inputB[gid.x];
    }
	
	kernel void Sub(device float* inputA    [[ buffer(0) ]],
					device float* inputB    [[ buffer(1) ]],
					device float* outputC   [[ buffer(2) ]],
					constant uint* count    [[ buffer(3) ]],
					uint2 gid             [[ thread_position_in_grid ]])
	{
		if (gid.x >= count[0]) return;
		outputC[gid.x] = inputA[gid.x] - inputB[gid.x];
	}
	
	kernel void Mult(device float* inputA    [[ buffer(0) ]],
					device float* inputB    [[ buffer(1) ]],
					device float* outputC   [[ buffer(2) ]],
					constant uint* count    [[ buffer(3) ]],
					uint2 gid             [[ thread_position_in_grid ]])
	{
		if (gid.x >= count[0]) return;
		outputC[gid.x] = inputA[gid.x] * inputB[gid.x];
	}
	
	kernel void Div(device float* inputA    [[ buffer(0) ]],
					 device float* inputB    [[ buffer(1) ]],
					 device float* outputC   [[ buffer(2) ]],
					 constant uint* count    [[ buffer(3) ]],
					 uint2 gid             [[ thread_position_in_grid ]])
	{
		if (gid.x >= count[0]) return;
		outputC[gid.x] = inputA[gid.x] / inputB[gid.x];
	}
	
	kernel void RDiv(device float* inputA    [[ buffer(0) ]],
					device float* inputB    [[ buffer(1) ]],
					device float* outputC   [[ buffer(2) ]],
					constant uint* count    [[ buffer(3) ]],
					uint2 gid             [[ thread_position_in_grid ]])
	{
		if (gid.x >= count[0]) return;
		outputC[gid.x] = inputB[gid.x] / inputA[gid.x];
	}
	
	kernel void Pow(device float* inputA    [[ buffer(0) ]],
					 device float* inputB    [[ buffer(1) ]],
					 device float* outputC   [[ buffer(2) ]],
					 constant uint* count    [[ buffer(3) ]],
					 uint2 gid             [[ thread_position_in_grid ]])
	{
		if (gid.x >= count[0]) return;
		outputC[gid.x] = pow(inputA[gid.x], inputB[gid.x]);
	}
}
