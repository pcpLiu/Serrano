//
//  transpose_op.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 6/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

namespace serrano_ios {
	
	typedef struct
	{
		uint M; // # of row in transposed matrix
		uint N; // # of cols in transposed matrix
		ushort stride; // stride of element
	} TransposeMatrixInfo;
	
	kernel void Transpose(const device float* in_tensor    [[ buffer(0) ]],
						  const device float* out_tensor    [[ buffer(1) ]],
						  constant TransposeMatrixInfo& matrix_info    [[ buffer(2) ]],
						  uint2 gid             [[ thread_position_in_grid ]]) {
		uint M = matrix_info.M;
		uint N = matrix_info.N;
		ushort stride = matrix_info.stride;
		
		if (gid.x >= M || gid.y >= N) return;
		
		const device float* src = (const device float*)((device char*)in_tensor + (M * gid.y + gid.x) * stride);
		device float* out = (device float*)((device char*)out_tensor + (N * gid.x + gid.y) * stride);
		out[0] = src[0];
	}
}
