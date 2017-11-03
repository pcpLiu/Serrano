//
//  fullyconnected_op.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 7/19/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

typedef struct
{
	uint M; 	 // number of rows in A
	uint N; 	 // number of cols in B
	uint K; 	 // number of cols in A, number of rows in B
	ushort stride;         // Element stride in bytes
	bool useBias; // 0 - false, 1 - true
} FCInfo;


kernel void Fullyconnected(constant float*       input_A    [[ buffer(0) ]],
							  constant float*       input_B    [[ buffer(1) ]], // should be transposed
							  const device float*   C          [[ buffer(2) ]],
							  constant float*       biasArray	   [[ buffer(3) ]],
							  constant FCInfo&  info        [[ buffer(4) ]],
							  uint2                     gid        [[ thread_position_in_grid ]]) {
	uint M = info.M;
	uint N = info.N;
	uint K = info.K;
	ushort stride = info.stride;
	bool useBias = info.useBias;
	
	// check boundary
	if (gid.x >= M || gid.y >= N) return;
	
	device float* c_reader = (device float*)((device char*)C + gid.x * N * stride + gid.y * stride);
	c_reader[0] = (useBias ? biasArray[gid.y] : 0.0f);
	
	// small case
	if (K < 4) {
		constant float* a_reader = (constant float*)((constant char*)input_A + gid.x * K * stride);
		constant float* b_reader = (constant float*)((constant char*)input_B + gid.y * K * stride);
		ushort i = 0;
		while (i < K) {
			c_reader[0] += a_reader[i] * b_reader[i];
			i++;
		}
		return;
	}
	
	// regular case, each time read 4 elements
	constant float4* a_reader = (constant float4*)((constant char*)input_A + gid.x * K * stride);
	constant float4* b_reader = (constant float4*)((constant char*)input_B + gid.y * K * stride);
	uint align_bound = K / 4;
	ushort i = 0;
	float4 result;
	while (i < align_bound) {
		result = a_reader[i] * b_reader[i];
		c_reader[0] += result.x;
		c_reader[0] += result.y;
		c_reader[0] += result.z;
		c_reader[0] += result.w;
		i++;
	}
	
	// rest
	if (K % 4 != 0) {
		constant float* a_reader_p = (constant float*)((constant char*)input_A + gid.x * K * stride);
		constant float* b_reader_p = (constant float*)((constant char*)input_B + gid.y * K * stride);
		i = align_bound * 4;
		while (i < K) {
			c_reader[0] += a_reader_p[i] * b_reader_p[i];
			i++;
		}
	}
}
