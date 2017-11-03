//
//  dot_product_op.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 6/16/17.
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
} MatrixDimInfo;

kernel void MatrixMult_Single(constant float*       input_A    [[ buffer(0) ]],
							  constant float*       input_B    [[ buffer(1) ]], // should be transposed
							  const device float*             C          [[ buffer(2) ]],
							  constant MatrixDimInfo&  dims        [[ buffer(3) ]],
							  uint2                     gid        [[ thread_position_in_grid ]]) {
	uint M = dims.M;
	uint N = dims.N;
	uint K = dims.K;
	ushort stride = dims.stride;
	// check boundary
	if (gid.x >= M || gid.y >= N) return;
	
	device float* c_reader = (device float*)((device char*)C + gid.x * N * stride + gid.y * stride);
	c_reader[0] = 0.0f;

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


kernel void MatrixMult_submatrix(constant float*  input_A    [[ buffer(0) ]], // should be transposed for convenience
					   constant float*            input_B    [[ buffer(1) ]],
					   const device float*        C          [[ buffer(2) ]],
					   constant MatrixDimInfo&    dims       [[ buffer(3) ]],
					   uint2                      gid        [[ thread_position_in_grid ]]) {
	uint M = dims.M;
	uint N = dims.N;
	uint K = dims.K;
	ushort stride = dims.stride;
	
	const ushort SUBMATRIX_SIZE = 4;

	// output element start position
	uint2 gid_out = uint2(gid.x * SUBMATRIX_SIZE, gid.y * SUBMATRIX_SIZE); // times 4

	// check boundary
	if (gid_out.x >= M || gid_out.y >= N) return;
	
	ushort row_bound = min(gid_out.x + SUBMATRIX_SIZE, M) - gid_out.x;
	ushort col_bound = min(gid_out.y + SUBMATRIX_SIZE, N) - gid_out.y;
	
	float4x4 output_c_m = float4x4(0.0f);
	constant float4* a_reader = (constant float4*)((constant char*)input_A + gid_out.x * stride);
	constant float4* b_reader = (constant float4*)((constant char*)input_B + gid_out.y * stride);
	for (uint i = 0; i < K; i++) { // loop block
		output_c_m[0] += a_reader[0].x * b_reader[0];
		output_c_m[1] += a_reader[0].y * b_reader[0];
		output_c_m[2] += a_reader[0].z * b_reader[0];
		output_c_m[3] += a_reader[0].w * b_reader[0];
		
		a_reader = (constant float4*)((constant char*)a_reader + M * stride);
		b_reader = (constant float4*)((constant char*)b_reader + N * stride);
	}

	
	// assign
	device float* c_reader = (device float*)((device char*)C + gid_out.x * N * stride + gid_out.y * stride);
	// reset boundary checker
	for (int out_row_index = 0; out_row_index < row_bound; out_row_index++) {
		for (int out_col_index = 0; out_col_index < col_bound; out_col_index++) {
//			c_reader[out_col_index] = output_c[out_row_index*SUBMATRIX_SIZE + out_col_index];
			c_reader[out_col_index] = output_c_m[out_row_index][out_col_index];
		}
		// skip row
		c_reader = (device float*)((device char*)c_reader + N * stride);
	}
}

