//
//  pooling_op.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 7/20/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

typedef struct {
	short channelPosition; // 0 --> First, 1 --> Last
	ushort kernelSizeHeight;
	ushort kernelSizeWidth;
	ushort strideHeight;
	ushort strideWidth;
	uint inHeight;
	uint inWidth;
	uint inChannel;
	uint outHeight;
	uint outWidth;
	uint outChannel;
} Pool2DInfo;

//TODO: Loop unrolling implementation for some specific kernel sizes: 2x2, 3x3, 5x5

void kernel MaxPool2D(constant float*      input    [[ buffer(0) ]],
					  device float*        output   [[ buffer(1) ]],
					  constant Pool2DInfo& info     [[ buffer(2) ]],
					  uint3                gid      [[ thread_position_in_grid ]]) {
	// check boundary
	if (gid.x >= info.outWidth || gid.y >= info.outHeight) {
		return;
	}
	
	// get pooling starting index in input
	uint elementStartIndex = gid.y * info.strideHeight * info.inWidth + gid.x * info.strideWidth + gid.z * info.inWidth * info.inHeight;
	if (info.channelPosition == 1) { // channel last
		elementStartIndex = gid.y * info.strideHeight * info.inWidth * info.inChannel + gid.x * info.strideWidth * info.inChannel + gid.z;
	}
	
	// valid input boundary
	int validHeightPoolCount = min(gid.y * info.strideHeight + info.kernelSizeHeight, info.inHeight) - gid.y * info.strideHeight;
	int validWidthPoolCount =  min(gid.x * info.strideWidth + info.kernelSizeWidth, info.inWidth) - gid.x * info.strideWidth;
	
	// pooling
	float max_val = -INFINITY;
	if (info.channelPosition == 0) { // channel first
		for (int i = 0; i < validHeightPoolCount; i++) {
			for (int j = 0; j < validWidthPoolCount; j++) {
				max_val = max(max_val, input[elementStartIndex + i * info.inWidth + j]);
			}
		}
		output[gid.z * info.outHeight * info.outWidth + gid.y * info.outWidth + gid.x] = max_val;
	} else { // channel last
		for (int i = 0; i < validHeightPoolCount; i++) {
			for (int j = 0; j < validWidthPoolCount; j++) {
				max_val = max(max_val, input[elementStartIndex + i * info.inWidth * info.inChannel + j * info.inChannel]);
			}
		}
		output[gid.y * info.outWidth * info.outChannel + gid.x * info.outChannel + gid.z] = max_val;
	}
}

void kernel AvgPool2D (constant float*      input    [[ buffer(0) ]],
					   device float*        output   [[ buffer(1) ]],
					   constant Pool2DInfo& info     [[ buffer(2) ]],
					   uint3                gid      [[ thread_position_in_grid ]]) {
	// check boundary
	if (gid.x >= info.outWidth || gid.y >= info.outHeight) {
		return;
	}
	
	// get pooling starting index in input
	uint elementStartIndex = gid.y * info.strideHeight * info.inWidth + gid.x * info.strideWidth + gid.z * info.inWidth * info.inHeight;
	if (info.channelPosition == 1) { // channel last
		elementStartIndex = gid.y * info.strideHeight * info.inWidth * info.inChannel + gid.x * info.strideWidth * info.inChannel + gid.z;
	}
	
	// valid input boundary
	int validHeightPoolCount = min(gid.y * info.strideHeight + info.kernelSizeHeight, info.inHeight) - gid.y * info.strideHeight;
	int validWidthPoolCount =  min(gid.x * info.strideWidth + info.kernelSizeWidth, info.inWidth) - gid.x * info.strideWidth;
	
	// pooling
	float sum = 0.0f;
	if (info.channelPosition == 0) { // channel first
		for (int i = 0; i < validHeightPoolCount; i++) {
			for (int j = 0; j < validWidthPoolCount; j++) {
				sum += input[elementStartIndex + i * info.inWidth + j];
			}
		}
		output[gid.z * info.outHeight * info.outWidth + gid.y * info.outWidth + gid.x] = sum / (info.kernelSizeHeight * info.kernelSizeWidth);
	} else { // channel last
		for (int i = 0; i < validHeightPoolCount; i++) {
			for (int j = 0; j < validWidthPoolCount; j++) {
				sum += input[elementStartIndex + i * info.inWidth * info.inChannel + j * info.inChannel];
			}
		}
		output[gid.y * info.outWidth * info.outChannel + gid.x * info.outChannel + gid.z] = sum / (info.kernelSizeHeight * info.kernelSizeWidth);
	}
	
}

void kernel SumPool2D (constant float*      input    [[ buffer(0) ]],
					   device float*        output   [[ buffer(1) ]],
					   constant Pool2DInfo& info     [[ buffer(2) ]],
					   uint3                gid      [[ thread_position_in_grid ]]) {
	// check boundary
	if (gid.x >= info.outWidth || gid.y >= info.outHeight) {
		return;
	}
	
	// get pooling starting index in input
	uint elementStartIndex = gid.y * info.strideHeight * info.inWidth + gid.x * info.strideWidth + gid.z * info.inWidth * info.inHeight;
	if (info.channelPosition == 1) { // channel last
		elementStartIndex = gid.y * info.strideHeight * info.inWidth * info.inChannel + gid.x * info.strideWidth * info.inChannel + gid.z;
	}
	
	// valid input boundary
	int validHeightPoolCount = min(gid.y * info.strideHeight + info.kernelSizeHeight, info.inHeight) - gid.y * info.strideHeight;
	int validWidthPoolCount =  min(gid.x * info.strideWidth + info.kernelSizeWidth, info.inWidth) - gid.x * info.strideWidth;
	
	// pooling
	float sum = 0.0f;
	if (info.channelPosition == 0) { // channel first
		for (int i = 0; i < validHeightPoolCount; i++) {
			for (int j = 0; j < validWidthPoolCount; j++) {
				sum += input[elementStartIndex + i * info.inWidth + j];
			}
		}
		output[gid.z * info.outHeight * info.outWidth + gid.y * info.outWidth + gid.x] = sum;
	} else { // channel last
		for (int i = 0; i < validHeightPoolCount; i++) {
			for (int j = 0; j < validWidthPoolCount; j++) {
				sum += input[elementStartIndex + i * info.inWidth * info.inChannel + j * info.inChannel];
			}
		}
		output[gid.y * info.outWidth * info.outChannel + gid.x * info.outChannel + gid.z] = sum;
	}
	
}
