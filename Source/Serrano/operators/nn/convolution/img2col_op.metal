//
//  img2col.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 8/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>

using namespace metal;
#include "Source/Serrano/utils/serrano_misc.h"

typedef struct {
	short channelPosition; // 0 --> First, 1 --> Last
	short paddingMode;     // 0 --> Valid, 1 --> Same
	float paddingValue;
	int channels;
	int inputWidth;
	int inputHeight;
	int kernelScanningXPatchCount;
	int strideWidth;
	int strideHeight;
	int patchWidth;
	int patchHeight;
} Img2ColInfo;

/// Each thread copy one value
void kernel Img2col (constant float*       input           [[ buffer(0) ]],
					 device float*         output          [[ buffer(1) ]],
					 constant Img2ColInfo& info            [[ buffer(2) ]],
					 uint3                 group_id        [[ threadgroup_position_in_grid ]],
					 uint3                 thread_id_group [[ thread_position_in_threadgroup ]]) {
	
	
	int strideWidth = info.strideWidth;
	int strideHeight = info.strideHeight;
	int channels = info.channels;
	int patchWidth = info.patchWidth;
	int patchHeight = info.patchHeight;
	float paddingValue = info.paddingValue;
	int inputWidth = info.inputWidth;
	int inputHeight = info.inputHeight;
	ChannelPosition channelPosition = getChannelPosition(info.channelPosition);
	PaddingMode paddingMode = getPaddingMode(info.paddingMode);

	// get thread group info
	int patchChannelIndex = group_id.z;
	int patchHightIndex = group_id.y / info.kernelScanningXPatchCount; // patchX in input
	int patchWidthIndex = group_id.y % info.kernelScanningXPatchCount; // patchY in input
	int patchElementHightIndex = thread_id_group.y;
	int patchElementWidthIndex = thread_id_group.x;
	
	// get input offset
	int inputOffset;
	if (channelPosition == First) {
		int channelCount = inputWidth * inputHeight; // no. of elements in one channel
		int channelOffset = channelCount * patchChannelIndex; // start of channel
		int patchOffset = patchHightIndex * (inputWidth * strideHeight) + patchWidthIndex * strideWidth; // start of patch
		int elementOffset = patchElementHightIndex * inputWidth + patchElementWidthIndex; // locate element in patch
		inputOffset = channelOffset + patchOffset + elementOffset;
	} else {
		int rowCount = channels * inputWidth; // num of element in each row
		int channelOffset = patchChannelIndex;
		int patchOffset = patchHightIndex * (rowCount * strideHeight) + patchWidthIndex * strideWidth * channels;
		int elementOffset = patchElementHightIndex * rowCount + patchElementWidthIndex * channels;
		inputOffset = channelOffset + patchOffset + elementOffset;
	}
	
	// get output offset
	int patchCount = patchHeight * patchWidth;
	int outputOffset = (group_id.y * patchCount * channels + patchChannelIndex * patchCount) +
						(patchElementHightIndex * patchWidth + patchElementWidthIndex);
	
	// assign
	int inputElementX = patchHightIndex * strideHeight + patchElementHightIndex;
	int inputElementY = patchWidthIndex * strideWidth + patchElementWidthIndex;
	if (inputElementX >= inputHeight || inputElementY >= inputWidth) { // boundary check
		if (paddingMode == Same) { // only same use padding value
			output[outputOffset] = paddingValue;
		}
	} else {
		output[outputOffset] = input[inputOffset];
	}
}

