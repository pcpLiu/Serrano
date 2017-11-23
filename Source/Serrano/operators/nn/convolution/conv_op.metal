//
//  conv_op.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 10/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>

using namespace metal;
#include "serrano_misc.h"


struct ConvInfo {
	short channelPosition; // 0 --> First, 1 --> Last
	short paddingMode;     // 0 --> Valid, 1 --> Same
	float paddingValue;
	int inChannels;
	int inputWidth;
	int inputHeight;
	int outChannels; // # of kernnels
	int outputWidth;
	int outputHeight;
	int strideWidth;
	int strideHeight;
	int kernelWidth;
	int kernelHeight;
};


/**
 Fetch value from input according to input_index (width_index, height_index, channel_index).
 input_dimension (inputWidth, inputHeight, inChannels).
 strides (strideWidth, strideHeight)
 */
float conv2d_fetch_input_value(constant float* input, thread uint3& input_index, thread int3& input_dimension,
								ChannelPosition channelPosition, float paddingValue) {
	// padding checking
	if (input_index.x >= uint(input_dimension.x) || input_index.y >= uint(input_dimension.y)) {
		return paddingValue;
	}
	
	int offset;
	if (channelPosition == Last) {
		offset = input_index.y * (input_dimension.x * input_dimension.z) + input_index.x * input_dimension.z + input_index.z;
	} else {
		offset = input_index.z * (input_dimension.x * input_dimension.y) + input_index.y * input_dimension.x + input_index.x;
	}
	return input[offset];
}

/**
 Fetch weight value according to weight_index (filter_index, input_channel_index, height_index, width_index)
 weight_dimension (num_filter，channel, kernelSize[0], kernelSize[1])
 */
 float conv2d_fetch_weight_value(constant float* weightValues, thread int4& weight_index, thread int4& weight_dimension) {
	 int offset = weight_index.x * (weight_dimension.y * weight_dimension.z * weight_dimension.w)
	 				+ weight_index.y * (weight_dimension.z * weight_dimension.w)
	 				+ weight_index.z * weight_dimension.w
	 				+ weight_index.w;
	return weightValues[offset];
}

/**
 Naive convolutional 2D.
 Each thread calculate one value in output tensor.
 */
void kernel conv2d_naive(constant float*       input           [[ buffer(0) ]],
						 device float*         output          [[ buffer(1) ]],
						 constant float*       weightValues    [[ buffer(2) ]],
						 constant float*       biasValues      [[ buffer(3) ]],
						 constant ConvInfo&    info            [[ buffer(4) ]],
						 uint3                 thread_id       [[ thread_position_in_grid ]]) {
	
	// output index, local indexed
	uint3 out_index = thread_id;
	
	// boundary check
	if (out_index.x >= uint(info.outputWidth) || out_index.y >= uint(info.outputHeight)) {
		return;
	}

	// get info
	int3 input_dimension = int3(info.inputWidth, info.inputHeight, info.inChannels);
	int3 output_dimension = int3(info.outputWidth, info.outputHeight, info.outChannels);
	int2 strides = int2(info.strideWidth, info.strideHeight);
	float paddingValue = info.paddingValue;
	ChannelPosition channelPosition = getChannelPosition(info.channelPosition);
	// `[num_filter，channel, kernelSize[0], kernelSize[1]]`,
	int4 kernel_dimension = int4(info.outChannels, info.inChannels, info.kernelHeight, info.kernelWidth);
	int4 weight_start_index = int4(thread_id.z, 0, 0, 0);
	uint3 input_start_index = uint3(out_index.x * strides.x, out_index.y * strides.y, 0);

//	// calcualte result
	float result = 0.0;
	uint3 input_index;
	int4 weight_index;
	if (channelPosition == First) {
		for (int input_channel = 0; input_channel < input_dimension.z; input_channel++) {
			for (int height_i = 0; height_i < kernel_dimension.z; height_i++) {
				for (int width_i = 0; width_i < kernel_dimension.w; width_i++) {
					input_index = uint3(input_start_index.x + width_i, input_start_index.y + height_i, input_start_index.z + input_channel);
					weight_index = int4(weight_start_index.x, weight_start_index.y + input_channel, weight_start_index.z + height_i, weight_start_index.w + width_i);
					result += conv2d_fetch_weight_value(weightValues, weight_index, kernel_dimension) *
						conv2d_fetch_input_value(input, input_index, input_dimension, channelPosition, paddingValue);
				}
			}
		}
	} else {
		for (int height_i = 0; height_i < kernel_dimension.z; height_i++) {
			for (int width_i = 0; width_i < kernel_dimension.w; width_i++) {
				for (int input_channel = 0; input_channel < input_dimension.z; input_channel++) {
					input_index = uint3(input_start_index.x + width_i, input_start_index.y + height_i, input_start_index.z + input_channel);
					weight_index = int4(weight_start_index.x, weight_start_index.y + input_channel, weight_start_index.z + height_i, weight_start_index.w + width_i);
					result += conv2d_fetch_weight_value(weightValues, weight_index, kernel_dimension) *
					conv2d_fetch_input_value(input, input_index, input_dimension, channelPosition, paddingValue);
				}
			}
		}
	}
	
	output[out_index.y * (output_dimension.x * output_dimension.z) + out_index.x * output_dimension.z + out_index.z] = result + biasValues[thread_id.z];
}

