//
//  conv_op.metal
//  serrano
//
//  Created by ZHONGHAO LIU on 10/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

/**
 Naive convolutional 2D.
 Each thread calculate one value in output tensor.
 */

//TODO: implement


//void kernel conv2d_naive(constant float*       input           [[ buffer(0) ]],
//						 device float*         output          [[ buffer(1) ]],
//						 constant Img2ColInfo& info            [[ buffer(2) ]],
//						 uint3                 group_id        [[ threadgroup_position_in_grid ]],
//						 uint3                 thread_id_group [[ thread_position_in_threadgroup ]]) {
//
//}

