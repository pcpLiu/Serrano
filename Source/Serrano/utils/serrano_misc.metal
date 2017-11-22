//
//  serrano_misc.metal
//  Serrano
//
//  Created by ZHONGHAO LIU on 11/18/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;
#include "serrano_misc.h"


ChannelPosition getChannelPosition(int channelPosition) {
	ChannelPosition pos = First;
	if (channelPosition == 1) {
		pos = Last;
	}
	return pos;
}

PaddingMode getPaddingMode(int paddingMode) {
	PaddingMode mode = Valid;
	if (paddingMode == 1) {
		mode = Same;
	}
	return mode;
}

