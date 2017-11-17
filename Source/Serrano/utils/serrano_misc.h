//
//  serrano_misc.h
//  Serrano
//
//  Created by ZHONGHAO LIU on 11/18/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#ifndef serrano_misc_h
#define serrano_misc_h

enum ChannelPosition {
	First,
	Last
};

ChannelPosition getChannelPosition(int channelPosition);

/// Get padding mode
enum PaddingMode {
	Valid,
	Same
};

PaddingMode getPaddingMode(int paddingMode); 

#endif /* serrano_misc_h */
