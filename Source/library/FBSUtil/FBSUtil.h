//
//  FBSUtil.h
//  serrano
//
//  Created by ZHONGHAO LIU on 7/5/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#ifndef FBSUtil_h
#define FBSUtil_h

#include <stdio.h>
#include <errno.h>
#include "SerranoSchema_reader.h"
#include "flatbuffers_common_reader.h"

char* FBSUtil_readFlatBuffer(const char* file_path);
void FBSUtil_releaseFlatBuffer(char* allocated_buffer);
size_t FBSUtil_tensorsCount(const char* buffer);
size_t FBSUtil_tensorValuesCount(const char* buffer, size_t tensor_index);
const char*  FBSUtil_tensorUID(const char* buffer, size_t tensor_index);
float FBSUtil_tensorValueAt(const char* buffer, size_t tensor_index, size_t value_index);


#endif /* FBSUtil_h */
