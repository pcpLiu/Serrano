//
//  FBSUtil.c
//  serrano
//
//  Created by ZHONGHAO LIU on 7/5/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

#include "FBSUtil.h"
char* FBSUtil_readFlatBuffer(const char* file_path) {
	FILE *bufferFile;
	bufferFile = fopen(file_path, "rb");
	if (bufferFile == NULL) {
		printf("Failed to open file. %s", strerror(errno));
		exit(1);
	}
	
	// check file size
	size_t filelen = 0;
	fseek(bufferFile, 0, SEEK_END);          // Jump to the end of the file
	filelen = ftell(bufferFile);             // Get the current byte offset in the file
	rewind(bufferFile);                      // Jump back to the beginning of the file
	
	// allocate buffer and read buffer file
	char *buffer;
	buffer = (char *) malloc((filelen) * sizeof(char)); // Enough memory for file + \0
	fread(buffer, filelen, 1, bufferFile); // Read in the entire file
	fclose(bufferFile); // Close the file
	
	return buffer;
}

void FBSUtil_releaseFlatBuffer(char* allocated_buffer) {
	free(allocated_buffer);
}

size_t FBSUtil_tensorsCount(const char* buffer) {
	// make params root
	Serrano_Params_Params_table_t params = Serrano_Params_Params_as_root(buffer);
	
	// tensor list
	Serrano_Params_Tensor_vec_t tensors = Serrano_Params_Params_tensors(params);
	
	return Serrano_Params_Params_vec_len(tensors);
}

const char* FBSUtil_tensorUID(const char* buffer, size_t tensor_index) {
	// make params root
	Serrano_Params_Params_table_t params = Serrano_Params_Params_as_root(buffer);
	
	// tensors list
	Serrano_Params_Tensor_vec_t tensors = Serrano_Params_Params_tensors(params);
	
	// get target tensor obj
	Serrano_Params_Tensor_table_t tensor = Serrano_Params_Tensor_vec_at(tensors, tensor_index);
	
	// get this tensor values list
	return Serrano_Params_Tensor_uid(tensor);
}

size_t FBSUtil_tensorValuesCount(const char* buffer, size_t tensor_index) {
	// make params root
	Serrano_Params_Params_table_t params = Serrano_Params_Params_as_root(buffer);
	
	// tensors list
	Serrano_Params_Tensor_vec_t tensors = Serrano_Params_Params_tensors(params);
	
	// get target tensor obj
	Serrano_Params_Tensor_table_t tensor = Serrano_Params_Tensor_vec_at(tensors, tensor_index);
	
	// get value list
	const float* values = Serrano_Params_Tensor_values(tensor);
	
	return flatbuffers_float_vec_len(values);
}

float FBSUtil_tensorValueAt(const char* buffer, size_t tensor_index, size_t value_index) {
	// make params root
	Serrano_Params_Params_table_t params = Serrano_Params_Params_as_root(buffer);
	
	// tensors list
	Serrano_Params_Tensor_vec_t tensors = Serrano_Params_Params_tensors(params);
	
	// get target tensor obj
	Serrano_Params_Tensor_table_t tensor = Serrano_Params_Tensor_vec_at(tensors, tensor_index);
	
	// get value list
	const float* values = Serrano_Params_Tensor_values(tensor);
	
	return flatbuffers_float_vec_at(values, value_index);
}

