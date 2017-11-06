//
//  flatbuffer.swift
//  Serrano
//
//  Created by ZHONGHAO LIU on 11/2/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import LibFBSUtil


/// Define APIs reading Flatbuffer
public class FlatbufferIO {
	/// Load a saved params flatbuffers from `flatbufferFile`
	///
	/// - Parameter flatbufferFile: flatbuffer file path
	/// - Returns: A dictionary with UID and corresponding `Tensor` object.
	public static func loadSavedParams(_ flatbufferFile: String) -> [String : Tensor]{
		var result = [String : Tensor]()
		
		let readBuffer = FBSUtil_readFlatBuffer(flatbufferFile)
		let tensorCount = FBSUtil_tensorsCount(readBuffer)
		for tensorIndex in 0..<tensorCount {
			let uid_C = FBSUtil_tensorUID(readBuffer, tensorIndex)
			let uid = String(cString: uid_C!)
			
			let valuesCount = FBSUtil_tensorValuesCount(readBuffer, tensorIndex)
			let tensor = Tensor(repeatingValue: 0.0,
								tensorShape: TensorShape(dataType: .float, shape: [valuesCount]))
			for valueIndex in 0..<valuesCount {
				tensor.floatValueReader[valueIndex] = FBSUtil_tensorValueAt(readBuffer, tensorIndex, valueIndex)
			}
			result[uid] = tensor
		}
		
		// !Release reading buffer!
		FBSUtil_releaseFlatBuffer(readBuffer)
		
		return result
	}
}
