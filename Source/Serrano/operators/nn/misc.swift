//
//  misc.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/20/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
#if  !((arch(i386)  || arch(x86_64)) && os(iOS))
	import MetalPerformanceShaders
#endif

/**
Padding mode for operators involving with kernel calculation, like pooling and convolution operators.
We follow same behavior as TensorFlow defined.

## Valid padding
No padding at all when moving kernel on input tensor.

## Same padding
Padding to cover all elements
*/
public enum PaddingMode: Int {
	case Valid = 0
	case Same  = 1
	
	var description: String {
		get { return String(reflecting: self) }
	}
}


/// Caculate output size for convolutiona like kernel scanning operation.
///
/// ## Valid
/// `Int(Float((inputSize - stride + 1) / patchSize).rounded(.up))`
///
/// ## Same
/// `Int(Float(inputSize / stride).rounded(.up))`
///
/// - Parameters:
///   - mode: mode
///   - inputSize: inputSize description
///   - kernelSize: kernel size
///	  - stride: stride
/// - Returns: return value
public func kernelScanningOutSize(_ mode: PaddingMode, inputSize: Int, kernelSize: Int, stride: Int) -> Int {
	switch mode {
	case .Same:
		let val = Float(inputSize) / Float(stride)
		return Int(val.rounded(.up))
	case .Valid:
		let val = Float(inputSize - kernelSize + 1) / Float(stride)
		return Int(val.rounded(.up))
	}
}


/**
The channel order in a N-D tensor.

## First
Coming before tensor shape.
For example an image with height `H` and width `W`, it will be represented as [C, H, W]

## Last
Coming after tensor shape.
For example an image with height `H` and width `W`, it will be represented as [C, H, W]

*/
public enum TensorChannelOrder: Int {
	case First = 0
	case Last  = 1
	
	var description: String {
		get { return String(reflecting: self) }
	}
	
	#if  !((arch(i386)  || arch(x86_64)) && os(iOS))
	@available(OSX 10.13, iOS 11.0, *)
	var MPSImageOrder: MPSDataLayout {
		get {
			if self == .First {
				return MPSDataLayout.featureChannelsxHeightxWidth
			} else {
				return MPSDataLayout.HeightxWidthxFeatureChannels
			}
		}
	}
	#endif
}


/// According to `channelOrder`, parse `inputShapeArray` to channel, height and width
///
/// - Note: rank of `inputShapeArray` should be `3`.
///
/// - Parameters:
///   - channelOrder: channelOrder
///   - shapeArray: inputShapeArray
/// - Returns: return value
public func parseImgChannelShapeInfo(_ channelOrder: TensorChannelOrder, shapeArray: [Int]) -> (channel:Int, height:Int, width: Int) {
	
	guard shapeArray.count == 3 else {
		SerranoLogging.errorLogging(message: "Input array should contain 3 element. Given \(shapeArray.count) elements",
		                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
		fatalError()
	}
	
	if channelOrder == .First {
		return (shapeArray[0], shapeArray[1], shapeArray[2])
	} else if channelOrder == .Last {
		return (shapeArray[2], shapeArray[0], shapeArray[1])
	} else {
		fatalError("Not implemented")
	}
}



