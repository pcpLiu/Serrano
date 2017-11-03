//
//  maxpool1d_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/20/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

func maxPoolVerify(input: Tensor, kernelSize: [Int], stride: [Int], padMode: PaddingMode, order: TensorChannelOrder) -> Tensor {
	let inputShapeArray = input.shape.shapeArray
	let (channel, inHeight, inWidth) = parseImgChannelShapeInfo(order, shapeArray: inputShapeArray)
	var outShapeArray = [channel,
	                     kernelScanningOutSize(padMode, inputSize: inHeight,
	                                           kernelSize: kernelSize[0], stride: stride[0]),
	                     kernelScanningOutSize(padMode, inputSize: inWidth,
	                                           kernelSize: kernelSize[1], stride: stride[1])]
	if order == TensorChannelOrder.Last {
		outShapeArray = [
		                 kernelScanningOutSize(padMode, inputSize: inHeight,
		                                       kernelSize: kernelSize[0], stride: stride[0]),
		                 kernelScanningOutSize(padMode, inputSize: inWidth,
		                                       kernelSize: kernelSize[1], stride: stride[1]),
		                 channel]
	}
	let tensor = Tensor(repeatingValue: 0.0, tensorShape: TensorShape(dataType: .float, shape: outShapeArray))
	let (_, outHeight, outWidth) = parseImgChannelShapeInfo(order, shapeArray: outShapeArray)
	
	for c in 0..<channel {
		for i in 0..<outHeight {
			for j in 0..<outWidth {
				// pooling
				let validHeightCount = min(i*stride[0] + kernelSize[0], inHeight) - i*stride[0]
				let validWidthCount = min(j*stride[1] + kernelSize[1], inWidth) - j*stride[1]
				var max_v: Float = -Float.infinity
				for m in 0..<validHeightCount {
					for n in 0..<validWidthCount {
						if order == TensorChannelOrder.Last {
							max_v = max(input[i*stride[0] + m, j*stride[1] + n, c], max_v)
						} else {
							max_v = max(input[c, i*stride[0] + m, j*stride[1] + n], max_v)
						}
					}
				}
				if order == TensorChannelOrder.Last {
					tensor[i, j, c] = max_v
				} else {
					tensor[c, i, j] = max_v
				}
			}
		}
	}
	return tensor
}

public class OperatorDelegateConvMaxPool2DOp: OperatorDelegateConvPool2DOp {
	override public func compare() {
		let kernelSize = self.kernelSize
		let stride = self.stride
		let padMode = self.padMode
		let inputTensors = self.veryfyTensors
		let resultTensors = self.resultTensors
		let channelOrder = self.channelOrder
		
		for (input, output) in zip(inputTensors, resultTensors) {
			let verifyTensor = maxPoolVerify(input: input, kernelSize: kernelSize,
			                                 stride: stride, padMode: padMode, order: channelOrder)

			XCTAssertTrue(verifyTensor.shape == output.shape)
			let outReader = output.floatValueReader
			let verifyReader = verifyTensor.floatValueReader
			for i in 0..<output.count {
				XCTAssertEqual(outReader[i], verifyReader[i], accuracy: abs(verifyReader[i]*0.01), "\(outReader[i]), \(verifyReader[i])")
			}
			
			
		}
	}
}

class maxpool2d_op_test: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
	func test() {
		let testCase = Pool2DOperatorTest<MaxPool2DOperator, OperatorDelegateConvMaxPool2DOp>()
		testCase.testAll()
	}
    
}
