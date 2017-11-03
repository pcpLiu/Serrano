//
//  avgpool1d_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/21/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano


func avgPoolVerify(input: Tensor, kernelSize: [Int], stride: [Int], padMode: PaddingMode, order: TensorChannelOrder) -> Tensor {
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
				var sum: Float = 0.0
				for m in 0..<validHeightCount {
					for n in 0..<validWidthCount {
						if order == TensorChannelOrder.Last {
							sum += input[i*stride[0] + m, j*stride[1] + n, c]
						} else {
							sum += input[c, i*stride[0] + m, j*stride[1] + n]
						}
					}
				}
				if order == TensorChannelOrder.Last {
					tensor[i, j, c] = sum / Float(kernelSize[0] * kernelSize[1])
				} else {
					tensor[c, i, j] = sum / Float(kernelSize[0] * kernelSize[1])
				}
			}
		}
	}
	return tensor
}

public class OperatorDelegateConvAvgPool2DOp: OperatorDelegateConvPool2DOp {
	override public func compare() {
		print("Run time:  \(CFAbsoluteTimeGetCurrent() - self.startTime) s")
		
		let kernelSize = self.kernelSize
		let stride = self.stride
		let padMode = self.padMode
		let inputTensors = self.veryfyTensors
		let resultTensors = self.resultTensors
		let order = self.channelOrder
		
		for (input, output) in zip(inputTensors, resultTensors) {
			let verifyTensor = avgPoolVerify(input: input, kernelSize: kernelSize, stride: stride, padMode: padMode, order: order)
			
			XCTAssertTrue(verifyTensor.shape == output.shape)
			let outReader = output.floatValueReader
			let verifyReader = verifyTensor.floatValueReader
			for i in 0..<output.count {
				XCTAssertEqual(outReader[i], verifyReader[i], accuracy: abs(verifyReader[i]*0.01), "\(outReader[i]), \(verifyReader[i])")
			}
		}
	}
}

class Avgpool2D_op_test: XCTestCase {
	
	override func setUp() {
		super.setUp()
		// Put setup code here. This method is called before the invocation of each test method in the class.
	}
	
	override func tearDown() {
		// Put teardown code here. This method is called after the invocation of each test method in the class.
		super.tearDown()
	}
	
	func test() {
		let testCase = Pool2DOperatorTest<AvgPool2DOperator, OperatorDelegateConvAvgPool2DOp>()
		testCase.testAll()
	}
	
}
