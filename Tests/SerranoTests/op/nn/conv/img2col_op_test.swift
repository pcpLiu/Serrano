//
//  img2col_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class OperatorDelegateConvImg2Col: OperatorDelegateConv {
	
	public var op: Img2ColOperator? = nil
	
	override public func compare() {
		XCTAssertNotNil(self.op)
		
		let inputTensors = self.op!.inputTensors!
		let patchHeight = self.op!.patchSize[0]
		let patchWidth = self.op!.patchSize[1]
		let strideHeight = self.op!.stride[0]
		let strideWidth = self.op!.stride[1]
		let patchElementCount = patchWidth * patchHeight
		
		// compute validation tensors
		var checkOutputTensors = [Tensor]()
		for inTensor in inputTensors {
			let outShape = self.op!.outputShape(shapeArray: [inTensor.shape])!.first!
			let checkTensor = Tensor(repeatingValue: -1.0, tensorShape: outShape)
			
			let inShapeArray = inTensor.shape.shapeArray
			let (channels, inHeight, inWidth) = parseImgChannelShapeInfo(self.op!.channelPosition, shapeArray: inShapeArray)
			
			// get out dim size
			let outHeight = kernelScanningOutSize(self.op!.padMode, inputSize: inHeight, kernelSize: patchHeight, stride: strideHeight)
			let outWidth = kernelScanningOutSize(self.op!.padMode, inputSize: inWidth, kernelSize: patchWidth, stride: strideWidth)
			
			for i in 0..<outHeight {
				for j in 0..<outWidth {
					for channelIndex in 0..<channels {
						for m in 0..<patchHeight {
							for n in 0..<patchWidth {
								let outIndex = [i * outWidth + j,
								                channelIndex * patchElementCount + (m * patchWidth + n)]
								if self.op!.channelPosition == .First {
									checkTensor[outIndex] = inTensor.fetchValueOrDefault([channelIndex, i*strideHeight + m, j*strideWidth + n],
									                                                   missingValue: self.op!.paddingValue)
								} else if self.op!.channelPosition == .Last {
									checkTensor[outIndex] = inTensor.fetchValueOrDefault([i*strideHeight + m, j*strideWidth + n, channelIndex],
									                                                   missingValue: self.op!.paddingValue)
								} else {
									fatalError("Not implemented")
								}
							}
						}
					}
				}
			}
			
			checkOutputTensors.append(checkTensor)
		}
		
		// validate
		for (checkTensor, outTensor) in zip(checkOutputTensors, self.resultTensors) {
//			print("outTensor: ", outTensor.nestedArrayFloat())
//			print("checkTensor: ", checkTensor.nestedArrayFloat())
//			XCTAssert(checkTensor.count == outTensor.count)
			let checkReader = checkTensor.floatValueReader
			let outReader = outTensor.floatValueReader
			for i in 0..<checkTensor.count {
				if abs(checkReader[i]) < 0.001 {
					XCTAssertEqualWithAccuracy(checkReader[i], outReader[i], accuracy: abs(checkReader[i]))
				} else {
					XCTAssertEqualWithAccuracy(checkReader[i], outReader[i], accuracy: abs(checkReader[i]) * 0.001)
				}
			}
		}
 	}
}


class Img2ColOpTest: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
	
	func makeValidPatchSize(_ range: [Int], same: Bool = true) -> [Int] {
		if same {
			let size = randomInt(range)
			return [size, size]
		} else {
			return [randomInt(range), randomInt(range)]
		}
	}
	
	func makeValidStride(_ range: [Int], same: Bool = true) -> [Int] {
		if same {
			let size = randomInt(range)
			return [size, size]
		} else {
			return [randomInt(range), randomInt(range)]
		}
	}
    
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	init
	*/
	func testInit() {
		let numCase = 100
		
		for i in 0..<numCase {
			print("Test case \(i+1)...")
			let patchWidth = randomInt([1, 5])
			let patchHeight = randomInt([1, 5])
			let strideWidth = randomInt([1, 5])
			let strideHeight = randomInt([1, 5])
			
			let op = Img2ColOperator(patchSize: [patchHeight, patchWidth], stride: [strideHeight, strideWidth], channelPosition: .First, padMode: .Same)
			
			XCTAssert(patchWidth == op.patchSize[1])
			XCTAssert(patchHeight == op.patchSize[0])
			
			XCTAssert(strideWidth == op.stride[1])
			XCTAssert(strideHeight == op.stride[0])
			
			print("FINISH Test case \(i+1)...\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
	*/
	func testOutputShape() {
		let numCase = 100
		
		for i in 0..<numCase {
			// make valid patch size
			var patchSize = makeValidPatchSize([2, 5])
			
			// make valid strides
			var stride = makeValidStride([2, 5])
			
			// make valid input shapes
			var inputShapes = [TensorShape]()
			for _ in 0..<randomInt([1, 4]) {
				inputShapes.append(TensorShape(dataType: .float, shape: [3, randomInt([5, 10]), randomInt([5, 10])]))
			}
			
			// mode
			var paddingMode = PaddingMode.Valid
			if i % 3 == 0 {
				paddingMode = .Same
			}
			
			// order, 
			let channelOrder = TensorChannelOrder.First
			
			// set invalid cases
			if i % 2 != 0 {
				let randCase = randomInt([0, 4]) % 4
				if randCase == 0 {
					// invalid patch size
					if i % 3 == 0 {
						patchSize.append(randomInt([1, 3]))
					} else {
						patchSize[1] = 0
					}
					print("Invalid patch size: ", patchSize)
				} else if randCase == 1 {
					// invalid stride
					if i % 3 == 0 {
						stride.append(randomInt([1, 3]))
					} else {
						stride[1] = 0
					}
					print("Invalid stride: ", stride)
				} else if randCase == 2 {
					// invalid inshape
					var shapeArray = inputShapes[0].shapeArray
					if i % 3 == 0 {
						shapeArray.append(randomInt([1, 3]))
					} else {
						shapeArray[randomInt([0, 3])] = 0
					}
					inputShapes[0] = TensorShape(dataType: .float, shape: shapeArray)
					print("Invalid inshape: ", inputShapes[0].description)
				} else {
					// valid mode, not enouhg inputs
					paddingMode = .Valid
					let shapeArray = [patchSize[0] - 1, patchSize[1] - 1]
					inputShapes[0] = TensorShape(dataType: .float, shape: shapeArray)
					print("not enouhg inputs: ", inputShapes[0].description)
					
				}
			}
			
			// print
			print("patchSize: ", patchSize)
			print("stride: ", stride)
			print("channelOrder: ", channelOrder)
			print("paddingMode: ", paddingMode)
			for shape in inputShapes {
				print("Input shape: ", shape.description)
			}
			
			let op = Img2ColOperator(patchSize: patchSize, stride: stride, channelPosition: channelOrder, padMode: paddingMode)
			let outputShapes = op.outputShape(shapeArray: inputShapes)
			if i % 2 == 0 {
				XCTAssertNotNil(outputShapes)
				XCTAssertEqual(outputShapes!.count, inputShapes.count)
				for (inShape, outShape) in zip(inputShapes, outputShapes!) {
					// check
					let (channels, height, width) = parseImgChannelShapeInfo(op.channelPosition, shapeArray: inShape.shapeArray)
					let outHeight = kernelScanningOutSize(op.padMode, inputSize: height, kernelSize: op.patchSize[0], stride: op.stride[0])
					let outWidth = kernelScanningOutSize(op.padMode, inputSize: width, kernelSize: op.patchSize[1], stride: op.stride[1])
					XCTAssertEqual(outShape.shapeArray[0], outHeight * outWidth)
					XCTAssertEqual(outShape.shapeArray[1], channels * op.patchSize[0] * op.patchSize[1])
				}
			} else {
				XCTAssertNil(outputShapes)
			}
			
			print("Finish test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
	*/
	func testInputOutputTensorsCheck() {
		let numCase = 1
		
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// make valid patch size
			let patchSize = makeValidPatchSize([2, 5])
			
			// make valid strides
			let stride = makeValidStride([2, 5])
			
			// mode
			var paddingMode = PaddingMode.Valid
			if i % 3 == 0 {
				paddingMode = .Same
			}
			
			// order
			let channelOrder = TensorChannelOrder.First
			
			// generate valid input tensors
			var inputTensors: [Tensor]? = [Tensor]()
			for _ in 0..<randomInt([1 , 4]) {
				let shape = TensorShape(dataType: .float, shape: [3, randomInt([5, 10]), randomInt([5, 10])])
				inputTensors!.append(randomTensor(fromShape: shape))
			}
			
			// generate valid output tensors
			var outputTensors: [Tensor]? = [Tensor]()
			for inputTensor in inputTensors! {
				let (channels, height, width) = parseImgChannelShapeInfo(channelOrder, shapeArray: inputTensor.shape.shapeArray)
				let outHeight = kernelScanningOutSize(paddingMode, inputSize: height, kernelSize: patchSize[0], stride: stride[0])
				let outWidth = kernelScanningOutSize(paddingMode, inputSize: width, kernelSize: patchSize[1], stride: stride[1])
				let outputShapeArray = [outHeight * outWidth, channels * patchSize[0] * patchSize[1]]
				let shape = TensorShape(dataType: .float, shape: outputShapeArray)
				outputTensors!.append(randomTensor(fromShape: shape))
			}
			
			// setup invalid cases
			if i % 2 != 0 {
				let randCase = randomInt([0, 5])
				if randCase == 0 {
					// input nil
					print("input nil")
					inputTensors = nil
				} else if randCase == 1 {
					// output nil
					print("output nil")
					outputTensors = nil
				} else if randCase == 2 {
					// input not valid
					print("input not valid")
					let newTensor = randomTensor(fromShape: TensorShape(dataType: .float, shape: [1,2]))
					inputTensors![0] = newTensor
				} else if randCase == 3 {
					// output shape not same count
					print("output shape not same count")
					outputTensors!.removeLast()
				} else {
					// output shape not valid
					print("output shape not valid")
					let newTensor = randomTensor(fromShape: TensorShape(dataType: .float, shape: [1,2]))
					outputTensors![0] = newTensor
				}
			}
			
			// print information
			print("patchSize: ", patchSize)
			print("stride: ", stride)
			print("paddingMode: ", paddingMode)
			print("channelOrder: ", channelOrder)
			if inputTensors != nil {
				for tensor in inputTensors! {
					print("Input tensor: ", tensor.description)
				}
			}
			if outputTensors != nil {
				for tensor in outputTensors! {
					print("Output tensor: ", tensor.description)
				}
			}
			
			let op = Img2ColOperator(patchSize: patchSize, stride: stride, channelPosition: channelOrder, padMode: paddingMode)
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			
			let (pass, msg) = op.inputOutputTensorsCheck()
			if i % 2 == 0 {
				XCTAssertTrue(pass)
			} else {
				XCTAssertFalse(pass)
				print(msg)
			}
			
			print("Finish test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func compute(_ computationMode: OperatorComputationMode)
	public func computeAsync(_ computationMode: OperatorComputationMode)
	*/
	func testCompute() {
		let numCase = 10
		
		let workGroup = DispatchGroup()
		let delegate = OperatorDelegateConvImg2Col()
		delegate.dispatchGroup = workGroup
		
		let _ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// make valid patch size
			let patchSize = makeValidPatchSize([1, 5])
			
			// make valid strides
			let stride = makeValidStride([1, 4])
//			let stride = [2, 2]
			
			// mode
			var paddingMode = PaddingMode.Valid
			if i % 3 == 0 {
				paddingMode = .Same
			}
			
			// order
			var channelOrder = TensorChannelOrder.First
			if i % 3 != 0 {
				channelOrder = .Last
			}
			
			// generate valid input tensors
			var inputTensors: [Tensor]? = [Tensor]()
			if i < 8 {
				var shape = TensorShape(dataType: .float, shape: [3, randomInt([20, 30]), randomInt([20, 30])])
				if channelOrder == .Last {
					 shape = TensorShape(dataType: .float, shape: [randomInt([20, 30]), randomInt([20, 30]), 3])
				}
				inputTensors!.append(randomTensor(fromShape: shape))
			} else {
				var shape = TensorShape(dataType: .float, shape: [3, randomInt([500, 500]), randomInt([500, 500])])
				if channelOrder == .Last {
					shape = TensorShape(dataType: .float, shape: [randomInt([500, 500]), randomInt([500, 500]), 3])
				}
				inputTensors!.append(randomTensor(fromShape: shape))
			}
			
			// generate valid output tensors
			var outputTensors: [Tensor]? = [Tensor]()
			for inputTensor in inputTensors! {
				let (channels, height, width) = parseImgChannelShapeInfo(channelOrder, shapeArray: inputTensor.shape.shapeArray)
				let outHeight = kernelScanningOutSize(paddingMode, inputSize: height, kernelSize: patchSize[0], stride: stride[0])
				let outWidth = kernelScanningOutSize(paddingMode, inputSize: width, kernelSize: patchSize[1], stride: stride[1])
				let outputShapeArray = [outHeight * outWidth, channels * patchSize[0] * patchSize[1]]
				let shape = TensorShape(dataType: .int, shape: outputShapeArray)
				outputTensors!.append(Tensor(repeatingValue: -1.0, tensorShape: shape))
				print("outWidth: ", outWidth)
			}
			
			// print information
			print("patchSize: ", patchSize)
			print("stride: ", stride)
			print("paddingMode: ", paddingMode)
			print("channelOrder: ", channelOrder)
			for tensor in inputTensors! {
				print("Input tensor: ", tensor.description)
//				print("Input tensor: ", tensor.nestedArrayFloat())
			}
			for tensor in outputTensors! {
				print("Output tensor: ", tensor.description)
			}
			
			let op = Img2ColOperator(patchSize: patchSize, stride: stride, channelPosition: channelOrder, padMode: paddingMode)
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			op.computationDelegate = delegate
			delegate.op = op
			op.paddingValue = 0.0
			
			var startTime: CFAbsoluteTime = 0.0
			if i % 2 == 0 {
				workGroup.enter()
				print("Run on CPU")
				startTime = CFAbsoluteTimeGetCurrent()
				op.computeAsync(.CPU)
			} else {
				if !SerranoEngine.configuredEngine.hasAvailableGPU() {
					print("No GPU available. Give up test.\n")
					continue
				}
				workGroup.enter()
				startTime = CFAbsoluteTimeGetCurrent()
				print("Run on GPU")
				op.computeAsync(.GPU)
			}
			
			workGroup.wait()
			
			let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
			print("Time elapsed: \(timeElapsed) s.")
			
			SerranoResourceManager.globalManager.releaseAllResources()
			
			print("Finish test \(i+1)\n\n")
		}
	}

}
