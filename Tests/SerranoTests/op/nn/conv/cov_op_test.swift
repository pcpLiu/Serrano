//
//  cov_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 10/17/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano


/// Do naive  convolution to get result
func naiveConvVerify(input: Tensor, weight: Tensor, outputShape: TensorShape, pad: PaddingMode, stride: [Int], channelOrder: TensorChannelOrder, kernelSize: [Int]) -> Tensor {
	let tensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(outputShape)
	let (channel, inHeight, inWidth) = parseImgChannelShapeInfo(channelOrder, shapeArray: input.shape.shapeArray)
	let numFilter = weight.shape.shapeArray[0]
	
	// out  bounary
	let outHeight = kernelScanningOutSize(pad, inputSize: inHeight, kernelSize: kernelSize[0], stride: stride[0])
	let outWidth = kernelScanningOutSize(pad, inputSize: inWidth, kernelSize: kernelSize[1], stride: stride[1])
	
	// calculate
	for h in 0..<outHeight {
		for w in 0..<outWidth {
			for x in 0..<kernelSize[0] {
				for y in 0..<kernelSize[1] {
					for n in 0..<numFilter {
						for c in 0..<channel {
							if channelOrder == .First {
								tensor[h, w, n] += input.fetchValueOrDefault([c, h*stride[0]+x, w*stride[1]+y], missingValue: Float(0.0)) * weight[n, c, x, y]
							} else {
								tensor[h, w, n] += input.fetchValueOrDefault([h*stride[0]+x, w*stride[1]+y, c], missingValue: Float(0.0)) * weight[n, c, x, y]
							}
						}
					}
				}
			}
		}
	}
	
	return tensor
}

class OperatorDelegateConv2D: OperatorDelegateConv {
	
	public var op: ConvOperator2D? = nil
	public var startTime: CFAbsoluteTime? = nil
	override public func compare() {
		print("Calculation time: \(CFAbsoluteTimeGetCurrent() - self.startTime!) s")
		
		for (input, output) in zip(op!.inputTensors!, op!.outputTensors!) {
			let veryTensor = naiveConvVerify(input: input, weight: self.op!.weight!,
			                                 outputShape: output.shape, pad: op!.padMode, stride: op!.stride, channelOrder: op!.channelPosition,
			                                 kernelSize: op!.kernelSize)
			let outReader = output.floatValueReader
			let verifyReader = veryTensor.floatValueReader
			for i in 0..<output.count {
				XCTAssertEqual(outReader[i], verifyReader[i], accuracy: abs(verifyReader[i]*0.001))
			}
		}
	}
	
}

class CovOpTest: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
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
			let numFilters = randomInt([1, 5])
			let kernelSize = [randomInt([1, 5]), randomInt([1, 5])]
			let paddingMode = PaddingMode.Same
			let imgOrder = TensorChannelOrder.Last
			let dilation = [randomInt([1, 5]), randomInt([1, 5])]
			let inputShape = randomShape(dimensions: 3, dimensionSizeRange: [1, 5], dataType: .float)
			
			let convOp = ConvOperator2D(numFilters: numFilters, kernelSize: kernelSize,
			                            padMode: paddingMode, channelPosition: imgOrder,
			                            weight: nil, dilation: dilation,
			                            computationDelegate: nil,
			                            inputTensors: nil, outputTensors: nil,
			                            operatorLabel: "d", inputShape: inputShape)
			
			XCTAssertEqual(numFilters, convOp.numFilters)
			XCTAssertEqual(kernelSize, convOp.kernelSize)
			XCTAssertEqual(paddingMode, convOp.padMode)
			XCTAssertEqual(dilation, convOp.dilation)
			XCTAssertEqual(inputShape, convOp.inputShape)
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
			print("Test case \(i+1)...")
			
			// valid num filters
			var numFilters = randomInt([1, 10])
			
			// valid kernel size
			var kernelSize = [randomInt([1, 5]), randomInt([1, 5])]
			
			// valid stride
			var stride = [randomInt([1, 5]), randomInt([1, 5])]
			
			// valid diliation
			// TODO: Later use different value when implemented diliation operation
			var diliation = [1, 1]
			
			// generate valid input shapes
			let shape = randomShape(dimensions: 3, dimensionSizeRange: [1, 10], dataType: TensorDataType.int)
			var inputShapes = [TensorShape]()
			for _ in 0..<randomInt([2, 5]) {
				inputShapes.append(TensorShape(dataType: .int, shape: shape.shapeArray))
			}
			
			// padding mode
			var padding = PaddingMode.Same
			if i % 3 == 0 {
				padding = PaddingMode.Valid
			}
			
			// channel order
			var channelOrder = TensorChannelOrder.Last
			if i % 4 == 0 {
				channelOrder = TensorChannelOrder.First
			}
			
			// make invalid case
			if i % 2 != 0 {
				let randCase = randomInt([0, 7])
				if randCase == 0 {
					// input shape empty
					inputShapes = [TensorShape]()
					print("Invalid case: Input shapes empty")
				} else if randCase == 1 {
					// invalid number of filters
					numFilters = 0
					print("Invalid case: invalid number of filters")
				} else if randCase == 2 {
					// invalid kernel size
					if randomInt([1, 100]) % 3 == 0{
						kernelSize = [3, 3, 5]
					} else {
						kernelSize = [0, -1]
					}
					print("Invalid case: invalid kernel size")
				} else if randCase == 3 {
					// invalid stride
					if randomInt([1, 100]) % 3 == 0{
						stride = [3, 3, 5]
					} else {
						stride = [0, -1]
					}
					print("Invalid case: invalid stride")
				} else if randCase == 4 {
					// deliation check. Skip now.
					diliation = [0, 1]
					print("Invalid case: invalid diliation")
				} else if randCase == 5 {
					// input shapes not all same shape
					var shapeArray = inputShapes.first!.shapeArray
					shapeArray[0] += randomInt([3, 7])
					inputShapes[0] = TensorShape(dataType: .int, shape: shapeArray)
					print("Invalid case: input shapes not equal")
				} else if randCase == 6 {
					// invalid input shape
					var shapeArray = inputShapes.first!.shapeArray
					if randomInt([1, 100]) % 3 == 0 {
						shapeArray.append(randomInt([3, 7]))
					} else {
						shapeArray[0] = -1
					}
					inputShapes[0] = TensorShape(dataType: .int, shape: shapeArray)
					print("Ivalid case: input shape not valid")
				}
			}
			
			print("numFilters: \(numFilters)")
			print("kernelSize: \(kernelSize)")
			print("stride: \(stride)")
			print("padMode: \(padding)")
			print("channelPosition: \(channelOrder)")
			print("dilation: \(diliation)")
			for shape in inputShapes {
				print("Input shape: \(shape.shapeArray)")
			}
			
			let convOp = ConvOperator2D(numFilters: numFilters, kernelSize: kernelSize, stride: stride, padMode: padding, channelPosition: channelOrder, dilation: diliation)
			let outShapes = convOp.outputShape(shapeArray: inputShapes)
			if i % 2 == 0 {
				XCTAssertNotNil(outShapes)
				for (inputShape, outputShape) in zip(inputShapes, outShapes!) {
					let (_, height, width) = parseImgChannelShapeInfo(convOp.channelPosition, shapeArray: inputShape.shapeArray)
					let outShapeCheckArray = [
					                     kernelScanningOutSize(convOp.padMode, inputSize: height,
					                                           kernelSize: convOp.kernelSize[0], stride: convOp.stride[0]),
					                     kernelScanningOutSize(convOp.padMode, inputSize: width,
					                                           kernelSize: convOp.kernelSize[1], stride: convOp.stride[1]),
					                     convOp.numFilters]
					XCTAssertEqual(outShapeCheckArray, outputShape.shapeArray)
				}
			} else {
				XCTAssertNil(outShapes)
			}
			
			print("Finish Test case \(i+1)\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
	*/
	func testInputOutputTensorsCheck() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test case \(i+1)...")
			
			// valid num filters
			let numFilters = randomInt([1, 10])
			
			// valid kernel size
			let kernelSize = [randomInt([1, 5]), randomInt([1, 5])]
			
			// valid stride
			let stride = [randomInt([1, 5]), randomInt([1, 5])]
			
			// valid diliation
			let diliation = [1, 1]
			
			// padding mode
			var padding = PaddingMode.Same
			if i % 3 == 0 {
				padding = PaddingMode.Valid
			}
			
			// channel order
			var channelOrder = TensorChannelOrder.Last
			if i % 4 == 0 {
				channelOrder = TensorChannelOrder.First
			}
			
			// decide input shape
			let channel = randomInt([1, 4])
			var inputShape = TensorShape(dataType: .int, shape: [randomInt([10, 20]), randomInt([10, 20]), channel])
			if channelOrder == TensorChannelOrder.First {
				inputShape = TensorShape(dataType: .int, shape: [channel, randomInt([10, 20]), randomInt([10, 20])])
			}
			
			// conv op
			let convOp = ConvOperator2D(numFilters: numFilters, kernelSize: kernelSize, stride: stride,
			                            padMode: padding, channelPosition: channelOrder, dilation: diliation)
			
			// generate input tensors
			var input:[Tensor]? = [Tensor]()
			for _ in 0..<randomInt([2, 5]) {
				input!.append(randomTensor(fromShape: inputShape))
			}
			
			// generate output tensors
			let outputShape = convOp.outputShape(shapeArray: input!.map {$0.shape})
			var output:[Tensor]? = [Tensor]()
			for shape in outputShape! {
				output!.append(Tensor(repeatingValue: 0.0, tensorShape: shape))
			}
			
			// weight
			var weight:Tensor? = randomTensor(fromShape: TensorShape(dataType: .int,
			                                                         shape: [numFilters, channel, kernelSize[0], kernelSize[1]]))
			
			// setup invalid cases
			if i % 2 != 0 {
				let randCase = randomInt([0, 7])
				if randCase == 0 {
					// input nil
					input = nil
					print("Invalid case: input nil")
				} else if randCase == 1 {
					// output nil
					output = nil
					print("Invalid case: output nil")
				} else if randCase == 2 {
					// weight nil
					weight = nil
					print("Invalid case: weight nil")
				} else if randCase == 3 {
					// weight shape invalid
					weight = randomTensor(fromShape: TensorShape(dataType: .int, shape: [3, kernelSize[0] + randomInt([1, 3]),
					                                                                     kernelSize[1] + randomInt([1, 3]), numFilters]))
					print("Invalid case: weight shape not valid")
				} else if randCase == 4 {
					// input shape not valid
					var shapeArray = input![0].shape.shapeArray
					shapeArray[0] += randomInt([1, 5])
					input![0] = randomTensor(fromShape: TensorShape(dataType: .int, shape: shapeArray))
					print("Invalid case: input shape not valid")
				} else if randCase == 5 {
					// output count invalid
					output!.removeLast()
					print("Invalid case: output count invalid")
				} else {
					// output shape invalid
					var shapeArray = output![0].shape.shapeArray
					shapeArray[0] += randomInt([1, 5])
					output![0] = randomTensor(fromShape: TensorShape(dataType: .int, shape: shapeArray))
					print("Invalid case: output shape invalid")
				}
			}
			
			convOp.inputTensors = input
			convOp.outputTensors = output
			convOp.weight = weight

			if convOp.inputTensors != nil {
				for t in convOp.inputTensors! {
					print("Input tensor: \(t.description)")
				}
			}
			if convOp.outputTensors != nil {
				for t in convOp.outputTensors! {
					print("Output tensor: \(t.description)")
				}
			}
			if weight != nil {
				print("Weight: \(convOp.weight!.description)")
			}
			
			let (pass, msg) = convOp.inputOutputTensorsCheck()
			if i % 2 == 0 {
				print(msg)
				XCTAssertTrue(pass)
			} else {
				XCTAssertFalse(pass)
				print(msg)
			}
			
			print("Finish Test case \(i+1)\n")
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
		let delegate = OperatorDelegateConv2D()
		delegate.dispatchGroup = workGroup
		
		let _ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		print(SerranoEngine.configuredEngine.GPUDevice!)
		
		for i in 0..<numCase {
			print("Test case \(i+1)...")
			
			// valid num filters
			let numFilters = randomInt([1, 10])
			
			// valid kernel size
			let kernelSize = [randomInt([1, 5]), randomInt([1, 5])]
			
			// valid stride
			let stride = [randomInt([1, 5]), randomInt([1, 5])]
			
			// valid diliation
			let diliation = [1, 1]
			
			// padding mode
			var padding = PaddingMode.Same
			if i % 3 == 0 {
				padding = PaddingMode.Valid
			}
			
			// channel order
			var channelOrder = TensorChannelOrder.Last
			if i % 4 == 0 {
				channelOrder = TensorChannelOrder.First
			}
			
			// decide input shape
			var range = [50, 50]
			if i >= 8 {
				range = [800, 800]
			}
			let channel = randomInt([1, 4])
			var inputShape = TensorShape(dataType: .int, shape: [randomInt(range), randomInt(range), channel])
			if channelOrder == TensorChannelOrder.First {
				inputShape = TensorShape(dataType: .int, shape: [channel, randomInt(range), randomInt(range)])
			}
			
			// conv op
			let convOp = ConvOperator2D(numFilters: numFilters, kernelSize: kernelSize, stride: stride,
										padMode: padding, channelPosition: channelOrder, dilation: diliation)
			convOp.computationDelegate = delegate
			
			// generate input tensors
			var input = [Tensor]()
			input.append(randomTensor(fromShape: inputShape))
			
			
			// generate output tensors
			let outputShape = convOp.outputShape(shapeArray: input.map {$0.shape})
			var output = [Tensor]()
			for shape in outputShape! {
				output.append(Tensor(repeatingValue: 0.0, tensorShape: shape))
			}
			
			// weight
			let weight:Tensor = randomTensor(fromShape: TensorShape(dataType: .int,
			                                                        shape: [numFilters, channel, kernelSize[0], kernelSize[1]]))
		
			convOp.inputTensors = input
			convOp.outputTensors = output
			convOp.weight = weight
			delegate.op = convOp
			
			if i % 2 == 0 {
				workGroup.enter()
				print("Run on CPU")
				delegate.startTime = CFAbsoluteTimeGetCurrent()
				convOp.computeAsync(.CPU)
			} else {
				if !SerranoEngine.configuredEngine.hasAvailableGPU() {
					print("No GPU available. Give up test.\n")
					continue
				}
				workGroup.enter()
				print("Run on GPU")
				delegate.startTime = CFAbsoluteTimeGetCurrent()
				convOp.computeAsync(.GPU)
			}
			
			workGroup.wait()
			
			print("Finish test case \(i+1)\n")

		}
	}
	
}
