//
//  pool1d_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/20/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

public class OperatorDelegateConvPool2DOp: OperatorDelegateConv {
	public var kernelSize: [Int] = [1]
	public var stride: [Int] = [1]
	public var padMode: PaddingMode = .Valid
	public var startTime : CFAbsoluteTime = 0.0
	public var channelOrder: TensorChannelOrder = .Last
	public required override init() {
		super.init()
	}
	
	override public func compare() {
		fatalError()
	}
}

class Pool2DOperatorTest<Op: Pooling2DOperator, Delegate: OperatorDelegateConvPool2DOp>: XCTestCase {
	
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
	
	func testAll() {
		testInit()
		testOutputShape()
		testCompute()
	}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	init....
	*/
	func testInit() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)..")
			
			let label = randomString(length: 5)
			let kernelSize = [randomInt([1, 5]), randomInt([1, 5])]
			let strideSize =  [randomInt([1, 5]), randomInt([1, 5])]
			let op = Op(kernelSize: kernelSize, stride: strideSize)
			op.operatorLabel = label
			
			XCTAssert(op.kernelSize == kernelSize)
			XCTAssert(op.stride == strideSize)
			XCTAssert(op.operatorLabel == label)
			
			print("Finish Test \(i+1)\n")
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
			print("Test \(i+1)..")
			// valid num filters
			var numFilters = randomInt([1, 10])
			
			// valid kernel size
			var kernelSize = [randomInt([1, 5]), randomInt([1, 5])]
			
			// valid stride
			var stride = [randomInt([1, 5]), randomInt([1, 5])]
			
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
			
			// generate valid input shapes
			var inputShapes = [TensorShape]()
			for _ in 0..<randomInt([2, 5]) {
				if channelOrder == TensorChannelOrder.First {
					inputShapes.append(TensorShape(dataType: .int, shape: [3, randomInt([100, 120]), randomInt([100, 120])]))
				} else {
					inputShapes.append(TensorShape(dataType: .int, shape: [ randomInt([100, 120]), randomInt([100, 120]), 3]))
				}
			}
			
			// setup invalid cases
			if i % 2 != 0 {
				let randCase = randomInt([0, 5])
				if randCase == 0 {
					// input empty
					inputShapes.removeAll()
					print("Invalid case: input empty")
				} else if randCase == 1 {
					// invalid kernel size
					if randomInt([1, 100]) % 3 == 0{
						kernelSize = [3, 3, 5]
					} else {
						kernelSize = [0, -1]
					}
					print("Invalid case: invalid kernel size")
				} else if randCase == 2 {
					// invalid stride
					if randomInt([1, 100]) % 3 == 0{
						stride = [3, 3, 5]
					} else {
						stride = [0, -1]
					}
					print("Invalid case: invalid stride")
				} else if randCase == 3 {
					// invalid input shape
					var shapeArray = inputShapes[0].shapeArray
					shapeArray[0] = 0
					inputShapes[0] = TensorShape(dataType: .int, shape: shapeArray)
					print("Invalid case: invalid input shape")
				} else if randCase == 4 {
					// invalid input shape to negative
					var shapeArray = inputShapes[0].shapeArray
					shapeArray[0] = -1
					inputShapes[0] = TensorShape(dataType: .int, shape: shapeArray)
					print("Invalid case: invalid input shape")
				}
			}
			
			print("numFilters: \(numFilters)")
			print("kernelSize: \(kernelSize)")
			print("stride: \(stride)")
			print("padMode: \(padding)")
			print("channelPosition: \(channelOrder)")
			for shape in inputShapes {
				print("Input shape: \(shape.shapeArray)")
			}
			
			
			let op = Op(kernelSize: kernelSize, stride: stride, channelPosition: channelOrder, paddingMode: padding)
			let outputShape = op.outputShape(shapeArray: inputShapes)
			if i % 2 == 0 {
				XCTAssertNotNil(outputShape)
			} else {
				XCTAssertNil(outputShape)
			}
			
			print("Finish Test \(i+1)\n")
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
			print("Test \(i+1)...")
			
			// valid kernel size
			let kernelSize = [randomInt([1, 5]), randomInt([1, 5])]
			
			// valid stride
			let stride = [randomInt([1, 5]), randomInt([1, 5])]
			
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
			
			// generate input tensors
			var input:[Tensor]? = [Tensor]()
			for _ in 0..<randomInt([2, 5]) {
				input!.append(randomTensor(fromShape: inputShape))
			}
			
			let op = Op(kernelSize: kernelSize, stride: stride, channelPosition: channelOrder, paddingMode: padding)

			// generate output tensors
			let outputShape = op.outputShape(shapeArray: input!.map {$0.shape})
			var output:[Tensor]? = [Tensor]()
			for shape in outputShape! {
				output!.append(Tensor(repeatingValue: 0.0, tensorShape: shape))
			}
			
			// setup invalid cases
			if i % 2 != 0 {
				let randCase = randomInt([0, 5])
				if randCase == 0 {
					// input nil
					input = nil
					print("Invalid case: input nil")
				} else if randCase == 1 {
					// output nil
					output = nil
					print("Invalid case: output nil")
				} else if randCase == 2 {
					// input shape not valid
					var shapeArray = input![0].shape.shapeArray
					shapeArray[0] = 0
					input![0] = randomTensor(fromShape: TensorShape(dataType: .int, shape: shapeArray))
					print("Invalid case: input shape not valid")
				} else if randCase == 3 {
					// output shape count not valid
					output!.removeLast()
					print("Invalid case: output count not valid")
				} else if randCase == 4 {
					// output shape not valid
					var shapeArray = output![0].shape.shapeArray
					shapeArray[0] += randomInt([1, 5])
					output![0] = randomTensor(fromShape: TensorShape(dataType: .int, shape: shapeArray))
					print("Invalid case: output shape invalid")
				}
			}
			
			op.inputTensors = input
			op.outputTensors = output
			
			let (pass, msg) = op.inputOutputTensorsCheck()
			if i % 2 == 0 {
				XCTAssertTrue(pass)
			} else {
				XCTAssertFalse(pass)
				print(msg)
			}
			
			print("Finish Test \(i+1)\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
	public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
	internal func cpu()
	internal func gpu()
	*/
	func testCompute() {
		let numCase = 10
		
		_ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		
		let workGroup = DispatchGroup()
		let delegate = Delegate()
		delegate.dispatchGroup = workGroup
		
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// valid kernel size
			let kernelSize = [randomInt([1, 5]), randomInt([1, 5])]
			
			// valid stride
			let stride = [randomInt([1, 5]), randomInt([1, 5])]
			
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
			var size = [10, 20]
			if i >= 8 {
				size = [500, 500]
			}
			var inputShape = TensorShape(dataType: .int, shape: [randomInt(size), randomInt(size), channel])
			if channelOrder == TensorChannelOrder.First {
				inputShape = TensorShape(dataType: .int, shape: [channel, randomInt(size), randomInt(size)])
			}
			
			// generate input tensors
			var input:[Tensor]? = [Tensor]()
			input!.append(randomTensor(fromShape: inputShape))
			
			
			let op = Op(kernelSize: kernelSize, stride: stride, channelPosition: channelOrder, paddingMode: padding)
			
			// generate output tensors
			let outputShape = op.outputShape(shapeArray: input!.map {$0.shape})
			var output:[Tensor]? = [Tensor]()
			for shape in outputShape! {
				output!.append(Tensor(repeatingValue: 0.0, tensorShape: shape))
			}
			
			
			op.inputTensors = input
			op.outputTensors = output
			op.kernelSize = kernelSize
			op.stride = stride
			op.paddingMode = padding
			op.computationDelegate = delegate

			delegate.veryfyTensors = input!
			delegate.kernelSize = kernelSize
			delegate.stride = stride
			delegate.padMode = padding
			delegate.channelOrder = channelOrder
			
			print("kernelSize: \(kernelSize)")
			print("stride: \(stride)")
			print("padMode: \(padding)")
			print("channel order: \(channelOrder)")
			for inputT in input! {
				print("Input: \(inputT.shape)")
			}
			for outputT in output! {
				print("Output: \(outputT.shape)")
			}
			
			
			if i % 2 == 0 {
				print("Run on CPU")
				workGroup.enter()
				delegate.startTime = CFAbsoluteTimeGetCurrent()
//				op.computeAsync(.GPU)
				op.computeAsync(.CPU)
			} else {
				if !SerranoEngine.configuredEngine.hasAvailableGPU() {
					print("No GPU available. Give up test.\n")
					continue
				}
				print("Run on GPU")
				workGroup.enter()
				delegate.startTime = CFAbsoluteTimeGetCurrent()
				op.computeAsync(.GPU)
			}
			
			workGroup.wait()
			
			SerranoResourceManager.globalManager.releaseAllResources()
			
			print("Finish Test \(i+1)\n")
		}
	}
}


