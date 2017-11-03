//
//  transpose_op.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/23/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

public class OperatorDelegateConvTransposeOperator: OperatorDelegateConv {
	
	override public func compare() {
		XCTAssertTrue(self.resultTensors.count == self.veryfyTensors.count)
		
		for tensorIndex in 0..<self.veryfyTensors.count {
			let resultTensor = self.resultTensors[tensorIndex]
			let inputTensor = self.veryfyTensors[tensorIndex]
			XCTAssertTrue(resultTensor.rank == 2)
			
			let resultShapeArray = resultTensor.shape.shapeArray
			let inputShapeArray = inputTensor.shape.shapeArray
			XCTAssertTrue(resultShapeArray[0] == inputShapeArray[1] && resultShapeArray[1] == inputShapeArray[0])
			
			for i in 0..<inputShapeArray[0] {
				for j in 0..<inputShapeArray[1] {
					XCTAssertEqualWithAccuracy(inputTensor[[i, j]], resultTensor[[j, i]], accuracy: abs(inputTensor[[i, j]]*0.001))
				}
			}
		}
	}
}

class transpose_op: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
//    func testExample() {
//        // This is an example of a functional test case.
//        // Use XCTAssert and related functions to verify your tests produce the correct results.
//    }
//    
//    func testPerformanceExample() {
//        // This is an example of a performance test case.
//        self.measure {
//            // Put the code you want to measure the time of here.
//        }
//    }
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target
		init
	*/
	func testInit() {
		let numCase = 100
		let op = TransposeOperator()
		for _ in 0..<numCase {
			let label = randomString(length: randomInt([2, 10]))
			op.operatorLabel = label
			XCTAssertEqual(label, op.operatorLabel)
			print("label: \(label), \(op.operatorLabel)")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
	*/
	func testOutputShape() {
		let numCase = 100
		let op = TransposeOperator()
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			var inputShapes = [TensorShape]()
			for _ in 0..<randomInt([3, 5]) {
				if i % 2 == 0 {
					// valid
					inputShapes.append(randomShape(dimensions: 2, dimensionSizeRange: [1, 10], dataType: .float))
				} else {
					// invalid
					if i % 3 == 0 {
						// nil
						continue
					} else {
						// rank is not 2
						inputShapes.append(randomShape(dimensions: randomInt([1, 10]), dimensionSizeRange: [1, 10], dataType: .float))
					}
				}
			}
			
			let outputShapes = op.outputShape(shapeArray: inputShapes)
			if i % 2 == 0 {
				XCTAssertNotNil(outputShapes)
				XCTAssertEqual(outputShapes!.count, inputShapes.count)
				for shapeIndex in 0..<outputShapes!.count {
					let outputShape = outputShapes![shapeIndex]
					let inputShape = inputShapes[shapeIndex]
					XCTAssertEqual(outputShape.shapeArray[0], inputShape.shapeArray[1])
					XCTAssertEqual(outputShape.shapeArray[1], inputShape.shapeArray[0])
				}
			} else {
				XCTAssertNil(outputShapes)
			}
			
			print("Finish test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
	*/
	func testInputOuputTensorsCheck() {
		let numCase = 100
		let op = TransposeOperator()
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			var inputTensors = [Tensor]()
			var outputTensors = [Tensor]()
			
			var skipInputAssgin = false
			var skipOutputAssgin = false
			
			if i % 2 == 0 {
				// valid
				for _ in 0..<randomInt([1, 4]) {
					inputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [1, 10], dataType: .float))
					let shape = inputTensors.last!.shape
					outputTensors.append(randomTensor(fromShape: TensorShape(dataType: .float, shape: [shape.shapeArray[1], shape.shapeArray[0]])))
				}
			} else {
				// invalid
				if i % 4 == 0 {
					// input tensors nil
					skipInputAssgin = true
					// output tensors nil
					for _ in 0..<randomInt([1, 4]) {
						outputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [1, 10], dataType: .float))
					}
				} else if i % 4 == 1 {
					// output tensors nil
					for _ in 0..<randomInt([1, 4]) {
						inputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [1, 10], dataType: .float))
					}
					skipOutputAssgin = true
				} else if i % 4 == 2 {
					// input tensor, output tensor not same count
					for _ in 0..<randomInt([1, 4]) {
						inputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [1, 10], dataType: .float))
					}
					
					for _ in 0..<randomInt([5, 7]) {
						outputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [1, 10], dataType: .float))
					}
				} else {
					// input tensors, output tensors shape not match
					for _ in 0..<randomInt([1, 4]) {
						inputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [1, 10], dataType: .float))
						let shape = inputTensors.last!.shape
						outputTensors.append(randomTensor(fromShape: TensorShape(dataType: .float,
						                                                         shape: [shape.shapeArray[1] + randomInt([1, 3]),
						                                                                 shape.shapeArray[0] + randomInt([1, 3])])))
					}
				}
			}
			
			
			
			if !skipInputAssgin {
				op.inputTensors = inputTensors
				print("Assign input tensors:")
				for tensor in inputTensors {
					print("Tensor: \(tensor.description)")
				}
				
			}
			
			if !skipOutputAssgin {
				op.outputTensors = outputTensors
				print("Assig  output tensors:")
				for tensor in outputTensors {
					print("Tensor: \(tensor.description)")
				}
			}
			
			let (pass, msg) = op.inputOutputTensorsCheck()
			if i % 2 == 0 {
				XCTAssertTrue(pass)
			} else {
				XCTAssertTrue(!pass)
				print(msg)
			}
			
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
	public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
	internal func cpu()
	internal func gpu()
	*/
	func testCompute() {
		let caseNum = 10
		let op = TransposeOperator()
		
		// configure engine
		let (_, _) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
		
		// setup delegate
		let delegate = OperatorDelegateConvTransposeOperator()
		let workingGroup = DispatchGroup()
		delegate.dispatchGroup = workingGroup
		op.computationDelegate = delegate
		
		for i in 0..<caseNum {
			print("Test \(i+1)...")
			
			// generate tensors
			var inputTensors = [Tensor]()
			var outputTensors = [Tensor]()
			if i < 8 {
				for _ in 0..<randomInt([1, 3]) {
					inputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float))
					let shape = inputTensors.last!.shape
					outputTensors.append(randomTensor(fromShape: TensorShape(dataType: .float, shape: [shape.shapeArray[1], shape.shapeArray[0]])))
				}
			} else {
				// larger tensors
				for _ in 0..<1 {
					inputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [1000, 1500], dataType: .float))
					let shape = inputTensors.last!.shape
					outputTensors.append(randomTensor(fromShape: TensorShape(dataType: .float, shape: [shape.shapeArray[1], shape.shapeArray[0]])))
				}
			}
			
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			delegate.veryfyTensors = inputTensors
			
			if i % 2 == 0 {
				print("Run on CPU")
				workingGroup.enter()
				op.computeAsync( .CPU)
			} else {
				print("Run on GPU")
				if !SerranoEngine.configuredEngine.hasAvailableGPU() {
					print("No gpu available, give up Test \(i+1)\n\n\n)")
					continue
				}
				workingGroup.enter()
				op.computeAsync( .GPU)
			}
			
			workingGroup.wait()
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish test \(i+1)\n\n")
		}
	}
	
}
