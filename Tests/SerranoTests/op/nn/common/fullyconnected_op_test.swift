//
//  fullyconnected_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano
import Dispatch

import Accelerate

public class OperatorDelegateConvFullyConnctOp: OperatorDelegateConv {
	
	override public func compare() {

		// Compare resutl
		let inputTensor = self.veryfyTensors[0]
		let weightTensor = self.veryfyTensors[1]
		let biasTensor = self.veryfyTensors[2]
		let tensorC = self.resultTensors[0]
		
	
		let readA = inputTensor.contentsAddress
		let readB = weightTensor.contentsAddress
		let verifyTensor = SerranoResourceManager.globalManager.allocateTensors( [tensorC.shape]).first!
		let verifyAddres = verifyTensor.contentsAddress
		let M = Int32(1)
		let N = Int32(weightTensor.shape.shapeArray[1])
		let K = Int32(inputTensor.count)
		cblas_sgemm(CblasRowMajor, cblasTrans(false), cblasTrans(false), M, N, K,
		            1.0, readA, K, readB, N, 0.0, verifyAddres, N)
		// plus bias
		vDSP_vadd(verifyAddres, 1, biasTensor.contentsAddress, 1, verifyAddres, 1, vDSP_Length(verifyTensor.count))
		
		
		let verifyReader = verifyTensor.floatValueReader
		let readC = tensorC.floatValueReader
		for i in 0..<tensorC.count {
			if verifyReader[i].isInfinite { continue }
			XCTAssertEqualWithAccuracy(verifyReader[i], readC[i], accuracy: abs(verifyReader[i]*0.001), "\(biasTensor.floatValueReader[i])")
		}
		
	}
}


class FullyconnectedOpTest: XCTestCase {
    
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
	init....
	*/
	func testInit() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			
			let label = randomString(length: 3)
			let num = randomInt([100, 250])
			let dim = randomInt([100, 250])
			let op = FullyconnectedOperator(inputDim: dim, numUnits: num, operatorLabel: label)
			XCTAssertEqual(op.operatorLabel, label)
			XCTAssertEqual(op.numUnits, num)
			
			print("Finish Test \(i+1)...\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
	*/
	func testOutputShape() {
		let numCase = 100
		let op = FullyconnectedOperator(inputDim:1, numUnits: 2)
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let num = randomInt([10, 100])
			op.numUnits = num
			
			let dim = randomInt([10, 100])
			op.inputDim = dim
			
			// generate input shapes
			var inputShapes = [TensorShape]()
			if i % 2 == 0 {
				// valid
				for _ in 0..<randomInt([1, 5]) {
					inputShapes.append(TensorShape(dataType: .int, shape: [dim]))
					print("Input shape: \(inputShapes.last!.description)")
				}
			} else {
				// invalid
				let caseRand = randomInt([0, 2])
				if caseRand % 2 == 0 {
					// nil
				} else {
					// not same
					inputShapes.append(TensorShape(dataType: .int, shape: [dim]))
					print("Input shape: \(inputShapes.last!.description)")
					for _ in 0..<randomInt([1, 5]) {
						var shape = TensorShape(dataType: .float, shape: [dim])
						while shape.count == inputShapes.last!.count {
							shape = randomShape(dimensions: randomInt([1,4]), dimensionSizeRange: [1, 10], dataType: .float)
						}
						inputShapes.append(shape)
						print("Input shape: \(inputShapes.last!.description)")
					}
				}
			}
			
			let outputShape = op.outputShape(shapeArray: inputShapes)
			if i % 2 == 0 {
				XCTAssertNotNil(outputShape)
				for shape in outputShape! {
					XCTAssertEqual(num, shape.count)
				}
			} else {
				XCTAssertNil(outputShape)
			}
			
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
	*/
	func testInputOutputTensorsCheck() {
		let numCase = 100
		let op = FullyconnectedOperator(inputDim:1, numUnits: 1)
		
		var numUnits = 0
		var inputDim = 0
		var weight:Tensor?
		var bias: Tensor?
		var inputTensors: [Tensor]?
		var outputTensors: [Tensor]?

		
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			inputTensors = [Tensor]()
			outputTensors = [Tensor]()
			
			// set up numUnit
			numUnits = randomInt([10, 150])
			op.numUnits = numUnits
			
			inputDim = randomInt([10, 150])
			op.inputDim = inputDim
			
			// generate valid input tensor
			for _ in 0..<randomInt([1, 3]) {
				inputTensors!.append(randomTensor(fromShape: TensorShape(dataType: .float, shape: [op.inputDim])))
			}
			
			// generate valid output tensor
			for _ in 0..<inputTensors!.count {
				outputTensors!.append(randomTensor(fromShape: TensorShape(dataType: .float, shape: [op.numUnits])))
			}
			
			// generate valid weight tensor
			weight = randomTensor(fromShape: TensorShape(dataType: .float, shape: [op.inputDim, op.numUnits]))
			
			// generate valid bias tensor
			bias = randomTensor(fromShape: TensorShape(dataType: .float, shape: [op.numUnits]))
			
			// setup invalid case
			if i % 4 != 0 {
				let randCase = i % 11
				
				if randCase == 0 {
					// input nil
					inputTensors = nil
					print("Set input tensors to nil")
				} else if randCase == 1 {
					// output nil
					outputTensors = nil
					print("Set output tensors to nil")
				} else if randCase == 2 {
					// weight nil
					weight = nil
					print("Set weight tensor to nil")
				} else if randCase == 3 {
					// bias nil
					bias = nil
					print("Set bias tensor to nil")
				} else if randCase == 4 {
					// input not valid
					var shape = randomShape(dimensions: randomInt([1,4]), dimensionSizeRange: [1, 10], dataType: .float)
					while shape.count == inputTensors!.first!.count {
						shape = randomShape(dimensions: randomInt([1,4]), dimensionSizeRange: [1, 10], dataType: .float)
					}
					inputTensors!.removeLast()
					inputTensors!.append(randomTensor(fromShape: shape))
					print("Set invalid input tensor: \(inputTensors!.last!.description)")
				} else if randCase == 5 {
					// output not same count
					outputTensors!.removeLast()
					print("Remove tensor from output tensors")
				} else if randCase == 6 {
					// output not valid
					var shapeArray = outputTensors!.last!.shape.shapeArray
					shapeArray[0] += randomInt([1, 5])
					outputTensors!.removeLast()
					outputTensors!.append(randomTensor(fromShape: TensorShape(dataType: .float, shape: shapeArray)))
					print("Set invalid output tensor: \(outputTensors!.last!.description)")
				} else if randCase == 7 {
					// weight rank not correct
					weight = randomTensor(fromShape: TensorShape(dataType: .float, shape: [3, 3, 3]))
					print("Set invalid weight tensor: \(weight!.description)")
				} else if randCase == 8 {
					 // weight shape not match
					var shapeArray = weight!.shape.shapeArray
					shapeArray[0] += randomInt([1, 5])
					weight = randomTensor(fromShape: TensorShape(dataType: .float, shape: [shapeArray[0], shapeArray[1]]))
					print("Set invalid weight tensor: \(weight!.description)")
				} else if randCase == 9 {
					// bias rank not correct
					bias = randomTensor(fromShape: TensorShape(dataType: .float, shape: [3, 3]))
					print("Set invalid bias tensor: \(bias!.description)")
				} else  {
					// bias shape not match
					var shapeArray = bias!.shape.shapeArray
					shapeArray[0] += randomInt([1, 5])
					bias = randomTensor(fromShape: TensorShape(dataType: .float, shape: [shapeArray[0]]))
					print("Set invalid bias tensor: \(bias!.description)")
				}
			}
			
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			op.weight = weight
			op.bias = bias
			
			if op.inputTensors == nil {
				print("Input tensor: nil")
			} else {
				for tensor in op.inputTensors! {
					print("Input tensor: \(tensor.description)")
				}
			}
			
			if op.outputTensors == nil {
				print("Output tensor: nil")
			} else {
				for tensor in op.outputTensors! {
					print("Output tensor: \(tensor.description)")
				}
			}
			
			print("Weight: \(op.weight?.description)")
			print("bias: \(op.bias?.description)")
			
			let (pass, msg) = op.inputOutputTensorsCheck()
			
			if i % 4 == 0 {
				XCTAssertTrue(pass)
			} else {
				XCTAssertFalse(pass)
				print(msg)
			}
			
			print("Finish Test \(i+1)\n\n")
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
		let numCase = 1
		let op = FullyconnectedOperator(inputDim: 1, numUnits: 1)
		
		var numUnits = 0
		var inputDim = 0
		var weight:Tensor?
		var bias: Tensor?
		var inputTensors: [Tensor]?
		var outputTensors: [Tensor]?
		
		// gpu initial
		_ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		
		
		let workGroup = DispatchGroup()
		let delegate = OperatorDelegateConvFullyConnctOp()
		op.computationDelegate = delegate
		delegate.dispatchGroup = workGroup
		
		for i in 0..<numCase {
			print("Test case \(i+1)...")
			print("Finish Test case \(i+1)\n\n")
			
			inputTensors = [Tensor]()
			outputTensors = [Tensor]()
			
			// set up numUnit
			numUnits = randomInt([100, 150])
			op.numUnits = numUnits
			
			inputDim = randomInt([10, 150])
			op.inputDim = inputDim
			
			// generate valid input tensor
			for _ in 0..<randomInt([1, 3]) {
				inputTensors!.append(randomTensor(fromShape: TensorShape(dataType: .float, shape: [op.inputDim])))
			}
			
			// generate valid output tensor
			for _ in 0..<inputTensors!.count {
				outputTensors!.append(randomTensor(fromShape: TensorShape(dataType: .float, shape: [op.numUnits])))
			}
			
			// generate valid weight tensor
			weight = randomTensor(fromShape: TensorShape(dataType: .float, shape: [op.inputDim, op.numUnits]))
			
			// generate valid bias tensor
			bias = randomTensor(fromShape: TensorShape(dataType: .float, shape: [op.numUnits]))
			
			op.inputTensors = inputTensors!
			op.outputTensors = outputTensors!
			op.weight = weight!
			op.bias = bias!
			
			delegate.veryfyTensors = [inputTensors!.first!, weight!, bias!]
			
			print("Input tensor: \(inputTensors!.first!.description)")
			print("Output tensor: \(outputTensors!.first!.description)")
			print("Weight: \(weight!.description)")
			print("Bias: \(bias!.description)")


			if i % 2 == 0 {
				print("Run CPU")
				workGroup.enter()
				op.computeAsync(.CPU)
			} else {
				if !SerranoEngine.configuredEngine.hasAvailableGPU() {
					print("No gpu available, give up gpu test \n\n")
					continue
				}
				workGroup.enter()
				op.computeAsync(.GPU)
			}
			
			workGroup.wait()
			
			SerranoResourceManager.globalManager.releaseAllResources()
			
			print("Finish test \(i+1)\n\n")
		}
	}

}
