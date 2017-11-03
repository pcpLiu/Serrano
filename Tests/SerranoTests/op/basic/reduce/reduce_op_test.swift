//
//  reduce_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/27/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

public class OperatorDelegateConvReduceOp: OperatorDelegateConv {
	
	public var compareBlock: (Tensor, Tensor) -> Void
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
			print()
		}
		self.init(block: blcok)
	}
	
	// override this func
	public init(block: @escaping (Tensor, Tensor) -> Void) {
		self.compareBlock = block
		super.init()
	}
	
	override public func compare() {
		XCTAssertTrue(self.resultTensors.first!.count == self.veryfyTensors.first!.count)
		
		for i in 0..<self.resultTensors.count {
			self.compareBlock(self.veryfyTensors[i], self.resultTensors[i])
		}
	}
}

class ReduceOpTest<OpDelegate: OperatorDelegateConvReduceOp, ReduceOp: ReduceOperator>: XCTestCase {
	
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
	
	
	func testAll() {
		print(String(repeating: "=", count: 80) + "\n\n")
		self.testOutputShape()
		
		print(String(repeating: "=", count: 80) + "\n\n")
		self.testInputOutputTensorCheck()
		
		print(String(repeating: "=", count: 80) + "\n\n")
		self.testCompute()
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Test init functions
	*/
	func testInit() {
		let numCase = 100
		let op = ReduceOp(axis: [Int]())
		for _ in 0..<numCase {
			let label = randomString(length: randomInt([2, 10]))
			op.operatorLabel = label
			XCTAssertEqual(label, op.operatorLabel)
			print("label: \(label), \(op.operatorLabel)")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
	*/
	func testOutputShape() {
		let numCase = 100
		let op = ReduceOp(axis: [Int]())
		for i in 0..<numCase {
			print("Test \(i+1)...")
			// generte input shape, just one
			var intputShapes = [randomShape(dimensions: randomInt([1, 5]), dimensionSizeRange: [1, 5], dataType: .float)]
			print("Input shape: \(intputShapes[0].description)")
			
			
			// keep dim set
			if i % 4 == 0 {
				op.keepDim = true
			} else {
				op.keepDim = false
			}
			print("Keep dim: \(op.keepDim)")
			
			// generate axis
			var axis = [Int]()
			if i % 2 == 0 {
				// valid
				for _ in 0..<randomInt([1, intputShapes[0].rank]) {
					let ax = randomInt([0, intputShapes[0].rank])
					if !axis.contains(ax) { axis.append(ax) }
				}
			} else {
				// not valid
				let caseRand = randomInt([0, 100])
				if caseRand % 2 == 0 {
					// rank not right
					print("rank not right")
					for dim in 0..<randomInt([intputShapes[0].rank+1, intputShapes[0].rank + 2]) {
						axis.append(dim)
					}
				} else {
					// axis dim value
					print("axis dim value")
					for _ in 0..<randomInt([1, intputShapes[0].rank]) {
						let ax = randomInt([0, intputShapes[0].rank]) + 3 + intputShapes[0].rank
						if !axis.contains(ax) { axis.append(ax) }
					}
				}
			}
			op.axis = axis
			print("Axis: \(axis)")
			
			// run
			let outputShapes = op.outputShape(shapeArray: intputShapes)
			if i % 2 == 0 {
				XCTAssertNotNil(outputShapes)
				let outputShape = outputShapes!.first!
				let inputShape = intputShapes.first!
				
				print("Output shape: \(outputShape.description)")

				if op.keepDim {
					XCTAssertTrue(outputShape.rank == inputShape.rank)
					for dimIndex in axis {
						XCTAssertTrue(outputShape.shapeArray[dimIndex] == 1)
					}
				} else {
					XCTAssertTrue(outputShape.rank == inputShape.rank - axis.count)
				}
			} else {
				XCTAssertNil(outputShapes)
			}
			
			
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
	*/
	func testInputOutputTensorCheck() {
		let numCase = 100
		let op = ReduceOp(axis: [Int]())
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			var inputTensors = [Tensor]()
			var outputTensors = [Tensor]()
			
			// keep dim set
			if i % 3 == 0 { op.keepDim = true }
			else		  { op.keepDim = false }
			print("Keep dim: \(op.keepDim)")
			
			// generte valid input tensor
			let intputShape = randomShape(dimensions: randomInt([1, 8]), dimensionSizeRange: [1, 5], dataType: .float)
			inputTensors.append(randomTensor(fromShape: intputShape))
			print("Input tensor: \(inputTensors.first!.description)")
			
			// generate valid axis
			var axis = [Int]()
			for _ in 0..<randomInt([1, intputShape.rank]) {
				let ax = randomInt([0, intputShape.rank])
				if !axis.contains(ax) { axis.append(ax) }
			}
			print("Axis: \(axis)")
			op.axis = axis
			
			// generate valid output tensor
			let outputShape = op.outputShape(shapeArray: [intputShape])!.first!
			outputTensors.append(randomTensor(fromShape: outputShape))
			print("Output tensor: \(outputTensors[0].description)")
			
			// set attributes
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			
			// set up invalid case
			if i % 2 != 0 {
				let randCase = randomInt([1, 12])
				if randCase % 5 == 0 {
					// input nil
					print("Input nil")
					op.inputTensors = nil
				} else if randCase % 5 == 1 {
					// output nil
					print("Output nil")
					op.outputTensors = nil
				} else if randCase % 5 == 2 {
					// count not match
					op.outputTensors!.append(randomTensor(fromShape: outputShape))
					print("Count not match. \(op.outputTensors!.count)")
				} else if randCase % 5 == 3 {
					// input not valid
					op.axis.append(inputTensors[0].rank + 2)
					print("Input invalid. New axis: \(op.axis)")
				} else {
					// outout not valid
					var shapeArrayInvalid = outputShape.shapeArray
					shapeArrayInvalid.append(inputTensors[0].rank + 2)
					op.outputTensors![0] = randomTensor(fromShape: TensorShape(dataType: .float, shape: shapeArrayInvalid))
					print("Output invalid. New output tensor: \(op.outputTensors![0].description)")
				}
			}
			
			let (pass, msg) = op.inputOutputTensorsCheck()
			
			if i % 2 == 0 {
				XCTAssertTrue(pass)
			} else {
				XCTAssertTrue(!pass)
				print(msg)
			}
			
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
	public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
	*/
	func testCompute() {
		let numCase = 10
		let op = ReduceOp(axis: [Int]())
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			var inputTensors = [Tensor]()
			var outputTensors = [Tensor]()
			

			let intputShape = randomShape(dimensions: randomInt([1, 6]), dimensionSizeRange: [2, 5], dataType: .float)
			inputTensors.append(randomTensor(fromShape: intputShape))
			print("Input tensor: \(inputTensors.first!.nestedArrayFloat())")
			
			// generate valid axis
			var axis = [Int]()
			for _ in 0..<randomInt([1, intputShape.rank]) {
				let ax = randomInt([0, intputShape.rank])
				if !axis.contains(ax) { axis.append(ax) }
			}
			print("Axis: \(axis)")
			op.axis = axis
			
			// generate valid output tensor
			let outputShape = op.outputShape(shapeArray: [intputShape])!.first!
			outputTensors.append(randomTensor(fromShape: outputShape))
			
			// set attributes
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			
			op.compute(.CPU)
			
			// FIXME: Needs a different code logic to verify the reduce result. Now just print out
			
			print(op.outputTensors!.first!.nestedArrayFloat())

			print("Finish Test \(i+1)\n\n")
		}
	}
	
}
