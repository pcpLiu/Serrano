//
//  broad_cast_op_test.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/13/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

fileprivate func verfyShape(targetShape: TensorShape, result: [TensorShape]) -> Bool {
	for shape in result {
		if !shape.shapeArray.elementsEqual(targetShape.shapeArray) {
			return false
		}
	}
	return true
}


fileprivate func generateShapes(targetShape: TensorShape, valid: Bool) -> [TensorShape] {
	let targetShapeReversed = Array(targetShape.shapeArray.reversed())
	var shapesReversed = [[Int]]()
	
	if valid {
		//valid
		for _ in 0..<randomInt([1, 8]) {
			var newShapeRevsered = Array(targetShapeReversed)
			
			if randomInt([100, 10000000]) % 3 == 0 { // remove last dim
				for _ in 0..<randomInt([1, targetShapeReversed.count]) {
					if newShapeRevsered.count == 1 { break }
					newShapeRevsered.removeLast()
				}
			} else if randomInt([100, 10000000]) % 2 == 0 { // decrease dim size  to 1
				let index = randomInt([0, targetShapeReversed.count - 1])
				newShapeRevsered[index] = 1
			}
			shapesReversed.append(newShapeRevsered)
		}
	} else {
		//invalid
		for _ in 0..<randomInt([1, 8]) {
			var newShapeRevsered = Array(targetShapeReversed)
			
			if randomInt([100, 10000000]) % 3 == 0 { // remove all dim
				newShapeRevsered.removeAll()
			} else { // make random dim size
				let index = randomInt([0, targetShapeReversed.count - 1])
				let randSize = randomInt([2, 100])
				if randSize == newShapeRevsered[index] || randSize == 1 {
					newShapeRevsered[index] += randSize
				} else {
					newShapeRevsered[index] = randSize
				}
			}
			shapesReversed.append(newShapeRevsered)
		}
	}
	
	// generate shapes
	var shapes = [TensorShape]()
	for reversedShape in shapesReversed {
		let tensorShape = TensorShape(dataType: .float, shape: Array(reversedShape.reversed()))
		shapes.append(tensorShape)
	}
	
	return shapes
}

/**
Generate tensor shapes according to input requriementd
*/
fileprivate func generateShapes() -> (targetShape: TensorShape, shapes: [TensorShape], valid: Bool) {
	
	// generate target shape
	let targetShape = randomShape(dimensions: randomInt([1, 5]), dimensionSizeRange: [1, 5], dataType: .float)
	let targetShapeReversed = Array(targetShape.shapeArray.reversed())
	
	var valid = true
	if randomInt([100, 10000000]) % 3 == 0 { valid = false }
	
	var shapesReversed = [[Int]]()
	if valid {
		//valid
		for _ in 0..<randomInt([1, 8]) {
			var newShapeRevsered = Array(targetShapeReversed)
			
			if randomInt([100, 10000000]) % 3 == 0 { // remove last dim
				for _ in 0..<randomInt([1, targetShapeReversed.count]) {
					if newShapeRevsered.count == 1 { break }
					newShapeRevsered.removeLast()
				}
			} else if randomInt([100, 10000000]) % 2 == 0 { // decrease dim size  to 1
				let index = randomInt([0, targetShapeReversed.count - 1])
				newShapeRevsered[index] = 1
			}
			shapesReversed.append(newShapeRevsered)
		}
	} else {
		//invalid
		for _ in 0..<randomInt([1, 8]) {
			var newShapeRevsered = Array(targetShapeReversed)
			
			if randomInt([100, 10000000]) % 3 == 0 { // remove all dim
				newShapeRevsered.removeAll()
			} else if randomInt([100, 10000000]) % 2 == 0 { // make random dim size
				let index = randomInt([0, targetShapeReversed.count - 1])
				let randSize = randomInt([2, 100])
				if randSize == newShapeRevsered[index] || randSize == 1 {
					newShapeRevsered[index] += randSize
				} else {
					newShapeRevsered[index] = randSize
				}
			} else { // add one more dim
				newShapeRevsered.append(randomInt([1, 10]))
			}
			shapesReversed.append(newShapeRevsered)
		}
	}
	
	// generate shapes
	var shapes = [TensorShape]()
	for reversedShape in shapesReversed {
		let tensorShape = TensorShape(dataType: .float, shape: Array(reversedShape.reversed()))
		shapes.append(tensorShape)
	}
	
	return (targetShape, shapes, valid)
}

class BroadcastOpTest: XCTestCase {
    
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
	Target:
		public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
	*/
	func testOutputShapes() {
		let numCase = 100
		
		for i in 0..<numCase {
			
			print("Test \(i+1)...")
			
			// generate shapes
			let (targetShape, shapes, valid) = generateShapes()
			for shape in shapes {
				print("Input shape: \(shape.shapeArray)")
			}
			print("Expect validation: \(valid), target shape: \(targetShape.shapeArray)")
			
			let op = BroadcastOperator(targetShape: targetShape)
			let result = op.outputShape(shapeArray: shapes)
			
			if valid {
				XCTAssertNotNil(result, "Given nil.")
				
				let veifyResult = verfyShape(targetShape: targetShape, result: result!)
				XCTAssertTrue(veifyResult, "Incorrect result")
			} else {
				XCTAssertNil(result, "Expect NIl, given \(result!)")
			}
			
			print("\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
		public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
	*/
	func testCheck() {
		let numCase = 100
		
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			var inputTensors = [Tensor]()
			var  outputTensors = [Tensor]()
			
			
			let targetShape = randomShape(dimensions: randomInt([1, 5]), dimensionSizeRange: [1, 5], dataType: .float)
			let op = BroadcastOperator(targetShape: targetShape)
			
			
			// generate input tensors
			var inputShapes = [TensorShape]()
			if i+1 % 2 == 0 {
				inputShapes = generateShapes(targetShape: targetShape, valid: false)
			} else {
				inputShapes = generateShapes(targetShape: targetShape, valid: true)
			}
			for shape in inputShapes {
				inputTensors.append(randomTensor(fromShape: shape))
			}
			
			// generate output tensors
			if i+1 % 2 == 0 {
				for _ in 0..<randomInt([1, inputTensors.count]) { // not enough tensor
					outputTensors.append(Tensor(repeatingValue: 0.0, tensorShape: targetShape))
				}
			} else {
				for _ in 0..<inputTensors.count {
					outputTensors.append(Tensor(repeatingValue: 0.0, tensorShape: targetShape))
				}
			}
			
			
			if i+1 % 3 == 0 { // null
				let (pass, msg) = op.inputOutputTensorsCheck()
				print(msg)
				XCTAssertTrue(!pass)
			} else {
				op.inputTensors = inputTensors
				op.outputTensors = outputTensors
				let (pass, msg) = op.inputOutputTensorsCheck()
				if i+1 % 2 == 0 {
					print(msg)
					XCTAssertTrue(!pass)
				} else {
					print(msg)
					XCTAssertTrue(pass)
				}
			}
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	func testCompute() {
		let numCase = 10
		
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let rawShape = TensorShape(dataType: .int, shape: [1, 4, 1])
			let targetShape = TensorShape(dataType: .int, shape: [2, 2,4, 3])
			let op = BroadcastOperator(targetShape: targetShape)
			
			op.inputTensors = [randomTensor(fromShape: rawShape)]
			op.outputTensors = [Tensor(repeatingValue: -32.0, tensorShape: targetShape)]
			
			op.cpu()
			
			print(op.inputTensors!.first!.nestedArrayFloat())
			print(op.outputTensors!.first!.nestedArrayFloat())
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
	
}
