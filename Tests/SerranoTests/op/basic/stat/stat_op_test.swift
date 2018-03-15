//
//  stat_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 1/4/18.
//  Copyright © 2018 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class stat_op_test: XCTestCase {
    
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
     public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
     */
    func testOutputShape() {
        let numCase = 100
        for i in 0..<numCase {
            print("Test case \(i+1)...")
            
            // valid input shapes
            let inShape = randomShape(dimensions: randomInt([1, 5]), dimensionSizeRange: [3, 5], dataType: .int)
            var inputShapes = [TensorShape]()
            for _ in 0..<randomInt([2, 5]) {
                inputShapes.append(inShape)
            }
            
            // setup invalid
            if i % 2 != 0 {
                if i % 3 == 0 {
                    // empty
                    inputShapes.removeAll()
                } else {
                    var shape = inputShapes.first!.shapeArray
                    shape[0] += randomInt([1, 3])
                    inputShapes[0] = TensorShape(dataType: .int, shape: shape)
                }
            }
            
            let op = StatOperator()
            let outputShapes = op.outputShape(shapeArray: inputShapes)
            if i % 2 == 0 {
                XCTAssertNotNil(outputShapes)
                XCTAssertEqual(outputShapes!.count, 1)
                XCTAssertEqual(outputShapes!.first!, inShape)
            } else {
                XCTAssertNil(outputShapes)
            }
            
            print("Finish Test case \(i+1)\n")
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
     */
    func testInputOutputTensorsCheck() {
        let numCase = 100
        for i in 0..<numCase {
            print("Test case \(i+1)...")
            
            // valid input tensors
            var inputs: [Tensor]? = [Tensor]()
            let inShape = randomShape(dimensions: randomInt([1, 5]), dimensionSizeRange: [3, 5], dataType: .int)
            for _ in 0..<randomInt([2, 5]) {
                inputs!.append(randomTensor(fromShape: inShape))
            }
            
            // valid output tensors
            var outputs: [Tensor]? = [randomTensor(fromShape: inShape)]
            
            // setup invalid cases
            if i % 2 != 0 {
                let randCase = randomInt([0, 5])
                if randCase == 0 {
                    // input nil
                    inputs = nil
                } else if randCase == 1 {
                    // output nil
                    outputs = nil
                } else if randCase == 2 {
                    // input invalid
                    var shape = inputs!.first!.shape.shapeArray
                    shape[0] += randomInt([1, 4])
                    let newShape = TensorShape(dataType: .int, shape: shape)
                    inputs!.append(randomTensor(fromShape: newShape))
                } else if randCase == 3 {
                    // INVALID output count
                    outputs!.append(randomTensor(fromShape: inShape))
                } else {
                    // output shape invalid
                    var shape = outputs!.first!.shape.shapeArray
                    shape[0] += randomInt([1, 4])
                    let newShape = TensorShape(dataType: .int, shape: shape)
                    outputs![0] = randomTensor(fromShape: newShape)
                }
            }
            
            let op = StatOperator()
            op.inputTensors = inputs
            op.outputTensors = outputs
            let (pass, msg) = op.inputOutputTensorsCheck()
            if i % 2 == 0 {
                XCTAssertTrue(pass)
            } else {
                XCTAssertFalse(pass)
                print(msg)
            }
            
            print("Finish Test case \(i+1)\n")
        }
    }
}
