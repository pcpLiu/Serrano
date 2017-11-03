//
//  binary_op_test.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/7/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
import Dispatch
import Metal
@testable import Serrano


public class OperatorDelegateConvBinaryOp: OperatorDelegateConv {
	
    public var compareBlock: ([Tensor], Tensor) -> Void
	
    required public convenience init(compareBlock: (([Tensor], Tensor) -> Void)?) {
        let blcok =  {(inputTensors: [Tensor], resultTensor: Tensor) -> Void in
            print("NEED OVERRIDE")
        }
        self.init(block: blcok)
    }
	
    // override this func
    public init(block: @escaping ([Tensor], Tensor) -> Void) {
        self.compareBlock = block
        super.init()
    }
	
    override public func compare() {
        XCTAssertTrue(self.resultTensors.count == 1)
        XCTAssertTrue(self.resultTensors.first!.count == self.veryfyTensors.first!.count)
		
        self.compareBlock(self.veryfyTensors, self.resultTensors.first!)
    }
}

public class BinaryOpTest<OpDelegate: OperatorDelegateConvBinaryOp, BinaryOp: BinaryOperator>: XCTestCase {
	
    override public func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
		
    }
	
    override public func tearDown() {
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
	
    public func  testAll() {
        self.testInit()
		
        self.testOuputShapesCheck()
		
        //        self.testCPU()
        //
        //        self.testGPU()
        //
        self.testCompute()
    }
	
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
    /**
     Test init functions
     */
    func testInit() {
        let numCase = 100
        let op = BinaryOp()
        for _ in 0..<numCase {
            let label = randomString(length: randomInt([2, 10]))
            op.operatorLabel = label
            XCTAssertEqual(label, op.operatorLabel)
            print("label: \(label), \(op.operatorLabel)")
        }
    }
	
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
    /**
     Test:
     public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
     */
    func testOuputShapesCheck() {
        let numCase = 100
        for i in 0..<numCase {
            print("Test case \(i+1)")
            let op = BinaryOp()
			
            var valid = true
            if i % 3 == 0 {
                valid = false
            }
			
            var shapes = [TensorShape]()
            if !valid {
                if i % 2 == 0 {
                    for _ in 0..<2 {
                        shapes.append(randomShape(dimensions: randomInt([1, 5]), dimensionSizeRange: [1, 1000], dataType: .float))
                    }
                } else {
                    for _ in 0..<randomInt([1, 5]) {
                        shapes.append(randomShape(dimensions: randomInt([1, 5]), dimensionSizeRange: [1, 1000], dataType: .float))
                    }
                }
				
            } else {
                let shape = randomShape(dimensions: 2, dimensionSizeRange: [10, 150], dataType: .float)
                for _ in 0..<2 {
                    shapes.append(shape)
                }
            }
			
            // compute
            let outShape = op.outputShape(shapeArray: shapes)?.first
			
            if valid {
                XCTAssertNotNil(outShape)
                XCTAssertTrue(outShape! == shapes[0])
                XCTAssertTrue(outShape! == shapes[1])
            } else {
                XCTAssertNil(outShape, "Should be nil.")
            }
        }
    }
	
	
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
    /**
     Test:
     func compute(withInputTensors tensors:[Tensor], computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) -> [Tensor]
     func compute(asyncWithInputTensors tensors:[Tensor], computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
     */
	
    func testCompute() {
        let caseNum = 10
        let op = BinaryOp()
		
		// configure engine
		let (_, _) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
		
        // setup delegate
        let delegate = OpDelegate(compareBlock: nil)
        let workingGroup = DispatchGroup()
        delegate.dispatchGroup = workingGroup
        op.computationDelegate = delegate
		
		
        for i in 0..<caseNum {
            print("Test case \(i+1)...")
			
            // generate tensors
			var inputTensors = [Tensor]()
			var outputTensors = [Tensor]()
            var shape: TensorShape
            if i < 8 { // smaller tensors
                shape = randomShape(dimensions: 2, dimensionSizeRange: [100, 200], dataType: .float)
            } else { // large tensors
                shape = randomShape(dimensions: 2, dimensionSizeRange: [1000, 2000], dataType: .float)
            }
            for _ in 0..<2 {
                inputTensors.append(randomTensor(fromShape: shape))
                print("Generate Input tensor: \(inputTensors.last!.description)")
            }
			outputTensors.append(randomTensor(fromShape: shape))
			
            delegate.veryfyTensors = inputTensors
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			
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
            print("Finish Test \(i+1)\n\n\n")
        }
    }
}


