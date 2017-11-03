//
//  copy_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 4/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
import Dispatch
@testable import Serrano


/**
 Covenient class for testing
 */
class OperatorDelegateConvCopy: OperatorDelegateConv {
    
    override public func compare() {
        XCTAssertTrue(self.resultTensors.count == self.veryfyTensors.count)
        
        for i in 0..<veryfyTensors.count {
            let tensor = resultTensors[i]
            let verifyTensor = veryfyTensors[i]
            
            XCTAssertTrue(tensor.flatArrayFloat().elementsEqual(verifyTensor.flatArrayFloat()), "Failed on test:\n \(tensor.flatArrayFloat()) \n \(verifyTensor.flatArrayFloat())")
        }
    }
}


class copy_op_test: XCTestCase {
    
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
        init(operatorLabel label: String, computationDelegate delegate: OperatorCalculationDelegate?)
        convenience init(operatorLabel label: String = "CopyOperator")
     */
    func testInit() {
        let numCase = 100
        
        for _ in 0..<numCase {
            let label = randomString(length: randomInt([2, 10]))
            let op = CopyOperator(operatorLabel: label)
            
            XCTAssertEqual(label, op.operatorLabel)
            print("label: \(label), \(op.operatorLabel)")
        }
        
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    /**
     Target:
        func compute(withInputTensors tensors: [Tensor], computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) -> [Tensor]
        func compute(asyncWithInputTensors tensors: [Tensor], computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
     */
    func testCompute() {
        let numCase = 100
        for i in 0..<numCase {
            let delegate = OperatorDelegateConvCopy()
            let workingGroup = DispatchGroup()
            
            let numTensor = randomInt([1, 5])
            var inputTensors = [Tensor]()
			var outputTensors =  [Tensor]()
            for _ in 0..<numTensor {
				let shape = randomShape(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
                inputTensors.append(randomTensor(fromShape: shape) )
				outputTensors.append(Tensor(repeatingValue: 0.0, tensorShape: shape))
            }
            delegate.veryfyTensors = inputTensors
            delegate.dispatchGroup = workingGroup
            
            let label = "copy_op_\(i+1)"
            let op = CopyOperator(inputTensors: inputTensors)
			op.outputTensors = outputTensors
            op.computationDelegate = delegate
            print("Test on op \(op.operatorLabel)")
			
			workingGroup.enter()
            op.computeAsync( .CPU)
            workingGroup.wait()
            print("\n\n")
        }
    }
}
