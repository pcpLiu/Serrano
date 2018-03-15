//
//  stat_mean_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 1/4/18.
//  Copyright © 2018 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

public class OperatorDelegateConvStatMeanOp: OperatorDelegateConv {
    
    public var inputs: [Tensor]? = nil
    public var output: Tensor? = nil
    
    override public func compare() {
        let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: self.output!.shape)
        let verifyReader = verifyTensor.floatValueReader
        for i in 0..<self.inputs!.count {
            let inReader = self.inputs![i].floatValueReader
            for eleIndex in 0..<verifyTensor.count {
                verifyReader[eleIndex] = (verifyReader[eleIndex] * Float(i) + inReader[eleIndex]) / Float(i+1)
            }
        }
        
        // check
        let outReader = output!.floatValueReader
        for i in 0..<verifyTensor.count {
            XCTAssertEqual(outReader[i], verifyReader[i], accuracy: max(0.001, abs(verifyReader[i] * 0.001)))
        }
        
        
    }
}

class stat_mean_op_test: XCTestCase {
    
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
     public func compute(_ computationMode: OperatorComputationMode = OperatorComputationMode.GPU)
     public func computeAsync(_ computationMode: OperatorComputationMode = OperatorComputationMode.GPU)
     */
    func testCompute() {
        let numCase = 10
        let delegate = OperatorDelegateConvStatMeanOp()
        let _ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
        
        for i in 0..<numCase {
            print("Test case \(i+1)...")
            
            // input tensors
            var inputs = [Tensor]()
            let inShape = randomShape(dimensions: randomInt([2, 5]), dimensionSizeRange: [10, 20], dataType: .int)
            for _ in 0..<randomInt([2, 5]) {
                inputs.append(randomTensor(fromShape: inShape))
            }
            
            // output tensor
            let output = randomTensor(fromShape: inShape)
            
            let op = StatMeanOperator()
            op.inputTensors = inputs
            op.outputTensors = [output]
            op.computationDelegate = delegate
            
            let workGroup = DispatchGroup()
            delegate.inputs = inputs
            delegate.output = output
            delegate.dispatchGroup = workGroup
            
            if i % 2 == 0 {
                workGroup.enter()
                print("Run on CPU")
                op.computeAsync(.CPU)
            } else {
                if !SerranoEngine.configuredEngine.hasAvailableGPU() {
                    print("No GPU available. Give up test.\n")
                    continue
                }
                workGroup.enter()
                print("Run on GPU")
                op.computeAsync(.GPU)
            }
            
            workGroup.wait()
            
            print("Finish test case \(i+1)\n")
        }
    }
}
