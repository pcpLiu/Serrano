////
////  add_op_test.swift
////  SerranoTests
////
////  Created by ZHONGHAO LIU on 6/7/17.
////  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
////

import XCTest
@testable import Serrano

class AddOpDelegate: OperatorDelegateConvBinaryOp {
    
    required public convenience init(compareBlock: (([Tensor], Tensor) -> Void)?)  {
        let blcok =  {(inputTensors: [Tensor], resultTensor: Tensor) -> Void in
            let inputReaderA = inputTensors[0].floatValueReader
            let inputReaderB = inputTensors[1].floatValueReader
            let resultReader = resultTensor.floatValueReader
            
            for i in 0..<resultTensor.count {
                let val = inputReaderA[i] + inputReaderB[i]
                if abs(val) < 0.001 {
                    XCTAssertEqual(val, resultReader[i], accuracy: 0.001)
                } else {
                    XCTAssertEqual(val, resultReader[i], accuracy: abs(val*0.001))
                }
            }
        }
        self.init(block: blcok)
        
        self.gradVerifyBlock = {(grads: [String : DataSymbolSupportedDataType], inputs:[Tensor]) -> Void in
            let A = inputs[0]
            let B = inputs[1]
            let gradA = grads["input_0"] as! Tensor
            
            let gradB = grads["input_1"] as! Tensor
            
            for i in 0..<A.count {
                XCTAssertEqual(1.0, gradA.floatValueReader[i], accuracy: abs(1.0*0.001))
            }
            
            for i in 0..<B.count {
                XCTAssertEqual(1.0, gradB.floatValueReader[i], accuracy: abs(1.0*0.001))
            }
        }
    }
}

class AddOpTest: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func test() {
        let testCase = BinaryOpTest<AddOpDelegate, AddOperator>()
        testCase.testAll()
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
//    func testClose() {
//        let op = AddOperator()
//        // generate tensors
//        var inputTensors = [Tensor]()
//        var outputTensors = [Tensor]()
//        var shape: TensorShape
//
//        // configure engine
//        let (_, msg) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
//
//        shape = randomShape(dimensions: 2, dimensionSizeRange: [2000, 2000], dataType: .float)
//        for _ in 0..<2 {
//            inputTensors.append(randomTensor(fromShape: shape))
//            print("Generate Input tensor: \(inputTensors.last!.description)")
//        }
//        outputTensors.append(randomTensor(fromShape: shape))
//
//        op.inputTensors = inputTensors
//        op.outputTensors = outputTensors
//
//        self.measure {
//            op.compute(.GPU)
//        }
//
//        SerranoResourceManager.globalManager.releaseAllResources()
//    }
    
//    func testAd() {
//        let (_,_) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
//        let tensorA = Tensor(randomRampTensor: TensorShape(dataType: .float, shape: [2,4]))
//        let tensorB = Tensor(randomRampTensor: TensorShape(dataType: .float, shape: [2,4]))
//        let tensorC = Tensor(randomRampTensor: TensorShape(dataType: .float, shape: [2,4]))
//        let op = AddOperator(inputTensors: [tensorA, tensorB], outputTensors: [tensorC])
//        op.compute(.GPU)
//
//    }
}

