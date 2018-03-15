//
//  linear_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/27/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class LinearOpDelegate: OperatorDelegateConvUnaryOp {
    
    required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
        let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
            XCTAssertEqual(rawTensor.count, resultTensor.count)
            let readerReader = rawTensor.floatValueReader
            let resultReader = resultTensor.floatValueReader
            for i in 0..<rawTensor.count {
                let val = readerReader[i] 
                if val.isNaN || val.isInfinite || resultReader[i].isInfinite || resultReader[i].isNaN {continue}
                XCTAssertEqual(val, resultReader[i], accuracy: max(0.001, abs(val*0.001)))
            }
        }
        self.init(block: blcok)
        
        // grad: 1
        self.gradVerifyBlock = {(grads: [String : DataSymbolSupportedDataType], inputs:[Tensor]) -> Void in
            for (index, input) in inputs.enumerated() {
                let resultGrad = grads["input_\(index)"]!.tensorValue
                for i in 0..<input.count {
                    let val:Float = 1.0
                    if val.isNaN || val.isInfinite { continue }
                    XCTAssertEqual(val, resultGrad.floatValueReader[i], accuracy: max(0.0001, abs(val*0.001)))
                }
            }
        }
    }
}

class LinearOpTest: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func test() {
        let testCase = UnarOpTest<LinearOpDelegate, LinearOperator>()
        testCase.testAll()
    }
}
