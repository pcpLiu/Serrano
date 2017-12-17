//
//  arctanh_op_test.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/6/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class ArctanhOpDelegate: OperatorDelegateConvUnaryOp {
    
    required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
        let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
            XCTAssertEqual(rawTensor.count, resultTensor.count)
            let readerReader = rawTensor.floatValueReader
            let resultReader = resultTensor.floatValueReader
            for i in 0..<rawTensor.count {
                let val = atanh(readerReader[i])
                if val.isNaN || val.isInfinite || resultReader[i].isNaN || resultReader[i].isInfinite { continue }
                if abs(val) < 0.001 {
                    XCTAssertEqualWithAccuracy(val, resultReader[i], accuracy: 0.001)
                } else {
                    XCTAssertEqualWithAccuracy(val, resultReader[i], accuracy: abs(val*0.001))
                }
            }
        }
        self.init(block: blcok)
        // grad: 1 / (1 - x^2)
        self.gradVerifyBlock = {(grads: [String : DataSymbolSupportedDataType], inputs:[Tensor]) -> Void in
            for (index, input) in inputs.enumerated() {
                let resultGrad = grads["input_\(index)"]!.tensorValue
                for i in 0..<input.count {
                    let val = 1.0 / (1 - input.floatValueReader[i] * input.floatValueReader[i])
                    if val.isNaN || val.isInfinite {
                        continue
                    }
                    XCTAssertEqual(val, resultGrad.floatValueReader[i], accuracy: max(0.001, abs(val*0.001)))
                }
            }
        }
    }
}

class ArctanhOpTest: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func test() {
        let testCase = UnarOpTest<ArctanhOpDelegate, ArctanhOperator>()
        testCase.testAll()
    }
}

