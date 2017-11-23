//
//  sin_op_test.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/5/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class SinOpDelegate: OperatorDelegateConvUnaryOp {
    
    required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
        let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
            XCTAssertEqual(rawTensor.count, resultTensor.count)
            let readerReader = rawTensor.floatValueReader
            let resultReader = resultTensor.floatValueReader
            for i in 0..<rawTensor.count {
                let val = sin(readerReader[i])
                if val.isNaN || val.isInfinite || resultReader[i].isNaN || resultReader[i].isInfinite { continue }
                if abs(val) < 0.001 {
                    XCTAssertEqual(val, resultReader[i], accuracy: 0.001)
                } else {
                    XCTAssertEqual(val, resultReader[i], accuracy: abs(val*0.001))
                }
                
            }
        }
        self.init(block: blcok)
        
        // grad: cos(x)
        self.gradVerifyBlock = {(grads: [String : DataSymbolSupportedDataType], inputs:[Tensor]) -> Void in
            for (index, input) in inputs.enumerated() {
                let resultGrad = grads["input_\(index)"]!.tensorValue
                for i in 0..<input.count {
                    let val = cos(input.floatValueReader[i])
                    if val.isNaN || val.isInfinite {
                        continue
                    }
                    XCTAssertEqual(val, resultGrad.floatValueReader[i], accuracy: abs(val*0.001))
                }
            }
        }
    }
}

class SinOpTest: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func test() {
        let testCase = UnarOpTest<SinOpDelegate, SinOperator>()
        testCase.testAll()
    }
}
