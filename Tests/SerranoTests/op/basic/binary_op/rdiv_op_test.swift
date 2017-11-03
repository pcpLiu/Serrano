//
//  rdiv_op_test.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/13/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class RDivOpDelegate: OperatorDelegateConvBinaryOp {
	
	required public convenience init(compareBlock: (([Tensor], Tensor) -> Void)?)  {
		let blcok =  {(inputTensors: [Tensor], resultTensor: Tensor) -> Void in
			let inputReaderA = inputTensors[0].floatValueReader
			let inputReaderB = inputTensors[1].floatValueReader
			let resultReader = resultTensor.floatValueReader
			
			for i in 0..<resultTensor.count {
				let val = inputReaderB[i] / inputReaderA[i]
				if val.isNaN || val.isInfinite || resultReader[i].isNaN || resultReader[i].isInfinite { continue }
				if abs(val) < 0.001 {
					XCTAssertEqualWithAccuracy(val, resultReader[i], accuracy: 0.001)
				} else {
					XCTAssertEqualWithAccuracy(val, resultReader[i], accuracy: abs(val*0.001))
				}
			}
		}
		self.init(block: blcok)
	}
}

class RDivOpTest: XCTestCase {
	
	override func setUp() {
		super.setUp()
		// Put setup code here. This method is called before the invocation of each test method in the class.
	}
	
	override func tearDown() {
		// Put teardown code here. This method is called after the invocation of each test method in the class.
		super.tearDown()
	}
	
	func test() {
		let testCase = BinaryOpTest<RDivOpDelegate, RDivOperator>()
		testCase.testAll()
	}
}

