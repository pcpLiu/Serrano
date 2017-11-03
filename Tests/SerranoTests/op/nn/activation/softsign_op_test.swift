//
//  softsign_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/27/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class SoftsignOpDelegate: OperatorDelegateConvUnaryOp {
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
			XCTAssertEqual(rawTensor.count, resultTensor.count)
			let readerReader = rawTensor.floatValueReader
			let resultReader = resultTensor.floatValueReader
			for i in 0..<rawTensor.count {
				let val = readerReader[i] / (abs(readerReader[i]) + 1)
				if val.isNaN || val.isInfinite || resultReader[i].isInfinite || resultReader[i].isNaN {continue}
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

class SoftsignOpTest: XCTestCase {
	
	override func setUp() {
		super.setUp()
		// Put setup code here. This method is called before the invocation of each test method in the class.
	}
	
	override func tearDown() {
		// Put teardown code here. This method is called after the invocation of each test method in the class.
		super.tearDown()
	}
	
	func test() {
		let testCase = UnarOpTest<SoftsignOpDelegate, SoftsignOperator>()
		testCase.testAll()
	}
}
