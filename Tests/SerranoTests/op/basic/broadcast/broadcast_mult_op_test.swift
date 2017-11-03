//
//  broadcast_mult_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/6/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class OperatorDelegateConvBroadcastMultOp: OperatorDelegateConvBroadcastArithmeticOp {
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
		let blcok =  {(veryfyTensors: [Tensor], resultTensors: [Tensor]) -> Void in
			let readerA = veryfyTensors[0].floatValueReader
			let readerB = veryfyTensors[1].floatValueReader
			let resultReader = resultTensors[0].floatValueReader
			XCTAssertEqual(veryfyTensors[0].count, resultTensors[0].count)
			XCTAssertEqual(veryfyTensors[1].count, resultTensors[0].count)
			for i in 0..<resultTensors[0].count {
				let val = readerA[i] * readerB[i]
				XCTAssertEqualWithAccuracy(val, resultReader[i], accuracy: abs(val*0.0001))
			}
		}
		self.init(block: blcok)
	}
}

class BroadcastMultOpTest: XCTestCase {
	
	override func setUp() {
		super.setUp()
		// Put setup code here. This method is called before the invocation of each test method in the class.
	}
	
	override func tearDown() {
		// Put teardown code here. This method is called after the invocation of each test method in the class.
		super.tearDown()
	}
	
	func test() {
		let testCase = BroadcastArithmeticOpTest<BroadcastMultOperator, OperatorDelegateConvBroadcastMultOp>()
		testCase.testAll()
	}
	
}
