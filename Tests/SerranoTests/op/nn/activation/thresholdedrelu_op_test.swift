//
//  thresholdedReLU_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/9/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//
import XCTest
@testable import Serrano

class ThresholdedReLUOpDelegate: OperatorDelegateConvUnaryOp {
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
		let alpha: Float = 1.0
		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
			XCTAssertEqual(rawTensor.count, resultTensor.count)
			let readerReader = rawTensor.floatValueReader
			let resultReader = resultTensor.floatValueReader
			for i in 0..<rawTensor.count {
				var val = readerReader[i]
				if val <= alpha { val = 0.0 }
				if val.isNaN || val.isInfinite || resultReader[i].isInfinite || resultReader[i].isNaN {continue}
				XCTAssertEqual(val, resultReader[i], accuracy: max(0.001, abs(val*0.001)))
			}
		}
		self.init(block: blcok)
	}
}

class ThresholdedReLUOpTest: XCTestCase {
	
	override func setUp() {
		super.setUp()
		// Put setup code here. This method is called before the invocation of each test method in the class.
	}
	
	override func tearDown() {
		// Put teardown code here. This method is called after the invocation of each test method in the class.
		super.tearDown()
	}
	
	func test() {
		let testCase = UnarOpTest<ThresholdedReLUOpDelegate, ThresholdedReLUOperator>()
		testCase.testAll()
	}
}
