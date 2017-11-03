//
//  pow_op_test.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/13/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
import Accelerate
@testable import Serrano

class PowOpDelegate: OperatorDelegateConvBinaryOp {
	
	required public convenience init(compareBlock: (([Tensor], Tensor) -> Void)?)  {
		let blcok =  {(inputTensors: [Tensor], resultTensor: Tensor) -> Void in
			let inputReaderA = inputTensors[0].floatValueReader
			let inputReaderB = inputTensors[1].floatValueReader
			let resultReader = resultTensor.floatValueReader
			
			for i in 0..<resultTensor.count {
				let val = powf(inputReaderA[i], inputReaderB[i])
				if val.isNaN || val.isInfinite || resultReader[i].isNaN || resultReader[i].isInfinite { continue }
				XCTAssertEqualWithAccuracy(val, resultReader[i], accuracy: abs(val*0.001), "\(val, resultReader[i], inputReaderA[i],inputReaderB[i], val.isNaN , resultReader[i].isNaN)")
			}
		}
		self.init(block: blcok)
	}
}

class PowOpTest: XCTestCase {
	
	override func setUp() {
		super.setUp()
		// Put setup code here. This method is called before the invocation of each test method in the class.
	}
	
	override func tearDown() {
		// Put teardown code here. This method is called after the invocation of each test method in the class.
		super.tearDown()
	}
	
	func test() {
		let testCase = BinaryOpTest<PowOpDelegate, PowOperator>()
		testCase.testAll()
	}

//	func testPof() {
//		let a = Tensor(repeatingValue: 0.0, tensorShape: TensorShape(dataType: .float, shape: [2, 3]))
//		let b = Tensor(repeatingValue: 0.0, tensorShape: TensorShape(dataType: .float, shape: [2, 3]))
//		let c = randomTensor(fromShape: TensorShape(dataType: .float, shape: [2, 3]))
//		print(a.flatArrayFloat())
//		print(b.flatArrayFloat())
//		print("==")
//		var count = Int32(c.count)
//		vvpowf(c.contentsAddress, b.contentsAddress, a.contentsAddress, &count)
//		print(a.flatArrayFloat())
//		print(b.flatArrayFloat())
//		print(c.flatArrayFloat())
//
//		print(powf(0, 0))
//	}
}
