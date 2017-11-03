//
//  reduce_min_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/1/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano


class ReduceMinOpDelegate: OperatorDelegateConvReduceOp {
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
			
		}
		self.init(block: blcok)
	}
}

class ReduceMinOpTest: XCTestCase {
	
	func test() {
		let testCase = ReduceOpTest<ReduceMinOpDelegate, ReduceMinOperator>()
		testCase.testAll()
	}
}
