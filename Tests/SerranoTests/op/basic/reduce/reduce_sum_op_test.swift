//
//  reduce_sum_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/28/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano


class ReduceSumOpDelegate: OperatorDelegateConvReduceOp {
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
			
		}
		self.init(block: blcok)
	}
}

class ReduceSumOpTest: XCTestCase {
	
	func test() {
		let testCase = ReduceOpTest<ReduceSumOpDelegate, ReduceSumOperator>()
		testCase.testAll()
	}
}
