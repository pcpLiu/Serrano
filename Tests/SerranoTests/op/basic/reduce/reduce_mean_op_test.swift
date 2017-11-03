//
//  reduce_mean_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/2/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano


class ReduceMeanOpDelegate: OperatorDelegateConvReduceOp {
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
			
		}
		self.init(block: blcok)
	}
}

class ReduceMeanOpTest: XCTestCase {
	
	func test() {
		let testCase = ReduceOpTest<ReduceMeanOpDelegate, ReduceMeanOperator>()
		testCase.testAll()
	}
}
