//
//  reduce_prod_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/1/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano


class ReduceProductOpDelegate: OperatorDelegateConvReduceOp {
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
			
		}
		self.init(block: blcok)
	}
}

class ReduceProductOpTest: XCTestCase {
	
	func test() {
		let testCase = ReduceOpTest<ReduceProductOpDelegate, ReduceProductOperator>()
		testCase.testAll()
	}
}
