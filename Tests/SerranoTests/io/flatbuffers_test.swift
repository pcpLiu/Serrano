//
//  flatbuffers_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 11/5/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class flatbuffers_test: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
	func testLoad() {
		let file = Bundle.main.path(forResource: "test_TensorValue_flatbuffer", ofType: "dat")
		print(file)
		let info = FlatbufferIO.loadSavedParams(file!);
		for (uid, tensor) in info {
			print(uid)
			print(tensor.description)
		}
	}
    
}
