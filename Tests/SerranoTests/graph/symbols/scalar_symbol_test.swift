//
//  scalar_symbol_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/11/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class scalar_symbol_test: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	init
	*/
	func testInit() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let label = randomString(length: 5)
			
			var dataSource = SymbolDataSource.User
			if i % 4 == 1 { dataSource = .Default }
			else if i % 4 == 2 { dataSource = .Calculation }
			else if i % 4 == 3 { dataSource = .Other }
			
			let scalarSymbol = SerranoScalarSymbol(label, dataType: .float, dataSource: dataSource)
			XCTAssertTrue(label == scalarSymbol.symbolLabel)
			XCTAssertTrue(dataSource == scalarSymbol.dataSource)
			
			print("Finish test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	@discardableResult
	public override func bindData(_ data:GraphSupportedBindingDataType) -> Bool
	*/
	func testBindData() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let symbol = SerranoScalarSymbol(dataType: .float, dataSource: .User)
			
			var result = false
			let val = randomFloat()
			if i % 2 == 0 {
				// valid
				result = symbol.bindData(val)
			} else {
				// invalid
				result = symbol.bindData(randomTensor(dimensions: 2, dimensionSizeRange: [1,5], dataType: .float))
			}
			
			if i % 2 == 0 {
				XCTAssertTrue(result)
				XCTAssertNotNil(symbol.bindedData)
				XCTAssertEqual(symbol.bindedData! as! Float, val)
			} else {
				XCTAssertFalse(result)
				XCTAssertNil(symbol.bindedData)
			}
			
			print("Finish test \(i+1)\n\n")
		}
	}
	
	
}
