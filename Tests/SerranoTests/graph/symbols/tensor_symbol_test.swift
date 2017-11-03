//
//  tensor_symbol_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/10/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class tensor_symbol_test: XCTestCase {
    
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
			
			let shape = randomShape(dimensions: 2, dimensionSizeRange: [1, 10], dataType: .float)
			
			let tensorSymbol = SerranoTensorSymbol(label, dataSource: dataSource, shape: shape)
			XCTAssertTrue(label == tensorSymbol.symbolLabel)
			XCTAssertTrue(dataSource == tensorSymbol.dataSource)
			XCTAssertTrue(shape .== tensorSymbol.shape)
			
			print("Finish test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public override func bindData(_ data:GraphSupportedBindingDataType)
	*/
	func testBindData() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let symbol = SerranoTensorSymbol(randomString(length: 6), dataSource: .User,
			                                 shape: randomShape(dimensions: 2, dimensionSizeRange: [1, 10], dataType: .float))
			
			// bind
			var result:Bool = false
			if i % 2 != 0 {
				// invalid cases
				if i % 3 == 0 {
					// not a tensor
					result = symbol.bindData(5.0)
				} else {
					// invalid shape
					let tensor = randomTensor(dimensions: randomInt([3, 5]), dimensionSizeRange: [1, 10], dataType: .float)
					result = symbol.bindData(tensor)
				}
			} else {
				// valid
				result = symbol.bindData(randomTensor(fromShape: symbol.shape))
			}
			
			if i % 2 != 0 {
				XCTAssertFalse(result)
				XCTAssertNil(symbol.bindedData)
			} else {
				XCTAssertTrue(result)
				XCTAssertNotNil(symbol.bindedData)
			}
			

			print("Finish test \(i+1)\n\n")
		}
	}
}
