//
//  graph_symbol_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/10/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class SerranoGraphSymbolTest: XCTestCase {
    
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
	public init(symbolType: SymbolType, label: String, inBounds:  [GraphSymbol], outBounds:  [GraphSymbol])
	public convenience init(symbolType: SymbolType, label: String = "")
	*/
	func testInit() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let label = randomString(length: 5)
			
			var type = SymbolType.Operator
			if i % 3 == 1 {
				type = .Scalar
			} else if i % 3 == 2 {
				type = .Tensor
			}
			
			let symbol = SerranoGraphSymbol(label, symbolType: type)
			XCTAssertTrue(symbol.symbolType == type)
			XCTAssertTrue(symbol.symbolLabel == label)
			
			print("Finish test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target: 
	public func generateOutputGraph(_ label: String? = nil) -> Graph in extension
	*/
	func testGraphGenerate() {
		//TODO: test
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func serranoSymbolUIDGenerate() -> String
	*/
	func testSerranoSymbolUIDGenerate() {
		let numCase = 10000
		var UIDSet = Set<String>()
		
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let newUID = serranoSymbolUIDGenerate()
			XCTAssertFalse(UIDSet.contains(newUID), "Generate duplciate UID \(newUID)")
			UIDSet.insert(newUID)
			print("New UID: \(newUID)")
			
			print("Finish test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func addToInBound(_ symbol: GraphSymbol)
	*/
	func testAddToInBound() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let symbol = SerranoGraphSymbol(symbolType: .Tensor)
			
			// generate symbols
			var symbols = [GraphSymbol]()
			for _ in 0..<randomInt([1, 5]) {
				symbols.append(SerranoTensorSymbol(dataSource: .User, shape: TensorShape(dataType: .float, shape: [3, 4])))
				print("Symbol:", (symbols.last! as! TensorSymbol).shape)
			}
			
			// add 
			for inBoundSymbol in symbols {
				// may add multi times
				for _ in 0..<randomInt([1, 5]) {
					symbol.addToInBound(inBoundSymbol)
				}
			}
			print("inBounds count:", symbol.inBounds.count)
			
			// check
			XCTAssertTrue(symbols.count == symbol.inBounds.count)
			for (inSymbol, symbolCheck) in zip(symbol.inBounds, symbols) {
				XCTAssertEqual(inSymbol as! SerranoTensorSymbol, symbolCheck as! SerranoTensorSymbol)
			}
			
			print("Finish test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func addToOutBound(_ symbol: GraphSymbol)
	*/
	func testAddToOutBound() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let symbol = SerranoGraphSymbol(symbolType: .Tensor)
			
			// generate symbols
			var symbols = [GraphSymbol]()
			for _ in 0..<randomInt([1, 5]) {
				symbols.append(SerranoTensorSymbol(dataSource: .User, shape: TensorShape(dataType: .float, shape: [3, 4])))
				print("Symbol:", (symbols.last! as! TensorSymbol).shape)
			}
			
			// add
			for inBoundSymbol in symbols {
				// may add multi times
				for _ in 0..<randomInt([1, 5]) {
					symbol.addToOutBound(inBoundSymbol)
				}
			}
			print("outBounds count:", symbol.outBounds.count)
			
			// check
			XCTAssertTrue(symbols.count == symbol.outBounds.count)
			for (inSymbol, symbolCheck) in zip(symbol.outBounds, symbols) {
				XCTAssertEqual(inSymbol as! SerranoTensorSymbol, symbolCheck as! SerranoTensorSymbol)
			}
			
			print("Finish test \(i+1)\n\n")
		}
	}

}
