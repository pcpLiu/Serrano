//
//  operator_symbol_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/11/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class operator_symbol_test: XCTestCase {
    
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
			print("Test case \(i+1)...")
			
			let label = randomString(length: 5)
			let op = PowOperator()
			
			let inputA = SerranoTensorSymbol("A", dataSource: .User, shape: TensorShape(dataType: .float, shape: [2,2]))
			let inputB = SerranoTensorSymbol("B", dataSource: .User, shape: TensorShape(dataType: .float, shape: [2,2]))
			
			let opSymbol = SerranoOperatorSymbol(label, serranoOperator: op, inputSymbols: [inputA, inputB])
			
			XCTAssertTrue(opSymbol.symbolLabel == label)
			XCTAssertTrue(ObjectIdentifier(opSymbol.serranoOperator as! PowOperator) == ObjectIdentifier(op))
			
			// inputs
			XCTAssertTrue(opSymbol.inputSymbols.contains(where: { (symbol) -> Bool in
				return symbol.UID == inputA.UID
			}))
			XCTAssertTrue(opSymbol.inputSymbols.contains(where: { (symbol) -> Bool in
				return symbol.UID == inputB.UID
			}))
			
			// inbouds
			XCTAssertTrue(opSymbol.inBounds.contains(where: { (symbol) -> Bool in
				return symbol.UID == inputA.UID
			}))
			XCTAssertTrue(opSymbol.inBounds.contains(where: { (symbol) -> Bool in
				return symbol.UID == inputB.UID
			}))
			
			print("Finish test case \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func outputSymbols() -> [TensorSymbol]
	*/
	func testOutputSymbols() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test case \(i+1)..")
			
			var inputSymbols = [TensorSymbol]()
			inputSymbols.append(SerranoTensorSymbol(dataSource: .User,
			                                        shape: randomShape(dimensions: 2, dimensionSizeRange: [1, 5], dataType: .double)))
			print("Input symbol: ", inputSymbols.last!.shape.description)
			for _ in 0..<randomInt([1, 6]) {
				inputSymbols.append(SerranoTensorSymbol(dataSource: .User, shape: inputSymbols.last!.shape))
				print("Input symbol: ", inputSymbols.last!.shape.description)
			}
			
			var opSymbol = SerranoOperatorSymbol(serranoOperator: SinOperator(), inputSymbols: inputSymbols)
			
			// get output symbols
			var outputSymbols = [TensorSymbol]()
			if i % 2 == 0 {
				// one-to-one
				print("OneToOne")
				outputSymbols = opSymbol.outputSymbols()
			} else {
				// constant
				print("Constant")
				opSymbol = SerranoOperatorSymbol(serranoOperator: AddOperator(),
				                                 inputSymbols: Array<TensorSymbol>(inputSymbols[0..<2]))
				outputSymbols = opSymbol.outputSymbols()
			}
			
			// get output symbols checking
			var outputSymbolsCheck = [TensorSymbol]()
			let inputShapes = opSymbol.inputSymbols.map({ (symbol) -> TensorShape in
				return symbol.shape
			})
			let outputShapes = opSymbol.serranoOperator.outputShape(shapeArray: inputShapes)
			for outputShape in outputShapes! {
				outputSymbolsCheck.append(SerranoTensorSymbol(dataSource: .User, shape: outputShape))
			}
			
			// CHECK
			XCTAssertTrue(outputSymbolsCheck.count == outputSymbols.count)
			for (outSymbolCheck, outSymbol) in zip(outputSymbolsCheck, outputSymbols) {
				print("Out symbol check:", outSymbolCheck.shape.description)
				print("Out symbol:", outSymbol.shape.description)
				XCTAssertTrue(outSymbolCheck.shape == outSymbol.shape)
			}
			
			print("Finish test case \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func addToParamSymbols(_ symbol: GraphSymbol)
	*/
	func testAddToParamSymbols() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let symbol = SerranoOperatorSymbol(serranoOperator: AddOperator(), inputSymbols: [TensorSymbol]())
			
			// generate param symbols
			var symbols = [GraphSymbol]()
			for _ in 0..<randomInt([1, 5]) {
				symbols.append(SerranoTensorSymbol(dataSource: .User, shape: TensorShape(dataType: .float, shape: [3, 4])))
				print("Symbol:", (symbols.last! as! TensorSymbol).shape)
			}
			
			// add
			for paramBoundSymbol in symbols {
				// may add multi times
				for _ in 0..<randomInt([1, 5]) {
					symbol.addToParamSymbols(paramBoundSymbol)
				}
			}
			print("paramSymbols count:", symbol.paramSymbols.count)
			
			// check
			XCTAssertTrue(symbols.count == symbol.paramSymbols.count)
			for (paramSymbol, symbolCheck) in zip(symbol.paramSymbols, symbols) {
				XCTAssertEqual(paramSymbol as! SerranoTensorSymbol, symbolCheck as! SerranoTensorSymbol)
			}
			
			print("Finish test \(i+1)\n\n")
		}
	}
}
