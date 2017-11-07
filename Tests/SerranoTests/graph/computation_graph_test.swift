//
//  computation_graph_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/11/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class ComputationGraphTest: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
	
	
	func constructGraph() -> ComputationGraph {
		let g = ComputationGraph()
		
		// input tensors
		var dataTensoSymbols = [TensorSymbol]()
		let shape = TensorShape(dataType: .float, shape: [randomInt([100, 100]), randomInt([100, 100]), 3])
		for _ in 0..<randomInt([3, 5]) {
			let symbol = g.tensor(shape: shape)
			dataTensoSymbols.append(symbol)
		}
		
		// add operations
		for _ in 0..<randomInt([1, 10]) {
			// random selece input tensors from previos tensors
			var inbounds = [ dataTensoSymbols[Int(arc4random_uniform(UInt32(dataTensoSymbols.count)))] ]
			while inbounds.count < randomInt([2, max(dataTensoSymbols.count, 5)]) {
				let randomIndex = Int(arc4random_uniform(UInt32(dataTensoSymbols.count)))
				let candidate = dataTensoSymbols[randomIndex]
				if (!inbounds.contains { $0.UID == candidate.UID }) && candidate.shape == inbounds.first!.shape { // make sure binary op work
					inbounds.append(candidate)
				}
			}
			
			// randomlly add new user input tensor symbols
			if randomInt([1, 10]) % 3 == 0 {
				for _ in 0..<randomInt([1, 3]) {
					let tensorSymbol = g.tensor(shape: shape)
					dataTensoSymbols.append(tensorSymbol)
				}
			}
			
			let caseRand = randomInt([1, 10]) % 3
			if caseRand == 0 {
				// unary
				let _ = g.operation(inputs: inbounds, op: ReLUOperator())
			} 
			else if caseRand == 1 {
				// binary
				let _ = g.operation(inputs: [inbounds[0], inbounds[1]], op: AddOperator())
			}
			else {
				// with param tensors
//				let conv = ConvOperator2D(numFilters: 10, kernelSize: [2, 2], inputShape: shape)
//				let _ = g.operation(inputs: inbounds, op: conv)
				let fc = FullyconnectedOperator(inputDim: inbounds[0].shape.shapeArray.reduce(1, *), numUnits: randomInt([100, 150]))
				let _ = g.operation(inputs: inbounds, op: fc)
			}
		}
		
		// remove isolated nodes
		for (UID, _) in (g.symbols.filter {$0.value.inBounds.count == 0 && $0.value.outBounds.count == 0}) {
			g.symbols.removeValue(forKey: UID)
		}
		
		return g
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
			let graph = ComputationGraph(label)
			
			XCTAssertTrue(graph.graphLabel == label)
			print("label", graph.graphLabel, label)
			
			print("Finish test case \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func tensor(_ label: String? = nil, shape: TensorShape) -> TensorSymbol
	*/
	func testTensor() {
		let numCase = 100
		let graph = ComputationGraph()
		
		for i in 0..<numCase {
			print("Test case \(i+1)...")
			
			let shape = randomShape(dimensions: 2, dimensionSizeRange: [1, 5], dataType: .float)
			
			var tensorSymbol: TensorSymbol
			if i % 2 == 0 {
				let label = randomString(length: 5)
				tensorSymbol = graph.tensor(label, shape: shape)
			} else {
				tensorSymbol = graph.tensor(shape: shape)
			}

			print("tensorSymbol:", tensorSymbol.shape.description)
			XCTAssertTrue(shape == tensorSymbol.shape)
			XCTAssertTrue(graph.symbols[tensorSymbol.UID] as! SerranoTensorSymbol == tensorSymbol as! SerranoTensorSymbol)
			XCTAssertTrue(tensorSymbol.dataSource == .User)
			XCTAssertFalse(graph.sorted)
			
			print("Finish test case \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func scalar(_ label: String? = nil, dataType: TensorDataType) -> ScalarSymbol
	*/
	func testScalar() {
		let numCase = 100
		let graph = ComputationGraph()

		for i in 0..<numCase {
			print("Test case \(i+1)...")
			
			var scalarSymbol: ScalarSymbol
			if i % 2 == 0 {
				let label = randomString(length: 5)
				scalarSymbol = graph.scalar(label, dataType: .double)
			} else {
				scalarSymbol = graph.scalar(dataType: .double)
			}

			
			print("scalarSymbol:", scalarSymbol.dataType)
			XCTAssertTrue(scalarSymbol.dataSource == .User)
			XCTAssertTrue(graph.symbols[scalarSymbol.UID] as! SerranoScalarSymbol == scalarSymbol as! SerranoScalarSymbol)
			XCTAssertFalse(graph.sorted)

			print("Finish test case \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func operation(_ label: String? = nil, inputs: [TensorSymbol], op: ComputableOperator) -> ([TensorSymbol], OperatorSymbol)
	*/
	func testOperation() {
		let numCase = 100
		let graph = ComputationGraph()

		for i in 0..<numCase {
			print("Test case \(i+1)...")
			
			// genreate input tensor symbols
			var inputTensorSymbols = [TensorSymbol]()
			let opSymbol: SerranoOperatorSymbol
			let outSymbols: [TensorSymbol]
			
			if i % 3 == 0 {
				// unary operator
				for _ in 0..<randomInt([1, 4]) {
					let tensorSymbol = graph.tensor(shape: randomShape(dimensions: 2, dimensionSizeRange: [1, 5], dataType: .int))
					inputTensorSymbols.append(tensorSymbol)
				}
				(outSymbols, opSymbol, _) = graph.operation(inputs: inputTensorSymbols, op: AbsOperator()) as! ([TensorSymbol], SerranoOperatorSymbol, [GraphSymbol])
			} else if i % 3 == 1 {
				// binary
				let shape = randomShape(dimensions: 2, dimensionSizeRange: [1, 5], dataType: .int)
				for _ in 0..<2 {
					let tensorSymbol = graph.tensor(shape: shape)
					inputTensorSymbols.append(tensorSymbol)
				}
				(outSymbols, opSymbol, _) = graph.operation(inputs: inputTensorSymbols, op: AddOperator()) as! ([TensorSymbol], SerranoOperatorSymbol, [GraphSymbol])
			} else {
				// conv
				let shape = TensorShape(dataType: .int, shape: [3, randomInt([3, 5]), randomInt([3, 5])])
				for _ in 0..<randomInt([1, 4]) {
					let tensorSymbol = graph.tensor(shape: shape)
					inputTensorSymbols.append(tensorSymbol)
				}
				let convOp = ConvOperator2D(numFilters: randomInt([5, 10]), kernelSize: [2, 2],
				                            padMode: .Valid, channelPosition: .First, inputShape: shape)
				(outSymbols, opSymbol, _) = graph.operation(inputs: inputTensorSymbols, op: convOp) as! ([TensorSymbol], SerranoOperatorSymbol, [GraphSymbol])
			}
			
			// check op added
			XCTAssertTrue(graph.symbols[opSymbol.UID] as! SerranoOperatorSymbol == opSymbol)
			
			// check bounds between opSymbol && inputTensorSymbols
			for tensorSymbol in inputTensorSymbols {
				XCTAssertTrue(opSymbol.inBounds.contains(where: { (symbol) -> Bool in
					symbol.UID == tensorSymbol.UID
				}))
				XCTAssertTrue(tensorSymbol.outBounds.contains(where: { (symbol) -> Bool in
					symbol.UID == opSymbol.UID
				}))
			}
			
			// check bounds between outSymbol && opSymbol
			for tensorSymbol in outSymbols {
				XCTAssertTrue(opSymbol.outBounds.contains(where: { (symbol) -> Bool in
					symbol.UID == tensorSymbol.UID
				}))
				XCTAssertTrue(tensorSymbol.inBounds.contains(where: { (symbol) -> Bool in
					symbol.UID == opSymbol.UID
				}))
			}
			
			// check bounds between opSymbol and parameter symbol if available
			if i % 3 == 2 {
				XCTAssertTrue(opSymbol.paramSymbols.count > 0)
				for paramSymbol in opSymbol.paramSymbols {
					XCTAssertTrue(opSymbol.inBounds.contains(where: { (symbol) -> Bool in
						symbol.UID == paramSymbol.UID
					}))
					XCTAssertTrue(paramSymbol.outBounds.contains(where: { (symbol) -> Bool in
						symbol.UID == opSymbol.UID
					}))
				}
			}
			
			print("Finish test case \(i+1)\n\n")
		}

	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func sortGraph()
	*/
	func testSortGraph() {
		let numCase = 100
		for i in 0..<numCase {
			//TODO: Update testing code for changed api
			
			print("Finish test case \(i+1)\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	internal func userInputBindCheck() -> (valid: Bool, msg: String)
	*/
	func testUserInputBindCheck() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test case \(i+1)")
			
			let g = constructGraph()
			
			let userInputSymbol = g.symbols.filter {$0.value.symbolType.isDataSymbol() && ($0.value as! DataSymbol).dataSource == .User}
			for (_, symbol) in userInputSymbol {
				let dataSymbol = symbol as! DataSymbol
				if symbol.symbolType == .Tensor {
					let result = dataSymbol.bindData(randomTensor(fromShape: (symbol as! TensorSymbol).shape))
					XCTAssertTrue(result)
				} else {
					let result = dataSymbol.bindData(randomFloat())
					XCTAssertTrue(result)
				}
			}
			
			// select to not bind
			if i % 2 != 0 {
				for index in 0..<randomInt([1, userInputSymbol.count]) {
					let symbol = userInputSymbol[index].value
					if symbol.symbolType == .Tensor {
						var tensorSymbol = symbol as! TensorSymbol
						tensorSymbol.bindedData = nil
					} else {
						var scalarSymbol = symbol as! ScalarSymbol
						scalarSymbol.bindedData = nil
					}
				}
			}
			
			let (result, msg) = g.userInputBindCheck()
			if i % 2 == 0 {
				XCTAssertTrue(result)
			} else {
				XCTAssertFalse(result)
				print(msg)
			}
			
			print("Finish test case \(i+1)\n")
		}
	}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func verifyGraph() -> (valid: Bool, msg: String)
	internal func checkShapeChain() -> (valid: Bool, msg: String)
	*/
	func testVerifyGraph() {
		let numCase = 50
		for i in 0..<numCase {
			print("Test case \(i+1)")
			
			let graph = constructGraph()
			
			// sort
			graph.sortGraph()
			
			// bind user input data
			let userInputSymbol = graph.symbols.filter {$0.value.symbolType.isDataSymbol() && ($0.value as! DataSymbol).dataSource == .User}
			for (_, symbol) in userInputSymbol {
				let dataSymbol = symbol as! DataSymbol

				if symbol.symbolType == .Tensor {
					
					let result = dataSymbol.bindData(randomTensor(fromShape: (symbol as! TensorSymbol).shape))
					print("Bind user input for tensor symbol: \((symbol as! TensorSymbol).bindedData!.description)")
					XCTAssertTrue(result)
				} else {
					let result = dataSymbol.bindData(randomFloat())
					XCTAssertTrue(result)
				}
			}
			
			// allocate tensor
			graph.allocateAllTensors()

			// set invalid cases
			if i % 2 != 0 {
				let randCase = randomInt([0, 3])
				if randCase % 3 == 0 {
					// input not bind
					for index in 0..<randomInt([1, userInputSymbol.count]) {
						let symbol = userInputSymbol[index].value
						if symbol.symbolType == .Tensor {
							var tensorSymbol = symbol as! TensorSymbol
							tensorSymbol.bindedData = nil
						} else {
							var scalarSymbol = symbol as! ScalarSymbol
							scalarSymbol.bindedData = nil
						}
					}
					print("Set invalid: input null")
				} else if randCase % 3 == 1 {
					// intermediate tensor not allocated
					let intermSymbols = graph.symbols.filter { $0.value.symbolType.isDataSymbol() && ($0.value as! DataSymbol).dataSource == .Calculation}
					for index in 0..<randomInt([1, intermSymbols.count]) {
						let symbol = intermSymbols[index].value
						if symbol.symbolType == .Tensor {
							var tensorSymbol = symbol as! TensorSymbol
							tensorSymbol.bindedData = nil
						} else {
							var scalarSymbol = symbol as! ScalarSymbol
							scalarSymbol.bindedData = nil
						}
					}
					print("Set invalid: intermediate tensor not allocated")
				} else {
					// tensor shape not match
					let tensorSymbols = graph.symbols.filter {$0.value.symbolType == SymbolType.Tensor}
					for _ in 0..<randomInt([1, tensorSymbols.count]) {
						var tensorSymbol = tensorSymbols[randomInt([0, tensorSymbols.count])].value as! TensorSymbol
						var newShapeArray = [Int]()
						for dim in tensorSymbol.shape.shapeArray {
							newShapeArray.append(dim + 1)
						}
						tensorSymbol.bindedData = SerranoResourceManager.globalManager.allocateUnamangedTensor(TensorShape(dataType: .float, shape: newShapeArray))
						
					}
					print("Set invalid: tensor shape not match")
				}
			}

			let (result, msg) = graph.verifyGraph()
			if i % 2 == 0 {
				XCTAssertTrue(result, msg)
			} else {
				XCTAssertFalse(result)
				print(msg)
			}

			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish test case \(i+1)\n")
		}
		
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func forward(mode: OperatorComputationMode) -> [Tensor]?
	internal func stageOrderCalculate(mode: OperatorComputationMode)
	*/
	func testForward() {
		let numCase = 50
		
		let _ =  SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		
		for i in 0..<numCase {
			print("Test case \(i+1)")
			
			let graph = constructGraph()
			
			graph.forwardPrepare()
			
			let _ = graph.forward(mode: .GPU)
			print("FINISH Test case \(i+1)\n")			
		}
		
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	func testBackwardQuick() {
		let _ =  SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		let g = ComputationGraph()
		g.trainable = true
		g.optimizer = SGDOptimizer(learningRate: 10)
		
		let shape = TensorShape(dataType: .float, shape: [1])
		var input = g.tensor(shape: shape)
		input.updatable = true
		input.bindedData = Tensor(fromFlatArray: [3.3], tensorShape: shape)
		var input2 = g.tensor(shape: shape)
		input2.updatable = true
		input2.bindedData = Tensor(fromFlatArray: [7.3], tensorShape: shape)

		
		let adOp = MultOperator()
		var (out, opsymbol, _) = g.operation(inputs: [input, input2], op: adOp)
		opsymbol.enabledParameterUpdate = true
		
		g.forwardPrepare()
		g.forward(mode: .GPU)
		g.backwardPrepare()
		g.backward(mode: .GPU)
		
		print("Input1:\((input.bindedData! as! Tensor).flatArrayFloat()) grad: \((input.currentGrad! as! Tensor).flatArrayFloat())")
		print("Input2:\((input2.bindedData! as! Tensor).flatArrayFloat()) grad: \((input2.currentGrad! as! Tensor).flatArrayFloat())")
	}
	
}
