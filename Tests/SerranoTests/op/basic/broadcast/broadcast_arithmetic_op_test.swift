//
//  broadcast_arithmetic_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/5/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
import Dispatch
@testable import Serrano


fileprivate func generateShapes(targetShape: TensorShape, valid: Bool) -> [TensorShape] {
	let targetShapeReversed = Array(targetShape.shapeArray.reversed())
	var shapesReversed = [[Int]]()
	
	if valid {
		//valid
		for _ in 0..<2 {
			var newShapeRevsered = Array(targetShapeReversed)
			
			if randomInt([100, 10000000]) % 3 == 0 { // remove last dim
				for _ in 0..<randomInt([1, targetShapeReversed.count]) {
					if newShapeRevsered.count == 1 { break }
					newShapeRevsered.removeLast()
				}
			} else if randomInt([100, 10000000]) % 2 == 0 { // decrease dim size  to 1
				let index = randomInt([0, targetShapeReversed.count - 1])
				newShapeRevsered[index] = 1
			}
			shapesReversed.append(newShapeRevsered)
		}
	} else {
		//invalid
		for _ in 0..<2 {
			var newShapeRevsered = Array(targetShapeReversed)
			
			if randomInt([100, 10000000]) % 3 == 0 &&  newShapeRevsered.count != 1 { // make random dim size
				let index = randomInt([0, targetShapeReversed.count - 1])
				let randSize = randomInt([100, 1000])
				if randSize == newShapeRevsered[index] || randSize == 1 {
					newShapeRevsered[index] += randSize
				} else {
					newShapeRevsered[index] = randSize
				}
			} else { // remove all
				newShapeRevsered.removeAll()
			}
			shapesReversed.append(newShapeRevsered)
		}
	}
	
	// generate shapes
	var shapes = [TensorShape]()
	for reversedShape in shapesReversed {
		let tensorShape = TensorShape(dataType: .float, shape: Array(reversedShape.reversed()))
		shapes.append(tensorShape)
	}
	
	return shapes
}

class OperatorDelegateConvBroadcastArithmeticOp: OperatorDelegateConv {
	
	public var compareBlock: ([Tensor], [Tensor]) -> Void
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)? = nil) {
		let blcok =  {(veryfyTensors: [Tensor], resultTensors: [Tensor]) -> Void in
			print("OVERRIDE")
		}
		self.init(block: blcok)
	}
	
	public init(block: @escaping ([Tensor], [Tensor]) -> Void) {
		self.compareBlock = block
		super.init()
	}
	
	override public func compare() {
		// do broadcasting
		var tensorA = self.veryfyTensors[0]
		var tensorB = self.veryfyTensors[1]
		if tensorA.shape != tensorB.shape {
			let broadcastOp = BroadcastOperator(targetShape: max(tensorA.shape, tensorB.shape))
			if tensorA.shape < tensorB.shape {
				// broadcast A
				broadcastOp.inputTensors = [ self.veryfyTensors[0]]
				tensorA = SerranoResourceManager.globalManager.allocateTensor(tensorB.shape)
				broadcastOp.outputTensors = [tensorA]
			} else if tensorA.shape > tensorB.shape {
				// broadcast B
				broadcastOp.inputTensors = [ self.veryfyTensors[1]]
				tensorB = SerranoResourceManager.globalManager.allocateTensor(tensorA.shape)
				broadcastOp.outputTensors = [tensorB]
			}
			broadcastOp.compute(.CPU)
		}
		
		self.compareBlock([tensorA, tensorB], self.resultTensors)
	}
}

class BroadcastArithmeticOpTest<Op: BroadcastArithmeticOperator, Delegate: OperatorDelegateConvBroadcastArithmeticOp>: XCTestCase {
	
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
	
	func testAll() {
		self.testInit()
		self.testOutputShape()
		self.testInputOutputTensorsCheck()
		self.testCompute()
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target
		init
	*/
	func testInit() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let label = randomString(length: 5)
			let op = Op(computationDelegate: nil)
			op.operatorLabel = label
			XCTAssertEqual(op.operatorLabel, label)
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
	*/
	func testOutputShape() {
		let numCase = 100
		let op = Op()
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// generate valid input shape
			var inputShapes = [TensorShape]()
			var maxShape: TensorShape?
			inputShapes.append(randomShape(dimensions: randomInt([1, 5]), dimensionSizeRange: [2, 10], dataType: .float))

			
			if i % 2 == 0 {
				// valid
				inputShapes.append(generateShapes(targetShape: inputShapes.last!, valid: true).first!)
				maxShape = inputShapes.max()
			} else {
				// invalid
				inputShapes.append(generateShapes(targetShape: inputShapes.last!, valid: false).first!)
			}
			print("Input shape A: \(inputShapes[0].description)")
			print("Input shape B: \(inputShapes[1].description)")

			
			let outputShape = op.outputShape(shapeArray: inputShapes)?.first
			if i % 2 == 0 {
				XCTAssertNotNil(outputShape)
				print("Output shape: \(outputShape!.description)")
				print("Max shape: \(maxShape!.description)")
				XCTAssertEqual(maxShape!, outputShape!)
			} else {
				XCTAssertNil(outputShape)
			}
			
			print("Finish Test  \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
	*/
	func testInputOutputTensorsCheck() {
		let numCase = 100
		let op = Op()
		for i in 0..<numCase {
			print("Test  \(i+1)...")
			
			var inputTensors = [Tensor]()
			var outputTensors = [Tensor]()
			
			// generate valid input tensors
			inputTensors.append(randomTensor(dimensions: randomInt([1, 5]), dimensionSizeRange: [1, 10], dataType: .float))
			print("Input tensor A: \(inputTensors.last!.description)")
			inputTensors.append(randomTensor(fromShape: generateShapes(targetShape: inputTensors.last!.shape, valid: true).first!))
			print("Input tensor B: \(inputTensors.last!.description)")
			
			// generate output tensors
			if i % 2 == 0 {
				// valid
				outputTensors.append(randomTensor(fromShape: inputTensors[0].shape))
			} else {
				// invalid
				outputTensors.append(randomTensor(fromShape: generateShapes(targetShape: inputTensors[0].shape, valid: false).first!))
			}
			print("Output tensor: \(outputTensors.last!.description)")
			
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			
			let (pass, msg) = op.inputOutputTensorsCheck()
			if i % 2 == 0 {
				XCTAssertTrue(pass)
			} else {
				XCTAssertFalse(pass)
				print(msg)
			}
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test  \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target
	public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) 
	public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
	*/
	func testCompute() {
		let numCase = 10
		let op = Op()
		
		// configure engine
		let (_, msg) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
		// setup delegate
		let delegate = Delegate()
		let workingGroup = DispatchGroup()
		delegate.dispatchGroup = workingGroup
		op.computationDelegate = delegate
		
		for i in 0..<numCase {
			print("Test case \(i+1)...")
			
			var inputTensors = [Tensor]()
			var outputTensors = [Tensor]()
			
			// generate input tensors
			var shapeA = randomShape(dimensions: randomInt([1, 4]), dimensionSizeRange: [1, 5], dataType: .float)
			if i >= 8 {
				shapeA = randomShape(dimensions: 2, dimensionSizeRange: [1000, 1500], dataType: .float)
			}
			let shapeB = generateShapes(targetShape: shapeA, valid: true).first!
			if i % 3 == 0 {
				inputTensors.append(randomTensor(fromShape: shapeA))
				inputTensors.append(randomTensor(fromShape: shapeB))
			} else {
				inputTensors.append(randomTensor(fromShape: shapeB))
				inputTensors.append(randomTensor(fromShape: shapeA))
			}
			print("Input tensor A: \(inputTensors[0].description)")
			print("Input tensor B: \(inputTensors[1].description)")

			// output tensor
			let maxShape = max(inputTensors[0].shape, inputTensors[1].shape)
			outputTensors.append(randomTensor(fromShape: maxShape))
			print("Output tensor: \(outputTensors.last!.description)")
			
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			delegate.veryfyTensors = inputTensors
			
			if i % 2 == 0 {
				print("Run on CPU")
				workingGroup.enter()
				op.computeAsync( .CPU)
			} else {
				print("Run on GPU")
				if !SerranoEngine.configuredEngine.hasAvailableGPU() {
					print("No gpu available, give up Test \(i+1)\n\n\n)")
					continue
				}
				workingGroup.enter()
				op.computeAsync( .GPU)
			}
			workingGroup.wait()
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test case \(i+1)\n\n")
		}
		
	}
}
