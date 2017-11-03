//
//  softmax_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/2/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
import Dispatch
import Metal
@testable import Serrano


import XCTest
@testable import Serrano

class SoftmaxOpDelegate: OperatorDelegateConvUnaryOp {
	
	public var dim: Int = -1
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
			print()
		}
		self.init(block: blcok)
	}
	
	public override func compare() {
		XCTAssertTrue(self.resultTensors.first!.count == self.veryfyTensors.first!.count)
		
		for tensorIndex in 0..<self.resultTensors.count {
			let rawTensor = self.veryfyTensors[tensorIndex]
			let resultTensor = self.resultTensors[tensorIndex]
			
			/// calcualte verify tensor
			// last dim convert
			var reduceDim = self.dim
			if reduceDim == -1 { reduceDim = rawTensor.rank - 1 }
			// exp
			let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: rawTensor.shape)
			let expOp = ExpOperator(inputTensors: [rawTensor], outputTensors: [verifyTensor])
			expOp.compute(.CPU)
			// reduce
			let reduceSumOp = ReduceSumOperator(axis: [reduceDim], keepDim: true)
			let intermediateShape = reduceSumOp.outputShape(shapeArray: [verifyTensor.shape])!.first!
			let intermediateTensor = SerranoResourceManager.globalManager.allocateTensor(intermediateShape)
			reduceSumOp.inputTensors = [verifyTensor]
			reduceSumOp.outputTensors = [intermediateTensor]
			reduceSumOp.compute(.CPU)
			//
			let _ = verifyTensor .&/ intermediateTensor
			
			// compare
			XCTAssertEqual(verifyTensor.count, resultTensor.count)
			let readerVerify = verifyTensor.floatValueReader
			let readerResult = resultTensor.floatValueReader
			for i in 0..<readerResult.count {
				let val = readerVerify[i]
				if val.isInfinite || val.isNaN { continue }
				XCTAssertEqualWithAccuracy(val, readerResult[i], accuracy: abs(val*0.001))
			}
		}
	}
}

class SoftmaxOpTest: XCTestCase {
	
	override func setUp() {
		super.setUp()
		// Put setup code here. This method is called before the invocation of each test method in the class.
	}
	
	override func tearDown() {
		// Put teardown code here. This method is called after the invocation of each test method in the class.
		super.tearDown()
	}
	
	func test() {
		let testCase = SoftmaxOverrideOpTest<SoftmaxOpDelegate, SoftmaxOperator>()
		testCase.testAll()
	}
}

class SoftmaxOverrideOpTest<OpDelegate: SoftmaxOpDelegate, SoftmaxOp: SoftmaxOperator>: UnarOpTest<OperatorDelegateConvUnaryOp, UnaryOperator> {
	/// Override output shapce check 
	override func testOuputShapesCheck() {
		let numCase = 100
		let op = SoftmaxOp()
		for i in 0..<numCase {
			print("Test case \(i+1)...")
			
			// generate shapes
			var shapes = [TensorShape]()
			var minRank = Int.max
			for _ in 0..<randomInt([2, 5]) {
				shapes.append(randomShape(dimensions: 3, dimensionSizeRange: [1, 10], dataType: .float))
				minRank = min(minRank, shapes.last!.rank)
				print("Generate shape: \(shapes.last!.description)")
			}
			
			// setup dim
			if i % 2 == 0 {
				// valid
				if i % 3 == 0 {
					// setup not as last dim
					op.dim = randomInt([0, minRank])
				} else {
					// setup as last dim
					op.dim = -1
				}
			} else {
				// invalid
				op.dim = randomInt([minRank + 1,  20])
			}
			
			let outputShapes = op.outputShape(shapeArray: shapes)
			if i % 2 == 0 {
				XCTAssertNotNil(outputShapes)
			} else {
				XCTAssertNil(outputShapes)
			}
			
			print("Finish Test case \(i+1)\n\n")
		}
	}
	
	/// Override to setup dim
	override func testCompute() {
		let caseNum = 10
		let op = SoftmaxOp()
		
		// configure engine
		let (_, msg) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
		
		// setup delegate
		let delegate = OpDelegate(compareBlock: nil)
		let workingGroup = DispatchGroup()
		delegate.dispatchGroup = workingGroup
		op.computationDelegate = delegate
		
		
		for i in 0..<caseNum {
			print("Test case \(i+1)...")
			
			// generate tensors
			var inputTensors = [Tensor]()
			var outputTensors = [Tensor]()
			if i < 8 { // smaller tensors
				for _ in 0..<randomInt([1, 3]) {
					let shape = randomShape(dimensions: 2, dimensionSizeRange: [100, 200], dataType: .float)
					inputTensors.append(randomTensor(fromShape: shape))
					outputTensors.append(randomTensor(fromShape: shape))
					print("Generate Input tensor: \(inputTensors.last!.description)")
					print("Generate Output tensor: \(outputTensors.last!.description)")
				}
			} else { // large tensors
				let shape = randomShape(dimensions: 2, dimensionSizeRange: [1000, 2000], dataType: .float)
				inputTensors.append(randomTensor(fromShape: shape))
				outputTensors.append(randomTensor(fromShape: shape))
				print("Generate Input tensor: \(inputTensors.last!.description)")
				print("Generate Output tensor: \(outputTensors.last!.description)")
			}
			
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			delegate.veryfyTensors = inputTensors
			
			let inputtensor_hash_values = op.inputTensors!.map { $0._dataMemoryBaseAdrress.hashValue }
			let outputtesnor_hash_values  = op.outputTensors!.map { $0._dataMemoryBaseAdrress.hashValue }
			print("inputtensor_hash_values: \(inputtensor_hash_values)")
			print("outputtesnor_hash_values: \(outputtesnor_hash_values)")
			
			// setup dim
			if i % 3 == 0 {
				// not last dim
				op.dim = 0
			}
			delegate.dim = op.dim
			
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
			print("Finish Test \(i+1)\n\n\n")
		}

	}
}

