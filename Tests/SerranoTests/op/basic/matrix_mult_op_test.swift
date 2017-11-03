//
//  dot_product_test.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano
import Dispatch
import Accelerate

public class OperatorDelegateConvDotMatrixMultOp: OperatorDelegateConv {
	public var transposeA = false
	public var transposeB = false
	
	override public func compare() {
		XCTAssertTrue(self.resultTensors.count == 1)
		XCTAssertTrue(self.veryfyTensors.count == 2)
		
		// Compare resutl
		let tensorA = self.veryfyTensors[0]
		let tensorB = self.veryfyTensors[1]
		let tensorC = self.resultTensors[0]
		
		

		let readA = tensorA.contentsAddress
		let readB = tensorB.contentsAddress
		let verifyTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(tensorC.shape)
		let verifyAddres = verifyTensor.contentsAddress
		var M = Int32(tensorA.shape.shapeArray[0])
		if self.transposeA {
			M = Int32(tensorA.shape.shapeArray[1])
		}
		
		var N = Int32(tensorB.shape.shapeArray[1])
		if self.transposeB {
			N = Int32(tensorB.shape.shapeArray[0])
		}

		var K = Int32(tensorA.shape.shapeArray[1])
		if self.transposeA {
			K = Int32(tensorA.shape.shapeArray[0])
		}
		
		let lda = Int32(tensorA.shape.shapeArray[1])
		let ldb = Int32(tensorB.shape.shapeArray[1])
		let ldc = Int32(verifyTensor.shape.shapeArray[1])
		
		
		cblas_sgemm(CblasRowMajor, cblasTrans(self.transposeA), cblasTrans(self.transposeB), M, N, K,
					1.0, readA, lda, readB, ldb, 0.0, verifyAddres, ldc)
		
		
		let verifyReader = verifyTensor.floatValueReader
		let readC = tensorC.floatValueReader
		for i in 0..<tensorC.count {
			if verifyReader[i].isInfinite || verifyReader[i].isNaN {
				continue
			}
			XCTAssertEqual(verifyReader[i], readC[i], accuracy: abs(verifyReader[i]*0.001))
		}

	}
}


class MatriMultOpTest: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
	

    
//    func testExample() {
//        // This is an example of a functional test case.
//        // Use XCTAssert and related functions to verify your tests produce the correct results.
//    }
//
//    func testPerformanceExample() {
//        // This is an example of a performance test case.
//        self.measure {
//            // Put the code you want to measure the time of here.
//        }
//    }
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
		init....
	*/
	func testInit() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			
			let label = randomString(length: 3)
			let op = MatrixMultOperator(operatorLabel: label)
			XCTAssertEqual(op.operatorLabel, label)
			
			
			print("Finish Test \(i+1)...")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
		public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
	*/
	func testOutputShape() {
		let numCase = 100
		
		for i in 0..<numCase {
			print("Test \(i+1)")
			
			var transposeA = false
			var transposeB = false
			
			var inputShapes = [TensorShape]()
			if i % 2 == 0 {
				// generate valid
				let dim = randomInt([1, 20])
				for _ in 0..<randomInt([1, 5]) {
					inputShapes.append(TensorShape(dataType: .int, shape: [randomInt([1, 20]), dim]))
					print("Generate valid 2D shapeA: \(inputShapes.last!)")
				}

				let shapeB = TensorShape(dataType: .float, shape: [dim, randomInt([1, 20])])
				inputShapes.append(shapeB)
				print("Generate valid 2D shapeB: \(shapeB)")
				
				// transpose
				if randomInt([0, 10]) % 2 == 0 {
					transposeA = true
					for i in 0..<inputShapes.count - 1 {
						inputShapes[i] = TensorShape(dataType: .float, shape: [inputShapes[i].shapeArray[1], inputShapes[i].shapeArray[0]])
						print("Transpose A: \(inputShapes[i])")
					}
				}
				if randomInt([0, 10]) % 4 == 0 {
					transposeB = true
					inputShapes[inputShapes.count - 1] = TensorShape(dataType: .float, shape: [shapeB.shapeArray[1], shapeB.shapeArray[0]])
					print("Transpose B: \(inputShapes.last!)")
				}
			} else {
				// generate invalid
				for _ in 0..<randomInt([1, 5]) {
					inputShapes.append(randomShape(dimensions: randomInt([1,6]), dimensionSizeRange: [1, 20], dataType: .float))
					print("Generate Ivalid shape: \(inputShapes.last!)")
				}
			}
			
			let op = MatrixMultOperator(transposeA: transposeA, transposeB: transposeB)

			let outShapes = op.outputShape(shapeArray: inputShapes)
			if i % 2 == 0 {
				XCTAssertNotNil(outShapes)
				XCTAssertEqual(outShapes!.count, inputShapes.count - 1)
				guard outShapes != nil else {
					print("FAIL \(i+1)")
					return
				}
				var inputB = inputShapes.last!
				if transposeB {
					inputB = inputB.transposed()
				}
				
				for i in 0..<inputShapes.count-1 {
					var inShape = inputShapes[i]
					if transposeA {
						inShape = inShape.transposed()
					}
					let verifyShapeArray = [inShape.shapeArray[0], inputB.shapeArray[1]]
					XCTAssertEqual(verifyShapeArray, outShapes![i].shapeArray)
				}
			} else {
				XCTAssertNil(outShapes)
			}
			
			print("Finish test \(i+1) \n\n")
		}
	}
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
	*/
	func testInputOutputCheck() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			var transposeA = false
			var transposeB = false
			
			var inputTensors: [Tensor]? = [Tensor]()
			var outputTensors: [Tensor]? = [Tensor]()
			
			// generate valid input tensor
			let dim = randomInt([1, 20])
			for _ in 0..<randomInt([1, 5]) {
				inputTensors!.append(randomTensor(fromShape: TensorShape(dataType: .int, shape: [randomInt([1, 20]), dim])))
				print("Input tensor A: \(inputTensors!.last!.description)")
			}
			inputTensors!.append(randomTensor(fromShape: TensorShape(dataType: .int, shape: [dim, randomInt([1, 20])])))
			print("Input tensor B: \(inputTensors!.last!.description)")
			
			// generate valid output tensors
			for input in inputTensors![0..<inputTensors!.count-1] {
				outputTensors!.append((randomTensor(fromShape: TensorShape(dataType: .int, shape: [input.shape.shapeArray[0], inputTensors!.last!.shape.shapeArray[1]]))))
			}
			
			// traspose
			if randomInt([0, 10]) % 2 == 0 {
				transposeA = true
				for i in 0..<inputTensors!.count - 1 {
					inputTensors![i] = randomTensor(fromShape: TensorShape(dataType: .float,
					                              shape: [inputTensors![i].shape.shapeArray[1],
					                                      inputTensors![i].shape.shapeArray[0]]))
					print("Transpose A: \(inputTensors![i].shape.description)")
				}
			}
			if randomInt([0, 10]) % 4 == 0 {
				transposeB = true
				inputTensors![inputTensors!.count - 1] = randomTensor(fromShape: TensorShape(dataType: .float,
				                                                 shape: [inputTensors!.last!.shape.shapeArray[1],
				                                                         inputTensors!.last!.shape.shapeArray[0]]))
				print("Transpose B: \(inputTensors!.last!.shape.description)")
			}
			
			
			// setup invalid cases
			if i % 2 != 0 {
				let randCase = randomInt([0, 6])
				if randCase == 0 {
					// input nil
					inputTensors = nil
					print("Invalid case: input nil")
				} else if randCase == 1 {
					// input count not valid
					let first = inputTensors!.first!
					inputTensors!.removeAll()
					inputTensors!.append(first)
					print("Invalid case: input count not valid")
				} else if randCase == 2 {
					// output nil
					outputTensors = nil
					print("Invalid case: output nil")
				} else if randCase == 3 {
					// output count invalid
					if randomInt([0, 10]) % 3 == 0 {
						outputTensors!.removeLast()
					} else {
						outputTensors!.append(randomTensor(fromShape: outputTensors!.last!.shape))
					}
					print("Invalid case: output count invalid")
				} else if randCase == 4 {
					// input not valid
					var shapeArray = inputTensors!.first!.shape.shapeArray
					shapeArray[0] += randomInt([2, 7])
					shapeArray[1] += randomInt([2, 7])
					inputTensors![0] = randomTensor(fromShape: TensorShape(dataType: .int, shape: shapeArray))
					print("Invalid case: input shape invalid")
				} else if randCase == 5 {
					// output shape not valid
					var shapeArray = outputTensors!.first!.shape.shapeArray
					shapeArray[0] += randomInt([2, 7])
					shapeArray[1] += randomInt([2, 7])
					outputTensors![0] = randomTensor(fromShape: TensorShape(dataType: .int, shape: shapeArray))
					print("Invalid case: output shape not valid")
				}
			}
			
			let op = MatrixMultOperator(transposeA: transposeA, transposeB: transposeB)
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			
			let (pass, msg) = op.inputOutputTensorsCheck()
			if i % 2 == 0 {
				XCTAssertTrue(pass)
			} else {
				XCTAssertTrue(!pass)
				print(msg)
			}
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
	public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
	internal func cpu()
	internal func gpu()
	*/
	func testCompute() {
		let numCase = 20
		let op = MatrixMultOperator()
		
		// gpu initial
		_ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		
		// deleagate
		let delegate = OperatorDelegateConvDotMatrixMultOp()
		op.computationDelegate = delegate
		let workGroup = DispatchGroup()
		delegate.dispatchGroup = workGroup
		
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			var outputTensors = [Tensor]()
			var inputTensors = [Tensor]()
			
			var transposeA = false
			var transposeB = false

			// generate tensors
			var dimRange: [Int] = [200, 400]
			if i >= 18 {
				dimRange = [1000, 1200]
			}
			
			// generate valid
			var shapeA = randomShape(dimensions: 2, dimensionSizeRange: dimRange, dataType: .int)
			var shapeB = TensorShape(dataType: .int, shape: [shapeA.shapeArray[1], randomInt(dimRange)])
			
			// transpose
			if randomInt([0, 10]) % 2 == 0 {
				transposeA = true
				shapeA = shapeA.transposed()
			}
			if randomInt([0, 10]) % 2 == 0 {
				transposeB = true
				shapeB = shapeB.transposed()
			}
			inputTensors.append(randomTensor(fromShape: shapeA))
			print("Input A: \(shapeA), transpose: \(transposeA)")
			
			inputTensors.append(randomTensor(fromShape: shapeB))
			print("Input B: \(shapeB), transpose: \(transposeB)")
			
			var AShapeArray = shapeA.shapeArray
			if transposeA {
				AShapeArray = [AShapeArray[1], AShapeArray[0]]
			}
			var BShapeArray = shapeB.shapeArray
			if transposeB {
				BShapeArray = [BShapeArray[1], BShapeArray[0]]
			}
			
			let outTensor = randomTensor(fromShape: TensorShape(dataType: .int, shape: [AShapeArray[0], BShapeArray[1]]))
			outputTensors.append(outTensor)
			print("Output C: \(outTensor.shape)")
			
			op.transposeA = transposeA
			op.transposeB = transposeB
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			delegate.veryfyTensors = inputTensors
			delegate.transposeA = transposeA
			delegate.transposeB = transposeB

			if i % 2 == 0 {
				print("Run CPU")
				workGroup.enter()
				op.computeAsync(.CPU)
			} else {
				if !SerranoEngine.configuredEngine.hasAvailableGPU() {
					print("No available GPU, give up test.\n\n")
					continue
				}
				workGroup.enter()
				op.computeAsync(.GPU)
			}
			
			workGroup.wait()
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	func testKernelPerformanceSingle() {
		
		let op = MatrixMultOperator()
		
		// gpu initial
		_ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		
		

		var outputTensors = [Tensor]()
		var inputTensors = [Tensor]()
		
		// generate tensors
		let dimRange: [Int] = [1200, 1200]
		
		// generate valid
		let shapeA = randomShape(dimensions: 2, dimensionSizeRange: dimRange, dataType: .int)
		let shapeB = TensorShape(dataType: .int, shape: [shapeA.shapeArray[1], randomInt(dimRange)])
		
	
		inputTensors.append(randomTensor(fromShape: shapeA))
		print("Input A: \(shapeA)")
		
		inputTensors.append(randomTensor(fromShape: shapeB))
		print("Input B: \(shapeB)")
		
		let AShapeArray = shapeA.shapeArray
		let BShapeArray = shapeB.shapeArray
		
		let outTensor = randomTensor(fromShape: TensorShape(dataType: .int, shape: [AShapeArray[0], BShapeArray[1]]))
		outputTensors.append(outTensor)
		print("Output C: \(outTensor.shape)")
		
		op.inputTensors = inputTensors
		op.outputTensors = outputTensors
		
		op.kernel = MatrixMultKernel.Single
		self.measure {
			op.compute(.GPU)
		}
	}
	
	func testKernelPerformanceSubMatrix() {
		
		let op = MatrixMultOperator()
		
		// gpu initial
		_ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		
		
		
		var outputTensors = [Tensor]()
		var inputTensors = [Tensor]()
		
		// generate tensors
		let dimRange: [Int] = [1200, 1200]
		
		// generate valid
		let shapeA = randomShape(dimensions: 2, dimensionSizeRange: dimRange, dataType: .int)
		let shapeB = TensorShape(dataType: .int, shape: [shapeA.shapeArray[1], randomInt(dimRange)])
		
		
		inputTensors.append(randomTensor(fromShape: shapeA))
		print("Input A: \(shapeA)")
		
		inputTensors.append(randomTensor(fromShape: shapeB))
		print("Input B: \(shapeB)")
		
		let AShapeArray = shapeA.shapeArray
		let BShapeArray = shapeB.shapeArray
		
		let outTensor = randomTensor(fromShape: TensorShape(dataType: .int, shape: [AShapeArray[0], BShapeArray[1]]))
		outputTensors.append(outTensor)
		print("Output C: \(outTensor.shape)")
		
		op.inputTensors = inputTensors
		op.outputTensors = outputTensors
		
		op.kernel = MatrixMultKernel.SubMatrix
		self.measure {
			op.compute(.GPU)
		}
	}
}
