////
////  PReLU_op_test.swift
////  serrano
////
////  Created by ZHONGHAO LIU on 7/10/17.
////  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
////
//
//import XCTest
//@testable import Serrano
//
//class PReLUOpDelegate: OperatorDelegateConvUnaryOp {
//	
//	public var alpha: [Tensor] = [Tensor]()
//	
//	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)? = nil) {
//		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
//			print("NOT USE")
//		}
//		self.init(block: blcok)
//	}
//	
//	public convenience init(alpha: [Tensor]) {
//		self.init(compareBlock: nil)
//		self.alpha = alpha
//	}
//	
//	override public func compare() {
//		XCTAssertEqual(self.alpha.count, self.resultTensors.count)
//		XCTAssertEqual(self.veryfyTensors.count, self.resultTensors.count)
//		for tensorIndex in 0..<self.resultTensors.count {
//			let inputTensor = self.veryfyTensors[tensorIndex]
//			let resultTensor = self.resultTensors[tensorIndex]
//			let alphaTensor = self.alpha[tensorIndex]
//			XCTAssertEqual(inputTensor.count, resultTensor.count)
//			XCTAssertEqual(alphaTensor.count, resultTensor.count)
//			for i in 0..<resultTensor.count {
//				var val = inputTensor.floatValueReader[i]
//				if val < 0.0 { val *= alphaTensor.floatValueReader[i] }
//				XCTAssertEqualWithAccuracy(val, resultTensor.floatValueReader[i], accuracy: abs(val*0.001))
//			}
//		}
//	}
//}
//
//
//class PReLUOpTest: XCTestCase {
//	
//	override func setUp() {
//		super.setUp()
//		// Put setup code here. This method is called before the invocation of each test method in the class.
//	}
//	
//	override func tearDown() {
//		// Put teardown code here. This method is called after the invocation of each test method in the class.
//		super.tearDown()
//	}
//	
//	func test() {
//		let testCase = UnarOpTest<PReLUOpDelegate, PReLUOperator>()
//		testCase.testInit()
//		testCase.testOuputShapesCheck()
//	}
//	
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	/**
//	Target:
//	public override func inputOutputTensorsCheck() -> (check: Bool, msg: String)
//	*/
//	func testInputOutputTensorsCheck() {
//		let numCase = 100
//		let op = PReLUOperator()
//		for i in 0..<numCase {
//			print("Test \(i+1)...")
//			
//			var inputTensors = [Tensor]()
//			var outputTensors = [Tensor]()
//			var alpha: [Tensor]? = [Tensor]()
//			let defaultAlpha = randomFloat()
//			
//			// generate valid tensors
//			for _ in 0..<randomInt([1, 3]) {
//				inputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [10, 20], dataType: .float))
//				print("Generate Input: \(inputTensors.last!.description)")
//				outputTensors.append(randomTensor(fromShape: inputTensors.last!.shape))
//				print("Generate Output: \(outputTensors.last!.description)")
//				alpha!.append(randomTensor(fromShape: inputTensors.last!.shape))
//				print("Generate Alpha: \(alpha!.last!.description)")
//			}
//			
//			if i % 2 == 0 {
//				// valid
//				if i % 3 == 0 {
//					// set alph to nil
//					alpha = nil
//				}
//			} else {
//				// invalid
//				let caseRand = randomInt([0, 3])
//				if caseRand % 3 == 0 {
//					// super check not passing
//					outputTensors.removeLast()
//				} else if caseRand % 3 == 1 {
//					// alpha count not matching
//					alpha!.removeLast()
//				} else {
//					// alph dim not match
//					var shape = alpha!.last!.shape.shapeArray
//					shape.removeLast()
//					alpha![alpha!.count - 1] = randomTensor(fromShape: TensorShape(dataType: .float, shape: shape))
//				}
//			}
//			
//			op.inputTensors = inputTensors
//			op.outputTensors = outputTensors
//			op.alpha = alpha
//			op.defaultAlphaValue = defaultAlpha
//			
//			let (pass, msg) = op.inputOutputTensorsCheck()
//			if i % 2 == 0 {
//				XCTAssertTrue(pass)
//				if i % 3 == 0 {
//					XCTAssertNotNil(op.alpha)
//					// check alpha
//					for alphaTensor in op.alpha! {
//						for eleIndex in 0..<alphaTensor.count {
//							XCTAssertEqual(defaultAlpha, alphaTensor.floatValueReader[eleIndex])
//						}
//					}
//				}
//			} else {
//				XCTAssertFalse(pass)
//				print(msg)
//			}
//			
//			SerranoResourceManager.globalManager.releaseAllResources()
//			print("Finish Test \(i+1)\n\n")
//		}
//	}
//	
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	/**
//	Target:
//	compute()
//	*/
//	func testCompute() {
//		let caseNum = 10
//		let op = PReLUOperator()
//		
//		// configure engine
//		let (_, msg) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
//
//		
//		for i in 0..<caseNum {
//			print("Test \(i+1)..")
//			
//			var inputTensors = [Tensor]()
//			var outputTensors = [Tensor]()
//			var alpha: [Tensor]? = [Tensor]()
//			let defaultAlpha = randomFloat()
//			
//			// generate valid tensors
//			if i < 8 {
//				// small
//				inputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [100, 200], dataType: .float))
//				print("Generate Input: \(inputTensors.last!.description)")
//				outputTensors.append(randomTensor(fromShape: inputTensors.last!.shape))
//				print("Generate Output: \(outputTensors.last!.description)")
//				alpha!.append(randomTensor(fromShape: inputTensors.last!.shape))
//				print("Generate Alpha: \(alpha!.last!.description)")
//			} else {
//				// large
//				inputTensors.append(randomTensor(dimensions: 2, dimensionSizeRange: [1000, 1500], dataType: .float))
//				print("Generate Input: \(inputTensors.last!.description)")
//				outputTensors.append(randomTensor(fromShape: inputTensors.last!.shape))
//				print("Generate Output: \(outputTensors.last!.description)")
//				alpha!.append(randomTensor(fromShape: inputTensors.last!.shape))
//				print("Generate Alpha: \(alpha!.last!.description)")
//			}
//			
//			op.inputTensors = inputTensors
//			op.outputTensors = outputTensors
//			op.alpha = alpha
//			op.defaultAlphaValue = defaultAlpha
//			
//			// setup delegate
//			let delegate = PReLUOpDelegate(compareBlock: nil)
//			let workingGroup = DispatchGroup()
//			delegate.dispatchGroup = workingGroup
//			delegate.alpha = alpha!
//			delegate.veryfyTensors = inputTensors
//			op.computationDelegate = delegate
//			
//			if i % 2 == 0 {
//				print("Run on CPU")
//				workingGroup.enter()
//				op.computeAsync(.CPU)
//			} else {
//				if !SerranoEngine.configuredEngine.hasAvailableGPU() {
//					print("No available GPU. Give up test.\n\n")
//					continue
//				}
//				workingGroup.enter()
//				op.computeAsync(.GPU)
//			}
//			
//			workingGroup.wait()
//			SerranoResourceManager.globalManager.releaseAllResources()
//			print("Finish Test \(i+1)\n\n")
//		}
//		
//	}
//}

//TODO: re-design
