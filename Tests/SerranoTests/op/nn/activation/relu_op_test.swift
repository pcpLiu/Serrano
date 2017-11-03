//
//  ReLU_op_test.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class ReLUOpDelegate: OperatorDelegateConvUnaryOp {
	
	public var alpha: Float = 0.0
	
	required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)? = nil) {
		let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
			print("NOT USE")
		}
		self.init(block: blcok)
	}
	
	override public func compare() {
		XCTAssertTrue(self.resultTensors.first!.count == self.veryfyTensors.first!.count)
		
		for i in 0..<self.resultTensors.count {
			let rawTensor = self.veryfyTensors[i]
			let resultTensor = self.resultTensors[i]
			XCTAssertEqual(rawTensor.count, resultTensor.count)
			let readerReader = rawTensor.floatValueReader
			let resultReader = resultTensor.floatValueReader
			for i in 0..<rawTensor.count {
				let val = max(self.alpha, readerReader[i])
				if abs(val) < 0.001 {
					XCTAssertEqualWithAccuracy(val, resultReader[i], accuracy: 0.001)
				} else {
					XCTAssertEqualWithAccuracy(val, resultReader[i], accuracy: abs(val*0.001))
				}
			}

		}
	}
}


class ReLUOpTest: XCTestCase {
	
	override func setUp() {
		super.setUp()
		// Put setup code here. This method is called before the invocation of each test method in the class.
	}
	
	override func tearDown() {
		// Put teardown code here. This method is called after the invocation of each test method in the class.
		super.tearDown()
	}
	
	func test() {
		let testCase = UnarOpTest<ReLUOpDelegate, ReLUOperator>() 
		testCase.testInit()
		testCase.testOuputShapesCheck()
		testCase.testInputOutputTensorsCheck()
	}
	
	func testAlpha() {
		let numCase = 10
		let op = ReLUOperator()
		
		let delegate = ReLUOpDelegate()
		let workingGroup = DispatchGroup()
		delegate.dispatchGroup = workingGroup
		op.computationDelegate = delegate
		
		// configure engine
		let (_, msg) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
		
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
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
				let shape = randomShape(dimensions: 2, dimensionSizeRange: [1000, 1500], dataType: .float)
				inputTensors.append(randomTensor(fromShape: shape))
				outputTensors.append(randomTensor(fromShape: shape))
				print("Generate Input tensor: \(inputTensors.last!.description)")
				print("Generate Output tensor: \(outputTensors.last!.description)")
			}
			
			
			op.inputTensors = inputTensors
			op.outputTensors = outputTensors
			delegate.veryfyTensors = inputTensors
			
			// random alpha
			let alpha = randomFloat()
			op.alpha = alpha
			delegate.alpha = alpha
			
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
			
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	
}
