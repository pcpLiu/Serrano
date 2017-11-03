//
//  tensor_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 3/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
import Metal
@testable import Serrano

class TensorTest: XCTestCase {
    
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
//        
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
     Targets:
        Tensor.init(dataArray array:[Any], tensorShape shape: TensorShape)
        flatArrayFloat() -> [Float]
     */
    func testCreateTensor() {
        let numCase = 100
        //TODO: more test
        for _ in 0..<numCase {
			let tensor = Tensor(repeatingValue: 1.5, tensorShape: TensorShape(dataType: .float, shape: [3,5]))

        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    /**
     Target:
        func nestedArrayFloat() -> [Any]
        func constructNestedArrayFrom(location: [Int]) -> [Any]
     */
    func testNestedFunc() {
        let numCase = 100
        
        for _ in 0..<numCase {
            let shape = randomShape(dimensions: randomInt([2, 5]), dimensionSizeRange: [1, 10], dataType: .float)
            let (nestedArray, flatArray) = generateArrayFromShape(shapeArray: shape.shapeArray, dataType: shape.dataType)
            
            let tensor = Tensor(dataArray: nestedArray, tensorShape: shape)
            let tensorNestedArray = tensor.nestedArrayFloat()
            
            XCTAssertTrue(flatArray.elementsEqual(TensorUtils.flattenArray(array: tensorNestedArray, dataType: shape.dataType)))
            
            print("pass on shape \(shape)!")
            print("tensor:", tensorNestedArray)
            print("verify:", nestedArray)
            print("\n\n")
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Generate random valid index list for target shape
     */
    func randomIndexList(forShape shape: [Int], valid: Bool = true) -> [Int] {
        var indices = [Int]()
        for dimSize in shape {
            if valid {
                indices.append(randomInt([0, dimSize-1]))
            } else {
                indices.append(randomInt([-1, dimSize+3]))
            }
        }
        return indices
    }
    
    /**
     Target:
        func indexIsValid(_ index: [Int]) -> Bool
     */
    func testIndexValid() {
        let numCase = 100
        
        for _ in 0..<numCase {
            let shape = randomShape(dimensions: randomInt([2, 5]), dimensionSizeRange: [1, 10], dataType: .float)
            let (nestedArray, flatArray) = generateArrayFromShape(shapeArray: shape.shapeArray, dataType: shape.dataType)
            let tensor = Tensor(dataArray: nestedArray, tensorShape: shape)
            
            
            for _ in 0..<10 {
                let indices = randomIndexList(forShape: shape.shapeArray, valid: false)
                
                let tensorCheck = tensor.indexIsValid(indices)
                
                var verifyCheck = true
                for i in 0..<indices.count {
                    if !(0 <= indices[i] && indices[i] < shape.shapeArray[i] ) {
                        verifyCheck = false
                        break
                    }
                }
                
                print("Test on shape \(shape.shapeArray), generated indices: \(indices), valid: \(verifyCheck), method checking: \(tensorCheck)")
                XCTAssertEqual(tensorCheck, verifyCheck)
            }
        }
    }
    
    /**
     Target:
        subscript(_ index: [Int]) -> Float
     */
    func testSubscript() {
        let numCase = 100
        
        for _ in 0..<numCase {
            let shape = randomShape(dimensions: randomInt([2, 5]), dimensionSizeRange: [1, 10], dataType: .float)
            let (nestedArray, flatArray) = generateArrayFromShape(shapeArray: shape.shapeArray, dataType: shape.dataType)
            let tensor = Tensor(dataArray: nestedArray, tensorShape: shape)
            
            print("Test on array:\n\(nestedArray)")
           
            for _ in 0..<10 {
                let indices = randomIndexList(forShape: shape.shapeArray)
                let tensorVal = tensor[indices]
                
                 print("Test with shape \(shape.shapeArray), indices:\(indices)")
                
                var offset = 0
                for i in 0..<(indices.count-1) {
                    offset += indices[i] * shape.shapeArray.suffix(from: i+1).reduce(1, *)
                }
                offset += indices.last!
                let verfiyVal = flatArray[offset]
                
                XCTAssertEqual(verfiyVal, tensorVal)
                print(verfiyVal, tensorVal)
            }
            
            print("\n\n")
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Test if could create MTLBuffer without copy for a tensor
     */
    func testBaseAddressForMtlBuffer() {
        let numCase = 100
        
        let (result, msg) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
        guard result == true else {
            print("No available gpu. Give up test. \(msg)")
            return
        }
        let engine = SerranoEngine.configuredEngine
        
        for i in 0..<numCase {
            print("Test \(i+1)")
            let tensor = randomTensor(dimensions: randomInt([2, 6]), dimensionSizeRange: [1, 10], dataType: .float)
            let tensorBuffer = engine.GPUDevice!.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(tensor.contentsAddress),
                                                            length: tensor.allocatedBytes,
                                                            options: [MTLResourceOptions.storageModeShared],
                                                            deallocator: nil)
            let bufferAddress = tensorBuffer.contents()
            XCTAssertTrue(bufferAddress == UnsafeMutableRawPointer(tensor.contentsAddress))
            print("Address: \(bufferAddress)")
			
            let tensorArray = tensor.flatArrayFloat()
            var bufferArray = [Float]()
            let mtlBufferPointer = UnsafeMutablePointer<Float>(OpaquePointer(bufferAddress))
            let mtlBufferReader = UnsafeMutableBufferPointer<Float>(start: mtlBufferPointer, count: tensor.capacity)
            for index in 0..<tensorArray.count {
                bufferArray.append(mtlBufferReader[index])
            }
            XCTAssertTrue(tensorArray.elementsEqual(bufferArray))
        }
    }
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: Arithmetic operation
	
	/**
		Target:
		Test tensor-scalar, scalar-tensor operations
	*/
	func testScalarOperations() {
		let numCase = 20
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// generate tensor
			let tensor = randomTensor(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
			
			// generate scalar
			let scalar = randomFloat()

			
			if i % 4 == 0 {
				// add
				print("Test add")
				let result = tensor + scalar
				let result2 = scalar + tensor
				let resultReader = result.floatValueReader
				let resultReader2 = result2.floatValueReader
				let rawReader = tensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(rawReader[index] + scalar, resultReader[index],
					                           accuracy: abs(0.0001 * (rawReader[index] + scalar)))
					XCTAssertEqualWithAccuracy(rawReader[index] + scalar, resultReader2[index],
					                           accuracy: abs(0.0001 * (rawReader[index] + scalar)))
				}
			} else if i % 4 == 1 {
				// substract
				print("Test substract")
				let result = tensor - scalar
				let result2 = scalar - tensor
				let resultReader = result.floatValueReader
				let resultReader2 = result2.floatValueReader
				let rawReader = tensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(rawReader[index] - scalar, resultReader[index],
					                           accuracy: abs(0.0001 * (rawReader[index] - scalar)))
					XCTAssertEqualWithAccuracy(scalar - rawReader[index], resultReader2[index],
					                           accuracy: abs(0.0001 * (scalar - rawReader[index])))
				}
			} else if i % 4 == 2 {
				// mult
				print("Test mult")
				let result = tensor * scalar
				let result2 = scalar * tensor
				let resultReader = result.floatValueReader
				let resultReader2 = result2.floatValueReader
				let rawReader = tensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(rawReader[index] * scalar, resultReader[index],
					                           accuracy: abs(0.0001 * (scalar * rawReader[index])))
					XCTAssertEqualWithAccuracy(rawReader[index] * scalar, resultReader2[index],
					                           accuracy: abs(0.0001 * (scalar * rawReader[index])))
				}
			} else {
				// div
				print("Test div")
				let result = tensor / scalar
				let result2 = scalar / tensor
				let resultReader = result.floatValueReader
				let resultReader2 = result2.floatValueReader
				let rawReader = tensor.floatValueReader
				for index in 0..<result.count {
					if (rawReader[index] / scalar).isInfinite || (scalar / rawReader[index]).isInfinite {continue}
					XCTAssertEqualWithAccuracy(rawReader[index] / scalar, resultReader[index],
					                           accuracy: abs(0.0001 * (rawReader[index] / scalar)))
					XCTAssertEqualWithAccuracy(scalar / rawReader[index], resultReader2[index],
					                           accuracy: abs(0.0001 * (scalar / rawReader[index])))

				}
			}
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	/**
	Target:
	Test tensor-scalar inplace operations
	*/
	func testTensorScalarInplaceOperations() {
		let numCase = 20
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// generate tensor
			let tensor = randomTensor(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
			
			// generate scalar
			let scalar = randomFloat()
			
			// copy raw tensor
			let copyOp = CopyOperator(inputTensors: [tensor])
			copyOp.outputTensors = SerranoResourceManager.globalManager.allocateTensors( [tensor.shape])
			copyOp.compute(.CPU)
			let rawTensor = copyOp.outputTensors!.first!
			
			if i % 4 == 0 {
				// plus
				let result = tensor &+ scalar
				XCTAssertEqual(result, tensor)
				let resultReader = result.floatValueReader
				let rawReader = rawTensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(rawReader[index] + scalar, resultReader[index],
					                           accuracy: abs(0.0001 * (rawReader[index] + scalar)))
				}
			} else if i % 4 == 1 {
				// substract
				let result = tensor &- scalar
				XCTAssertEqual(result, tensor)
				let resultReader = result.floatValueReader
				let rawReader = rawTensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(rawReader[index] - scalar, resultReader[index],
					                           accuracy: abs(0.0001 * (rawReader[index] - scalar)))
				}
			} else if i % 4 == 2 {
				// mult
				let result = tensor &* scalar
				XCTAssertEqual(result, tensor)
				let resultReader = result.floatValueReader
				let rawReader = rawTensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(rawReader[index] * scalar, resultReader[index],
					                           accuracy: abs(0.0001 * (rawReader[index] * scalar)))
				}
			} else {
				// mult
				let result = tensor &/ scalar
				XCTAssertEqual(result, tensor)
				let resultReader = result.floatValueReader
				let rawReader = rawTensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(rawReader[index] / scalar, resultReader[index],
					                           accuracy: abs(0.0001 * (rawReader[index] / scalar)))
				}
			}
			
			
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	/**
	Target:
	Test scalar-scalar inplace operations
	*/
	func testScalarTensorInplaceOperations() {
		let numCase = 20
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// generate tensor
			let tensor = randomTensor(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
			
			// generate scalar
			let scalar = randomFloat()
			
			// copy raw tensor
			let copyOp = CopyOperator(inputTensors: [tensor])
			copyOp.outputTensors = SerranoResourceManager.globalManager.allocateTensors( [tensor.shape])
			copyOp.compute(.CPU)
			let rawTensor = copyOp.outputTensors!.first!
			
			if i % 4 == 0 {
				// plus
				let result = scalar &+ tensor
				XCTAssertEqual(result, tensor)
				let resultReader = result.floatValueReader
				let rawReader = rawTensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(scalar + rawReader[index], resultReader[index],
					                           accuracy: abs(0.0001 * (scalar + rawReader[index])))
				}
			} else if i % 4 == 1 {
				// substract
				let result = scalar &- tensor
				XCTAssertEqual(result, tensor)
				let resultReader = result.floatValueReader
				let rawReader = rawTensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(scalar - rawReader[index], resultReader[index],
					                           accuracy: abs(0.0001 * (scalar - rawReader[index])))
				}
			} else if i % 4 == 2 {
				// mult
				let result = scalar &* tensor
				XCTAssertEqual(result, tensor)
				let resultReader = result.floatValueReader
				let rawReader = rawTensor.floatValueReader
				for index in 0..<result.count {
					XCTAssertEqualWithAccuracy(scalar * rawReader[index], resultReader[index],
					                           accuracy: abs(0.0001 * (scalar * rawReader[index])))
				}
			} else {
				// mult
				let result = scalar &/ tensor
				XCTAssertEqual(result, tensor)
				let resultReader = result.floatValueReader
				let rawReader = rawTensor.floatValueReader
				for index in 0..<result.count {
					if (scalar / rawReader[index]).isInfinite || (scalar / rawReader[index]).isNaN { continue }
					XCTAssertEqualWithAccuracy(scalar / rawReader[index], resultReader[index],
					                           accuracy: abs(0.0001 * (scalar / rawReader[index])))
				}
			}
			
			
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	Target:
	Test tensor-tensor operations. not broadcasting
	*/
	func testTensorTensorOperation() {
		let numCase = 20
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// generate tensor
			let tensorA = randomTensor(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
			let tensorB = randomTensor(fromShape: tensorA.shape)
			
			if i % 4 == 0 {
				// plus
				let result = tensorA + tensorB
				let resultReader = result.floatValueReader
				let AReader = tensorA.floatValueReader
				let BReader = tensorB.floatValueReader
				for index in 0..<result.count {
					let val = AReader[index] + BReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: abs(val*0.001))
				}
			} else if i % 4 == 1 {
				// substract
				let result = tensorA - tensorB
				let resultReader = result.floatValueReader
				let AReader = tensorA.floatValueReader
				let BReader = tensorB.floatValueReader

				for index in 0..<result.count {
					let val = AReader[index] - BReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy:abs(val*0.001))
				}
			} else if i % 4 == 2 {
				// mult
				let result = tensorA * tensorB
				let resultReader = result.floatValueReader
				let AReader = tensorA.floatValueReader
				let BReader = tensorB.floatValueReader
				for index in 0..<result.count {
					let val = AReader[index] * BReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: abs(val*0.001))
				}
			} else {
				// div
				let result = tensorA / tensorB
				let resultReader = result.floatValueReader
				let AReader = tensorA.floatValueReader
				let BReader = tensorB.floatValueReader
				for index in 0..<result.count {
					let val = AReader[index] / BReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: abs(val*0.001))
				}
			}
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}

	}
	
	/**
	Target:
	Test tensor-tensor inplace operations. not broadcasting
	*/
	func testTensorTensorInplaceOperation() {
		let numCase = 20
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// generate tensor
			let tensorA = randomTensor(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
			let tensorB = randomTensor(fromShape: tensorA.shape)
		
			
			// copy A
			let copyA  = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
			let cpOp = CopyOperator(inputTensors: [tensorA], outputTensors: [copyA])
			cpOp.compute()

			let AReader = copyA.floatValueReader
			let BReader = tensorB.floatValueReader
			if i % 4 == 0 {
				// plus
				let result = tensorA &+ tensorB
				XCTAssertEqual(result, tensorA)
				let resultReader = result.floatValueReader
				for index in 0..<result.count {
					let val = AReader[index] + BReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: abs(val*0.0001))
				}
			} else if i % 4 == 1 {
				// substract
				let result = tensorA &- tensorB
				XCTAssertEqual(result, tensorA)
				let resultReader = result.floatValueReader
				for index in 0..<result.count {
					let val = AReader[index] - BReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: abs(val*0.0001))
				}
			} else if i % 4 == 2 {
				// mult
				let result = tensorA &* tensorB
				XCTAssertEqual(result, tensorA)
				let resultReader = result.floatValueReader
				for index in 0..<result.count {
					let val = AReader[index] * BReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: abs(val*0.0001))
				}
			} else {
				// mult
				let result = tensorA &/ tensorB
				XCTAssertEqual(result, tensorA)
				let resultReader = result.floatValueReader
				for index in 0..<result.count {
					let val = AReader[index] / BReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: abs(val*0.0001))
				}
			}
			
			
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
		
	}
	
	/**
	Target:
	Test tensor-tensor operations. broadcasting
	*/
	func testTensorTensorBroadcastOperation() {
		let numCase = 20
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// generate tensor
			let tensorA = randomTensor(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
			let tensorB = randomTensor(fromShape: TensorShape(dataType: .float, shape: [tensorA.shape.shapeArray[1]]) )
			
			if i % 4 == 0 {
				// plus
				let result = tensorA .+ tensorB
				let resultReader = result.floatValueReader
				let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
				let verifyTensorReader = verifyTensor.floatValueReader
				let op = BroadcastAddOperator(inputTensors: [tensorA, tensorB], outputTensors: [verifyTensor])
				op.compute()
				for index in 0..<result.count {
					let val = verifyTensorReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: 0.0001)
				}
			} else if i % 4 == 1 {
				// substract
				let result = tensorA .- tensorB
				let resultReader = result.floatValueReader
				let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
				let verifyTensorReader = verifyTensor.floatValueReader
				let op = BroadcastSubOperator(inputTensors: [tensorA, tensorB], outputTensors: [verifyTensor])
				op.compute()
				for index in 0..<result.count {
					let val = verifyTensorReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: 0.0001)
				}
			} else if i % 4 == 2 {
				// mult
				let result = tensorA .* tensorB
				let resultReader = result.floatValueReader
				let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
				let verifyTensorReader = verifyTensor.floatValueReader
				let op = BroadcastMultOperator(inputTensors: [tensorA, tensorB], outputTensors: [verifyTensor])
				op.compute()
				for index in 0..<result.count {
					let val = verifyTensorReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: 0.0001)
				}
			} else {
				// div
				let result = tensorA ./ tensorB
				let resultReader = result.floatValueReader
				let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
				let verifyTensorReader = verifyTensor.floatValueReader
				let op = BroadcastDivOperator(inputTensors: [tensorA, tensorB], outputTensors: [verifyTensor])
				op.compute()
				for index in 0..<result.count {
					let val = verifyTensorReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: 0.0001)
				}
			}
			
			
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	/**
	Target:
	Test tensor-tensor inplace operations. broadcasting
	*/
	func testTensorTensorBroadcastInplaceOperation() {
		let numCase = 20
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			// generate tensor
			let tensorA = randomTensor(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
			let tensorB = randomTensor(fromShape: TensorShape(dataType: .float, shape: [tensorA.shape.shapeArray[1]]) )
			let copyA = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
			let cp = CopyOperator(inputTensors: [tensorA], outputTensors: [copyA])
			cp.compute()
			
			if i % 4 == 0 {
				// plus
				let result = tensorA .&+ tensorB
				XCTAssertEqual(result, tensorA)
				let resultReader = result.floatValueReader
				let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
				let verifyTensorReader = verifyTensor.floatValueReader
				let op = BroadcastAddOperator(inputTensors: [copyA, tensorB], outputTensors: [verifyTensor])
				op.compute()
				for index in 0..<result.count {
					let val = verifyTensorReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: 0.0001)
				}
			} else if i % 4 == 1 {
				// substract
				let result = tensorA .&- tensorB
				XCTAssertEqual(result, tensorA)
				let resultReader = result.floatValueReader
				let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
				let verifyTensorReader = verifyTensor.floatValueReader
				let op = BroadcastSubOperator(inputTensors: [copyA, tensorB], outputTensors: [verifyTensor])
				op.compute()
				for index in 0..<result.count {
					let val = verifyTensorReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: 0.0001)
				}
			} else if i % 4 == 2 {
				// mult
				let result = tensorA .&* tensorB
				XCTAssertEqual(result, tensorA)
				let resultReader = result.floatValueReader
				let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
				let verifyTensorReader = verifyTensor.floatValueReader
				let op = BroadcastMultOperator(inputTensors: [copyA, tensorB], outputTensors: [verifyTensor])
				op.compute()
				for index in 0..<result.count {
					let val = verifyTensorReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: 0.0001)
				}
			} else {
				// div
				let result = tensorA .&/ tensorB
				XCTAssertEqual(result, tensorA)
				let resultReader = result.floatValueReader
				let verifyTensor = Tensor(repeatingValue: 0.0, tensorShape: tensorA.shape)
				let verifyTensorReader = verifyTensor.floatValueReader
				let op = BroadcastDivOperator(inputTensors: [copyA, tensorB], outputTensors: [verifyTensor])
				op.compute()
				for index in 0..<result.count {
					let val = verifyTensorReader[index]
					if val.isInfinite || val.isNaN {continue}
					XCTAssertEqualWithAccuracy(val, resultReader[index], accuracy: 0.0001)
				}
			}
			
			
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	+ - * /
	*/
	public func testTensorTensorOpreator() {
		let numCase = 100
		SerranoEngine.configuredEngine.defaultComputationMode = .CPU
		for i in 0..<numCase {
			print("Test \(i+1)...")
			let tensorA = randomTensor(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
			let tensorB = randomTensor(fromShape: tensorA.shape)
			var result: Tensor
			if i % 4 == 0 {
				result = tensorA + tensorB
			} else if i % 4 == 1 {
				result = tensorA - tensorB
			} else if i % 4 == 2 {
				result = tensorA * tensorB
			} else {
				result = tensorA / tensorB
			}
			
			// verify
			let floatReadeA = tensorA.floatValueReader
			let floatReadeB = tensorB.floatValueReader
			let floatReaderResult = result.floatValueReader
			for elementIndex in 0..<result.count {
				var value:Float = 0.0
				if i % 4 == 0 { value = floatReadeA[elementIndex] + floatReadeB[elementIndex] }
				else if i % 4 == 1 { value = floatReadeA[elementIndex] - floatReadeB[elementIndex] }
				else if i % 4 == 2 { value = floatReadeA[elementIndex] * floatReadeB[elementIndex] }
				else { value = floatReadeA[elementIndex] / floatReadeB[elementIndex] }
				if value.isNaN || value.isInfinite { continue }
				XCTAssertEqualWithAccuracy(value, floatReaderResult[elementIndex], accuracy: abs(value*0.0001))
			}
			print("Finish Test \(i+1)\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: slice
	/**
	Target:
	public func batchSlice(_ batchIndex: Int) -> Tensor?
	*/
	func testTensorBatchSlice() {
		let numCase = 100
		for i in 0..<numCase {
			print("Test \(i+1)...")
			
			let tensor = randomTensor(dimensions: randomInt([1, 5]), dimensionSizeRange: [2, 5], dataType: .float)
			print("Genereate tensor \(tensor.description)")
			
			// tensor dimension
			if tensor.shape.shapeArray.count < 2 {
				let slice = tensor.batchSlice(0)
				XCTAssertNil(slice)
			} else {
				for batchIndex in 0..<tensor.shape.shapeArray[0] {
					let slice = tensor.batchSlice(batchIndex)
					XCTAssertNotNil(slice)
					print("Genereate slice \(slice!.description)")
					
					// count
					XCTAssertEqual(slice!.count, tensor.shape.shapeArray.suffix(from: 1).reduce(1, *))
					
					// address
					XCTAssertEqual(slice!.contentsAddress, tensor.contentsAddress + slice!.count * batchIndex)
					XCTAssertEqual(slice!.sliceRootTensor!, tensor)
					XCTAssertEqual(slice!.sliceIndex!, batchIndex)
					
					// value
					let sliceReader = slice!.floatValueReader
					let tensorReader = tensor.floatValueReader
					for eleIndex in 0..<slice!.count {
						XCTAssertEqual(sliceReader[eleIndex], tensorReader[eleIndex + slice!.count * batchIndex])
					}
				}
			}
			
			SerranoResourceManager.globalManager.releaseAllResources()
			print("Finish Test \(i+1)\n\n")
		}
	}
}
