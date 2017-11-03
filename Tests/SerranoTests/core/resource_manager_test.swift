//
//  resource_manager_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/9/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class resource_manager_test: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

	/**
	Target:
	public func allocateTensors(forShapes shapes: [TensorShape]) -> [Tensor]
	public func returnTensors(_ tensors: [Tensor])
	public func isManagingTensor(_ tensor: Tensor) -> Bool
	public func isTensorAvailable(_ tensor: Tensor) -> Bool
	*/
	func testTensorManagement() {
		let numCase = 100
		let manager = SerranoResourceManager()

		for i in 0..<numCase {
			print("Test case \(i+1)..")
			let shape1 = randomShape(dimensions: 2, dimensionSizeRange: [100, 150], dataType: .float)
			// allocate
			let newTensor = manager.allocateTensors([shape1]).first
			XCTAssertNotNil(newTensor)
			XCTAssertTrue(newTensor!.shape == shape1)
			XCTAssertTrue(manager.isManagingTensor(newTensor!))
			XCTAssertTrue(!manager.isTensorAvailable(newTensor!))
			let tensor1AllocateBytes = newTensor!.allocatedBytes
			
			// return
			manager.returnTensors([newTensor!])
			XCTAssertTrue(manager.isManagingTensor(newTensor!))
			XCTAssertTrue(manager.isTensorAvailable(newTensor!))

			var shape2 = randomShape(dimensions: 2, dimensionSizeRange: [40, 50], dataType: .float)
			if i % 2 == 0 {
				// generate shap2 should be small than shape1
				var shape2NeedBytes = shape2.shapeArray.reduce(1, *) * MemoryLayout<Float>.stride
				while shape2NeedBytes > tensor1AllocateBytes {
					shape2 = randomShape(dimensions: 2, dimensionSizeRange: [40, 50], dataType: .float)
					shape2NeedBytes = shape2.shapeArray.reduce(1, *) * MemoryLayout<Float>.stride
				}
				print("Tensor 2 require bytes: \(shape2NeedBytes), tensor 1 allocate bytes: \(tensor1AllocateBytes)")
			} else {
				// generate shape2 should be larger than shape1
				var shape2NeedBytes = shape2.shapeArray.reduce(1, *) * MemoryLayout<Float>.stride
				while shape2NeedBytes <= tensor1AllocateBytes {
					shape2 = randomShape(dimensions: 2, dimensionSizeRange: [200, 210], dataType: .float)
					shape2NeedBytes = shape2.shapeArray.reduce(1, *) * MemoryLayout<Float>.stride
				}
				print("Tensor 2 require bytes: \(shape2NeedBytes), tensor 1 allocate bytes: \(tensor1AllocateBytes)")
			}
			
			
			let newTensor2 = manager.allocateTensors([shape2]).first
			XCTAssertNotNil(newTensor2)
			XCTAssertTrue(newTensor2!.shape == shape2)
			XCTAssertTrue(manager.isManagingTensor(newTensor2!))
			XCTAssertTrue(!manager.isTensorAvailable(newTensor2!))
			
			if i % 2 == 0 {
				// should reuse
				XCTAssertTrue(!manager.isTensorAvailable(newTensor!))
				XCTAssertTrue(newTensor2! == newTensor!)
			} else {
				// should allocate a new tensor
				XCTAssertTrue(manager.isTensorAvailable(newTensor!))
				XCTAssertTrue(newTensor2 != newTensor!)
				print(newTensor!.description)
				print(newTensor2!.description)
			}
			
			
			manager.releaseAllResources()
			
			print("Finish Test \(i+1)\n\n\n")
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	Target:
	public func allocateMTLBuffers(forTensors tensors: [Tensor]) -> [MTLBuffer]
	public func releaseTensorAttachedBuffers(_ tensors: [Tensor])
	public func isManagingBufferr(_ buffer: MTLBuffer) -> Bool
	*/
	func testBufferManagement() {
		// TODO:
	}

}
