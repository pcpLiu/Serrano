//
//  SimpleCNN_imperative.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 11/3/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class SimpleCNN_imperative: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testImperative() {
		SerranoLogging.release = true
		let _ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		
        let inputTensor = randomTensor(fromShape: TensorShape(dataType: .float, shape: [244, 244, 3]))
		
		// conv
		let convOp = ConvOperator2D(numFilters: 20,
									kernelSize: [11,11],
									stride: [4, 4],
									padMode: PaddingMode.Valid,
									channelPosition: TensorChannelOrder.Last,
									weight: randomTensor(fromShape: TensorShape(dataType: .float, shape: [20, 3, 11, 11])),
									bias: randomTensor(fromShape: TensorShape(dataType: .float, shape: [20])))
		
		// Initialize a tensor object to store convOp's result.
		// In serrano, operator cannot allocate memeory for output tensors so that it can control memory allcoation precisely.
		let convOutputs = SerranoResourceManager.globalManager.allocateUnamangedTensors(convOp.outputShape(shapeArray: [inputTensor.shape])!)
		convOp.inputTensors = [inputTensor]
		convOp.outputTensors = convOutputs
		convOp.compute(.GPU)
		
		// pooling
		let pool = MaxPool2DOperator(kernelSize: [2, 2], stride: [2, 2],
									 channelPosition: TensorChannelOrder.Last,
									 paddingMode: PaddingMode.Valid)
		let poolOutputs = SerranoResourceManager.globalManager.allocateUnamangedTensors(pool.outputShape(shapeArray: convOutputs.map {$0.shape})!)
		pool.inputTensors = convOutputs
		pool.outputTensors = poolOutputs
		pool.compute(.GPU)
		
		// fully connected
		let fc = FullyconnectedOperator(inputDim: poolOutputs.first!.shape.count,
										numUnits: 200,
										weight: randomTensor(fromShape: TensorShape(dataType: .float, shape: [poolOutputs.first!.shape.count,
																											  200])),
										bias: randomTensor(fromShape: TensorShape(dataType: .float, shape: [200])))
		let outputs = SerranoResourceManager.globalManager.allocateUnamangedTensors(fc.outputShape(shapeArray: poolOutputs.map {$0.shape})!)
		fc.inputTensors = poolOutputs
		fc.outputTensors = outputs
		fc.compute(.GPU)
    }

    
}
