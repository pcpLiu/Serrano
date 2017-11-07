//
//  SimpleCNN.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 11/3/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

func configureSimpleCNN() -> ForwardGraph {
	let g = ForwardGraph()
	let shape = TensorShape(dataType: .float, shape: [244, 244, 3]) // shape of the tensor
	let input = g.tensor(shape: shape) // add an input tensor
	
	let convOp  = ConvOperator2D(numFilters: 96,
								 kernelSize: [11,11],
								 stride: [4, 4],
								 padMode: PaddingMode.Valid,
								 channelPosition: TensorChannelOrder.Last,
								 inputShape: input.shape)
	let (convOut, _, _) = g.operation(inputs: [input], op: convOp)
	
	let maxPool = MaxPool2DOperator(kernelSize: [2, 2], stride: [2, 2],
									channelPosition: TensorChannelOrder.Last,
									paddingMode: PaddingMode.Valid)
	let (poolOut, _, _) = g.operation(inputs: convOut, op: maxPool)
	
	let fc = FullyconnectedOperator(inputDim: poolOut.first!.shape.count,
									numUnits: 200)
	let _ = g.operation(inputs: poolOut, op: fc)
	
	return g
}

class SimpleCNN: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testSimpleCNN() {
		
		SerranoLogging.release = true
		let _ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		
		let g = configureSimpleCNN()
		g.forwardPrepare()
		
		// calculate
		let start = CFAbsoluteTimeGetCurrent()
		let results = g.forward(mode: .GPU)
		print("Forward Execution Time : \((CFAbsoluteTimeGetCurrent() - start) * 100) ms")
    }
	
    
}
