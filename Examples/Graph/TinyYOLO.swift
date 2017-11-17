//
//  TinyYOLOV1.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 11/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//


// credit: https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo-voc.cfg


import XCTest
@testable import Serrano

func configureGraph() -> ForwardGraph {
	let g = ForwardGraph()
	
	// input
	let input = g.tensor(shape: TensorShape(dataType: .int, shape: [416, 416, 3]))
	
	// conv block 1
	let (out_conv1,_, _) = g.operation(inputs: [input],
									   op: ConvOperator2D(numFilters: 16, kernelSize: [3, 3],
														  padMode: PaddingMode.Same,
														  channelPosition: TensorChannelOrder.Last,
														  inputShape: input.shape))
	
	let (out_act1, _, _) = g.operation(inputs: out_conv1,
									   op: LeakyReLUOperator(alpha: 0.1))
	
	let (out_poo1, _, _) = g.operation(inputs: out_act1,
									   op: MaxPool2DOperator(kernelSize: [2, 2],
															 channelPosition: TensorChannelOrder.Last,
															 paddingMode: PaddingMode.Same))

	// conv block 2
	let (out_conv2,_, _) = g.operation(inputs: out_poo1,
									   op: ConvOperator2D(numFilters: 32, kernelSize: [3, 3],
														  padMode: PaddingMode.Same,
														  channelPosition: TensorChannelOrder.Last,
														  inputShape: out_poo1.first!.shape))

	let (out_act2, _, _) = g.operation(inputs: out_conv2,
									   op: LeakyReLUOperator(alpha: 0.1))

	let (out_poo2, _, _) = g.operation(inputs: out_act2,
									   op: MaxPool2DOperator(kernelSize: [2, 2],
															 channelPosition: TensorChannelOrder.Last,
															 paddingMode: PaddingMode.Same))

	// conv block 3
	let (out_conv3,_, _) = g.operation(inputs: out_poo2,
									   op: ConvOperator2D(numFilters: 64, kernelSize: [3, 3],
														  padMode: PaddingMode.Same,
														  channelPosition: TensorChannelOrder.Last,
														  inputShape: out_poo2.first!.shape))

	let (out_act3, _, _) = g.operation(inputs: out_conv3,
									   op: LeakyReLUOperator(alpha: 0.1))

	let (out_poo3, _, _) = g.operation(inputs: out_act3,
									   op: MaxPool2DOperator(kernelSize: [2, 2],
															 channelPosition: TensorChannelOrder.Last,
															 paddingMode: PaddingMode.Same))

	// conv block 4
	let (out_conv4,_, _) = g.operation(inputs: out_poo3,
									   op: ConvOperator2D(numFilters: 128, kernelSize: [3, 3],
														  padMode: PaddingMode.Same,
														  channelPosition: TensorChannelOrder.Last,
														  inputShape: out_poo3.first!.shape))

	let (out_act4, _, _) = g.operation(inputs: out_conv4,
									   op: LeakyReLUOperator(alpha: 0.1))

	let (out_poo4, _, _) = g.operation(inputs: out_act4,
									   op: MaxPool2DOperator(kernelSize: [2, 2],
															 channelPosition: TensorChannelOrder.Last,
															 paddingMode: PaddingMode.Same))

	// conv block 5
	let (out_conv5,_, _) = g.operation(inputs: out_poo4,
									   op: ConvOperator2D(numFilters: 256, kernelSize: [3, 3],
														  padMode: PaddingMode.Same,
														  channelPosition: TensorChannelOrder.Last,
														  inputShape: out_poo4.first!.shape))

	let (out_act5, _, _) = g.operation(inputs: out_conv5,
									   op: LeakyReLUOperator(alpha: 0.1))

	let (out_poo5, _, _) = g.operation(inputs: out_act5,
									   op: MaxPool2DOperator(kernelSize: [2, 2],
															 channelPosition: TensorChannelOrder.Last,
															 paddingMode: PaddingMode.Same))

	// conv block 6
	let (out_conv6,_, _) = g.operation(inputs: out_poo5,
									   op: ConvOperator2D(numFilters: 512, kernelSize: [3, 3],
														  padMode: PaddingMode.Same,
														  channelPosition: TensorChannelOrder.Last,
														  inputShape: out_poo5.first!.shape))

	let (out_act6, _, _) = g.operation(inputs: out_conv6,
									   op: LeakyReLUOperator(alpha: 0.1))

	let (out_poo6, _, _) = g.operation(inputs: out_act6,
									   op: MaxPool2DOperator(kernelSize: [2, 2],
															 channelPosition: TensorChannelOrder.Last,
															 paddingMode: PaddingMode.Same))

	// conv block 7
	let (out_conv7,_, _) = g.operation(inputs: out_poo6,
									   op: ConvOperator2D(numFilters: 1024, kernelSize: [3, 3],
														  padMode: PaddingMode.Same,
														  channelPosition: TensorChannelOrder.Last,
														  inputShape: out_poo6.first!.shape))

	let (out_act7, _, _) = g.operation(inputs: out_conv7,
									   op: LeakyReLUOperator(alpha: 0.1))

	/////////

	// conv block  8
	let (out_conv8,_, _) = g.operation(inputs: out_act7,
									   op: ConvOperator2D(numFilters: 512, kernelSize: [3, 3],
														  padMode: PaddingMode.Same,
														  channelPosition: TensorChannelOrder.Last,
														  inputShape: out_act7.first!.shape))

	let (out_act8, _, _) = g.operation(inputs: out_conv8,
									   op: LeakyReLUOperator(alpha: 0.1))

	//  output
	let (out_conv9,_, _) = g.operation(inputs: out_act8,
									   op: ConvOperator2D(numFilters: 425, kernelSize: [1, 1],
														  padMode: PaddingMode.Same,
														  channelPosition: TensorChannelOrder.Last,
														  inputShape: out_act8.first!.shape))

	print(out_conv9.first!.shape)

	return g
}

class TinyYOLO: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
	func testYOLO() {
		SerranoLogging.release = true
		
		let _ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		let yolo = configureGraph()
		yolo.forwardPrepare()
		
		for _ in 0..<1 {
			let start = CFAbsoluteTimeGetCurrent()
			yolo.forward(mode: .GPU)
			print("Forward Execution Time : \((CFAbsoluteTimeGetCurrent() - start) * 100) ms")
			print("===================")
		}
	}
    
}
