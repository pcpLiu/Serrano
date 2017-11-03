import XCTest
import Serrano

/**
This code shows how to construct a VGG16 network using graph's low-level API.

[vgg16](http://book.paddlepaddle.org/03.image_classification/image/vgg16.png)
*/
func configureVGG16() -> ComputationGraph {
	let g = ComputationGraph()
	
	// input [244, 244, 3]
	let shape = TensorShape(dataType: .float, shape: [244, 244, 3])
	let input = g.tensor(shape: shape)
	
	// block 1
	let convOp  = ConvOperator2D(numFilters: 64,
	                             kernelSize: [3, 3],
	                             padMode: PaddingMode.Same,
	                             channelPosition: TensorChannelOrder.Last,
	                             inputShape: input.shape)
	let (out, _, _) = g.operation(inputs: [input], op: convOp)
	
	let convOp1  = ConvOperator2D(numFilters: 64,
	                              kernelSize: [3, 3],
	                              padMode: PaddingMode.Same,
	                              channelPosition: TensorChannelOrder.Last,
	                              inputShape: out.first!.shape)
	let (out1, _, _) = g.operation(inputs: out, op: convOp1)
	
	let poo1 = MaxPool2DOperator(kernelSize: [2, 2],
	                             channelPosition: TensorChannelOrder.Last,
	                             paddingMode: PaddingMode.Valid)
	let (out_block_1, _, _) = g.operation(inputs: out1, op: poo1)
	
	// block 2
	let convOp2  = ConvOperator2D(numFilters: 128,
	                              kernelSize: [3, 3],
	                              padMode: PaddingMode.Same,
	                              channelPosition: TensorChannelOrder.Last,
	                              inputShape: out_block_1.first!.shape)
	let (out2, _, _) = g.operation(inputs: out_block_1, op: convOp2)
	
	let convOp3  = ConvOperator2D(numFilters: 128,
	                              kernelSize: [3, 3],
	                              padMode: PaddingMode.Same,
	                              channelPosition: TensorChannelOrder.Last,
	                              inputShape: out2.first!.shape)
	let (out3, _, _) = g.operation(inputs: out2, op: convOp3)
	
	let poo2 = MaxPool2DOperator(kernelSize: [2, 2],
	                             channelPosition: TensorChannelOrder.Last,
	                             paddingMode: PaddingMode.Valid)
	let (out_block_2, _, _) = g.operation(inputs: out3, op: poo2)
	
	// block 3
	let convOp4  = ConvOperator2D(numFilters: 256,
	                              kernelSize: [3, 3],
	                              padMode: PaddingMode.Same,
	                              channelPosition: TensorChannelOrder.Last,
	                              inputShape: out_block_2.first!.shape)
	let (out4, _, _) = g.operation(inputs: out_block_2, op: convOp4)
	
	let convOp5  = ConvOperator2D(numFilters: 256,
	                              kernelSize: [3, 3],
	                              padMode: PaddingMode.Same,
	                              channelPosition: TensorChannelOrder.Last,
	                              inputShape: out4.first!.shape)
	let (out5, _, _) = g.operation(inputs: out4, op: convOp5)
	
	let convOp6  = ConvOperator2D(numFilters: 256,
	                              kernelSize: [3, 3],
	                              padMode: PaddingMode.Same,
	                              channelPosition: TensorChannelOrder.Last,
	                              inputShape: out5.first!.shape)
	let (out6, _, _) = g.operation(inputs: out5, op: convOp6)
	
	let poo3 = MaxPool2DOperator(kernelSize: [2, 2],
	                             channelPosition: TensorChannelOrder.Last,
	                             paddingMode: PaddingMode.Valid)
	let (out_block_3, _, _) = g.operation(inputs: out6, op: poo3)
	
	// bloack 4
	let convOp7  = ConvOperator2D(numFilters: 512,
	                              kernelSize: [3, 3],
	                              padMode: PaddingMode.Same,
	                              channelPosition: TensorChannelOrder.Last,
	                              inputShape: out_block_3.first!.shape)
	let (out7, _, _) = g.operation(inputs: out_block_3, op: convOp7)
	
	let convOp8  = ConvOperator2D(numFilters: 512,
	                              kernelSize: [3, 3],
	                              padMode: PaddingMode.Same,
	                              channelPosition: TensorChannelOrder.Last,
	                              inputShape: out7.first!.shape)
	let (out8, _, _) = g.operation(inputs: out7, op: convOp8)
	
	let convOp9  = ConvOperator2D(numFilters: 512,
	                              kernelSize: [3, 3],
	                              padMode: PaddingMode.Same,
	                              channelPosition: TensorChannelOrder.Last,
	                              inputShape: out8.first!.shape)
	let (out9, _, _) = g.operation(inputs: out8, op: convOp9)
	
	let poo4 = MaxPool2DOperator(kernelSize: [2, 2],
	                             channelPosition: TensorChannelOrder.Last,
	                             paddingMode: PaddingMode.Valid)
	let (out_block_4, _, _) = g.operation(inputs: out9, op: poo4)
	
	// block 5
	let convOp10  = ConvOperator2D(numFilters: 512,
	                               kernelSize: [3, 3],
	                               padMode: PaddingMode.Same,
	                               channelPosition: TensorChannelOrder.Last,
	                               inputShape: out_block_4.first!.shape)
	let (out10, _, _) = g.operation(inputs: out_block_4, op: convOp10)
	
	let convOp11  = ConvOperator2D(numFilters: 512,
	                               kernelSize: [3, 3],
	                               padMode: PaddingMode.Same,
	                               channelPosition: TensorChannelOrder.Last,
	                               inputShape: out10.first!.shape)
	let (out11, _, _) = g.operation(inputs: out10, op: convOp11)
	
	let convOp12  = ConvOperator2D(numFilters: 512,
	                               kernelSize: [3, 3],
	                               padMode: PaddingMode.Same,
	                               channelPosition: TensorChannelOrder.Last,
	                               inputShape: out11.first!.shape)
	let (out12, _, _) = g.operation(inputs: out11, op: convOp12)
	
	let poo5 = MaxPool2DOperator(kernelSize: [2, 2],
	                             channelPosition: TensorChannelOrder.Last,
	                             paddingMode: PaddingMode.Valid)
	let (out_block_5, _, _) = g.operation(inputs: out12, op: poo5)
	
	// block 6
	let fc13 = FullyconnectedOperator(inputDim: 25088, numUnits: 4096)
	let (out13, _, _) = g.operation(inputs: out_block_5, op: fc13)
	
	let fc14 = FullyconnectedOperator(inputDim: 4096, numUnits: 4096)
	let (out14, _, _) = g.operation(inputs: out13, op: fc14)
	
	let fc15 = FullyconnectedOperator(inputDim: 4096, numUnits: 4096)
	let (out15, _, _) = g.operation(inputs: out14, op: fc15)
	
	return g
}



class Example_VGG16: XCTestCase {
	// Test vgg16 network forward
	// Suggestion: run this function on macOS or real iOS devices supporting GPU. It goona be very slow on CPU mode.
	func testVGG16Forawad() {
		SerranoLogging.release = true
		
		let _ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
		let vgg16 = configureVGG16()
		vgg16.allocateAllTensors()
		vgg16.forwardPrepare()
		
		let start = CFAbsoluteTimeGetCurrent()
//		vgg16.forward(mode: .CPU)
		vgg16.forward(mode: .GPU)
		print("Forward Execution Time : \((CFAbsoluteTimeGetCurrent() - start) * 100) ms")
	}
}
