//
//  conv_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/14/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Dispatch
import Accelerate

// TODO: figure out if needs inputshape for same on all inputs

/**
2D Convolution operator.

## Input tensors shapes
All tensors in `inputTensors` should have same shapes

## Shape specification
- inputTensors: each tensor: `[channel, height, width]` or `[height, width, channel]` according to `channelPosition`
- weight: `[num_filter，channel, kernelSize[0], kernelSize[1]]`,
- outputTensors: each tensor: `[out_height, out_width, num_filter]`, i.e., `TensorChannelOrder.Last`

## nil weight
At declaration, `weight` could be `nil`.
However, if you add this operator through a graph's `operation(_,inputs:,op:)` API (i.e. symbolic constructing graph).
You must indicate the `inputShape` attribute so that the graph could estimate the input and output shape information.

## Batch process
This operator does not directly support batch data.
However, user could use `TensorSlice` to do the batch processing.
Details can be found in [Slice tensor]() and [Batch calculation with operators]().

## Dilation
Currently, calculation with `dilation > 1` has not been implemented and supported.

*/
public class ConvOperator2D: ComputableOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Operator label. Conforms to `ComputableOperator`
	public var operatorLabel: String
	
	/// This operator does not operator on GPU. Conforms to `ComputableOperator`
	public var metalKernelFuncLabel:String = "Conv2D"
	
	/// Conforms to `ComputableOperator`
	public var computationDelegate: OperatorCalculationDelegate?
	
	/// Conforms to `ComputableOperator`
	public var inputTensors: [Tensor]?
	
	/// Conforms to `ComputableOperator`
	public var outputTensors: [Tensor]?
	
	/// If `true`, operator will not call `inputOutputTensorsCheck()` before doing calculation.
	/// This is used inside framework to speed up in situation we know it will not be wrong.
	public var disableInputOutputCheck: Bool
	
	/// Indicate if this operator would do paramter update.
	public var trainable: Bool = true
	
	/// The mapping type of this operator.
	/// `OneToOne` for this operator.
	public var mapType: OperatorMappingType {
		get {
			return OperatorMappingType.OneToOne
		}
	}
	
	/// The number of fitlers
	public var numFilters: Int
	
	/// The kernel size. `[height, width]`
	public var kernelSize: [Int]
	
	/// Stride. `[height, width]`
	/// Default is `[1, 1]`.
	public var stride: [Int]
	
	/// Dilate values. 2D vecotr indicating the dilation value in height and width.
	/// Default is `[1, 1]`, i.e. without dilation.
	/// TODO: Support dilation
	public var dilation: [Int]
	
	/// Padding mode. Default `PaddingMode.valid`
	public var padMode: PaddingMode = PaddingMode.Valid
	
	/// Channel position. Default is `ImageChannelOrder.First`
	public var channelPosition: TensorChannelOrder =  .First
	
	/// The weight tensor.
	public var weight: Tensor?
	
	/// The input shape
	/// Used to indicate the input tensors' shape.
	/// Should not be `nil` construction from scratch.
	public var inputShape: TensorShape?
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Init
	
	/// Designated init.
	///
	/// - Parameters:
	///   - numFilters:
	///   - kernelSize:
	///   - stride:
	///   - padMode:
	///   - channelPosition:
	///   - weight:
	///   - dilation:
	///   - computationDelegate:
	///   - inputTensors:
	///   - outputTensors:
	///   - operatorLabel:
	///   - inputShape:
	///   - disableInputOutputCheck: 
	public init(numFilters: Int, kernelSize: [Int],
	            stride: [Int] = [1, 1],
	            padMode: PaddingMode = .Valid,
	            channelPosition: TensorChannelOrder = .First,
	            weight: Tensor? = nil,
	            dilation: [Int] = [1, 1],
	            computationDelegate: OperatorCalculationDelegate? = nil,
	            inputTensors: [Tensor]? = nil, outputTensors: [Tensor]? = nil,
	            operatorLabel: String = "Conv2DOp",
	            inputShape: TensorShape? = nil,
	            disableInputOutputCheck: Bool = false) {
		self.numFilters = numFilters
		self.kernelSize = kernelSize
		self.stride = stride
		self.dilation = dilation
		self.padMode = padMode
		self.channelPosition = channelPosition
		self.weight = weight
		self.computationDelegate = computationDelegate
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
		self.operatorLabel = operatorLabel
		self.inputShape = inputShape
		self.disableInputOutputCheck = disableInputOutputCheck
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Compute output shape according `numFilters`, `kernelSize`, `stride` and `dilation`.
	///
	/// - Parameter shapes: shapes description
	/// - Returns: return value description
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		// shape empty check
		guard shapes.count >= 1 else {
			SerranoLogging.errorLogging(message: "Input shapes array is empty",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		// numFilters check
		guard numFilters > 0 else {
			SerranoLogging.errorLogging(message: "numFilters (\(self.numFilters)) should be a positive integer",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		// kernelSize check
		guard self.kernelSize.count == 2 && self.kernelSize[0] > 0 && self.kernelSize[1] > 0 else {
			SerranoLogging.errorLogging(message: "Invalid kernelSize (\(self.kernelSize)).",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		// stride check
		guard self.stride.count == 2 && self.stride[0] > 0 && self.stride[1] > 0 else {
			SerranoLogging.errorLogging(message: "Invalid stride (\(self.stride)).",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		// dilation check
		guard self.dilation.count == 2 && self.dilation[0] >= 1 && self.dilation[1] >= 1 else {
			SerranoLogging.errorLogging(message: "Invalid dilation (\(self.dilation)).",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		// input shapes same check 
		let checkShape = shapes.first!
		for shape in shapes {
			guard shape == checkShape else {
				SerranoLogging.errorLogging(message: "Input shapes should have same shape.",
				                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
		}
		
		var outputShapes = [TensorShape]()
		for inShape in shapes {
			// check input shapes
			let shapeArray = inShape.shapeArray
			guard inShape.rank == 3 || shapeArray[0] > 0 || shapeArray[1] > 0 || shapeArray[2] > 0 else {
				SerranoLogging.errorLogging(message: "Invalid input shape \(inShape.description)",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
			let (_, height, width) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: shapeArray)
			
			if self.dilation[0] >= 2 || self.dilation[1] >= 2 {
				fatalError("Not implemented")
			} else {
				let outShapeArray = [kernelScanningOutSize(self.padMode, inputSize: height,
														  kernelSize: self.kernelSize[0], stride: self.stride[0]),
									kernelScanningOutSize(self.padMode, inputSize: width,
														  kernelSize: self.kernelSize[1], stride: self.stride[1]),
									self.numFilters]
				outputShapes.append(TensorShape(dataType: inShape.dataType, shape: outShapeArray))
			}
		}
		return outputShapes
	}
	
	
	/// Check input and output tensors.
	///
	/// - Returns: return value description
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
		// input nil check
		guard self.inputTensors != nil && self.inputTensors!.count > 0 else {
			return (false, "inputTensors is nil or empty.")
		}
		
		// output nil check
		guard self.outputTensors != nil && self.outputTensors!.count > 0 else {
			return (false, "outputTensors is nil or empty.")
		}
		
		guard self.weight != nil else {
			return (false, "weight is nil.")
		}
		
		// weight shape check
		// [channel, kernelSize[0], kernelSize[1], num_filter]
		let weightShapeArray = self.weight!.shape.shapeArray
		let (channel, _, _) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: self.inputTensors!.first!.shape.shapeArray)
		guard weightShapeArray.count == 4 && weightShapeArray[0] == self.numFilters
			&& weightShapeArray[2] == self.kernelSize[0] && weightShapeArray[3] == self.kernelSize[1]
			&& weightShapeArray[1] == channel else {
			return (false, "Invalid weight shape, Expect \([self.numFilters, channel, self.kernelSize[0], self.kernelSize[1]]). " +
				"Given \(weightShapeArray).")
		}
		
		// inputShape check
		let inputShapes = self.inputTensors!.map { $0.shape }
		let outputShapesCheck = self.outputShape(shapeArray: inputShapes)
		guard outputShapesCheck != nil else {
			return (false, "Input tensors' shapes are not valid. Check logs for detail.")
		}
		
		// outputshape check
		let outputShapes = self.outputTensors!.map { $0.shape }
		guard outputShapes.count == outputShapesCheck!.count else {
			return (false, "Output tensor's count is not valid. Expect \(outputShapesCheck!.count), given \(outputShapes.count).")
		}
		for (outputShape, outputShapeCheck) in zip(outputShapes, outputShapesCheck!) {
			guard outputShape == outputShapeCheck else {
				return (false, "OutputTensor with shape [\(outputShape.description)] is not valid. Expect [\(outputShapeCheck.description)].")
			}
		}
		
		return (true, "")
	}
	
	/// Compute sync way.
	///
	/// - Parameter computationMode: mode
	public func compute(_ computationMode: OperatorComputationMode) {
		// check
		let (pass, msg) = self.inputOutputTensorsCheck()
		guard pass else {
			SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) aborts calculation cause given invalid data: \(msg)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		self.computationDelegate?.operatorWillBeginComputation(self)
		
		switch computationMode {
		case .GPU:
			if !SerranoEngine.configuredEngine.hasAvailableGPU() {
				SerranoLogging.warningLogging(message: "Serrano Engine has no available configured GPU device. Use CPU doing calculation instead.", file: "\(#file)", function: "\(#function)", line: "\(#line)")
				self.cpu()
			} else {
				self.gpu()
			}
		case .CPU:
			self.cpu()
		case .Auto:
			// TODO: More intelligent way to decide
			if self.inputTensors![0].count > 1000000 && SerranoEngine.configuredEngine.hasAvailableGPU(){
				self.gpu()
			} else {
				self.cpu()
			}
		}
		self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
	}
	
	/// Compute async
	///
	/// - Parameter computationMode: computationMode
	public func computeAsync(_ computationMode: OperatorComputationMode) {
		// check delegate
		OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
		
		DispatchQueue.global(qos: .userInitiated).async {
			self.compute(computationMode)
		}
	}
	
	public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType] {
		fatalError()
	}
	
	public func gradComputAsync(_ computationMode: OperatorComputationMode) {
		fatalError()
	}
	
	public func updateParams(grads: [Tensor], LR: Float) {
		fatalError()
	}
	
	/// Bind according to labels.
	///
	/// -Note: if cannot bind all needed parameters. `fatalError` will be raised.
	public func bindParamSymbols(_ symbols: [GraphSymbol]) {
		let paramsLabels = ["weight"]
		
		for label in paramsLabels {
			let symbol = (symbols.filter {$0.symbolLabel == label}).first
			guard symbol != nil else{
				SerranoLogging.errorLogging(message: "\(label) symbol does not exist.",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Faltal error raised by Serrano. Check log for details.")
			}
			guard symbol!.symbolType == SymbolType.Tensor else {
				SerranoLogging.errorLogging(message: "\(label) symbol should be a tensor symbol.",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Faltal error raised by Serrano. Check log for details.")
			}
			let dataSymbol = symbol! as! TensorSymbol
			guard dataSymbol.bindedData != nil else {
				SerranoLogging.errorLogging(message: "\(label) symbol has no binded data.",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Faltal error raised by Serrano. Check log for details.")
			}
		
//			self.weight = dataSymbol.bindedData! as! Tensor
			guard let weightTensor = dataSymbol.bindedData! as? Tensor else {
				SerranoLogging.errorLogging(message: "Cov2D operator \(self.operatorLabel) was trying to bind to data symbol \(dataSymbol). But seems this symbol is not a tensor symbol as expected.",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Faltal error raised by Serrano. Check log for details.")
			}
			self.weight = weightTensor
		}
	}
	
	/// `Weight` as `TensorSymbol`
	///
	/// - Returns:
	public func paramSymbols() -> [GraphSymbol] {
		let weightTensorSymbol: SerranoTensorSymbol
		if self.weight == nil {
			// refer from input shape
			
			// not nil
			guard self.inputShape != nil else {
				SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) attribute inputShape is nil while its weight tensor is also ni. Need one of them assigned to construct the graph",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Serrano error. Check log.")
			}
			
			// dim check
			guard self.inputShape!.rank == 3 else {
				SerranoLogging.errorLogging(message: "Attribute inputShape has invalid rank. Expect 3. Given \(self.inputShape!.rank)",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Serrano error. Check log.")
			}
			let (channel, _, _) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: self.inputShape!.shapeArray)
			let weightShape = TensorShape(dataType: .float, shape: [self.numFilters, channel, self.kernelSize[0], self.kernelSize[1]])
			weightTensorSymbol = SerranoTensorSymbol("weight", dataSource: SymbolDataSource.Parameter, shape: weightShape)
		} else {
			weightTensorSymbol = SerranoTensorSymbol("weight", dataSource: SymbolDataSource.Parameter, shape: self.weight!.shape)
		}
		return [weightTensorSymbol as GraphSymbol]
	}
	
	/// Cpu calculation
	internal func cpu() {
		// call dilation function
		if self.dilation[0] > 1 || self.dilation[1] > 1 {
			self.cpu_dilation()
		}
	
		regular(OperatorComputationMode.CPU)
	}
	
	/// GPU calcualtion
	internal func gpu() {
		// call dilation function
		if self.dilation[0] > 1 || self.dilation[1] > 1 {
			self.gpu_dilation()
		}
		
		regular(OperatorComputationMode.GPU)
	}
	
	/// Use Img2Col to calcualte result.
	/// 1. Convert each input tensor via Img2Col to a 2D tensor `A` with shape `[out_Height x out_Width, channel x kernelSize[0] x kernelSize[1]]`;
	/// 2. We view weight tensor as a 2D tensor `B` with shape `[num_filter，channel x kernelSize[0] x kernelSize[1]]`;
	/// 3. Do matrix multiplication `AxB` with `transposeB` setting as `true`.
	///
	/// - Parameter mode: computation mode
	internal func regular(_ mode: OperatorComputationMode) {
		// temp make weight's shape 2D faking it as a 2D tensor
		let originWeightShapeArray = self.weight!.shape.shapeArray
		self.weight!.shape = TensorShape(dataType: self.weight!.shape.dataType,
		                                 shape: [originWeightShapeArray[0],
		                                         originWeightShapeArray[1] * originWeightShapeArray[2] * originWeightShapeArray[3]])
		
		let workGroup = DispatchGroup()
		for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				// img2col
				let img2colOP = Img2ColOperator(patchSize: self.kernelSize, stride: self.stride,
				                                channelPosition: self.channelPosition,
				                                padMode: self.padMode,
				                                inputTensors: [input],
				                                disableInputOutputCheck: true)
				let outShape = img2colOP.outputShape(shapeArray: [input.shape])!.first!
				let colTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(outShape)
				img2colOP.outputTensors = [colTensor]
				img2colOP.compute(mode)
				
				// fake output tensor shape as a 2D shape
				let outputShape = output.shape
				output.shape = TensorShape(dataType: outputShape.dataType,
				                           shape: [outputShape.shapeArray[0] * outputShape.shapeArray[1], outputShape.shapeArray[2]])
				
				// matrix mult
				let matrixMultOp = MatrixMultOperator(transposeB: true,
				                                      inputTensors: [colTensor, self.weight!],
				                                      outputTensors: [output],
				                                      disableInputOutputCheck: true)
				matrixMultOp.compute(mode)
				
				// change back output tensor shape
				output.shape = outputShape
				
				workGroup.leave()
			}
		}
		workGroup.wait()
		
		// change weight's shape info back
		self.weight!.shape = TensorShape(dataType: self.weight!.shape.dataType, shape: originWeightShapeArray)
	}
	
	internal func cpu_dilation() {
		//TODO: implementaion
	}
	
	internal func gpu_dilation() {
		//TODO: implementaion
	}
	
}
