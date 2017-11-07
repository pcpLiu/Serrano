//
//  pooling_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/20/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Dispatch
import Accelerate


/// Mirror struct of `Pool2DInfo` in `pooling_op.metal`
public struct Pool2DInfo {
	var channelPosition: MetalShort
	var kernelSizeHeight: MetalUShort
	var kernelSizeWidth: MetalUShort
	var strideHeight: MetalUShort
	var strideWidth: MetalUShort
	var inHeight: MetalUInt
	var inWidth: MetalUInt
	var inChannel: MetalUInt
	var outHeight: MetalUInt
	var outWidth: MetalUInt
	var outChannel: MetalUInt
	
	/// Generate `Pool2DInfo` from a `Pooling2DOperator`'s information.
	///
	/// - Parameters:
	///   - op: operator
	///   - inputSize: input size
	///   - outputSize: output size
	/// - Returns: `Pool2DInfo`
	static func makePool2DInfo(op: Pooling2DOperator, inputSize: [Int], outputSize: [Int]) -> Pool2DInfo {
		let (inChannel, inHeight, inWidth) = parseImgChannelShapeInfo(op.channelPosition, shapeArray: inputSize)
		let (outChannel, outHeight, outWidth) = parseImgChannelShapeInfo(op.channelPosition, shapeArray: outputSize)
		return Pool2DInfo(channelPosition: MetalShort(op.channelPosition.rawValue),
		                  kernelSizeHeight: MetalUShort(op.kernelSize[0]), kernelSizeWidth: MetalUShort(op.kernelSize[1]),
		                  strideHeight: MetalUShort(op.stride[0]), strideWidth: MetalUShort(op.stride[1]),
		                  inHeight: MetalUInt(inHeight), inWidth: MetalUInt(inWidth), inChannel: MetalUInt(inChannel),
		                  outHeight: MetalUInt(outHeight), outWidth: MetalUInt(outWidth), outChannel: MetalUInt(outChannel))
	}
}

/**
This class is an abstract class for 2D pooling operators.
*/
public class Pooling2DOperator: ComputableOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Operator label. Conforms to `ComputableOperator`
	public var operatorLabel: String
	
	/// This operator does not operator on GPU. Conforms to `ComputableOperator`
	public var metalKernelFuncLabel:String = "" // need override by child class
	
	/// Conforms to `ComputableOperator`
	public var computationDelegate: OperatorCalculationDelegate?
	
	/// Conforms to `ComputableOperator`
	public var inputTensors: [Tensor]?
	
	/// Conforms to `ComputableOperator`
	public var outputTensors: [Tensor]?
	
	/// Pad mode, default is `PaddingMode.Valid`.
	public var paddingMode: PaddingMode = PaddingMode.Valid
	
	/// Kernel size array which contains the kernel size in each dimension.
	public var kernelSize: [Int]
	
	/// Stride array which contains the stride in each dimension.
	public var stride: [Int]
	
	/// CPU computation block.
	/// Two tensors are input and output tensors.
	public lazy var cpuComputeBlock: ((Tensor, Tensor) -> Void)? = nil
	
	/// Channel position. Default is `ImageChannelOrder.First`
	public var channelPosition: TensorChannelOrder =  .First
	
	/// If `true`, operator will not call `inputOutputTensorsCheck()` before doing calculation.
	/// This is used inside framework to speed up in situation we know it will not be wrong.
	public var disableInputOutputCheck: Bool = false
	
	/// Indicate if this operator would do paramter update.
	public var trainable: Bool = false
	
	/// The mapping type of this operator.
	/// `OneToOne` for this operator.
	public var mapType: OperatorMappingType {
		get {
			return OperatorMappingType.OneToOne
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Initializer.
	///
	/// - Parameters:
	///   - kernelSize: Array of int values. Should has 2 elemetns for height and width dimesnsions.
	///   - stride: Array of int values. If `stride` is `nil`, it will be assigned as same value as `kernelSize`
	///   - channelPosition: channel position in input data
	///   - paddingMode: paddingMode
	///   - inputTensors: inputTensors
	///   - outputTensors: outputTensors
	///   - computationDelegate: computationDelegate
	///   - operatorLabel: operatorLabel
	required public init(kernelSize: [Int],
	            stride: [Int]? = nil,
	            channelPosition: TensorChannelOrder = TensorChannelOrder.First,
	            paddingMode: PaddingMode = PaddingMode.Valid,
	            inputTensors: [Tensor]? = nil,
	            outputTensors: [Tensor]? = nil,
	            computationDelegate: OperatorCalculationDelegate? = nil,
	            operatorLabel: String = "Pooling") {
		self.kernelSize = kernelSize
		if stride == nil {
			self.stride = kernelSize
		} else {
			self.stride = stride!
		}
		self.channelPosition = channelPosition
		self.paddingMode = paddingMode
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
		self.computationDelegate  = computationDelegate
		self.operatorLabel = operatorLabel
		self.kernelSize = kernelSize
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Compute output shape according to `kernelSize`,  `stride` and `paddingMode`
	///
	/// - Parameter shapes: shapes description
	/// - Returns: return value description
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		// input not empty
		guard shapes.count != 0 else {
			SerranoLogging.errorLogging(message: "Input shapes array is empty",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		// kernel valid
		guard self.kernelSize.count == 2 && self.kernelSize[0] > 0 && self.kernelSize[1] > 0 else {
			SerranoLogging.errorLogging(message: "Invalid kernelSize: \(self.kernelSize)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		// stride check
		guard self.stride.count == 2 && self.stride[0] > 0 && self.stride[0] > 0 else {
			SerranoLogging.errorLogging(message: "Invalid stride: \(self.stride)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		var outputShapes = [TensorShape]()
		for shape in shapes {
			// check valid shape
			guard shape.rank == 3 && shape.shapeArray[0] > 0 && shape.shapeArray[1] > 0 && shape.shapeArray[2] > 0 else {
				SerranoLogging.errorLogging(message: "Input shape is not valid \(shape.description).",
				                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
			let (channel, height, width) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: shape.shapeArray)
			var outShapeArray = [kernelScanningOutSize(self.paddingMode, inputSize: height,
													   kernelSize: self.kernelSize[0], stride: self.stride[0]),
								 kernelScanningOutSize(self.paddingMode, inputSize: width,
													   kernelSize: self.kernelSize[1], stride: self.stride[1]),
								 channel]
			if self.channelPosition == TensorChannelOrder.First {
				outShapeArray = [channel,
				                 kernelScanningOutSize(self.paddingMode, inputSize: height,
				                                       kernelSize: self.kernelSize[0], stride: self.stride[0]),
				                 kernelScanningOutSize(self.paddingMode, inputSize: width,
				                                       kernelSize: self.kernelSize[1], stride: self.stride[1])]
			}
	
			// valid out shape
			guard outShapeArray[0] > 0 && outShapeArray[1] > 0 && outShapeArray[2] > 0 else {
				SerranoLogging.errorLogging(message: "Input shape \(shape.description) is not valid which will lead to negative output dimension.",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
			outputShapes.append(TensorShape(dataType: shape.dataType, shape: outShapeArray))
		}
		
		return outputShapes
	}
	
	/// Check validation between `inputTensors`/`outputTensors` and `stride`, `kernelSize`.
	///
	/// - Returns: check indicating if pass checking, msg for error message.
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
		// input not nil
		guard self.inputTensors != nil else {
			return (false, "Attribute inputTensors is nil")
		}
		
		// output not nil
		guard self.outputTensors != nil else {
			return (false, "Attribute outputTensors is nil")
		}
		
		// input shape check
		let inputShapes = self.inputTensors!.map { $0.shape }
		let outputShapeCheck = self.outputShape(shapeArray: inputShapes)
		guard outputShapeCheck != nil else {
			return (false, "Input tensors are not valid. Check log for details.")
		}
		
		// output shape check
		let outputShapes = self.outputTensors!.map { $0.shape }
		guard outputShapes.count == outputShapeCheck!.count else {
			return (false, "Attribute outputTensors should have same number of tensors as inputTensors. " +
			               "Expect \(self.inputTensors!.count) tensors, given \(self.outputTensors!.count) tensors.")
		}
		for (outputShape, checkShape) in zip(outputShapes, outputShapeCheck!) {
			guard outputShape == checkShape else {
				return (false, "One of outputTensors has invalid shape. Expect shape \(checkShape.description), given \(outputShape.description)")
			}
		}
		
		return (true, "")
	}
	
	
	/// Compute sync
	///
	/// - Parameter computationMode: computationMode
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
	
	/// Calulate grads sync.
	/// All unary operator return grads tensor with same number and shape as attribute `inputTensors`.
	///
	/// - Parameters:
	///   - computationMode: computationMode
	/// - Returns: return grads tensor
	public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType] {
		//TODO: Implementation
		fatalError("Not implemented")
	}
	
	/// Cal grads async
	///
	/// - Parameters:
	///   - computationMode: computationMode
	public func gradComputAsync(_ computationMode: OperatorComputationMode) {
		// check delegate
		OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
		
		DispatchQueue.global(qos: .userInitiated).async {
			_ = self.gradCompute(computationMode)
		}
	}
	
	/// This operator has no parameters. Do nothing
	///
	public func bindParamSymbols(_ symbols: [GraphSymbol]) {
		
	}
	
	/// This operator has no parameters.
	///
	/// - Returns: An empty array
	public func paramSymbols() -> [GraphSymbol] {
		return [GraphSymbol]()
	}
	
	/// CPU calcualtion. Call `cpuComputeBlock` which is defined in subclass
	internal func cpu() {
		let workGroup = DispatchGroup()
		for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				self.cpuComputeBlock!(input, output)
				workGroup.leave()
			}
		}
		workGroup.wait()
	}
	
	/// GPU calculation
	internal func gpu() {
		// prepare resources
		let engine = SerranoEngine.configuredEngine
		// kernel
		let (kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
		guard kernel != nil else {
			fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
		}
		
		// command buffer
		let commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
		guard commandBuffer != nil else {
			fatalError("[Serrano] Failed to make new command buffer.")
		}
		
		// encoder
		for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
			let inputBufferResource = input.gpuBufferResource()
			let outputBufferResource = output.gpuBufferResource()
			var info = Pool2DInfo.makePool2DInfo(op: self, inputSize: input.shape.shapeArray, outputSize: output.shape.shapeArray)

			let encoder = commandBuffer!.makeComputeCommandEncoder()
			encoder.setComputePipelineState(kernel!)
			encoder.setBuffer(inputBufferResource.buffer, offset: inputBufferResource.offset, at: 0)
			encoder.setBuffer(outputBufferResource.buffer, offset: outputBufferResource.offset, at: 1)
			encoder.setBytes(&info, length: MemoryLayout<Pool2DInfo>.stride, at: 2)
			
			/// Calculate grid
			let (channel, outHeight, outWidth) = parseImgChannelShapeInfo(self.channelPosition,
																		  shapeArray: input.shape.shapeArray)
			let threadsPerThreadgroup = MTLSizeMake(16,
													Int(Float(kernel!.threadExecutionWidth / 16).rounded(FloatingPointRoundingRule.up)), // incase threadExecutionWidth is very samll
													1)
			let threadgroupsPerGrid = MTLSizeMake((outWidth + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
												  (outHeight + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
												  channel)
			encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
			encoder.endEncoding()
		}
		
		// commit command buffer
		commandBuffer!.commit()
		commandBuffer!.waitUntilCompleted()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
2D Max pooling.

## Input shape
Input data should be a tensor with 3D shape `[channel, height, width]` or `[height, width, channel]`
according to the `channelPosition`.

## Output shape
Output shape follow input shape's order.
*/
public class MaxPool2DOperator: Pooling2DOperator {
	public required init(kernelSize: [Int], stride: [Int]? = nil,
	                     channelPosition: TensorChannelOrder = TensorChannelOrder.First,
	                     paddingMode: PaddingMode = PaddingMode.Valid,
	                     inputTensors: [Tensor]? = nil,
	                     outputTensors: [Tensor]? = nil,
	                     computationDelegate: OperatorCalculationDelegate? = nil,
	                     operatorLabel: String = "Pooling")  {
		super.init(kernelSize: kernelSize, stride: stride, channelPosition: channelPosition,
		           paddingMode: paddingMode,
		           inputTensors: inputTensors, outputTensors: outputTensors,
		           computationDelegate: computationDelegate, operatorLabel: operatorLabel)
		self.metalKernelFuncLabel = "MaxPool2D"
		self.operatorLabel = "MaxPool2DOperator"

		self.cpuComputeBlock = { (input, output) -> Void in
			let workGroup = DispatchGroup()
			let outShapeArray = output.shape.shapeArray
			let inShapeArray = input.shape.shapeArray
			let stride = self.stride
			let kernelSize = self.kernelSize
			let (channel, outHeight, outWidth) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: outShapeArray)
			let (_, inHeight, inWidth) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: inShapeArray)
			
			// according to channel order, do pooling.
			// Here we are trying to taking advantage of spatial locality for input
			// TODO: I doubt if this really improve performance. Should do profiling and verify.
			if self.channelPosition == TensorChannelOrder.First {
				for c in 0..<channel{
					for i in 0..<outHeight {
						for j in 0..<outWidth {
							// pooling
							workGroup.enter()
							DispatchQueue.global(qos: .userInitiated).async {
								let validHeightCount = min(i * stride[0] + kernelSize[0], inHeight) - i * stride[0]
								let validWidthCount = min(j * stride[1] + kernelSize[1], inWidth) - j * stride[1]
								var max_v: Float = -Float.infinity
								for m in 0..<validHeightCount {
									for n in 0..<validWidthCount {
										max_v = max(input[withoutChecking:[c, i*stride[0] + m, j*stride[1] + n]], max_v)
									}
								}
								output[c, i, j] = max_v;
								workGroup.leave()
							}
						}
					}
				}
			} else {
				for i in 0..<outHeight {
					for j in 0..<outWidth {
						for c in 0..<channel {
							// pooling
							workGroup.enter()
							DispatchQueue.global(qos: .userInitiated).async {
								let validHeightCount = min(i * stride[0] + kernelSize[0], inHeight) - i * stride[0]
								let validWidthCount = min(j * stride[1] + kernelSize[1], inWidth) - j * stride[1]
								var max_v: Float = -Float.infinity
								for m in 0..<validHeightCount {
									for n in 0..<validWidthCount {
										max_v = max(input[withoutChecking:[i*stride[0] + m, j*stride[1] + n, c]], max_v)
									}
								}
								output[i, j, c] = max_v;
								workGroup.leave()
							}
						}
					}
					
				}
			}
			workGroup.wait()
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
2D Average pooling.

## Input shape
Input data should be a tensor with 3D shape `[channel, height, width]` or `[height, width, channel]`
according to the `channelPosition`.
*/
public class AvgPool2DOperator: Pooling2DOperator {
	public required init(kernelSize: [Int], stride: [Int]? = nil,
	                     channelPosition: TensorChannelOrder = TensorChannelOrder.First,
	                     paddingMode: PaddingMode = PaddingMode.Valid,
	                     inputTensors: [Tensor]? = nil,
	                     outputTensors: [Tensor]? = nil,
	                     computationDelegate: OperatorCalculationDelegate? = nil,
	                     operatorLabel: String = "Pooling")  {
		super.init(kernelSize: kernelSize, stride: stride, channelPosition: channelPosition,
		           paddingMode: paddingMode,
		           inputTensors: inputTensors, outputTensors: outputTensors,
		           computationDelegate: computationDelegate, operatorLabel: operatorLabel)
		self.metalKernelFuncLabel = "AvgPool2D"
		self.operatorLabel = "AvgPool2DOperator"
		
		self.cpuComputeBlock = { (input, output) -> Void in
			let workGroup = DispatchGroup()
			let outShapeArray = output.shape.shapeArray
			let inShapeArray = input.shape.shapeArray
			let stride = self.stride
			let kernelSize = self.kernelSize
			let (channel, outHeight, outWidth) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: outShapeArray)
			let (_, inHeight, inWidth) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: inShapeArray)
			
			// according to channel order, do pooling.
			// Here we are trying to taking advantage of spatial locality for input
			// TODO: I doubt if this really improve performance. Should do profiling and verify.
			if self.channelPosition == TensorChannelOrder.First {
				for c in 0..<channel{
					for i in 0..<outHeight {
						for j in 0..<outWidth {
							// pooling
							workGroup.enter()
							DispatchQueue.global(qos: .userInitiated).async {
								let validHeightCount = min(i * stride[0] + kernelSize[0], inHeight) - i * stride[0]
								let validWidthCount = min(j * stride[1] + kernelSize[1], inWidth) - j * stride[1]
								var sum: Float = 0.0
								for m in 0..<validHeightCount {
									for n in 0..<validWidthCount {
										sum += input[withoutChecking:[c, i*stride[0] + m, j*stride[1] + n]]
									}
								}
								output[c, i, j] = sum / Float(kernelSize[0] * kernelSize[1]);
								workGroup.leave()
							}
						}
					}
				}
			} else {
				for i in 0..<outHeight {
					for j in 0..<outWidth {
						for c in 0..<channel {
							// pooling
							workGroup.enter()
							DispatchQueue.global(qos: .userInitiated).async {
								let validHeightCount = min(i * stride[0] + kernelSize[0], inHeight) - i * stride[0]
								let validWidthCount = min(j * stride[1] + kernelSize[1], inWidth) - j * stride[1]
								var sum: Float = 0.0
								for m in 0..<validHeightCount {
									for n in 0..<validWidthCount {
										sum += input[withoutChecking:[i*stride[0] + m, j*stride[1] + n, c]]
									}
								}
								output[i, j, c] = sum / Float(kernelSize[0] * kernelSize[1]);
								workGroup.leave()
							}
						}
					}
					
				}
			}
			workGroup.wait()
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
2D Sum Pooling.

## Input shape
Input data should be a tensor with 3D shape `[channel, height, width]` or `[height, width, channel]`
according to the `channelPosition`.
*/
public class SumPool2DOperator: Pooling2DOperator {
	public required init(kernelSize: [Int], stride: [Int]? = nil,
	                     channelPosition: TensorChannelOrder = TensorChannelOrder.First,
	                     paddingMode: PaddingMode = PaddingMode.Valid,
	                     inputTensors: [Tensor]? = nil,
	                     outputTensors: [Tensor]? = nil,
	                     computationDelegate: OperatorCalculationDelegate? = nil,
	                     operatorLabel: String = "Pooling")  {
		super.init(kernelSize: kernelSize, stride: stride, channelPosition: channelPosition,
		           paddingMode: paddingMode,
		           inputTensors: inputTensors, outputTensors: outputTensors,
		           computationDelegate: computationDelegate, operatorLabel: operatorLabel)
		self.metalKernelFuncLabel = "SumPool2D"
		self.operatorLabel = "SumPool2DOperator"
		
		self.cpuComputeBlock = { (input, output) -> Void in
			let workGroup = DispatchGroup()
			let outShapeArray = output.shape.shapeArray
			let inShapeArray = input.shape.shapeArray
			let stride = self.stride
			let kernelSize = self.kernelSize
			let (channel, outHeight, outWidth) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: outShapeArray)
			let (_, inHeight, inWidth) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: inShapeArray)
			
			// according to channel order, do pooling.
			// Here we are trying to taking advantage of spatial locality for input
			// TODO: I doubt if this really improve performance. Should do profiling and verify.
			if self.channelPosition == TensorChannelOrder.First {
				for c in 0..<channel{
					for i in 0..<outHeight {
						for j in 0..<outWidth {
							// pooling
							workGroup.enter()
							DispatchQueue.global(qos: .userInitiated).async {
								let validHeightCount = min(i * stride[0] + kernelSize[0], inHeight) - i * stride[0]
								let validWidthCount = min(j * stride[1] + kernelSize[1], inWidth) - j * stride[1]
								var sum: Float = 0.0
								for m in 0..<validHeightCount {
									for n in 0..<validWidthCount {
										sum += input[withoutChecking:[c, i*stride[0] + m, j*stride[1] + n]]
									}
								}
								output[c, i, j] = sum ;
								workGroup.leave()
							}
						}
					}
				}
			} else {
				for i in 0..<outHeight {
					for j in 0..<outWidth {
						for c in 0..<channel {
							// pooling
							workGroup.enter()
							DispatchQueue.global(qos: .userInitiated).async {
								let validHeightCount = min(i * stride[0] + kernelSize[0], inHeight) - i * stride[0]
								let validWidthCount = min(j * stride[1] + kernelSize[1], inWidth) - j * stride[1]
								var sum: Float = 0.0
								for m in 0..<validHeightCount {
									for n in 0..<validWidthCount {
										sum += input[withoutChecking:[i*stride[0] + m, j*stride[1] + n, c]]
									}
								}
								output[i, j, c] = sum;
								workGroup.leave()
							}
						}
					}
					
				}
			}
			workGroup.wait()
		}
	}
}


