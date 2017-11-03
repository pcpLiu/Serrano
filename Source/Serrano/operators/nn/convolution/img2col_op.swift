//
//  img2col.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Dispatch
import Accelerate

/// Corresponding struct `Img2ColInfo` in img2col_op.metal
public struct Img2ColInfo {
	/// 0 --> First, 1 --> Last
	var channelPosition: MetalShort
	
	/// 0 --> Valid, 1 --> Same
	var paddingMode: MetalShort
	
	/// padding value
	var paddingValue: MetalFloat
	
	/// number of channels
	var channels: MetalInt
	
	/// input width
	var inputWidth: MetalInt
	
	/// input height
	var inputHeight: MetalInt
	
	/// kernel scanning patch count in X direction
	var kernelScanningXPatchCount: MetalInt
	
	/// stride width
	var strideWidth: MetalInt
	
	/// stride height
	var strideHeight: MetalInt
	
	/// patch width
	var patchWdith: MetalInt
	
	/// patch height
	var patchHeight: MetalInt
}

/**
Operator works like img2col in matlab.
It converts any 3D tensor (`[H, W, C]` or `[C, H, W]`) into a 2D tensor (`[H*W, C*M*N]`) according to patchSize `[M, N]` and stride.
*/
public class Img2ColOperator: ComputableOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Operator label. Conforms to `ComputableOperator`
	public var operatorLabel: String
	
	/// This operator does not operator on GPU. Conforms to `ComputableOperator`
	public var metalKernelFuncLabel:String = "Img2col"
	
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
	
	/// The patch size. 2D vecotr. `[patchHeight, patchWidth]`
	public var patchSize: [Int]
	
	/// The stride. 2D vector. Default is `[1, 1]`. `[strideHeight, strideWidth]`
	public var stride: [Int] = [1, 1]
	
	/// Channel position. Default is `ImageChannelOrder.First`
	public var channelPosition: TensorChannelOrder =  .First
	
	/// Padding mode. Default is `PaddingMode.Valid`
	public var padMode: PaddingMode = .Valid
	
	/// Padding value
	public var paddingValue: Float = 0.0
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Init
	
	/// Designated init
	///
	/// - Parameters:
	///   - patchSize:  [patchHeight, patchWidth]
	///   - channelPosition: channelPosition description
	///   - padMode: padMode
	///	  - stride: [strideHeight, strideWidth]
	///   - computationDelegate: computationDelegate description
	///   - inputTensors: inputTensors description
	///   - outputTensors: outputTensors description
	///   - operatorLabel: operatorLabel description
	public init(patchSize: [Int], stride: [Int],
	            channelPosition: TensorChannelOrder = .First,
	            padMode: PaddingMode = PaddingMode.Valid,
				computationDelegate: OperatorCalculationDelegate? = nil,
				inputTensors: [Tensor]? = nil, outputTensors: [Tensor]? = nil,
				operatorLabel: String = "Img2ColOp",
				disableInputOutputCheck: Bool = false) {
		self.patchSize = patchSize
		self.channelPosition = channelPosition
		self.padMode = padMode
		self.stride = stride
		self.computationDelegate = computationDelegate
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
		self.operatorLabel = operatorLabel
		self.disableInputOutputCheck = disableInputOutputCheck
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Compute output shape according `numFilters`, `kernelSize`, `stride` and `dilation`.
	///
	/// - Parameter shapes: shapes description
	/// - Returns: return value description
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		// patch size validation check
		guard self.patchSize.count == 2 && self.patchSize[0] > 0 && self.patchSize[1] > 0 else {
			SerranoLogging.errorLogging(message: "Invalid patchSize \(self.patchSize).",
										file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		// stride check
		guard self.stride.count == 2 && self.stride[0] > 0 && self.stride[1] > 0 else {
			SerranoLogging.errorLogging(message: "Invalid stride \(self.stride).",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		var outShapes = [TensorShape]()
		for inShape in shapes {
			let shapeArray = inShape.shapeArray
			guard inShape.rank == 3  && shapeArray[0] > 0 && shapeArray[1] > 0 && shapeArray[2] > 0 else {
				SerranoLogging.errorLogging(message: "Invalid input shape \(inShape.description)",
											file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
			
			// get input shape values
			let (channel, height, width) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: shapeArray)
			
			// compute output shape array
			var outWidth = 0
			var outHeight = 0
			if self.padMode == .Valid {
				// In valid mode, if the input size is less than patchSize. We do not accept it
				guard height >= self.patchSize[0] && width >= self.patchSize[1] else {
					SerranoLogging.errorLogging(message: "Padding mode is Valid and the input shape \(inShape.description) with width \(width) and height \(height) is not valid with patchSize \(self.patchSize).",
					                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
					return nil
				}
			}
			outHeight = kernelScanningOutSize(self.padMode, inputSize: height, kernelSize: self.patchSize[0], stride: self.stride[0])
			outWidth = kernelScanningOutSize(self.padMode, inputSize: width, kernelSize: self.patchSize[1], stride: self.stride[1])
			// valid out shape
			guard outHeight > 0 && outWidth > 0 else {
				SerranoLogging.errorLogging(message: "Input shape \(inShape.description) is not valid which will lead to negative output dimension.",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
			outShapes.append(TensorShape(dataType: inShape.dataType, shape: [outHeight * outWidth,
			                                                                 self.patchSize[0] * self.patchSize[1] * channel]))
			
		}
		return outShapes
	}
	
	
	/// Check input and output tensors.
	///
	/// - Returns: return value description
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
	
	/// Compute sync way.
	///
	/// - Parameter computationMode: mode
	public func compute(_ computationMode: OperatorComputationMode) {
		// check
		if !self.disableInputOutputCheck {
			let (pass, msg) = self.inputOutputTensorsCheck()
			guard pass else {
				SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) aborts calculation cause given invalid data: \(msg)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError()
			}
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
	
	
	/// This operator itself does not do any grad update.
	///
	/// - Parameters:
	///   - computationMode: computationMode description
	/// - Returns: return value description
	public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType] {
		return [:]
	}
	
	
	/// This operator itself does not do any grad update.
	///
	/// - Parameters:
	///   - computationMode: computationMode description
	///   - upGrads: upGrads description
	public func gradComputAsync(_ computationMode: OperatorComputationMode) {
		// check delegate
		OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
		
		DispatchQueue.global(qos: .userInitiated).async {
			self.computationDelegate?.operatorWillBeginGradsComputation(self)
			let result = self.gradCompute(computationMode)
			self.computationDelegate?.operatorDidEndGradsComputation(self, grads: result)
		}
	}
	
	
	/// Do nothing. No param to update
	///
	/// - Parameters:
	///   - grads: grads description
	///   - LR: LR description
	public func updateParams(grads: [Tensor], LR: Float) {
		// no param to update
		return
	}
	
	/// This operator has no parameters. Do nothing
	///
	public func bindParamSymbols(_ symbols: [GraphSymbol]) {
		
	}
	
	/// This operator returns no param symbols
	///
	/// - Returns: empty array
	public func paramSymbols() -> [GraphSymbol] {
		return []
	}
	
	/// CPU calculation
	internal func cpu() {
		let patchHeight = self.patchSize[0]
		let patchWidth = self.patchSize[1]
		let patchElementCount = patchWidth * patchHeight
		
		let strideHeight = self.stride[0]
		let strideWidth = self.stride[1]
		
		let workGroup = DispatchGroup()
		for (inTensor, outTensor) in zip(self.inputTensors!, self.outputTensors!) {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				let inShapeArray = inTensor.shape.shapeArray
				let (channels, inHeight, inWidth) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: inShapeArray)
				
				// get out dim size
				let outHeight = kernelScanningOutSize(self.padMode, inputSize: inHeight, kernelSize: patchHeight, stride: strideHeight)
				let outWidth = kernelScanningOutSize(self.padMode, inputSize: inWidth, kernelSize: patchWidth, stride: strideWidth)
				
				let patchCalculationGroup = DispatchGroup()
				for i in 0..<outHeight {
					for j in 0..<outWidth {
						for channelIndex in 0..<channels {
							patchCalculationGroup.enter()
							DispatchQueue.global(qos: .userInitiated).async {
								for m in 0..<patchHeight {
									for n in 0..<patchWidth {
										let outIndex = [i * outWidth + j,
										                channelIndex * patchElementCount + (m * patchWidth + n)]
										if self.channelPosition == .First {
											outTensor[outIndex] = inTensor.fetchValueOrDefault([channelIndex, i*strideHeight + m, j*strideWidth + n],
											                                                   missingValue: self.paddingValue)
										} else if self.channelPosition == .Last {
											outTensor[outIndex] = inTensor.fetchValueOrDefault([i*strideHeight + m, j*strideWidth + n, channelIndex],
											                                                   missingValue: self.paddingValue)
										} else {
											fatalError("Not implemented")
										}
									}
								}
								patchCalculationGroup.leave()
							}
						}
					}
				}
				patchCalculationGroup.wait()
				
				workGroup.leave()
			}
		}
		workGroup.wait()
	}
	
	/// make `Img2ColInfo`
	///
	/// - Parameters:
	///   - inputShape: input
	///   - outputShape: output
	/// - Returns: struct
	internal func makeImg2ColInfo(inputShape: TensorShape, outputShape: TensorShape) -> Img2ColInfo {
		let (channel, inputHeight, inputWidth) = parseImgChannelShapeInfo(self.channelPosition, shapeArray: inputShape.shapeArray)
		let kernelScanningXPatchCount = kernelScanningOutSize(self.padMode, inputSize: inputWidth,
		                                                      kernelSize: self.patchSize[1], stride: self.stride[1])
		return Img2ColInfo(channelPosition: MetalShort(self.channelPosition.rawValue),
		                   paddingMode: MetalShort(self.padMode.rawValue), paddingValue: self.paddingValue,
		                   channels: MetalInt(channel), inputWidth: MetalInt(inputWidth), inputHeight: MetalInt(inputHeight),
		                   kernelScanningXPatchCount: MetalInt(kernelScanningXPatchCount),
		                   strideWidth: MetalInt(self.stride[1]), strideHeight: MetalInt(self.stride[0]),
		                   patchWdith: MetalInt(self.patchSize[1]), patchHeight: MetalInt(self.patchSize[0]))
	}
	
	/// GPU calculation.
	///
	/// Split the output 2D matrix into threadgroups based on each patch and channel index.
	/// Calculate each element in each group independently.
	internal func gpu() {
		// prepare resource
		let resourcePrepareGroup = DispatchGroup()
		let engine = SerranoEngine.configuredEngine
		var kernel: MTLComputePipelineState?
		var commandBuffers: [MTLCommandBuffer] = [MTLCommandBuffer]()
		var inputBuffers: [MTLBufferResource] = [MTLBufferResource]()
		var resultBuffers: [MTLBufferResource] = [MTLBufferResource]()
		
		//// kernel
		resourcePrepareGroup.enter()
		DispatchQueue.global(qos: .userInitiated).async {
			var info = ""
			(kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
			guard kernel != nil else {
				fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
			}
			resourcePrepareGroup.leave()
		}
		
		//// command buffers
		resourcePrepareGroup.enter()
		DispatchQueue.global(qos: .userInitiated).async {
			for _ in 0..<self.inputTensors!.count {
				let commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
				guard commandBuffer != nil else {
					fatalError("[Serrano] Failed to make new command buffer.")
				}
				commandBuffers.append(commandBuffer!)
			}
			resourcePrepareGroup.leave()
		}
		
		/// input and output buffers
		resourcePrepareGroup.enter()
		DispatchQueue.global(qos: .userInitiated).async {
			inputBuffers = SerranoResourceManager.globalManager.allocateMTLBufferResources(self.inputTensors!)
			resultBuffers = SerranoResourceManager.globalManager.allocateMTLBufferResources(self.outputTensors!)
			resourcePrepareGroup.leave()
		}
		
		resourcePrepareGroup.wait()
		
		for bufferIndex in 0..<inputBuffers.count {
			resourcePrepareGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				let buffer = commandBuffers[bufferIndex]
				let encoder = buffer.makeComputeCommandEncoder()
				encoder.setComputePipelineState(kernel!)
				
				// set data
				encoder.setBuffer(inputBuffers[bufferIndex].buffer, offset: inputBuffers[bufferIndex].offset, at: 0)
				encoder.setBuffer(resultBuffers[bufferIndex].buffer, offset: resultBuffers[bufferIndex].offset, at: 1)
				var info = self.makeImg2ColInfo(inputShape: self.inputTensors![bufferIndex].shape,
				                                outputShape: self.outputTensors![bufferIndex].shape)
				encoder.setBytes(&info, length: MemoryLayout<Img2ColInfo>.stride, at: 2)
				
				// View output matrix as a [outputHeight * outputWidth, patchCount, channels] matrix, las dim is channels
				// Each group, spawn thread for each point
				let outHeight = self.outputTensors![bufferIndex].shape.shapeArray[0]
				let channels = Int(info.channels)
				let threadsPerThreadgroup = MTLSizeMake(self.patchSize[1], // width
				                                        self.patchSize[0], // height
				                                        1)
				let threadgroupsPerGrid = MTLSizeMake(1,
				                                      outHeight,
					                                  channels)
				SerranoLogging.stdLogging(message: "Dispatch group configured with threadgroupsPerGrid: \(threadgroupsPerGrid), threadsPerThreadgroup: \(threadsPerThreadgroup) requested by operator \(self.operatorLabel)",
					file: "\(#file)", function: "\(#function)", line: "\(#line)",  loggingLevel: .LowLevel)
				encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

				encoder.endEncoding()
				
				buffer.commit()
				buffer.waitUntilCompleted()
				
				resourcePrepareGroup.leave()
			}
		}
		
		resourcePrepareGroup.wait()
	}


}
