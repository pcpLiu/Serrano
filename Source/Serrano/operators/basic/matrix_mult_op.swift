//
//  dot_product_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Accelerate
import Metal
#if  !((arch(i386)  || arch(x86_64)) && os(iOS)) // prevent build error on simulaor
	import MetalPerformanceShaders
#endif

/// This struct corresponds to the `MatrixDimInfo` struct in file 'matrix_mult_op.metal'
public struct MatrixDimInfo {
	var M: MetalUInt // number of rows in A
	var N: MetalUInt // number of cols in B
	var K: MetalUInt // number of cols in A, number of rows in B
	var stride: MetalUShort // element stride in bytes
}


/// Two kernels
///
/// - Single: Caclulate a single element each thread
/// - SubMatrix: Caculate a submatrix each thread
public enum MatrixMultKernel {
	case Single
	case SubMatrix
}


/**
matrix multiplication.

## Transpose input tensors
Opertor's attributes `transposeA` and `transposeB` indicating if tranposing input tensors before doing calculation.
And if any or both of them are set  to `true`, all caulcation and input/output validation will be doing __after__ transposing.

## Metal performance shader support
By default, operator tries to use [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication) in `MetalPerformanceShaders`.
But on some devices which [do not support `MetalPerformanceShaders`](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), we use self kernel.

## Multiple input
This operator could takein multiple input. Currently, it support multiple input `A` and single input `B`.
If `inputTensors` contains more than 2 elements, operator will view last element as input `B` and all previous element as input `A`s.

*/
public class MatrixMultOperator: ComputableOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// The submatrix size in Metal calcualtion.
	/// Should be the same as `SUBMATRIX_SIZE` in `matrix_mult_op.metal`
	public static let METAL_SUBMATRIX_SIZE = 4
	
	/// Operator label. Conforms to `ComputableOperator`
	public var operatorLabel: String
	
	/// This operator does not operator on GPU. Conforms to `ComputableOperator`
	public var metalKernelFuncLabel:String = "MatrixMult"
	
	/// Conforms to `ComputableOperator`
	public var computationDelegate: OperatorCalculationDelegate?
	
	/// Conforms to `ComputableOperator`
	public var inputTensors: [Tensor]?
	
	/// Conforms to `ComputableOperator`
	public var outputTensors: [Tensor]?
	
	/// Whether transpose inputA before calculation
	public var transposeA: Bool
	
	/// Whether transpose inputB before calcualtion
	public var transposeB: Bool
	
	/// If `true`, operator will not call `inputOutputTensorsCheck()` before doing calculation.
	/// This is used inside framework to speed up in situation we know it will not be wrong.
	public var disableInputOutputCheck: Bool = false
	
	/// Indicate if this operator would do paramter update.
	public var trainable: Bool = false
	
	/// Kernel to choose
	public var kernel: MatrixMultKernel
	
	/// The mapping type of this operator.
	/// `Constant` for this operator.
	public var mapType: OperatorMappingType {
		get {
			return OperatorMappingType.Constant
		}
	}
	
	/// Matrix multiplication cannot do in-place calculation 
	public var inPlaceble: Bool = false
	
	/// If use MPS. Default is `false`
	internal var disabledMPS: Bool = false
	
	/// beta for mstrix calculation, if want to add into output tensor, set this to 1
	internal var matrixBeta: Float = 0.0
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializer
	
	public init(operatorLabel: String = "MatrixMultOperator",
	            computationDelegate: OperatorCalculationDelegate? = nil,
	            transposeA: Bool = false, transposeB: Bool = false,
	            inputTensors: [Tensor]? = nil, outputTensors: [Tensor]? = nil,
	            kernel: MatrixMultKernel = MatrixMultKernel.Single,
	            disableInputOutputCheck: Bool = false) {
		self.operatorLabel = operatorLabel
		self.computationDelegate = computationDelegate
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
		self.transposeA = transposeA
		self.transposeB = transposeB
		self.kernel = kernel
		self.disableInputOutputCheck = disableInputOutputCheck
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Conforms to `ComputableOperator`
	
	/// Check the input shapes. Following same rule of matrix multiplication.
	///
	/// - Note: This function will transpose shape first and then calcualte output shape.
	///
	/// - Parameter shapes: input shapes
	/// - Returns: return shapes
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		// >=2 shapes
		guard shapes.count >= 2 else {
			SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) expect more than 2 shapes, given \(shapes.count)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		var shapeAs = shapes[0..<shapes.count-1]
		var shapeB = shapes.last!
		var outputShapes = [TensorShape]()

		// do transpose
		if self.transposeA {
			for (i,shapeA) in shapeAs.enumerated() {
				shapeAs[i] = shapeA.transposed()
			}
		}
		if self.transposeB {
			shapeB = shapeB.transposed()
		}
		
		for shapeA in shapeAs {
			// shapeA validation
			guard shapeA.rank == 2 && shapeB.rank == 2 else {
				SerranoLogging.errorLogging(message: "Shape A and shape B should have ranks as 2. Given shapeA rank \(shapeA.rank) , shapeB rank \(shapeB.rank) ",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
			
			// multiplication rule validation
			guard shapeA.shapeArray[1] == shapeB.shapeArray[0] else {
				SerranoLogging.errorLogging(message: "Shape A and shape B does not follow matrix multiply rule. Given shapeA \(shapeA.shapeArray) , shapeB \(shapeB.shapeArray) ",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
			
			outputShapes.append(TensorShape(dataType: shapeA.dataType, shape: [shapeA.shapeArray[0], shapeB.shapeArray[1]]))
		}
		
		
		return outputShapes
	}
	
	
	/// Check if assigned `inputTensors` and `outputTensors` valid
	///
	/// - Note: This function will first transposes input tensors' shapes according to attributes `transposeA` and `transposeB`,
	///		    and then validate the shapes.
	///
	/// - Returns: the result and erro message if possible
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
		// not nil inputs
		guard self.inputTensors != nil else {
			return (false, "Input tensors are nil")
		}
		
		// >=2 input tensors
		guard self.inputTensors!.count >= 2 else {
			return (false, "Invalid input tneosr. Requires more than 2, given \(self.inputTensors!.count)")
		}
		
		// output not nil
		guard self.outputTensors != nil else {
			return (false, "Output tensors are nil")
		}
		
		// output tensors count match
		guard self.outputTensors!.count == self.inputTensors!.count - 1 else {
			return (false, "Invalid number of output tensors. Requires \(self.inputTensors!.count - 1), given \(self.inputTensors!.count)")
		}
		
		// check dimension
		let inputShapes = self.inputTensors!.map { $0.shape }
		let checkResult = self.outputShape(shapeArray: inputShapes)
		guard checkResult != nil else {
			return (false, "Invalid input tneosrs. See logs for details.")
		}
		
		// outptu dimension
		let outputShapes = self.outputTensors!.map {$0.shape}
		for (outShape, checkShape) in zip(outputShapes, checkResult!) {
			guard outShape.shapeArray == checkShape.shapeArray else {
				return (false, "Invalid output tensor. Expecting shape \(checkShape.shapeArray), given \(outShape.shapeArray)")
			}
		}
		
		return (true, "")
	}
	
	
	/// Compute sync
	///
	/// - Parameter computationMode: mode
	public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
		// check
		if self.disableInputOutputCheck {
			let (pass, msg) = self.inputOutputTensorsCheck()
			guard pass else {
				SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) calculation aborted cause invalid input tensors or output tensors: \(msg)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError()
			}
		}
		
		self.computationDelegate?.operatorWillBeginComputation(self)
		
		switch computationMode {
		case .GPU:
			if !SerranoEngine.configuredEngine.hasAvailableGPU() {
				SerranoLogging.warningLogging(message: "Serrano Engine has no available configured GPU device. Use CPU doing calculation instead.",
				                              file: "\(#file)", function: "\(#function)", line: "\(#line)")
				self.cpu()
			} else {
				self.gpu()
			}
		case .CPU:
			self.cpu()
		case .Auto:
			if self.outputTensors![0].count > 4000000 && SerranoEngine.configuredEngine.hasAvailableGPU() {
				self.gpu()
			} else {
				self.cpu()
			}
		}
		self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
	}
	
	
	/// Compute async
	///
	/// - Parameter computationMode: mode
	public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
		// delegation check
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
	///   - upGrds: upGrds
	/// - Returns: return grads tensor
	public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType] {
		//TODO: Implementation
		fatalError("Not implemented")
	}
	
	/// Cal grads async
	///
	/// - Parameters:
	///   - computationMode: computationMode
	///   - upGrds: upGrds
	public func gradComputAsync(_ computationMode: OperatorComputationMode) {
		// check delegate
		OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
		
		DispatchQueue.global(qos: .userInitiated).async {
			_ = self.gradCompute(computationMode)
		}
	}
	
	/// Update params if possible.
	/// No update parameters for binary operators.
	///
	/// - Parameters:
	///   - grads: grads tensor list
	///   - LR: learning rate
	public func updateParams(grads: [Tensor], LR: Float) {
		return
	}
	
	/// Do nothing 
	public func bindParamSymbols(_ symbols: [GraphSymbol]) {
		// DO NOTHING
	}
	
	/// This operator has no parameters.
	///
	/// - Returns: An empty array
	public func paramSymbols() -> [GraphSymbol] {
		return [GraphSymbol]()
	}
	
	/// Get M, N, K attributes
	///
	/// - Returns: M, N, K
	internal func MNKFetch(tensorA: Tensor, tensorB: Tensor) -> (M: Int, N: Int, K: Int) {
		var M = tensorA.shape.shapeArray[0]
		if self.transposeA {
			M = tensorA.shape.shapeArray[1]
		}
		var N = tensorB.shape.shapeArray[1]
		if self.transposeB {
			N = tensorB.shape.shapeArray[0]
		}
		var K = tensorA.shape.shapeArray[1]
		if self.transposeA {
			K = tensorA.shape.shapeArray[0]
		}
		return (M, N, K)
	}
	
	/// Use `BLAS` `cblas_sgemm` to do calculation
	internal func cpu() {
		let workGroup = DispatchGroup()
		let inputB = self.inputTensors!.last!
		let inputBAddress = inputB.contentsAddress
		
		for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
			let inputAAddress = input.contentsAddress
			let outCAddress = output.contentsAddress
			
			let (M, N, K) = self.MNKFetch(tensorA: input, tensorB: inputB)
			
			let lda = Int32(input.shape.shapeArray[1])
			let ldb = Int32(inputB.shape.shapeArray[1])
			let ldc = Int32(output.shape.shapeArray[1])
			
			cblas_sgemm(CblasRowMajor, cblasTrans(self.transposeA), cblasTrans(self.transposeB), Int32(M), Int32(N), Int32(K),
						1.0, inputAAddress, lda, inputBAddress, ldb, self.matrixBeta, outCAddress, ldc)
		}
		workGroup.wait()
	}
	
	
	/// This method choose proper kernel or MPS to do calculation
	internal func gpu() {
		// Use MPS if possible
		if !self.disabledMPS && MetalHardwareChecker.supportMPS() {
			if #available(OSX 10.13, iOS 10.0, *) {
				self.gpu_kernel_MPS()
				return
			}
		}
		
		// choose kernel
		if self.transposeA && !self.transposeB {
			self.gpu_kernel_submatrix()
		} else {
			self.gpu_kernel_single()
		}
	}

	/// Do calculation of inputA and transpoedB.
	/// `transposedInputB` is supposed already transposed
	///
	/// - Parameters:
	///   - inputABuffer: inputA
	///   - inputBBufferTransposed: transposedInputB
	///   - outputCBuffer: outputCBuffer
	///   - dimInfo: dimInfo
	///   - kernel: kernel
	internal func gpu_single(inputABuffer: MTLBufferResource, inputBBufferTransposed: MTLBufferResource, outputCBuffer: MTLBufferResource,
	                         dimInfo: inout MatrixDimInfo, kernel: MTLComputePipelineState) {
		let commandBuffer = SerranoEngine.configuredEngine.serranoCommandQueue?.makeCommandBuffer()
		guard commandBuffer != nil else {
			fatalError("[Serrano] Failed to make new command buffer.")
		}
		
		// Encoders.
		let encoder = commandBuffer!.makeComputeCommandEncoder()
		encoder.setComputePipelineState(kernel)
		encoder.setBuffer(inputABuffer.buffer, offset: inputABuffer.offset, at: 0)
		encoder.setBuffer(inputBBufferTransposed.buffer, offset: inputBBufferTransposed.offset, at: 1)
		encoder.setBuffer(outputCBuffer.buffer, offset: outputCBuffer.offset, at: 2)
		encoder.setBytes(&dimInfo, length: MemoryLayout<MatrixDimInfo>.stride, at: 3)
		
		/// Calculate grid
		let threadsPerThreadgroup = MTLSizeMake(kernel.threadExecutionWidth,
		                                        kernel.maxTotalThreadsPerThreadgroup / kernel.threadExecutionWidth,
		                                        1)
		let threadgroupsPerGrid = MTLSizeMake((Int(dimInfo.M) + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
		                                      (Int(dimInfo.N) + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
		                                      1)
		encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
		encoder.endEncoding()

		if !SerranoLogging.release {
			SerranoLogging.stdLogging(message: "Dispatch group configured with threadgroupsPerGrid: \(threadgroupsPerGrid), threadsPerThreadgroup: \(threadsPerThreadgroup) requested by operator \(self.operatorLabel)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)",  loggingLevel: .LowLevel)
		}
		
		
		// commit command buffer
		commandBuffer!.commit()
		commandBuffer!.waitUntilCompleted()
	}
	
	internal func gpu_kernel_single() {
		let (kernel, info) = SerranoEngine.configuredEngine.loadGPUKernel(kernelLabel: "MatrixMult_Single")
		guard kernel != nil else {
			fatalError("[Serrano] Failed to load kernel MatrixMult_Single. Info: \(info)")
		}
		
		// transpose B process
		var inputBTransposed = self.inputTensors!.last!
		if !self.transposeB {
			let transposeB = SerranoResourceManager.globalManager.allocateUnamangedTensor(inputBTransposed.shape.transposed())
			let transOp = TransposeOperator(inputTensors: [inputBTransposed], outputTensors: [transposeB])
			transOp.disableInputOutputCheck = true
			transOp.compute(.GPU)
			inputBTransposed = transposeB
		}
		let inputBBufferTransposed = inputBTransposed.gpuBufferResource()
		
		// do calcualtion
		let workGroup = DispatchGroup()
		if self.transposeA {
			for (inputA, output) in zip(self.inputTensors![0..<self.inputTensors!.count-1], self.outputTensors!) {
				workGroup.enter()
				DispatchQueue.global(qos: .userInitiated).async {
					let transA = SerranoResourceManager.globalManager.allocateUnamangedTensor(inputA.shape.transposed())
					let transOp = TransposeOperator(inputTensors: [inputA], outputTensors: [transA])
					transOp.disableInputOutputCheck = true
					transOp.compute(.GPU)
					
					let (M,N,K) = self.MNKFetch(tensorA: inputA, tensorB: self.inputTensors!.last!) // here use raw inputA, inputB to get M,N,K
					var info = MatrixDimInfo(M: MetalUInt(M), N: MetalUInt(N), K: MetalUInt(K), stride: MetalUShort(MemoryLayout<Float>.stride))
					self.gpu_single(inputABuffer: transA.gpuBufferResource(), inputBBufferTransposed: inputBBufferTransposed,
					                outputCBuffer: output.gpuBufferResource(), dimInfo: &info, kernel: kernel!)
					workGroup.leave()
				}
			}
		} else {
			for (inputA, output) in zip(self.inputTensors![0..<self.inputTensors!.count-1], self.outputTensors!) {
				workGroup.enter()
				DispatchQueue.global(qos: .userInitiated).async {
					let (M,N,K) = self.MNKFetch(tensorA: inputA, tensorB: self.inputTensors!.last!) // here use raw inputB to get M,N,K
					var info = MatrixDimInfo(M: MetalUInt(M), N: MetalUInt(N), K: MetalUInt(K), stride: MetalUShort(MemoryLayout<Float>.stride))
					self.gpu_single(inputABuffer: inputA.gpuBufferResource(), inputBBufferTransposed: inputBBufferTransposed,
					                outputCBuffer: output.gpuBufferResource(), dimInfo: &info, kernel: kernel!)
					workGroup.leave()
				}
			}
		}
		
		workGroup.wait()
	}
	
	
	/// Do matrix multiplication with submatrix kernel.
	///
	/// - Parameters:
	///   - inputATransposeBuffer: transposed A buffer
	///   - inputBBuffer: inputBBuffer
	///   - outputCBuffer: outputCBuffer
	///   - dimInfo: dimInfo
	///   - kernel: kernel
	internal func gpu_submatrix(inputATransposeBuffer: MTLBufferResource, inputBBuffer: MTLBufferResource,
	                            outputCBuffer: MTLBufferResource,
	                            dimInfo: inout MatrixDimInfo, kernel: MTLComputePipelineState) {
		let commandBuffer = SerranoEngine.configuredEngine.serranoCommandQueue?.makeCommandBuffer()
		guard commandBuffer != nil else {
			fatalError("[Serrano] Failed to make new command buffer.")
		}
		
		// Encoders.
		let encoder = commandBuffer!.makeComputeCommandEncoder()
		encoder.setComputePipelineState(kernel)
		encoder.setBuffer(inputATransposeBuffer.buffer, offset: inputATransposeBuffer.offset, at: 0)
		encoder.setBuffer(inputBBuffer.buffer, offset: inputBBuffer.offset, at: 1)
		encoder.setBuffer(outputCBuffer.buffer, offset: outputCBuffer.offset, at: 2)
		encoder.setBytes(&dimInfo, length: MemoryLayout<MatrixDimInfo>.stride, at: 3)
		
		/// Calculate grid
		let threadsPerThreadgroup = MTLSizeMake(MatrixMultOperator.METAL_SUBMATRIX_SIZE,
		                                        MatrixMultOperator.METAL_SUBMATRIX_SIZE,
		                                        1)
		let threadgroupsPerGrid = MTLSizeMake((Int(dimInfo.M) + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
		                                      (Int(dimInfo.N) + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
		                                      1)
		encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
		encoder.endEncoding()
		
		if !SerranoLogging.release {
			SerranoLogging.stdLogging(message: "Dispatch group configured with threadgroupsPerGrid: \(threadgroupsPerGrid), threadsPerThreadgroup: \(threadsPerThreadgroup) requested by operator \(self.operatorLabel)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)",  loggingLevel: .LowLevel)
		}
		
		// commit command buffer
		commandBuffer!.commit()
		commandBuffer!.waitUntilCompleted()
	}
	
	/// - Note:  There's no any transposing processing in this function cause
	///          in function `gpu()` it only dispatches suitable inputs to this function.
	internal func gpu_kernel_submatrix() {
		let (kernel, info) = SerranoEngine.configuredEngine.loadGPUKernel(kernelLabel: "MatrixMult_submatrix")
		guard kernel != nil else {
			fatalError("[Serrano] Failed to load kernel MatrixMult_Single. Info: \(info)")
		}
		
		let inputBBuffer = self.inputTensors!.last!.gpuBufferResource()
		
		// do calcualtion
		let workGroup = DispatchGroup()

		for (inputA, output) in zip(self.inputTensors![0..<self.inputTensors!.count-1], self.outputTensors!) {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				let (M,N,K) = self.MNKFetch(tensorA: inputA, tensorB: self.inputTensors!.last!) // here use raw inputA and inputB to get M,N,K
				var info = MatrixDimInfo(M: MetalUInt(M), N: MetalUInt(N), K: MetalUInt(K), stride: MetalUShort(MemoryLayout<Float>.stride))
				self.gpu_submatrix(inputATransposeBuffer: inputA.gpuBufferResource(), inputBBuffer: inputBBuffer,
								   outputCBuffer: output.gpuBufferResource(),
				                   dimInfo: &info, kernel: kernel!)
				workGroup.leave()
			}
		}
		workGroup.wait()
	}
	
	@available(OSX 10.13, iOS 10.0, *)
	internal func gpu_kernel_MPS() {
		#if  !((arch(i386)  || arch(x86_64)) && os(iOS))
		// input B
		let inputB = self.inputTensors!.last!
		let inputBBuffer = SerranoResourceManager.globalManager.allocateUnmanagedMTLBuffe(inputB)
		let inputBMatrixDescript =  MPSMatrixDescriptor(dimensions: inputB.shape.shapeArray[0], columns: inputB.shape.shapeArray[1],
												rowBytes:  inputB.shape.shapeArray[1] * MemoryLayout<Float>.stride,
												dataType: .float32)
		let inputBMatrix = MPSMatrix(buffer: inputBBuffer, descriptor: inputBMatrixDescript)
		
		for (inputA, output) in zip(self.inputTensors![0..<self.inputTensors!.count-1], self.outputTensors!) {
			// INPUT A
			let inputABuffer = SerranoResourceManager.globalManager.allocateUnmanagedMTLBuffe(inputA)
			let inputAMatrixDescript = MPSMatrixDescriptor(dimensions: inputA.shape.shapeArray[0], columns: inputA.shape.shapeArray[1],
															rowBytes:  inputA.shape.shapeArray[1] * MemoryLayout<Float>.stride,
															dataType: .float32)
			let inputAMatrix = MPSMatrix(buffer: inputABuffer, descriptor: inputAMatrixDescript)
			
			// output
			let outputBuffer = SerranoResourceManager.globalManager.allocateUnmanagedMTLBuffe(output)
			let outputMatrixDescript = MPSMatrixDescriptor(dimensions: output.shape.shapeArray[0], columns: output.shape.shapeArray[1],
														   rowBytes:  output.shape.shapeArray[1] * MemoryLayout<Float>.stride,
														   dataType: .float32)
			let outputAMatrix = MPSMatrix(buffer: outputBuffer, descriptor: outputMatrixDescript)
			
			// do calculation
			let (M, N, K) = MNKFetch(tensorA: inputA, tensorB: inputB)
			let kernel = MPSMatrixMultiplication(device: SerranoEngine.configuredEngine.GPUDevice!,
												 transposeLeft: self.transposeA, transposeRight: self.transposeB,
												 resultRows: M, resultColumns: N,
												 interiorColumns: K, alpha: 1, beta: Double(self.matrixBeta))
			
			let commandBuffer = SerranoEngine.configuredEngine.serranoCommandQueue!.makeCommandBuffer()
			kernel.encode(commandBuffer: commandBuffer, leftMatrix: inputAMatrix, rightMatrix: inputBMatrix, resultMatrix: outputAMatrix)
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		}
		#endif
	}
	
}
