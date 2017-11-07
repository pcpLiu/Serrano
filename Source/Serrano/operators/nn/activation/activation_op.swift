//
//  activation_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Dispatch
import Accelerate

/**
The abstract class for all activation operator.
An activation operator is actually a special kind of `UnaryOperator` doing a little bit more complex element-wise computation.
*/
public class ActivationOperator: UnaryOperator {
	
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, oututTensor: Tensor) -> Void in
			fatalError("Not implemented")
		}
		let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			fatalError("Not implemented")
		}
		let defaultLabel = "NEED OVERRIDE"
		let kernelLabel = "NEED OVERRIDE"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block,  gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
	}
	
	/// Construct an `ActivationOperator` from a `String` name.
	public convenience init(activationOpName: String) {
		fatalError()
		//TODO: Implementation
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
Rectified Linear Unit, 
`y=max(x,alpha=0.0)`
*/
public class ReLUOperator: ActivationOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// hyperparameter, default is `0.0`
	public var alpha: Float = 0.0
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, oututTensor: Tensor) -> Void in
			fatalError("Not implemented")
		}
		let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "ReLUOperator"
		let kernelLabel = "ReLU"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
	}
	
	/// Initial by setting `alpha` value
	///
	/// - Parameters:
	///   - computationDelegate: computationDelegate description
	///   - alpha: alpha description
	public convenience init(computationDelegate: OperatorCalculationDelegate? = nil, alpha: Float) {
		self.init(computationDelegate: computationDelegate)
		self.alpha = alpha
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Attribute `alpha` as a `ScalarSymbol`.
	///
	/// - Returns:  Array  of GraphSymbol
	public override func paramSymbols() -> [GraphSymbol] {
		let alpha = SerranoScalarSymbol("alpha", dataType: .float, dataSource: .Default)
		alpha.bindedData = Float(0.0)
		return [alpha]
	}
	
	/// Override CPU
	internal override func cpu() {
		let workGroup = DispatchGroup()
		for tensorIndex in 0..<self.inputTensors!.count {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				let inputPointer = self.inputTensors![tensorIndex].contentsAddress
				let outputPointer = self.outputTensors![tensorIndex].contentsAddress
				let count = vDSP_Length(self.outputTensors![tensorIndex].count)
				let alphaTensor = Tensor(repeatingValue: self.alpha, tensorShape: self.outputTensors![tensorIndex].shape)
				let alphaTensorPointer = alphaTensor.contentsAddress
				// vDSP_vmax
				vDSP_vmax(inputPointer, 1, alphaTensorPointer, 1, outputPointer, 1, count)
				workGroup.leave()
			}
		}
		workGroup.wait()
	}
	
	/// Override GPU
	internal override func gpu() {
		let engine = SerranoEngine.configuredEngine
		
		//// kernel
		let (kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
		guard kernel != nil else {
			fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
		}
		
		
		//// command buffer
		let commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
		guard commandBuffer != nil else {
			fatalError("[Serrano] Failed to make new command buffer.")
		}
		
		var alphaM = MetalFloat(self.alpha)
		for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
			let inputBufferResource = input.gpuBufferResource()
			let outputBufferResource = output.gpuBufferResource()
			var count = MetalUInt(input.count)
			// encoder
			let encoder = commandBuffer!.makeComputeCommandEncoder()
			encoder.setComputePipelineState(kernel!)
			encoder.setBuffer(inputBufferResource.buffer, offset: inputBufferResource.offset, at: 0)
			encoder.setBuffer(outputBufferResource.buffer, offset: outputBufferResource.offset, at: 1)
			encoder.setBytes(&count, length: MemoryLayout<MetalUInt>.stride, at: 2)
			encoder.setBytes(&alphaM, length: MemoryLayout<MetalFloat>.stride, at: 3)
			// dispatch
			let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
			                                        1,
			                                        1)
			let threadgroupsPerGrid = MTLSizeMake((input.count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
			                                      1,
			                                      1)
			encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
			encoder.endEncoding()
		}
		
		// commit command buffer
		commandBuffer!.commit()
		commandBuffer!.waitUntilCompleted()
	}
	
}

/**
Sigmoid.`y = 1 / (1 + exp(-x))`
*/
public class SigmoidOperator: ActivationOperator {
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
			let inputPointer = inputTensor.contentsAddress
			let outputPointer = outputTensor.contentsAddress
			let oneValueTensor = Tensor(repeatingValue: 1.0, tensorShape: outputTensor.shape)
			let oneValuePointer = oneValueTensor.contentsAddress
			var count32 = Int32(outputTensor.count)
			let count = vDSP_Length(outputTensor.count)
			//copy value from input to output
			cblas_scopy(count32, inputPointer, 1, outputPointer, 1)
			// negative
			vDSP_vneg(outputPointer, 1, outputPointer, 1, count)
			// exp
			vvexpf(outputPointer, outputPointer, &count32)
			// plus
			cblas_saxpy(count32, 1.0, oneValuePointer, 1, outputPointer, 1)
			// reciprocal
			vvrecf(outputPointer, outputPointer, &count32)
		}
		let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "SigmoidOperator"
		let kernelLabel = "Sigmoid"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
	}
}

/**
Softplus.`y = log(exp(x) + 1)`
*/
public class SoftplusOperator: ActivationOperator {
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
			let inAddress = inputTensor.contentsAddress
			let outAddress = outputTensor.contentsAddress
			let count = vDSP_Length(inputTensor.count)
			var count32 = Int32(inputTensor.count)
			var one: Float = 1.0
			// exp
			vvexpf(outAddress, inAddress, &count32)
			// plus 1
			vDSP_vsadd(outAddress, 1, &one, outAddress, 1, count)
			// log
			vvlogf(outAddress, outAddress, &count32) 
		}
		let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "SoftplusOperator"
		let kernelLabel = "Softplus"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
	}
}


/**
Softsign.`y = x / (1 + abs(x))`
*/
public class SoftsignOperator: ActivationOperator {
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
			let inputAddress = inputTensor.contentsAddress
			let outAddress = outputTensor.contentsAddress
			let count = vDSP_Length(outputTensor.count)
			var one:Float = 1.0
			// abs
			vDSP_vabs(inputAddress, 1, outAddress, 1, count)
			// add
			vDSP_vsadd(outAddress, 1, &one, outAddress, 1, count)
			// div
			vDSP_vdiv(outAddress, 1, inputAddress, 1, outAddress, 1, count)
		}
		let gradBlock = { (inputs: [Tensor],  mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "SoftsignOperator"
		let kernelLabel = "Softsign"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
	}
}


/**
Do nothing. Output identify tensor.
`y = x`
*/
public class LinearOperator: ActivationOperator {
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
			let inputPointer = inputTensor.contentsAddress
			let outputPointer = outputTensor.contentsAddress
			let count32 = Int32(outputTensor.count)
			cblas_scopy(count32, inputPointer, 1, outputPointer, 1)
		}
		let gradBlock = { (inputs: [Tensor],  mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "LinearOperator"
		let kernelLabel = "Linear"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
	}
}

/**
Exponential Linear Units.

`y = x if x > 0, else y = alpha*(exp(x) - 1)`

## Reference
[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
*/
public class ELUOperator: ActivationOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes

	/// hyperparameter, default is `1.0`
	public var alpha:Float = 1.0
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializer
	
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
			print("NOT USE") // will override cpu(). this block will not be used
		}
		let gradBlock = { (inputs: [Tensor],  mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "ELUOperator"
		let kernelLabel = "ELU"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
	}
	
	
	/// Initial by setting `alpha` value
	///
	/// - Parameters:
	///   - computationDelegate: computationDelegate description
	///   - alpha: alpha description
	public convenience init(computationDelegate: OperatorCalculationDelegate? = nil, alpha: Float) {
		self.init(computationDelegate: computationDelegate)
		self.alpha = alpha
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Attribute `alpha` as a `ScalarSymbol`.
	///
	/// - Returns:  Array  of GraphSymbol
	public override func paramSymbols() -> [GraphSymbol] {
		let alpha = SerranoScalarSymbol("alpha", dataType: .float, dataSource: .Default)
		alpha.bindedData = Float(1.0)
		return [alpha]
	}
	
	/// Override CPU
	internal override func cpu() {
		let workGroup = DispatchGroup()
		for tensorIndex in 0..<self.inputTensors!.count {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				let inputReader = self.inputTensors![tensorIndex].floatValueReader
				let outputReadre = self.outputTensors![tensorIndex].floatValueReader
				for i in 0..<self.outputTensors![tensorIndex].count {
					if inputReader[i] > 0.0 { outputReadre[i] = inputReader[i] }
					else { outputReadre[i] =  self.alpha * (exp(inputReader[i]) - 1.0) }
				}
				workGroup.leave()
			}
		}
		
		workGroup.wait()
	}
	
	/// Override GPU calculation cause there's a hyperparameter to pass-in
	internal override func gpu() {
		let engine = SerranoEngine.configuredEngine
		
		//// kernel
		let (kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
		guard kernel != nil else {
			fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
		}
		
		
		//// command buffer
		let commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
		guard commandBuffer != nil else {
			fatalError("[Serrano] Failed to make new command buffer.")
		}
		
		var alphaM = MetalFloat(self.alpha)
		for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
			let inputBufferResource = input.gpuBufferResource()
			let outputBufferResource = output.gpuBufferResource()
			var count = MetalUInt(input.count)
			// encoder
			let encoder = commandBuffer!.makeComputeCommandEncoder()
			encoder.setComputePipelineState(kernel!)
			encoder.setBuffer(inputBufferResource.buffer, offset: inputBufferResource.offset, at: 0)
			encoder.setBuffer(outputBufferResource.buffer, offset: outputBufferResource.offset, at: 1)
			encoder.setBytes(&count, length: MemoryLayout<MetalUInt>.stride, at: 2)
			encoder.setBytes(&alphaM, length: MemoryLayout<MetalFloat>.stride, at: 3)
			// dispatch
			let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
													1,
													1)
			let threadgroupsPerGrid = MTLSizeMake((input.count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
												  1,
												  1)
			encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
			encoder.endEncoding()
		}
		
		// commit command buffer
		commandBuffer!.commit()
		commandBuffer!.waitUntilCompleted()
	}
}

/**
Scaled Exponential Linear Unit.

`y = scale * elu(x)`

## alpha
default value is `1.673263`

## scale
default value is `1.050701`

## Reference
[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
*/
public class SELUOperator: ActivationOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// alpha for ELU operation. Default is `1.673263`
	public var alpha:Float = 1.673263

	/// scalue factor. Defualt is `1.050701`
	public var scale:Float = 1.050701
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
			print("NOT USE")
		}
		let gradBlock = { (inputs: [Tensor],  mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "SELUOperator"
		let kernelLabel = "SELU"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
	}
	
	/// Initial by setting `alpha` value and `scale` values.
	///
	/// - Parameters:
	///   - computationDelegate: computationDelegate description
	///   - alpha: alpha description
	public convenience init(computationDelegate: OperatorCalculationDelegate? = nil, alpha: Float, scale:Float) {
		self.init(computationDelegate: computationDelegate)
		self.alpha = alpha
		self.scale = scale
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Attribute `alpha` as a `ScalarSymbol`.
	/// Attribute `scale` as a `ScalarSymbol`
	///
	/// - Returns:  Array  of GraphSymbol
	public override func paramSymbols() -> [GraphSymbol] {
		let alpha = SerranoScalarSymbol("alpha", dataType: .float, dataSource: .Default)
		alpha.bindedData = Float(1.673263)
		
		let scale = SerranoScalarSymbol("scale", dataType: .float, dataSource: .Default)
		alpha.bindedData = Float(1.050701)
		
		return [alpha, scale]
	}
	
	/// Override CPU
	internal override func cpu() {
		let workGroup = DispatchGroup()
		for tensorIndex in 0..<self.inputTensors!.count {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				let inputReader = self.inputTensors![tensorIndex].floatValueReader
				let outputReadre = self.outputTensors![tensorIndex].floatValueReader
				for i in 0..<self.outputTensors![tensorIndex].count {
					if inputReader[i] > 0.0 { outputReadre[i] = inputReader[i] }
					else { outputReadre[i] =  self.alpha * (exp(inputReader[i]) - 1.0) }
					outputReadre[i] *= self.scale
				}
				workGroup.leave()
			}
		}
		
		workGroup.wait()
	}
	
	/// Override GPU calculation cause there's a hyperparameter to pass-in
	internal override func gpu() {
		let engine = SerranoEngine.configuredEngine
		
		//// kernel
		let (kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
		guard kernel != nil else {
			fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
		}
		
		
		//// command buffer
		let commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
		guard commandBuffer != nil else {
			fatalError("[Serrano] Failed to make new command buffer.")
		}
		
		var alphaM = MetalFloat(self.alpha)
		var scaleM = MetalFloat(self.scale)
		for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
			let inputBufferResource = input.gpuBufferResource()
			let outputBufferResource = output.gpuBufferResource()
			var count = MetalUInt(input.count)
			// encoder
			let encoder = commandBuffer!.makeComputeCommandEncoder()
			encoder.setComputePipelineState(kernel!)
			encoder.setBuffer(inputBufferResource.buffer, offset: inputBufferResource.offset, at: 0)
			encoder.setBuffer(outputBufferResource.buffer, offset: outputBufferResource.offset, at: 1)
			encoder.setBytes(&count, length: MemoryLayout<MetalUInt>.stride, at: 2)
			encoder.setBytes(&alphaM, length: MemoryLayout<MetalFloat>.stride, at: 3)
			encoder.setBytes(&scaleM, length: MemoryLayout<MetalFloat>.stride, at: 4)
			// dispatch
			let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
													1,
													1)
			let threadgroupsPerGrid = MTLSizeMake((input.count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
												  1,
												  1)
			encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
			encoder.endEncoding()
		}
		
		// commit command buffer
		commandBuffer!.commit()
		commandBuffer!.waitUntilCompleted()
	}
}

/**
Softmax activations.

`y = exp(x) / reduce_sum(exp(x), dim)`

Default `dim` is the last dimension of input tensors. 

- Note: The operator assumens all input tensors have __same rank__.
If not, it could not do the calculation.
*/
public class SoftmaxOperator: ActivationOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Reduce summing dimension. 
	/// The value should be `>=0`. 
	/// Any negative value will be automatically making this attribute value to `-1`.
	/// - Note: `-1` is a special value indicating last dim.
	public var dim: Int = -1 {
		didSet {
			if self.dim < 0 { self.dim = -1 }
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
			print("NOT USE")
		}
		let gradBlock = { (inputs: [Tensor],  mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "SoftmaxOperator"
		let kernelLabel = ""
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
	}
	
	/// Initial by setting `dim` value.
	///
	/// - Parameters:
	///   - computationDelegate: computationDelegate description
	///   - dim: reduce dim
	public convenience init(computationDelegate: OperatorCalculationDelegate? = nil, dim: Int) {
		self.init(computationDelegate: computationDelegate)
		self.dim = dim
		if self.dim < 0 { self.dim = -1 }
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Override this function from `UnaryOperaotr` to do additional checking validation bewtween `dim` and `shapes`.
	///
	/// - Parameter shapes: shapes description
	/// - Returns: return value description
	public override func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		// super check
		let outShapes = super.outputShape(shapeArray: shapes)
		
		// check dim and shapes
		for shape in shapes {
			guard self.dim < shape.shapeArray.count else {
				SerranoLogging.errorLogging(message: "Invalid shape [\(shape.description)] for dim \(self.dim).",
				                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
		}
		return outShapes
	}
	
	/// Override computaion methods. 
	/// This computation just call other operators.
	///
	/// - Parameter computationMode: computationMode
	public override func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
		// check
		let (pass, msg) = self.inputOutputTensorsCheck()
		guard pass else {
			SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) calculation aborted cause invalid input tensors or output tensors: \(msg)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		
		self.computationDelegate?.operatorWillBeginComputation(self)
		
		let computeGroup = DispatchGroup()
		for tensorIndex in 0..<self.inputTensors!.count {
			computeGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				let inputTensor = self.inputTensors![tensorIndex]
				let outputTensor = self.outputTensors![tensorIndex]
				// last dim convert
				var reduceDim = self.dim
				if reduceDim == -1 { reduceDim = inputTensor.rank - 1 }
				// exp
				let expOp = ExpOperator(inputTensors: [inputTensor], outputTensors: [outputTensor])
				expOp.compute(computationMode)
				// reduce
				let reduceSumOp = ReduceSumOperator(axis: [reduceDim], keepDim: true)
				let intermediateShape = reduceSumOp.outputShape(shapeArray: [outputTensor.shape])!.first!
				let intermediateTensor = SerranoResourceManager.globalManager.allocateTensor(intermediateShape)
				reduceSumOp.inputTensors = [outputTensor]
				reduceSumOp.outputTensors = [intermediateTensor]
				reduceSumOp.compute(computationMode)
				// brodcast in-place div
				let _ = outputTensor .&/ intermediateTensor
				// return tesnor
				SerranoResourceManager.globalManager.returnTensor(intermediateTensor)
				computeGroup.leave()
			}
		}
		computeGroup.wait()
		
		self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
	}
	
	/// Attribute `dim` as a `ScalarSymbol`.
	///
	/// - Returns: Array of GraphSymbol
	public override func paramSymbols() -> [GraphSymbol] {
		let dim = SerranoScalarSymbol("dim", dataType: .int, dataSource: .Default)
		dim.bindedData = -1
		
		return [dim]
	}
}

/**
LeakyReLU activation operator.
```
y = alpha * x (x <   0)
y = x         (x >=  0)
```

## scale factor alpha
Default value is `0.3`

## Reference 
[Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/ReLU_hybrid_icml2013_final.pdf)
*/
public class LeakyReLUOperator: ELUOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializer
	
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
			print("NOT USE") // will override cpu(). this block will not be used
		}
		let gradBlock = { (inputs: [Tensor],  mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "LeakyReLUOperator"
		let kernelLabel = "LeakyReLU"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
		self.alpha = 0.3
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Override CPU
	internal override func cpu() {
		let workGroup = DispatchGroup()
		for tensorIndex in 0..<self.inputTensors!.count {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				let inputReader = self.inputTensors![tensorIndex].floatValueReader
				let outputReadre = self.outputTensors![tensorIndex].floatValueReader
				for i in 0..<self.outputTensors![tensorIndex].count {
					if inputReader[i] >= 0.0 { outputReadre[i] = inputReader[i] }
					else { outputReadre[i] = self.alpha * inputReader[i] }
				}
				workGroup.leave()
			}
		}
		workGroup.wait()
	}

}


/**
Thresholded Rectified Linear Unit.
`f(x) = x if x > alpha, else f(x) = 0`

## Threshold `alpha`
Default value is `1.0`

## Reference
[Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](https://arxiv.org/abs/1402.3337)
*/
public class ThresholdedReLUOperator: ELUOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializer
	
	/// Convenience initializer
	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
			print("NOT USE") // will override cpu(). this block will not be used
		}
		let gradBlock = { (inputs: [Tensor],  mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
			//TODO: implemented
			fatalError("Not implemented")
		}
		let defaultLabel = "ThresholdedReLUOperator"
		let kernelLabel = "ThresholdedReLU"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil)
		self.alpha = 1.0
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Override CPU
	internal override func cpu() {
		let workGroup = DispatchGroup()
		for tensorIndex in 0..<self.inputTensors!.count {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				let inputReader = self.inputTensors![tensorIndex].floatValueReader
				let outputReadre = self.outputTensors![tensorIndex].floatValueReader
				for i in 0..<self.outputTensors![tensorIndex].count {
					if inputReader[i] > self.alpha { outputReadre[i] = inputReader[i] }
					else                           { outputReadre[i] = 0.0 }
				}
				workGroup.leave()
			}
		}
		workGroup.wait()
	}
}

/**
Parametric Rectified Linear Unit.

```
y_i = alpha_i * x_i (x_i< 0)
y_i = x_i           (x_i >= 0)
```

## Reference
[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
*/
//public class PReLUOperator: ActivationOperator {
	//TODO: RE-IMPLEMENTATION

//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	// MARK: - Attributes
//	
//
//	/// A tensor
//	/// - Note: If this attribute is `nil` when doing calculation,
//	///         operator will use `defaultAlpha` as the default values.
//	public var alpha: Tensor
//	
//	/// Default alpha values when `alpha` is `nil`
//	public var defaultAlphaValue: Float = 0.3
//	
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	// MARK: - Initializers
//	
//	/// Convenience initializer
//	/// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
//	///
//	/// - Parameter computationDelegate: computationDelegate
//	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
//		let block = { (inputTensor: Tensor, outputTensor: Tensor) -> Void in
//			print("NOT USE") // will override cpu(). this block will not be used
//		}
//		let gradBlock = { (inputs: [Tensor],  mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
//			//TODO: implemented
//			fatalError("Not implemented")
//		}
//		let defaultLabel = "PReLUOperator"
//		let kernelLabel = "PReLU"
//		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
//		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
//		          inputTensors: nil, outputTensors: nil)
//		self.alpha = nil
//	}
//	
//	/// Initial by setting `alpha` value
//	///
//	/// - Parameters:
//	///   - computationDelegate: computationDelegate
//	///   - alpha: alpha tensor object
//	public convenience init(computationDelegate: OperatorCalculationDelegate? = nil, alpha: [Tensor]) {
//		self.init(computationDelegate: computationDelegate)
//		self.alpha = alpha
//	}
//	
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	// MARK: - Methods
//	
//	/// Additional checking needs to verify the diemsions mathcing of `alpha` and `inputTensors`.
//	///
//	/// - Note: If `alpha` is `nil`, this function will generate default `alpha` following `defaultAlphaValue`.
//	///
//	/// - Returns: return pass or not and error message.
//	public override func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
//		// super check
//		let (pass, msg) = super.inputOutputTensorsCheck()
//		guard pass else {
//			// false return
//			return (pass, msg)
//		}
//		
//		// check alpha
//		if self.alpha == nil {
//			// generate default alpha
//			SerranoLogging.stdLogging(message: "Operator \(self.operatorLabel) has nil alpha. Will generate default alpha tensors with value \(self.defaultAlphaValue).",
//			                          file: "\(#file)", function: "\(#function)", line: "\(#line)",
//									  loggingLevel: SerranoLoggingType.LowLevel)
//			self.alpha = [Tensor]()
//			for tensor in self.inputTensors! {
//				self.alpha!.append(Tensor(repeatingValue: self.defaultAlphaValue, tensorShape: tensor.shape))
//			}
//		} else {
//			// check matching
//			guard self.alpha!.count == self.inputTensors!.count else {
//				return (false, "Attribute alpha does not have same number of tensor objects as input tensors does.")
//			}
//			for (alphaTensor, inputTensor) in zip(self.alpha!, self.inputTensors!) {
//				guard alphaTensor.shape .== inputTensor.shape else {
//					return (false, "Alpha has shape \(alphaTensor.shape.description) while input tensor has shape \(inputTensor.shape.description).")
//				}
//			}
//		}
//		
//		return (true, "")
//	}
//	
//	/// Attribute `alpha` as an array of `TensorSymbol` if not `nil`.
//	/// Attribute `defaultAlphaValue` as a `ScalarSymbol`.
//	///
//	/// - Returns:  Array  of GraphSymbol
//	public override func paramSymbols() -> [GraphSymbol] {
//		let defaultAlphaValue = SerranoScalarSymbol("defaultAlphaValue", dataType: .float, dataSource: .Default)
//		defaultAlphaValue.bindedData = Float(0.3)
//		
//		var symbols = [defaultAlphaValue]
//		
//		if self.alpha != nil {
//			for tensor in self.alpha! {
//				symbols.append(SerranoTensorSymbol( )
//			}
//		}
//		
//		return symbols
//	}
//	
//	/// Override
//	internal override func cpu() {
//		let workGroup = DispatchGroup()
//		for i in 0..<self.outputTensors!.count {
//			workGroup.enter()
//			DispatchQueue.global(qos: .userInitiated).async {
//				let inputReader = self.inputTensors![i].floatValueReader
//				let outputReader = self.outputTensors![i].floatValueReader
//				let alphaReader = self.alpha![i].floatValueReader
//				for eleIndex in 0..<self.outputTensors![i].count {
//					outputReader[eleIndex] = inputReader[eleIndex]
//					if inputReader[eleIndex] < 0.0 { outputReader[eleIndex] = inputReader[eleIndex] * alphaReader[eleIndex]}
//				}
//				workGroup.leave()
//			}
//		}
//		workGroup.wait()
//	}
//	
//	/// Override
//	internal override func gpu() {
//		// prepare resource
//		let resourcePrepareGroup = DispatchGroup()
//		let engine = SerranoEngine.configuredEngine
//		var kernel: MTLComputePipelineState?
//		var commandBuffer: MTLCommandBuffer?
//		var inputBuffers: [MTLBufferResource] = [MTLBufferResource]()
//		var resultBuffers: [MTLBufferResource] = [MTLBufferResource]()
//		var alphaBuffers: [MTLBufferResource] = [MTLBufferResource]()
//		
//		//// kernel
//		resourcePrepareGroup.enter()
//		DispatchQueue.global(qos: .userInitiated).async {
//			var info = ""
//			(kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
//			guard kernel != nil else {
//				fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
//			}
//			resourcePrepareGroup.leave()
//		}
//		
//		//// command buffer
//		resourcePrepareGroup.enter()
//		DispatchQueue.global(qos: .userInitiated).async {
//			commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
//			guard commandBuffer != nil else {
//				fatalError("[Serrano] Failed to make new command buffer.")
//			}
//			resourcePrepareGroup.leave()
//		}
//		
//		/// input buffers
//		resourcePrepareGroup.enter()
//		DispatchQueue.global(qos: .userInitiated).async {
//			inputBuffers = SerranoResourceManager.globalManager.allocateMTLBufferResources(self.inputTensors!)
//			resourcePrepareGroup.leave()
//		}
//		
//		/// output buffers
//		resourcePrepareGroup.enter()
//		DispatchQueue.global(qos: .userInitiated).async {
//			resultBuffers = SerranoResourceManager.globalManager.allocateMTLBufferResources(self.outputTensors!)
//			resourcePrepareGroup.leave()
//		}
//		
//		/// alpha buffers
//		resourcePrepareGroup.enter()
//		DispatchQueue.global(qos: .userInitiated).async {
//			alphaBuffers = SerranoResourceManager.globalManager.allocateMTLBufferResources(self.alpha!)
//			resourcePrepareGroup.leave()
//		}
//		
//		resourcePrepareGroup.wait()
//		
//		for bufferIndex in 0..<inputBuffers.count {
//			// dimensionBuffer
//			var count = UInt32(self.inputTensors![bufferIndex].count)
//			let countBuffer = engine.GPUDevice?.makeBuffer(bytes: &count,
//			                                               length: MemoryLayout<UInt32>.size)
//			guard countBuffer != nil else {
//				fatalError("[Serrano] Failed to careate MTLBuffer.")
//			}
//			SerranoLogging.stdLogging(message: "Allocated a Metal buffer [\(countBuffer!.length) bytes] requested for count info \(count) by operator \(self.operatorLabel)", file: "\(#file)", function: "\(#function)", line: "\(#line)",  loggingLevel: .LowLevel)
//		
//			// encoder
//			let encoder = commandBuffer!.makeComputeCommandEncoder()
//			encoder.setComputePipelineState(kernel!)
//			encoder.setBuffer(inputBuffers[bufferIndex].buffer, offset: inputBuffers[bufferIndex].offset, at: 0)
//			encoder.setBuffer(resultBuffers[bufferIndex].buffer, offset: resultBuffers[bufferIndex].offset, at: 1)
//			encoder.setBuffer(countBuffer, offset: 0, at: 2)
//			encoder.setBuffer(alphaBuffers[bufferIndex].buffer, offset: alphaBuffers[bufferIndex].offset, at: 3)
//			
//			// dispatch
//			let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
//			                                        1,
//			                                        1)
//			let threadgroupsPerGrid = MTLSizeMake((self.inputTensors![bufferIndex].count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
//			                                      1,
//			                                      1)
//			encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
//			SerranoLogging.stdLogging(message: "Dispatch group configured with threadgroupsPerGrid: \(threadgroupsPerGrid), threadsPerThreadgroup: \(threadsPerThreadgroup) requested by operator \(self.operatorLabel)", file: "\(#file)", function: "\(#function)", line: "\(#line)",  loggingLevel: .LowLevel)
//			encoder.endEncoding()
//		}
//		
//		// commit command buffer
//		commandBuffer!.commit()
//		commandBuffer!.waitUntilCompleted()
//	}
//}

