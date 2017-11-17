//
//  binary_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/6/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Dispatch
import Metal
import Accelerate

/**
 Abstract class define the standard binary operator working flow.
 This class should not be used directly.
 Any class inheritance this class is doing computation on exactly __two__ input tensors in element-wise way and return __one__ result tensor, i.e.
 `x + y ->>> z`
 This `BinaryOperator` does __not__ support broadcasting
 */
public class BinaryOperator: ComputableOperator {

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - attributes
	
    /// Operator label. Conforms to `ComputableOperator`
    public var operatorLabel: String = ""
    
    /// This operator does not operator on GPU. Conforms to `ComputableOperator`
    public var metalKernelFuncLabel:String
    
    /// Conforms to `ComputableOperator`
    public var computationDelegate: OperatorCalculationDelegate?
    
    /// Conforms to `ComputableOperator`
    public var inputTensors: [Tensor]?
    
    /// Conforms to `ComputableOperator`
    public var outputTensors: [Tensor]?
    
    /// The element compuation block in CPU mode.
    /// In most cases, subclass should just override this part in `init` method instead overriding the whole `cpu(inputTensors:[Tensor], resultTensor: Tensor)` method.
	/// The fisr tensor is input tensor A;
	/// the seconds tensor is input tensor B;
	/// the third tensor is output tensor C.
    /// This block should do some computation and assign value back to result tensor's reader
    public var cpuElementComputationBlock: (Tensor, Tensor, Tensor) -> Void
	
	
	/// The grad compuation block.
	/// parameter: inputA, inputB, mode
	/// returns: An array of tensor. Should just have 2 object corresponding to two inputs
	public var gradComputationBlock: (Tensor, Tensor, OperatorComputationMode) -> [Tensor]
	
	/// If `true`, operator will not check the `upGrads`'s shape.
	/// This is used inside framework to speed up in situation we know it will not be wrong.
	/// Cases like auto generated differentiation graph.
	public var disableUpGradShapeCheck: Bool = false
	
	/// If `true`, operator will not call `inputOutputTensorsCheck()` before doing calculation.
	/// This is used inside framework to speed up in situation we know it will not be wrong.
	public var disableInputOutputCheck: Bool = false
	
	/// Indicate if this operator would do paramter update.
	///
	/// - Note: All `UnaryOperators` are not trainable.
	public var trainable: Bool = false

	/// The mapping type of this operator.
	/// `Constant` for this operator.
	public var mapType: OperatorMappingType {
		get {
			return OperatorMappingType.Constant
		}
	}
	
	/// Binary operator can do in-place calculation
	public var inPlaceble: Bool = true
	
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Initializers
    
    /// Designated init function
    ///
    /// - Parameters:
    ///   - label: label description
    ///   - delegate: delegate description
    public init(operatorLabel label: String,
                cpuComputeBlock block: @escaping (Tensor, Tensor, Tensor) -> Void ,
                gradComputationBlock gradBlock: @escaping (Tensor, Tensor, OperatorComputationMode) -> [Tensor],
                metalKernelFuncLabel kernelLabel: String,
                computationDelegate: OperatorCalculationDelegate?) {
        self.operatorLabel = label
        self.computationDelegate = computationDelegate
        self.metalKernelFuncLabel = kernelLabel
        self.cpuElementComputationBlock = block
		self.gradComputationBlock = gradBlock
    }
    
    /// Convenience initializer
    /// Subclass required to override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
    ///
    required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputA: Tensor,  inputB: Tensor, outputC: Tensor) -> Void in
            fatalError("NEED OVERRIDE")
        }
		let gradBlock = { (inputA: Tensor, inputB: Tensor, mode: OperatorComputationMode) -> [Tensor] in
			fatalError("NEED OVERRIDE")
		}
        let defaultLabel = "NEED OVERRIDE"
        let kernelLabel = "NEED OVERRIDE"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock, 
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate)
    }
	
	
	/// Convenience initializer
	///
	/// - Parameters:
	///   - computationDelegate: computationDelegate description
	///   - inputTensors: inputTensors description
	///   - outputTensors: outputTensors description
	public convenience init(computationDelegate: OperatorCalculationDelegate? = nil,
	                        inputTensors: [Tensor], outputTensors: [Tensor]) {
		self.init(computationDelegate: computationDelegate)
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
	}
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Methods
    
    /// This operator should just receive two tensors with same dimensions (dataType could be different).
    /// Return shape is exactly the same shape as input.
    ///
    /// - Parameter shapes: input shapes
    /// - Returns: return shapes
    public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
        guard shapes.count == 2 else {
            SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) could just receive two input tensors. Given \(shapes.count)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return nil
        }
        
        // dimension size should be the same. Ignore type
        guard shapes[0] == shapes[1] else {
            SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) receive two shapes not dimension equal. Given \(shapes[0].shapeArray) and \(shapes[1].shapeArray)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return nil
        }
        
        return [shapes[0]]
    }
    
    /// The `inputTensors` should have exactly two tensors and same dimensions.
    /// The `outputTensors` should have exactly one tensora and same dimension with input tensors.
    ///
    ///
    public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
        // inpu tensor and output tensor not nil
        guard self.inputTensors != nil && self.outputTensors != nil else {
            return (false, "Operator \(self.operatorLabel) should non-nil inputTensors and outputTensors.")
        }
        
        // input tensor shapes checck
        let inputShapes = self.inputTensors!.map { $0.shape }
        guard self.outputShape(shapeArray: inputShapes) != nil else {
            return (false, "Operator \(self.operatorLabel) does not have valid input tensors.")
        }
        
        // output tensors check
        guard self.outputTensors!.count == 1 else {
            return (false, "Operator \(self.operatorLabel) does not have valid number of output tensors. Require 1, given \(self.outputTensors!.count)")
        }
        
        // input tensor and output tensor shape match checking
        guard self.outputTensors![0].shape == inputShapes[0] else {
            return (false, "Operator \(self.operatorLabel) does not have valid output tensor shapes. Require \(inputShapes[0]), given \(self.outputTensors![0].count)")
        }
        
        return (true, "")
    }
    
    /// Compute asynclly
    ///
    /// - Parameters:
    ///   - tensors: input tensors
    ///   - computationMode: computation mode
     public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
		// check delegate
        OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
        DispatchQueue.global(qos: .userInitiated).async {
            self.compute(computationMode)
        }
    }
    
    /// Compute synclly.
    /// - Parameters:
    ///   - tensors: input tensors
    ///   - computationMode: cmputation mode. If choose `GPU` but haven't configued a GPU SerranoEngine, operator will use `CPU` to compute.
    /// - Returns: result tensors
    public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
        // check
        let (pass, msg) = self.inputOutputTensorsCheck()
        guard pass else {
            SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) calculation aborted cause invalid input tensors or output tensors: \(msg)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
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
	
	/// Calulate grads sync.
	/// All unary operator return grads tensor with same number and shape as attribute `inputTensors`.
	///
	/// - Parameters:
	///   - computationMode: computationMode
	///   - upGrds: upGrds
	/// - Returns: return grads tensor
	public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType] {
		let grads =  self.gradComputationBlock(self.inputTensors![0], self.inputTensors![1], computationMode)
		var result = [String: DataSymbolSupportedDataType]()
		for (i, grad) in grads.enumerated() {
			result["input_\(i)"] = grad
		}
		return result
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
			self.computationDelegate?.operatorWillBeginGradsComputation(self)
			let result = self.gradCompute(computationMode)
			self.computationDelegate?.operatorDidEndGradsComputation(self, grads: result)
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
	
	/// Binary operator has no parameters. Do nothing
	public func bindParamSymbols(_ symbols: [GraphSymbol]) {
	}
	
	/// This operator has no parameters.
	///
	/// - Returns: An empty array
	public func paramSymbols() -> [GraphSymbol] {
		return [GraphSymbol]()
	}
    
    /// Use cpu do the inplace computation. This function always do inPlace computation for inputTensorA.
    /// It's caller function to decide the tensor's assignment.
    /// Default, `UnaryOperator` defines a workflow. Subclass just needs to override `cpuElementComputationBlock`.
    /// If subclass needs custom flow, it could just override this function.
    ///
    /// - Note: This function should not be called from outside.
    ///
    /// - Parameter tensors: the operation tensors
    internal func cpu()  {
        self.cpuElementComputationBlock(self.inputTensors![0], self.inputTensors![1], self.outputTensors![0])
    }
    
    /// Let GPU call the Metal kernel to do the inplace computation.This function always do inPlace computation for inputTensorA.
    /// It's caller function to decide the tensor's assignment.
    /// Default, `UnaryOperator` defines a workflow. Subclass just needs to override `metalKernelFuncLabel` attribute.
    /// If subclass needs custom flow, it could just override this function.
    ///
    /// - Note: This function should not be called from outside.
    ///
    /// - Parameter tensors: the operation tensors
    internal func gpu()  {
        // prepare resources
        let engine = SerranoEngine.configuredEngine
        var kernel: MTLComputePipelineState?
        var commandBuffer: MTLCommandBuffer?
		let inputABufferResource = self.inputTensors![0].gpuBufferResource()
		let inputBBufferResource = self.inputTensors![1].gpuBufferResource()
		let resultBufferResource = self.outputTensors![0].gpuBufferResource()
		
        // kernel
		var info = ""
		(kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
		guard kernel != nil else {
			fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
		}
		
        // command buffer
		commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
		guard commandBuffer != nil else {
			fatalError("[Serrano] Failed to make new command buffer.")
		}
		
        // dimensionBuffer
        var count = MetalUInt(self.inputTensors![0].count)
		
        //// Prepare encoders.
        let encoder = commandBuffer!.makeComputeCommandEncoder()
        encoder.setComputePipelineState(kernel!)
        encoder.setBuffer(inputABufferResource.buffer, offset: inputABufferResource.offset, at: 0)
        encoder.setBuffer(inputBBufferResource.buffer, offset: inputBBufferResource.offset, at: 1)
        encoder.setBuffer(resultBufferResource.buffer, offset: resultBufferResource.offset, at: 2)
        encoder.setBytes(&count, length: MemoryLayout<MetalUInt>.stride, at: 3)
        
        // dispatch
        let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
                                                1,
                                                1)
        let threadgroupsPerGrid = MTLSizeMake((self.inputTensors![0].count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                              1,
                                              1)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
        
        // commit command buffer
        commandBuffer!.commit()
        commandBuffer!.waitUntilCompleted()
    }
}

/**
 Do `a+b`. Not support broadcasting.
 */
public class AddOperator: BinaryOperator {
	
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputA: Tensor,  inputB: Tensor, outputC: Tensor) -> Void in
			let inputAAddress = inputA.contentsAddress
			let inputBAddress = inputB.contentsAddress
			let outputCAddress = outputC.contentsAddress
			let count = vDSP_Length(outputC.count)
			vDSP_vadd(inputAAddress, 1, inputBAddress, 1, outputCAddress, 1, count)
		}
		
		// dc/da = 1; dc/db = 1
		let gradBlock = { (inputA: Tensor,  inputB: Tensor, mode: OperatorComputationMode) -> [Tensor] in
			let gradA = Tensor(repeatingValue: 1.0, tensorShape: inputA.shape)
			let gradB = Tensor(repeatingValue: 1.0, tensorShape: inputB.shape)
			return [gradA, gradB]
		}

		
        let defaultLabel = "AddOperator"
        let kernelLabel = "Add"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate)
    }
}

/**
Do `a-b`. Not support broadcasting.
*/
public class SubOperator: BinaryOperator {
	
	/// Override init
	///
	/// - Parameter computationDelegate: delegate
	public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputA: Tensor,  inputB: Tensor, outputC: Tensor) -> Void in
			let inputAAddress = inputA.contentsAddress
			let inputBAddress = inputB.contentsAddress
			let outputCAddress = outputC.contentsAddress
			let count = vDSP_Length(outputC.count)
			vDSP_vsub(inputBAddress, 1, inputAAddress, 1, outputCAddress, 1, count)
		}
		
		// dc/da = 1; dc/db = -1
		let gradBlock = { (inputA: Tensor,  inputB: Tensor, mode: OperatorComputationMode) -> [Tensor] in
			let gradA = Tensor(repeatingValue: 1.0, tensorShape: inputA.shape)
			let gradB = Tensor(repeatingValue: -1.0, tensorShape: inputB.shape)
			return [gradA, gradB]
		}
		
		let defaultLabel = "SubOperator"
		let kernelLabel = "Sub"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate)
	}
}

/**
Do `a*b` in element-wise way. Not support broadcasting.
*/
public class MultOperator: BinaryOperator {
	
	/// Override init
	///
	/// - Parameter computationDelegate: delegate
	public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputA: Tensor,  inputB: Tensor, outputC: Tensor) -> Void in
			let inputAAddress = inputA.contentsAddress
			let inputBAddress = inputB.contentsAddress
			let outputCAddress = outputC.contentsAddress
			let count = vDSP_Length(outputC.count)
			vDSP_vmul(inputAAddress, 1, inputBAddress, 1, outputCAddress, 1, count)
		}
		
		// dc/da = b; dc/db = a
		let gradBlock = { (inputA: Tensor,  inputB: Tensor, mode: OperatorComputationMode) -> [Tensor] in
			/// First allocate as managed to speed up incase using GPU with reusing MTLBuffers
			let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors([inputA.shape, inputB.shape])
			let copyOp = CopyOperator(inputTensors: [inputA, inputB], outputTensors: [grads[1], grads[0]])
			copyOp.disableInputOutputCheck = true
			copyOp.compute(mode)
			return grads
		}
		
		let defaultLabel = "MultOperator"
		let kernelLabel = "Mult"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate)
	}
}

/**
Do `a/b` in element-wise way. Not support broadcasting.
*/
public class DivOperator: BinaryOperator {
	
	/// Override init
	///
	/// - Parameter computationDelegate: delegate
	public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputA: Tensor,  inputB: Tensor, outputC: Tensor) -> Void in
			let inputAAddress = inputA.contentsAddress
			let inputBAddress = inputB.contentsAddress
			let outputCAddress = outputC.contentsAddress
			let count = vDSP_Length(outputC.count)
			vDSP_vdiv(inputBAddress, 1, inputAAddress, 1, outputCAddress, 1, count)
		}
		
		// dc/da = 1/b; dc/db = -a/b^2
		let gradBlock = { (inputA: Tensor,  inputB: Tensor, mode: OperatorComputationMode) -> [Tensor] in
			let workGroup = DispatchGroup()
			
			// A
			var gradA: Tensor?
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				gradA = 1.0 / inputB
				workGroup.leave()
			}
			
			// B 
			var gradB: Tensor?
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				// a / b^2
				gradB = inputB * inputB
				let divOp = DivOperator(inputTensors: [inputA, gradB!], outputTensors: [gradB!])
				divOp.disableInputOutputCheck = true
				divOp.compute(mode)
				// -1.0
				-1.0 &* gradB!
				workGroup.leave()
			}
			workGroup.wait()
			return [gradA!, gradB!]
		}
		
		let defaultLabel = "DivOperator"
		let kernelLabel = "Div"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate)
	}
}


/**
Do `b/a` in element-wise way. Not support broadcasting.
*/
public class RDivOperator: BinaryOperator {
	
	/// Override init
	///
	/// - Parameter computationDelegate: delegate
	public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputA: Tensor,  inputB: Tensor, outputC: Tensor) -> Void in
			let inputAAddress = inputA.contentsAddress
			let inputBAddress = inputB.contentsAddress
			let outputCAddress = outputC.contentsAddress
			let count = vDSP_Length(outputC.count)
			vDSP_vdiv(inputAAddress, 1, inputBAddress, 1, outputCAddress, 1, count)
		}
		
		// dc/da = -b/a^2; dc/db = 1/a
		let gradBlock = { (inputA: Tensor,  inputB: Tensor, mode: OperatorComputationMode) -> [Tensor] in
			let workGroup = DispatchGroup()
			
			// A
			var gradA: Tensor?
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				// b / a^2
				gradA = inputA * inputA
				let divOp = DivOperator(inputTensors: [inputB, gradA!], outputTensors: [gradA!])
				divOp.disableInputOutputCheck = true
				divOp.compute(mode)
				// -1.0
				-1.0 &* gradA!
				workGroup.leave()
			}
			
			// B
			var gradB: Tensor?
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				gradB = 1.0 / inputA
				workGroup.leave()
			}
			
			workGroup.wait()
			return [gradA!, gradB!]
		}

		
		let defaultLabel = "RDivOperator"
		let kernelLabel = "RDiv"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate)
	}
}

/**
Do `a^b` in element-wise way. Not support broadcasting.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
*/
public class PowOperator: BinaryOperator {
	
	/// Override init
	///
	/// - Parameter computationDelegate: delegate
	public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let block = { (inputA: Tensor,  inputB: Tensor, outputC: Tensor) -> Void in
			let inputAAddress = inputA.contentsAddress
			let inputBAddress = inputB.contentsAddress
			let outputCAddress = outputC.contentsAddress
			var count = Int32(outputC.count)
			vvpowf(outputCAddress, inputBAddress, inputAAddress, &count)
		}
		
		// dc/da = b * a^(b-1); dc/db = (a^b) * ln(a)
		let gradBlock = { (inputA: Tensor,  inputB: Tensor, mode: OperatorComputationMode) -> [Tensor] in
			let workGroup = DispatchGroup()
			
			// A
			var gradA: Tensor?
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				gradA = SerranoResourceManager.globalManager.allocateUnamangedTensor(inputA.shape)
				let bm1 = inputB - 1.0
				// a^(b-1)
				let powOp = PowOperator(inputTensors: [inputA, bm1], outputTensors: [gradA!])
				powOp.disableInputOutputCheck = true
				powOp.compute(mode)
				// * b
				gradA! &* inputB
				workGroup.leave()
			}
			
			// B
			var gradB: Tensor?
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				gradB = SerranoResourceManager.globalManager.allocateUnamangedTensor(inputB.shape)
				// lna
				let group = DispatchGroup()
				let lna = Tensor(repeatingValue: 0.0, tensorShape: inputA.shape)
				DispatchQueue.global(qos: .userInitiated).async {
					group.enter()
					let logOp = LogOperator(inputTensors: [inputA], outputTensors: [lna])
					logOp.disableInputOutputCheck = true
					logOp.compute(mode)
					group.leave()
				}
				
				// (a^b)
				let powOp = PowOperator(inputTensors: [inputA, inputB], outputTensors: [gradB!])
				powOp.disableInputOutputCheck = true
				powOp.compute(mode)
				
				// *
				group.wait()
				gradB! &* lna
				
				workGroup.leave()
			}
			
			workGroup.wait()
			return [gradA!, gradB!]
		}
		
		let defaultLabel = "PowOperator"
		let kernelLabel = "Pow"
		self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
		          metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate)
	}
}

