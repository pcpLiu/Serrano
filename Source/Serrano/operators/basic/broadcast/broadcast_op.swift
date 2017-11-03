//
//  broadcast_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/13/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation

/**
Doing broadcasting on input tensors and store result in output tensors.
Serrano follows the broadcasting rule of [Scipy](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc).
*/
public class BroadcastOperator: ComputableOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
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
	
	/// Target shape.
	/// - Note: This atrribute could be `nil` when initialize an object.
	///         If it is `nil` doing calculation, a `fatalError()` will be raise.
	public var targetShape: TensorShape?
	
	/// If `true`, operator will not check the `upGrads`'s shape.
	/// This is used inside framework to speed up in situation we know it will not be wrong.
	/// Cases like auto generated differentiation graph.
	public var disableUpGradShapeCheck: Bool = false
	
	/// If `true`, operator will not call `inputOutputTensorsCheck()` before doing calculation.
	/// This is used inside framework to speed up in situation we know it will not be wrong.
	public var disableInputOutputCheck: Bool = false
	
	/// Indicate if this operator would do paramter update.
	///
	/// - Note: `BroadcastOperator` is not trainable.
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
	
	/// Initializers
	///
	/// - Note: In most cases, this initializer should not be called.
	///         Call one of the convenience initializers instead.
	///
	/// - Parameters:
	///   - label:
	///   - kernelLabel:
	///   - computationDelegate:
	///   - inputTensors:
	///   - outputTensors:
	///   - targetShape:
	public init(operatorLabel label: String,
	            metalKernelFuncLabel kernelLabel: String,
	            computationDelegate: OperatorCalculationDelegate?,
	            inputTensors: [Tensor]?,
	            outputTensors: [Tensor]?,
	            targetShape: TensorShape?) {
		self.operatorLabel = label
		self.computationDelegate = computationDelegate
		self.metalKernelFuncLabel = kernelLabel
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
		self.targetShape = targetShape
	}
	
	
	///
	///
	/// - Parameters:
	///   - computationDelegate:
	///   - targetShape:
	public convenience init(computationDelegate: OperatorCalculationDelegate, targetShape: TensorShape?) {
		let defaultLabel = "BroadCastOp"
		let kernelLabel = ""
		self.init(operatorLabel: defaultLabel, metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
		          inputTensors: nil, outputTensors: nil, targetShape: targetShape)
	}
	
	
	/// Initializer
	///
	/// - Parameter targetShape: target broadcast shape
	public convenience init(targetShape: TensorShape? = nil) {
		let defaultLabel = "BroadCastOp"
		let kernelLabel = ""
		self.init(operatorLabel: defaultLabel, metalKernelFuncLabel: kernelLabel, computationDelegate: nil,
		          inputTensors: nil, outputTensors: nil, targetShape: targetShape)
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Conforms to `ComputableOperator`
	
	/// The input shapes and target shape should follow the [`scipy rule`](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc)
	///
	/// - Parameter shapes: input shapes
	/// - Returns: return shapes. `nil` if not valid
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		// target shape set
		guard self.targetShape != nil else {
			SerranoLogging.errorLogging(message: "Did not setup targetShape.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		let targetShapeReversed = Array(self.targetShape!.shapeArray.reversed())
		
		let transformedShape = shapes.flatMap { tensorShape -> TensorShape? in
			// dimensions check
			guard tensorShape.shapeArray.count > 0 && tensorShape.shapeArray.count <= self.targetShape!.shapeArray.count else {
				SerranoLogging.errorLogging(message: "Invalid shape: \(tensorShape.shapeArray) for target shape: \(self.targetShape!.shapeArray)",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
			
			// dim size check
			let tensorShapeReversed = Array(tensorShape.shapeArray.reversed())
			for (dimInput, dimTarget) in zip(tensorShapeReversed, targetShapeReversed) {
				guard dimInput == dimTarget || dimInput == 1 else {
					SerranoLogging.errorLogging(message: "Invalid shape: \(tensorShape.shapeArray) for target shape: \(self.targetShape!.shapeArray)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
					return nil
				}
			}
			
			return self.targetShape
		}
		
		if transformedShape.count != shapes.count {
			return nil
		} else {
			return transformedShape
		}
	}
	
	
	/// Check if shapes of all input tensors and output tensors are compatible with `targetShape`
	///
	/// - Returns: check and message
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
		guard self.inputTensors != nil else {
			return (false, "Input tensors are nil")
		}
		
		guard self.outputTensors != nil else {
			return (false, "Output tensors are nil")
		}
		
		guard self.inputTensors!.count == self.outputTensors!.count else {
			return (false, "Input and output tensors should have same number of tensors. Given input: \(self.inputTensors!.count) and output: \(self.outputTensors!.count)")
		}
		
		// check each tensor's shape
		for tensor in self.inputTensors! {
			guard self.outputShape(shapeArray: [tensor.shape]) != nil else {
				return (false, "Input tensor \(tensor.description) cannot be broadcast to target shape \(self.targetShape!.description)")
			}
		}
		
		// check each tensor's shape
		for tensor in self.outputTensors! {
			guard self.outputShape(shapeArray: [tensor.shape]) != nil else {
				return (false, "Output tensor \(tensor.description) cannot be broadcast to target shape \(self.targetShape!.description)")
			}
		}
		
		return (true, "")
	}
	
	public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
		// check
		let (pass, msg) = self.inputOutputTensorsCheck()
		guard pass else {
			SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) calculation aborted cause invalid input tensors or output tensors: \(msg)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		self.computationDelegate?.operatorWillBeginComputation(self)
		self.cpu()
		self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
	}
	
	public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
		// check delegate
		OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
		
		DispatchQueue.global(qos: .userInitiated).async {
			self.compute(computationMode)
		}
	}
	
	/// Calulate grads sync.
	/// Broadcast operator itself does not generate any grads. Should be ignored in gaph AD.
	///
	/// - Parameters:
	///   - computationMode: computationMode
	/// - Returns: return `upGrads` if not nil. Else return an empty array.
	public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType] {
		return [:]
	}
	
	
	/// Cal grads async
	///
	/// - Parameters:
	///   - computationMode: computationMode
	public func gradComputAsync(_ computationMode: OperatorComputationMode) {
		// check delegate
		OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
		
		DispatchQueue.global(qos: .userInitiated).async {
			self.computationDelegate?.operatorWillBeginGradsComputation(self)
			let result = self.gradCompute(computationMode)
			self.computationDelegate?.operatorDidEndGradsComputation(self, grads: result)
		}
	}
	
	
	/// No updatable parameters.
	/// This function just returns.
	///
	/// - Parameters:
	///   - grads: grads
	///   - LR: LR
	public func updateParams(grads: [Tensor], LR: Float) {
		return
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
	
	public func cpu() {
		for (tensor, targetTensor) in zip(self.inputTensors!, self.outputTensors!) {
			
			// get reverse shapes for convenience
			let targetShapeReversed = Array(self.targetShape!.shapeArray.reversed())
			var rawShapeReversed =  Array(tensor.shape.shapeArray.reversed())
			
			for i in 0..<targetShapeReversed.count {
				// If raw shape size is less than target shape, fill those position with 1s
				if i >= rawShapeReversed.count {
					rawShapeReversed.append(0)
				}
				let rawDim = rawShapeReversed[i]
				let targetDim = targetShapeReversed[i]
				if i == 0 { // 1st dim
					if rawDim == targetDim { // just copy each element
						let cpFromAddress = tensor.contentsAddress
						let cpToAddress = UnsafeMutableRawPointer(targetTensor.contentsAddress)
						memcpy(cpToAddress, cpFromAddress, tensor.count * MemoryLayout<Float>.stride)
					} else { // copy each element for targetDim times
						let reader = tensor.floatValueReader
						let writer = targetTensor.floatValueReader
						var writerIndex = 0
						for elementCount in 0..<tensor.count {
							for _ in 0..<targetDim {
								writer[writerIndex] = reader[elementCount]
								writerIndex += 1
							}
						}
					}
				} else { // other dims
					if rawDim == targetDim {
						continue
					} else if rawDim != 0 { // inner dims
						let numBlock = rawShapeReversed.suffix(from: i+1).reduce(1, *) // cols block count should be use rawShapeReversed
						let blockElementCount = targetShapeReversed.prefix(upTo: i).reduce(1, *)
						let blockSize = blockElementCount * MemoryLayout<Float>.stride
						
						// Needs to move first avoid over writing
						for i in 0..<numBlock {
							let startAddress = UnsafeRawPointer(targetTensor.contentsAddress + blockElementCount * i)
							let toAddress = UnsafeMutableRawPointer(targetTensor.contentsAddress + blockElementCount * targetDim * i)
							memcpy(toAddress, startAddress, blockSize)
						}
						
						// copy for repeat
						for i in 0..<numBlock {
							let cpFromAddress =  UnsafeRawPointer(targetTensor.contentsAddress + blockElementCount * targetDim * i)
							var cpToAddress = UnsafeMutableRawPointer(targetTensor.contentsAddress + blockElementCount * targetDim * i) + blockSize
							for _ in 1..<targetDim {
								memcpy(cpToAddress, cpFromAddress, blockSize)
								cpToAddress += blockSize
							}
						}
					} else { // outter dim, just repeat all existing elements
						let blockSize = targetShapeReversed.prefix(upTo: i).reduce(1, *) * MemoryLayout<Float>.stride
						let cpFromAddress = UnsafeRawPointer(targetTensor.contentsAddress)
						var cpToAddress = UnsafeMutableRawPointer(targetTensor.contentsAddress) + blockSize
						for _ in 1..<targetDim {
							memcpy(cpToAddress, cpFromAddress, blockSize)
							cpToAddress += blockSize
						}
					}
				}
			}
		}
	}
}
