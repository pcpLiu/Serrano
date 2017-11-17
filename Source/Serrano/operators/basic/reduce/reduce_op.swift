//
//  reduce_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/27/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Accelerate
import Dispatch


/**
The abstract class for all reduce operators. Should not be used directly.
A reduce operator do some aggregate calculation along given axes.
An example given by TensorFlow is [here](https://www.tensorflow.org/api_docs/python/tf/reduce_sum).
*/
public class ReduceOperator: ComputableOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	public var computationDelegate: OperatorCalculationDelegate?
	
	public var metalKernelFuncLabel: String
	
	public var operatorLabel: String
	
	public var inputTensors: [Tensor]?
	
	public var outputTensors: [Tensor]?
	
	/// The axes to do the computation
	public var axis: [Int] {
		didSet {
			self.axis = Array(Set(self.axis)) // remove duplicates
		}
	}
	
	/// Indicate if keep dimensions in result tensor.
	/// This just affects result tensor's `shape` attributes.
	/// Default `false`
	public var keepDim: Bool = false
	
	/// The element compuation block in CPU mode.
	/// In most cases, subclass should just override this part in `init` method instead overriding the whole `cpu()` method.
	/// The firat pointer is the input tensor,
	//// the second is the output tensor
	public lazy var cpuComputeBlock:  (Tensor, Tensor, [Int]) -> Void = { (inputTensor: Tensor, outputTensor: Tensor, axis: [Int]) -> Void in
		print("NEED OVERLOAD")
	}
	
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
	/// `OneToOne` for this operator.
	public var mapType: OperatorMappingType {
		get {
			return OperatorMappingType.OneToOne
		}
	}
	
	/// Reduce operator cannot do in-place calculation
	public var inPlaceble: Bool = false
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	init(computationDelegate: OperatorCalculationDelegate?,
	            operatorLabel: String = "NEED OVERRIDE",
	            metalKernelFuncLabel kernelLabel: String = "NEED OVERRIDE",
	            inputTensors: [Tensor]?, outputTensors: [Tensor]?,
	            axis: [Int]) {
		self.computationDelegate = computationDelegate
		self.operatorLabel = operatorLabel
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
		self.metalKernelFuncLabel = kernelLabel
		self.axis = Array(Set(axis))
		self.cpuComputeBlock = { (inputTensor: Tensor, outputTensor: Tensor, axis: [Int]) -> Void in
			print("NEED OVERLOAD")
		}
	}
	
	
	/// Conenience init.
	///
	/// - Parameters:
	///   - inputTensors: inputTensors description
	///   - outputTensors: outputTensors description
	///   - axis: axis description
	public convenience required init(inputTensors: [Tensor]? = nil, outputTensors: [Tensor]? = nil, axis: [Int]) {
		self.init(computationDelegate: nil, inputTensors: inputTensors, outputTensors: outputTensors, axis: axis)
	}
	
	
	/// Conenience init.
	///
	/// - Parameters:
	///   - axis: axis
	///   - keepDim: keepDim
	public convenience init(axis: [Int], keepDim: Bool) {
		self.init(computationDelegate: nil, inputTensors: nil, outputTensors: nil, axis: axis)
		self.keepDim = keepDim
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	
	/// Check if for each input shape operator could do reduce operation with `axis` attribute value.
	///
	/// - Parameter shapes: shapes description
	/// - Returns: return value description
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		var outShapes = [TensorShape]()
		
		// empty input warning
		if shapes.count == 0 {
			SerranoLogging.warningLogging(message: "The input shapes contains no element.",
			                              file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return outShapes
		}
		
		for shape in shapes {
			// check rank
			guard shape.rank >= self.axis.count else {
				SerranoLogging.errorLogging(message: "Input shape [\(shape)] is not valid on target axis: \(self.axis)",
				                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
				return nil
			}
			
			
			var outputShapeArray = shape.shapeArray
			// ax dim value check
			for ax in self.axis {
				guard ax < outputShapeArray.count && ax >= 0 else {
					SerranoLogging.errorLogging(message: "Input shape [\(shape)] is not valid on target axis: \(self.axis)",
						file: "\(#file)", function: "\(#function)", line: "\(#line)")
					return nil
				}
				outputShapeArray[ax] = 1
			}
			
			// not keeping dim, del dim
			if !self.keepDim {
				outputShapeArray = outputShapeArray
					.enumerated()
					.filter{ !self.axis.contains($0.offset) }
					.map { $0.element }
			}
			outShapes.append(TensorShape(dataType: shape.dataType, shape: outputShapeArray))
		}
		
		return outShapes
	}
	
	
	/// Validate shapes`inputTensors` and `outputTensors`.
	///
	/// - Returns: check passing and message if not passing
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
		// input not nil
		guard self.inputTensors != nil else {
			return (false, "Input tensors are nil.")
		}
		
		// output not nil
		guard self.outputTensors != nil else {
			return (false, "Output tensors are nil.")
		}
		
		// count
		guard self.inputTensors!.count == self.outputTensors!.count else {
			return (false, "Input and output tensors should have same amount of tensors. " +
						   "Input: \(self.inputTensors!.count), output: \(self.outputTensors!.count).")
		}
		
		// check input shapes
		let inputShapes = self.inputTensors!.map { $0.shape }
		let outputShapesValid = self.outputShape(shapeArray: inputShapes)
		guard outputShapesValid != nil else {
			return (false, "Input tensors' shapes are not valid. Check log for detail.")
		}
		
		// check output shapes
		let outputShapes = self.outputTensors!.map { $0.shape }
		for i in 0..<outputShapesValid!.count {
			guard outputShapesValid![i] == outputShapes[i] else {
				return (false, "Shape of Output tensor \(self.outputTensors![i]) is not valid. " +
							   "Expect \(outputShapesValid![i]). Given \(outputShapes[i])")
			}
		}
		
		return (true, "")
	}
	
	
	/// Comput sync
	///
	/// - Parameter computationMode: computationMode description
	public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
		// check
		let (pass, msg) = self.inputOutputTensorsCheck()
		guard pass else {
			SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) calculation aborted cause invalid input tensors or output tensors: \(msg)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		self.computationDelegate?.operatorWillBeginComputation(self)
		
		self.cpu()
		
//		switch computationMode {
//		case .GPU:
//			if !SerranoEngine.configuredEngine.hasAvailableGPU() {
//				SerranoLogging.warningLogging(message: "Serrano Engine has no available configured GPU device. Use CPU doing calculation instead.", file: "\(#file)", function: "\(#function)", line: "\(#line)")
//				self.cpu()
//			} else {
//				self.gpu()
//			}
//		case .CPU:
//			self.cpu()
//		}
		self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
	}
	
	
	/// Comput async
	///
	/// - Parameter computationMode: computationMode description
	public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
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
	
	/// Update params if possible.
	/// No update parameters for binary operators.
	///
	/// - Parameters:
	///   - grads: grads tensor list
	///   - LR: learning rate
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
	
	internal func cpu() {
	 let workGroup = DispatchGroup()
		for tensorIndex in 0..<self.inputTensors!.count {
			workGroup.enter()
			DispatchQueue.global(qos: .userInitiated).async {
				self.cpuComputeBlock(self.inputTensors![tensorIndex], self.outputTensors![tensorIndex], self.axis)
				workGroup.leave()
			}
		}
		
		workGroup.wait()
	}
	
	internal func gpu() {
		//TODO: IMPLEMENTATION
		fatalError()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
Computes the sum of array elements over given axes.
*/
public class ReduceSumOperator: ReduceOperator {
	
	override init(computationDelegate: OperatorCalculationDelegate?,
	     operatorLabel: String = "ReduceSumOperator",
	     metalKernelFuncLabel kernelLabel: String = "ReduceSum",
	     inputTensors: [Tensor]?, outputTensors: [Tensor]?,
	     axis: [Int]) {
		super.init(computationDelegate: computationDelegate, operatorLabel: operatorLabel, metalKernelFuncLabel: kernelLabel,
		           inputTensors: inputTensors, outputTensors: outputTensors, axis: axis)
		// custom block
		self.cpuComputeBlock = {(inputTensor: Tensor, outputTensor: Tensor, axis: [Int]) -> Void in
			let op = ReduceOperator(axis: [Int]())
			op.keepDim = true
			
			var inputAddress = inputTensor.contentsAddress
			var outptuAddress: UnsafeMutablePointer<Float>
			var inputShape = inputTensor.shape
			var intermediateTensor: Tensor = inputTensor
			var nextTensor: Tensor = outputTensor
			
			let sortedAxis = axis.sorted()
			for (axIndex, ax) in sortedAxis.enumerated() {
				// decide output address
				if axIndex != sortedAxis.count - 1 {
					// intermediate
					op.axis = [ax]
					let nextShape = op.outputShape(shapeArray: [inputShape])!.first!
					nextTensor = Tensor(repeatingValue: 0.0, tensorShape: nextShape)
					outptuAddress = nextTensor.contentsAddress
				} else {
					nextTensor = outputTensor
				}
				outptuAddress = nextTensor.contentsAddress
				
				// do reduce on intermediateTensor and store result to newTensor
				let preDimCount = intermediateTensor.shape.shapeArray.prefix(upTo: ax).reduce(1, *)
				let nextEntryCount = intermediateTensor.shape.shapeArray.suffix(from: ax+1).reduce(1, *)
				let entryCount = intermediateTensor.shape.shapeArray.suffix(from: ax).reduce(1, *)
				let axDim = intermediateTensor.shape.shapeArray[ax]
				for preDimIndex in 0..<preDimCount {
					for entryIndex in 0..<nextEntryCount {
						// calcualte sum for each element in nextTensor.
						// vDSP_sve sums  [vDSP_Length(ax)] number of floats from [inputAddress] to [outptuAddress] with stride [entryCount]
						vDSP_sve(inputAddress + preDimIndex*entryCount + entryIndex,
						         nextEntryCount,
						         outptuAddress,
						         vDSP_Length(axDim))
						outptuAddress += 1
					}
				}
				
				// update input address and input shape
				intermediateTensor = nextTensor // keep it alive
				inputAddress = intermediateTensor.contentsAddress
				inputShape = intermediateTensor.shape
				
			}
		}
	}
}

/**
Computes the product of array elements over given axes.
*/
public class ReduceProductOperator: ReduceOperator {
	
	override init(computationDelegate: OperatorCalculationDelegate?,
	              operatorLabel: String = "ReduceProductOperator",
	              metalKernelFuncLabel kernelLabel: String = "ReduceProduct",
	              inputTensors: [Tensor]?, outputTensors: [Tensor]?,
	              axis: [Int]) {
		super.init(computationDelegate: computationDelegate, operatorLabel: operatorLabel, metalKernelFuncLabel: kernelLabel,
		           inputTensors: inputTensors, outputTensors: outputTensors, axis: axis)
		// custom block
		self.cpuComputeBlock = {(inputTensor: Tensor, outputTensor: Tensor, axis: [Int]) -> Void in
			let op = ReduceOperator(axis: [Int]())
			op.keepDim = true
			
			var inputAddress = inputTensor.contentsAddress
			var outptuAddress: UnsafeMutablePointer<Float>
			var inputShape = inputTensor.shape
			var intermediateTensor: Tensor = inputTensor
			var nextTensor: Tensor = outputTensor
			
			let sortedAxis = axis.sorted()
			for (axIndex, ax) in sortedAxis.enumerated() {
				// decide output address
				if axIndex != sortedAxis.count - 1 {
					// intermediate
					op.axis = [ax]
					let nextShape = op.outputShape(shapeArray: [inputShape])!.first!
					nextTensor = Tensor(repeatingValue: 0.0, tensorShape: nextShape)
					outptuAddress = nextTensor.contentsAddress
				} else {
					nextTensor = outputTensor
				}
				outptuAddress = nextTensor.contentsAddress
				
				// do reduce on intermediateTensor and store result to newTensor
				let preDimCount = intermediateTensor.shape.shapeArray.prefix(upTo: ax).reduce(1, *)
				let nextEntryCount = intermediateTensor.shape.shapeArray.suffix(from: ax+1).reduce(1, *)
				let entryCount = intermediateTensor.shape.shapeArray.suffix(from: ax).reduce(1, *)
				let axDim = intermediateTensor.shape.shapeArray[ax]
				for preDimIndex in 0..<preDimCount {
					for entryIndex in 0..<nextEntryCount {
						outptuAddress[0] = 1.0
						// calcualte sum for each element in nextTensor.
						for j in 0..<axDim {
							outptuAddress[0] *= (inputAddress + preDimIndex*entryCount + entryIndex + nextEntryCount * j).pointee
						}
						
						outptuAddress += 1
					}
				}
				
				// update input address and input shape
				intermediateTensor = nextTensor // keep it alive
				inputAddress = intermediateTensor.contentsAddress
				inputShape = intermediateTensor.shape
				
			}
		}
	}
}

/**
Computes the max of array elements over given axes.
*/
public class ReduceMaxOperator: ReduceOperator {
	
	override init(computationDelegate: OperatorCalculationDelegate?,
	              operatorLabel: String = "ReduceMaxOperator",
	              metalKernelFuncLabel kernelLabel: String = "ReduceMax",
	              inputTensors: [Tensor]?, outputTensors: [Tensor]?,
	              axis: [Int]) {
		super.init(computationDelegate: computationDelegate, operatorLabel: operatorLabel, metalKernelFuncLabel: kernelLabel,
		           inputTensors: inputTensors, outputTensors: outputTensors, axis: axis)
		// custom block
		self.cpuComputeBlock = {(inputTensor: Tensor, outputTensor: Tensor, axis: [Int]) -> Void in
			let op = ReduceOperator(axis: [Int]())
			op.keepDim = true
			
			var inputAddress = inputTensor.contentsAddress
			var outptuAddress: UnsafeMutablePointer<Float>
			var inputShape = inputTensor.shape
			var intermediateTensor: Tensor = inputTensor
			var nextTensor: Tensor = outputTensor
			
			let sortedAxis = axis.sorted()
			for (axIndex, ax) in sortedAxis.enumerated() {
				// decide output address
				if axIndex != sortedAxis.count - 1 {
					// intermediate
					op.axis = [ax]
					let nextShape = op.outputShape(shapeArray: [inputShape])!.first!
					nextTensor = Tensor(repeatingValue: 0.0, tensorShape: nextShape)
					outptuAddress = nextTensor.contentsAddress
				} else {
					nextTensor = outputTensor
				}
				outptuAddress = nextTensor.contentsAddress
				
				// do reduce on intermediateTensor and store result to newTensor
				let preDimCount = intermediateTensor.shape.shapeArray.prefix(upTo: ax).reduce(1, *)
				let nextEntryCount = intermediateTensor.shape.shapeArray.suffix(from: ax+1).reduce(1, *)
				let entryCount = intermediateTensor.shape.shapeArray.suffix(from: ax).reduce(1, *)
				let axDim = intermediateTensor.shape.shapeArray[ax]
				for preDimIndex in 0..<preDimCount {
					for entryIndex in 0..<nextEntryCount {
						outptuAddress[0] = 1.0
						// vDSP_maxv get max from all  [vDSP_Length(ax)] number of floats from [inputAddress] to [outptuAddress] with stride [entryCount]
						vDSP_maxv(inputAddress + preDimIndex*entryCount + entryIndex,
						          nextEntryCount,
						          outptuAddress,
						          vDSP_Length(axDim))
						outptuAddress += 1
					}
				}
				
				// update input address and input shape
				intermediateTensor = nextTensor // keep it alive
				inputAddress = intermediateTensor.contentsAddress
				inputShape = intermediateTensor.shape
				
			}
		}
	}
}

/**
Computes the min of array elements over given axes.
*/
public class ReduceMinOperator: ReduceOperator {
	
	override init(computationDelegate: OperatorCalculationDelegate?,
	              operatorLabel: String = "ReduceMinOperator",
	              metalKernelFuncLabel kernelLabel: String = "ReduceMin",
	              inputTensors: [Tensor]?, outputTensors: [Tensor]?,
	              axis: [Int]) {
		super.init(computationDelegate: computationDelegate, operatorLabel: operatorLabel, metalKernelFuncLabel: kernelLabel,
		           inputTensors: inputTensors, outputTensors: outputTensors, axis: axis)
		// custom block
		self.cpuComputeBlock = {(inputTensor: Tensor, outputTensor: Tensor, axis: [Int]) -> Void in
			let op = ReduceOperator(axis: [Int]())
			op.keepDim = true
			
			var inputAddress = inputTensor.contentsAddress
			var outptuAddress: UnsafeMutablePointer<Float>
			var inputShape = inputTensor.shape
			var intermediateTensor: Tensor = inputTensor
			var nextTensor: Tensor = outputTensor
			
			let sortedAxis = axis.sorted()
			for (axIndex, ax) in sortedAxis.enumerated() {
				// decide output address
				if axIndex != sortedAxis.count - 1 {
					// intermediate
					op.axis = [ax]
					let nextShape = op.outputShape(shapeArray: [inputShape])!.first!
					nextTensor = Tensor(repeatingValue: 0.0, tensorShape: nextShape)
					outptuAddress = nextTensor.contentsAddress
				} else {
					nextTensor = outputTensor
				}
				outptuAddress = nextTensor.contentsAddress
				
				// do reduce on intermediateTensor and store result to newTensor
				let preDimCount = intermediateTensor.shape.shapeArray.prefix(upTo: ax).reduce(1, *)
				let nextEntryCount = intermediateTensor.shape.shapeArray.suffix(from: ax+1).reduce(1, *)
				let entryCount = intermediateTensor.shape.shapeArray.suffix(from: ax).reduce(1, *)
				let axDim = intermediateTensor.shape.shapeArray[ax]
				for preDimIndex in 0..<preDimCount {
					for entryIndex in 0..<nextEntryCount {
						outptuAddress[0] = 1.0
						// vDSP_minv get max from all  [vDSP_Length(ax)] number of floats from [inputAddress] to [outptuAddress] with stride [entryCount]
						vDSP_minv(inputAddress + preDimIndex*entryCount + entryIndex,
						          nextEntryCount,
						          outptuAddress,
						          vDSP_Length(axDim))
						outptuAddress += 1
					}
				}
				
				// update input address and input shape
				intermediateTensor = nextTensor // keep it alive
				inputAddress = intermediateTensor.contentsAddress
				inputShape = intermediateTensor.shape
				
			}
		}
	}
}

/**
Computes the mean of array elements over given axes.
*/
public class ReduceMeanOperator: ReduceOperator {
	
	override init(computationDelegate: OperatorCalculationDelegate?,
	              operatorLabel: String = "ReduceMeanOperator",
	              metalKernelFuncLabel kernelLabel: String = "ReduceMean",
	              inputTensors: [Tensor]?, outputTensors: [Tensor]?,
	              axis: [Int]) {
		super.init(computationDelegate: computationDelegate, operatorLabel: operatorLabel, metalKernelFuncLabel: kernelLabel,
		           inputTensors: inputTensors, outputTensors: outputTensors, axis: axis)
		// custom block
		self.cpuComputeBlock = {(inputTensor: Tensor, outputTensor: Tensor, axis: [Int]) -> Void in
			let op = ReduceOperator(axis: [Int]())
			op.keepDim = true
			
			var inputAddress = inputTensor.contentsAddress
			var outptuAddress: UnsafeMutablePointer<Float>
			var inputShape = inputTensor.shape
			var intermediateTensor: Tensor = inputTensor
			var nextTensor: Tensor = outputTensor
			var placeHolder: Float = 1.0
			
			let sortedAxis = axis.sorted()
			for (axIndex, ax) in sortedAxis.enumerated() {
				// decide output address
				if axIndex != sortedAxis.count - 1 {
					// intermediate
					op.axis = [ax]
					let nextShape = op.outputShape(shapeArray: [inputShape])!.first!
					nextTensor = Tensor(repeatingValue: 0.0, tensorShape: nextShape)
					outptuAddress = nextTensor.contentsAddress
				} else {
					nextTensor = outputTensor
				}
				outptuAddress = nextTensor.contentsAddress
				
				// do reduce on intermediateTensor and store result to newTensor
				let preDimCount = intermediateTensor.shape.shapeArray.prefix(upTo: ax).reduce(1, *)
				let nextEntryCount = intermediateTensor.shape.shapeArray.suffix(from: ax+1).reduce(1, *)
				let entryCount = intermediateTensor.shape.shapeArray.suffix(from: ax).reduce(1, *)
				let axDim = intermediateTensor.shape.shapeArray[ax]
				for preDimIndex in 0..<preDimCount {
					for entryIndex in 0..<nextEntryCount {
						outptuAddress[0] = 1.0
						// vDSP_normalize get mean from all  [vDSP_Length(ax)] number of floats from [inputAddress] to [outptuAddress] with stride [entryCount]
						vDSP_normalize(inputAddress + preDimIndex*entryCount + entryIndex,
						               nextEntryCount,
						               nil, 0,
						               outptuAddress,
						               &placeHolder,
						               vDSP_Length(axDim))
						outptuAddress += 1
					}
				}
				
				// update input address and input shape
				intermediateTensor = nextTensor // keep it alive
				inputAddress = intermediateTensor.contentsAddress
				inputShape = intermediateTensor.shape
				
			}
		}
	}
}
