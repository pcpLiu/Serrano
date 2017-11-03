//
//  tensor_arithmetic_ops.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/4/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation


/**
The abstract parent class for all broadcast arithmetic oeprators.
Any child class of this operator support element-wise calculation between two `Tensor` objects with broadcasting support.
*/
public class BroadcastArithmeticOperator: ComputableOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	public var computationDelegate: OperatorCalculationDelegate?
	
	public var metalKernelFuncLabel: String
	
	public var operatorLabel: String
	
	public var inputTensors: [Tensor]?
	
	public var outputTensors: [Tensor]?
	
	/// Computation logic block. 
	/// First param is input tensors, seconds param is output tensors.
	/// Operator just needs override this in requried init function instead of override whole computation methods.
	/// All input tensors are already broadcasted.
 	public lazy var calculationBlock: ([Tensor],  [Tensor], OperatorComputationMode) -> Void = {_,_,_ in }
	
	/// The grad compuation block.
	/// parameter: inputA, inputB,
	/// returns: An array of tensor. Should just have 2 object corresponding to two inputs
	public var gradComputationBlock: (Tensor, Tensor,  OperatorComputationMode) -> [Tensor] = {(_,_,_) -> [Tensor] in
		fatalError("Not implemented")
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
	/// - Note: `BroadcastArithmeticOperator` is not trainable.
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
	/// - Note: In most cases this initializer should not be called directly.
	///			call one of the convenience initializers instead.
	///
	/// - Parameters:
	///   - computationDelegate: computationDelegate
	///   - inputTensors: inputTensors
	///   - outputTensors: outputTensors
	///   - metalKernelFuncLabel: metalKernelFuncLabel
	///   - operatorLabel: operatorLabel
	public init(computationDelegate: OperatorCalculationDelegate?,
	     inputTensors: [Tensor]?, outputTensors: [Tensor]?,
	     metalKernelFuncLabel: String,
	     operatorLabel: String) {
		self.computationDelegate = computationDelegate
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
		self.metalKernelFuncLabel = metalKernelFuncLabel
		self.operatorLabel = operatorLabel
	}
	
	/// Conevnience initializer.
	///
	/// - Note: All subclass should override this convenience initializer to setup `metalKernelFuncLabel` and default `operatorLabel`.
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		// no need metal. Just use other existing operators
		let metalKernelFuncLabel = ""
		let defaultLable = "OVERRIDE"
		self.init(computationDelegate: computationDelegate, inputTensors: nil, outputTensors: nil,
		          metalKernelFuncLabel: metalKernelFuncLabel, operatorLabel: defaultLable)
	}
	
	/// Conenience initializer.
	///
	/// - Parameters:
	///   - computationDelegate: computationDelegate
	///   - inputTensors: inputTensors
	///   - outputTensors: outputTensors
	public convenience init(computationDelegate: OperatorCalculationDelegate? = nil, inputTensors: [Tensor], outputTensors: [Tensor]) {
		self.init(computationDelegate: computationDelegate)
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
	}
	
	
	/// Convenience initializer
	///
	/// - Parameters:
	///   - computationDelegate: computationDelegate
	///   - operatorLabel: operatorLabel
	public convenience init(computationDelegate: OperatorCalculationDelegate? = nil, operatorLabel: String) {
		self.init(computationDelegate: computationDelegate)
		self.operatorLabel = operatorLabel
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	
	/// Check the input shapes.
	/// Should exactly contains two shapes and the two shapes should be have same dimensions with or without broadcasting.
	///
	/// - Parameter shapes: input shapes
	/// - Returns: output shapes, maybe `nil` if not valid
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		// two input shapes
		guard shapes.count == 2 else {
			SerranoLogging.errorLogging(message: "Input shapes should have exactly 2 shapes. Given \(shapes.count)",
			                              file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		// shape match
		let shapeA = shapes[0]
		let shapeB = shapes[1]
		if shapeA == shapeB {
			// same dim
			return [shapeA]
		} else {
			// if not same dims, check if could broadcast to higher rank shape
			let broadcastOp = BroadcastOperator(targetShape: max(shapeA, shapeB))
			if shapeA < shapeB {
				return broadcastOp.outputShape(shapeArray: [shapeA])
			} else {
				return broadcastOp.outputShape(shapeArray: [shapeB])
			}
		}
	}
	
	
	/// Check validation of shapes mathcing between `inpuTensors` and `outputTensors`.
	///
	/// - Returns: `check` indicates if validation, `msg` containing error information.
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
		// input not nil
		guard self.inputTensors != nil else {
			return (false, "Input tensors are nil")
		}
		
		// output not nil
		guard self.outputTensors != nil else {
			return (false, "Output tensors are nil")
		}
		
		// check input shapes
		let inputShapes = self.inputTensors!.map { $0.shape }
		let outputShapeCheck = self.outputShape(shapeArray: inputShapes)
		guard outputShapeCheck != nil else {
			return (false, "Invalid shapes from input tensors. Check log for details.")
		}
		
		// check output shape match 
		let outputShapes = self.outputTensors!.map { $0.shape }
		guard outputShapeCheck!.count == outputShapes.count else {
			return (false, "Ouput tensors amount is not valid. Expect \(outputShapeCheck!.count), given \(outputShapes.count).")
		}
		for (shape, shapeCheck) in zip(outputShapes, outputShapeCheck!) {
			guard shape == shapeCheck else {
				return (false, "Invalid shape in output tensors. Expect \(shapeCheck.description), given \(shape.description)")
			}
		}
		
		return (true, "")
	}
	
	
	/// Usually, a `BroadcastArithmeticOperator` just call `BroadcastOperator` and other calculation operators.
	/// This methods will first do braodcast on input tensors if needs and then call `calculationBlock`.
	/// - Parameter computationMode: computationMode
	public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
		// check
		let (pass, msg) = self.inputOutputTensorsCheck()
		guard pass else {
			SerranoLogging.errorLogging(message: msg, file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		self.computationDelegate?.operatorWillBeginComputation(self)
		
		var inputA: Tensor = self.self.inputTensors![0]
		var inputB: Tensor = self.self.inputTensors![1]

		// Do broadcast if needs
		let shapeA = self.inputTensors![0].shape
		let shapeB = self.inputTensors![1].shape
		if shapeA != shapeB {
			let broadcastOp = BroadcastOperator(targetShape: max(shapeA, shapeB))
			if shapeA < shapeB {
				// broadcast A
				broadcastOp.inputTensors = [self.inputTensors![0]]
				inputA = SerranoResourceManager.globalManager.allocateTensor(shapeB)
				broadcastOp.outputTensors = [inputA]
			} else if shapeA > shapeB {
				// broadcast B
				broadcastOp.inputTensors = [self.inputTensors![1]]
				inputB = SerranoResourceManager.globalManager.allocateTensor(shapeA)
				broadcastOp.outputTensors = [inputB]
			}
			broadcastOp.compute(computationMode)
		}

		// call calculation block
		self.calculationBlock([inputA, inputB], self.outputTensors!, computationMode)
		
		// return intermediate tensors
		if inputA != self.inputTensors![0] { SerranoResourceManager.globalManager.returnTensor(inputA) }
		if inputB != self.inputTensors![1] { SerranoResourceManager.globalManager.returnTensor(inputB) }
		
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
	///
	/// - Parameters:
	///   - computationMode: computationMode
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
	/// No update parameters for broadcast arithmetic operators.
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
}

/**
Broadcasting addition.

- Note: The `inputTensors` of this operator should just have exactly 2 tensors and the `outputTensors` should just have 1 tensor.
*/
public class BroadcastAddOperator: BroadcastArithmeticOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Override requred inializer
	///
	/// - Parameter computationDelegate: computationDelegate 
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let metalKernelFuncLabel = ""
		let defaultLable = "BroadcastAddOp"
		self.init(computationDelegate: computationDelegate, inputTensors: nil, outputTensors: nil,
		          metalKernelFuncLabel: metalKernelFuncLabel, operatorLabel: defaultLable)
		self.calculationBlock = { (inputTensors: [Tensor], outputTensors: [Tensor], computationMode: OperatorComputationMode) -> Void in
			let inputA = inputTensors[0]
			let inputB = inputTensors[1]
			let output = outputTensors[0]
			// element wise add
			let addOp = AddOperator(inputTensors: [inputA, inputB], outputTensors: [output])
			addOp.compute(computationMode)
		}
		
		//TODO: Implement gradBlock
	}
}

/**
Broadcasting substraction.

- Note: The `inputTensors` of this operator should just have exactly 2 tensors and the `outputTensors` should just have 1 tensor.
*/
public class BroadcastSubOperator: BroadcastArithmeticOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Override requred inializer
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let metalKernelFuncLabel = ""
		let defaultLable = "BroadcastSubOp"
		self.init(computationDelegate: computationDelegate, inputTensors: nil, outputTensors: nil,
		          metalKernelFuncLabel: metalKernelFuncLabel, operatorLabel: defaultLable)
		self.calculationBlock = { (inputTensors: [Tensor], outputTensors: [Tensor], computationMode: OperatorComputationMode) -> Void in
			let inputA = inputTensors[0]
			let inputB = inputTensors[1]
			let output = outputTensors[0]
			// element wise add
			let subOp = SubOperator(inputTensors: [inputA, inputB], outputTensors: [output])
			subOp.compute(computationMode)
		}
		
		//TODO: Implement gradBlock
	}
}

/**
Broadcasting multiplication.

- Note: The `inputTensors` of this operator should just have exactly 2 tensors and the `outputTensors` should just have 1 tensor.
*/
public class BroadcastMultOperator: BroadcastArithmeticOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Override requred inializer
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let metalKernelFuncLabel = ""
		let defaultLable = "BroadcastMultOp"
		self.init(computationDelegate: computationDelegate, inputTensors: nil, outputTensors: nil,
		          metalKernelFuncLabel: metalKernelFuncLabel, operatorLabel: defaultLable)
		self.calculationBlock = { (inputTensors: [Tensor], outputTensors: [Tensor], computationMode: OperatorComputationMode) -> Void in
			let inputA = inputTensors[0]
			let inputB = inputTensors[1]
			let output = outputTensors[0]
			// element wise add
			let multOp = MultOperator(inputTensors: [inputA, inputB], outputTensors: [output])
			multOp.compute(computationMode)
		}
		
		//TODO: Implement gradBlock
	}
}

/**
Broadcasting division.

- Note: The `inputTensors` of this operator should just have exactly 2 tensors and the `outputTensors` should just have 1 tensor.
*/
public class BroadcastDivOperator: BroadcastArithmeticOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Override requred inializer
	///
	/// - Parameter computationDelegate: computationDelegate
	required public convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
		let metalKernelFuncLabel = ""
		let defaultLable = "BroadcastDivOp"
		self.init(computationDelegate: computationDelegate, inputTensors: nil, outputTensors: nil,
		          metalKernelFuncLabel: metalKernelFuncLabel, operatorLabel: defaultLable)
		self.calculationBlock = { (inputTensors: [Tensor], outputTensors: [Tensor], computationMode: OperatorComputationMode) -> Void in
			let inputA = inputTensors[0]
			let inputB = inputTensors[1]
			let output = outputTensors[0]
			// element wise add
			let divOp = DivOperator(inputTensors: [inputA, inputB], outputTensors: [output])
			divOp.compute(computationMode)
		}
		
		//TODO: Implement gradBlock
	}
}
