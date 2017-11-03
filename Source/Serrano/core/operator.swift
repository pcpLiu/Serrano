//
//  Operator.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 3/17/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal

/**
The delegate support methods tracking the calculation status and result of associated `Operator` object.
Operator could assign attribute `computationDelegate` to a instance conforms to this protocol.
The instanced could track the calculation status of operator.
 */
public protocol OperatorCalculationDelegate {
    /**
     Tell the delegate this `Operator` will begin to calcualte the output tensor
     
     - Parameters:
     - op: The calculation operator
     */
    func operatorWillBeginComputation(_ op: ComputableOperator)
    
    /**
     Tell the delegate this `Operator` has done the calcualtion.
     
     - Parameters:
     - op: The calcaulation operator.
     - tensor: The calcualted output tensors object.
     */
    func operatorDidEndComputation(_ op: ComputableOperator, outputTensors tensors: [Tensor])
	
	/// Tell the delegate this operator will begin grads calculation
	///
	/// - Parameter op: op
	func operatorWillBeginGradsComputation(_ op: ComputableOperator)
	
	/// Tell the delegate this operator end grads claculation
	///
	/// - Parameters:
	///   - op: op
	///   - tensors: grads tensor
	func operatorDidEndGradsComputation(_ op: ComputableOperator, grads: [String: DataSymbolSupportedDataType])

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


public enum OperatorComputationMode {
    case CPU
    case GPU
	case Auto
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

public class OperatorUtils {
    
    /// If target operator's `computationDelegate` is `nil`, logging warning.
    ///
    /// - Parameters:
    ///   - op: target operator conforms to `ComputableOperator`
    ///   - file: checking file name
    ///   - function: checking function name
    ///   - line: checking code line number
    public static func delegateNilWarning<T: ComputableOperator>(op: T, file: String, function: String, line: Int) {
        if op.computationDelegate == nil {
            SerranoLogging.warningLogging(message: "Call async computation without assigning computate delegate. The output will never be fetched.", file: "\(#file)", function: "\(#function)", line: "\(#line)")
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/// Types of operator's input-output mapping.
public enum OperatorMappingType {
	/// N-to-N
	case OneToOne
	
	/// N-to-1
	case Constant
}


/**
 This protocol defines the common computation APIs of `Operator`.
 */
public protocol ComputableOperator {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// MARK: - Attributes
	
	/// Computation delegate.
	///
	/// The assigned delegate can track the computation status and \
	/// result through methods from `OperatorComputationDelegate`.
	///
	/// Usually the delegate is a `Flow` object.
	var computationDelegate: OperatorCalculationDelegate? {get set}
	
	/// Kernel function name
	var metalKernelFuncLabel: String {get}
	
	/// Operator readable label
	var operatorLabel: String {get set}
	
	/// Input tensors to operate
	var inputTensors: [Tensor]? {get set}
	
	/// Output tensors
	var outputTensors: [Tensor]? {get set}
	
	/// If `true`, operator will not call `inputOutputTensorsCheck()` before doing calculation.
	/// This is used inside framework to speed up in situation we know it will not be wrong.
	var disableInputOutputCheck: Bool {get set}

	/// Indicate if this operator would do paramter update
	var trainable: Bool {get set}
	
	/// The mapping type of this operator
	var mapType: OperatorMappingType {get}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// MARK: - Computation realted methods (forward)
	
    /**
	Calulate the output tensor shape given an input tensor shape.
	If the operator cannot operate on the input tensor shape, return `nil`.
     
     - Parameters:
        - shapeArray: An array of `TensorShape`
     
     - Returns: A `TensorShape` object or `nil` if the operator could not operate on the input shape.
     */
    func outputShape(shapeArray shapes:[TensorShape]) -> [TensorShape]?
    
    /// Check if the input tensors and output tensors's shape matching
    ///
    /// - Returns: `check` is `true` if match; `msg` error info if not match
    func inputOutputTensorsCheck() -> (check: Bool, msg: String)
	
    /// Compute sync
    ///
    /// - Parameter computationMode: computationMode
    func compute(_ computationMode: OperatorComputationMode)
	
    /// Compute the output tensor asyncally. Output result will be passed to `computationDelegate`.
    ///
    /// - note: If the `computationDelegate` is nil, the computed output will be lost.
    func computeAsync(_ computationMode: OperatorComputationMode)
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// MARK: - Differentiaion realted methods (backward)
	
	
	/// Compute grads from output against each input tensor and involving parameters.
	///
	/// ## Identify corresponding input
	/// The returned label of data could be used to identify its correspoding input
	/// following below rules:
	/// - __Input tensor__. `input_{i}` where `i` is the corresponding input tensor's index in `inputTensors`
	///	- __Parameter__. The parameter's name.
	///
	/// - Note: Operator will not store grads tensor. If the returned value not used,
	///         grads will lost.
	///
	/// - Parameter computationMode: computationMode description
	/// - Returns: grads list for each input tensor and involving parameters with label
	func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType]
	
	/// Compute async grads from output against each input tensor and involving parameters.
	///
	/// - Parameter computationMode: computationMode
	/// - Parameter upGrds: Optional. Grads from upstream operators in a Graph computation.
	func gradComputAsync(_ computationMode: OperatorComputationMode)
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// MARK: - Support symbolic graph computation
	
	/// This function is called when add an operator to a `Graph` with function `operation()`.
	/// A `Graph` object is returned by this function representing the inner structure of this operator.
	/// Some complex operators may consist of other simple operators and we want explicitly show the structure
	/// in the graph it added to.
	/// So the returned graph will be merged into the graph calling this function.
	///
	/// - Parameter InputSymbols: input symbols for this graph
	/// - Returns: a graph object
//	func addedToGraph(with InputSymbols: [TensorSymbol]) -> Graph
	
	/// Bind data from symbol to parameter of this operator.
	///
	/// - Parameters:
	///   - symbols: binded symbols
	func bindParamSymbols(_ symbols: [GraphSymbol])
	
	/// An array of `GraphSymbol` for this operator's parameters.
	/// This array may be empty if operator needs no parameter.
	/// This function is used in constructing computaion graph.
	///
 	/// - Returns: An array.
	func paramSymbols() -> [GraphSymbol]

}

