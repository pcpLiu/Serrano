//
//  fullyconnected_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Dispatch
import Accelerate

/**
The regular fully connected operator.
Operator do the `1D_dot(inputTensor.flatten, weights) + bias` calculation on all input tensors.

## Weight tensor layout
The `weight` is a 2D tensor with shape: `[n, m]` where :
- `n`: the flattened dimension of `inputTensors`, i.e. same value as `count` of a input tensor.
- `m`: the number of hidden nodes, same value as `numUnits`;

Thus, each column stores the weights of corresponding hidden unit.

## Bias tensor layout
The `bias` is a 1D tensor with shape `[m]` where:
- `m`: the number of hidden nodes, same value as `numUnits`;

Thus, each value in the tensor is the bias value of corresponding hidden unit.

## Input tensors auto flatten
For input tensor with rank `>=2`, the operator will automatically flatten the tensor and then do calcualtion.

## Multiple input tensors
If `inputTensors` has more than 1 tensor object, 
the operator applies calculation on each input tensor independently 
and stores the results in corresponding tensor of `outputTensors`.

- Note: All input tensor should have same `count`.

## Bias enable choice
Bias could be disabled by setting the `biasEnabled` to `false`.

## Batch calculation
This operator itself does not explicitly support batch calculation.
But user can use slice tensor to do the same thing. 
Details can be found in [Slice tensor]() and [Batch calculation with operators]()
*/
public class FullyconnectedOperator: ComputableOperator {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Attributes
    
    /// Operator label. Conforms to `ComputableOperator`
    public var operatorLabel: String
    
    /// This operator does not operator on GPU. Conforms to `ComputableOperator`
    public var metalKernelFuncLabel:String = "Fullyconnected"
    
    /// Conforms to `ComputableOperator`
    public var computationDelegate: OperatorCalculationDelegate?
    
    /// Conforms to `ComputableOperator`
    public var inputTensors: [Tensor]?
    
    /// Conforms to `ComputableOperator`
    public var outputTensors: [Tensor]?
    
    /// If use `bias`. Default is `true`.
    public var biasEnabled: Bool = true
    
    /// Weight tensor.
    ///
    /// ## Shape specific
    /// The tensor should be with shape `[inputDim, numUnits]`.
    ///
    /// - Note: If `weight` is `nil` when calculation, `fataError` will be raised.
    public var weight: Tensor?
    
    /// Bias tensor
    ///
    /// ## Shape specific
    /// The tensor should be with shape `[numUnits]`.
    ///
    /// - Note:  If `bias` is `nil` when calculation, `fataError` will be raised.
    public var bias: Tensor?
    
    /// Number of input units. Must be a positive integer.
    public var inputDim: Int = 1 {
        didSet {
            if numUnits <= 0 {
                SerranoLogging.errorLogging(message: "Attribute inputDim of FullyconnectedOperator must be a positive integer. " +
                    "Given \(numUnits).",
                    file: "\(#file)", function: "\(#function)", line: "\(#line)")
                fatalError()
            }
        }
    }
    
    /// Number of hidden units. Must be a positive integer.
    public var numUnits: Int = 1 {
        didSet {
            if numUnits <= 0 {
                SerranoLogging.errorLogging(message: "Attribute numUnits of FullyconnectedOperator must be a positive integer. " +
                    "Given \(numUnits).",
                    file: "\(#file)", function: "\(#function)", line: "\(#line)")
                fatalError()
            }
        }
    }
    
    /// If `true`, operator will not check the `upGrads`'s shape.
    /// This is used inside framework to speed up in situation we know it will not be wrong.
    /// Cases like auto generated differentiation graph.
    public var disableUpGradShapeCheck: Bool = false
    
    /// If `true`, operator will not call `inputOutputTensorsCheck()` before doing calculation.
    /// This is used inside framework to speed up in situation we know it will not be wrong.
    public var disableInputOutputCheck: Bool = false
    
    /// Indicate if this operator would do paramter update.
    public var trainable: Bool = true
    
    /// The mapping type of this operator.
    /// `OneToOne` for this operator.
    public var mapType: OperatorMappingType {
        get {
            return OperatorMappingType.OneToOne
        }
    }
    
    /// fully connected operator cannot do in-place calculation
    public var inPlaceble: Bool = false
    
    /// If disable using MPS
    public var disabledMPS: Bool = false
    
    /// Default in training mode
    public var forwadMode: GraphForwardMode = GraphForwardMode.training
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Initializers
    
    /// Designated init
    ///
    /// - Parameters:
    ///   - inputDim: inputDim
    ///   - numUnits: numUnits
    ///   - operatorLabel: operatorLabel
    ///   - inputTensors: inputTensors
    ///   - outputTensors: outputTensors
    ///   - computationDelegate: computationDelegate
    ///   - weight: weight
    ///   - bias: bias
    public init(inputDim: Int, numUnits: Int,
                operatorLabel: String = "FullyconnectedOperator",
                inputTensors: [Tensor]? = nil, outputTensors: [Tensor]? = nil,
                computationDelegate: OperatorCalculationDelegate? = nil,
                weight: Tensor? = nil, bias: Tensor? = nil) {
        guard numUnits >= 1 else {
            SerranoLogging.errorLogging(message: "Attribute numUnits of FullyconnectedOperator must be a positive integer. " +
                                                 "Given \(numUnits).",
                                        file: "\(#file)", function: "\(#function)", line: "\(#line)")
            fatalError()
        }
        guard inputDim >= 1 else {
            SerranoLogging.errorLogging(message: "Attribute inputDim of FullyconnectedOperator must be a positive integer. " +
                "Given \(numUnits).",
                file: "\(#file)", function: "\(#function)", line: "\(#line)")
            fatalError()
        }
        self.inputDim = inputDim
        self.numUnits = numUnits
        self.operatorLabel = operatorLabel
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors
        self.computationDelegate = computationDelegate
        self.weight = weight
        self.bias = bias
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Methods
    
    /// The output shape of `FullyConnectedOperator` is decides by `numUnits` and `inputDim`.
    ///
    /// - Parameter shapes: input shapes
    /// - Returns: result shapes
    public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
        // empty check
        guard shapes.count >= 1 else {
            SerranoLogging.errorLogging(message: "Input shapes array is empty.",
                                        file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return nil
        }
        
        // all shapes should equal to inputDim
        for shape in shapes {
            guard self.inputDim == shape.count else {
                SerranoLogging.errorLogging(message: "Input shape's count(\(shape.count)) should equal to inputDim(\(self.inputDim)).",
                                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
                return nil
            }
        }
        
        var outShapes = [TensorShape]()
        for shape in shapes {
            outShapes.append(TensorShape(dataType: shape.dataType, shape: [self.numUnits]))
        }
        return outShapes
    }
    
    
    /// Check validation of `inputTensors`, `outputTensors`, `weight` and `bias`.
    ///
    /// - Returns: check, if pass. msg, error message.
    public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
        // input tensor not nil
        guard self.inputTensors != nil else {
            return (false, "Input tensors array is nil.")
        }
        
        // output tensor not nil
        guard self.outputTensors != nil else {
            return (false, "Output tensors array is nil.")
        }
        
        // weight not nil
        guard self.weight != nil else {
            return (false, "Weight tensor is nil.")
        }
        
        // bias not nil if enabled
        if self.biasEnabled {
            guard self.bias != nil else {
                return (false, "Bias is nil.")
            }
        }
        
        // check input shapes
        let inputShapes = self.inputTensors!.map { $0.shape }
        let outputShapesCheck = self.outputShape(shapeArray: inputShapes)
        guard outputShapesCheck != nil else {
            return (false, "Input tensors are invalid. Check log for details.")
        }
        
        // check output shape
        let outputShapes = self.outputTensors!.map { $0.shape }
        guard outputShapes.count == inputShapes.count else {
            // same count
            return (false, "Input tensors and output tensors have different number of objects. " +
                           "Input tensors contains \(inputShapes.count) tensors, output tensor contains \(outputShapes.count) tensors.")
        }
        for (shape, shapeCheck) in zip(outputShapes, outputShapesCheck!) {
            // same shape
            guard shape == shapeCheck else {
                return (false, "One of output tensors has invalid shape. Expect \(shapeCheck.description), give \(shape.description).")
            }
        }
        
        // check weights dim
        guard self.weight!.rank == 2 else {
            // rank
            return (false, "Weight tensor has invalid rank. Expect 2, given \(self.weight!.rank).")
        }
        guard self.weight!.shape.shapeArray[0] == self.inputDim && self.weight!.shape.shapeArray[1] == self.numUnits else {
            // match with input
            return (false, "Weight shape is invalid. Expect [\(self.inputDim), \(self.numUnits)], given \(self.weight!.shape.shapeArray).")
        }
        
        // check bias dim
        if self.biasEnabled {
            guard self.bias!.rank == 1 else {
                // rank
                return (false, "Bias tensor has invalid rank. Expect 1, given \(self.bias!.rank).")
            }
            guard self.bias!.shape.shapeArray[0] == self.numUnits else {
                // shape match with numUnits
                return (false, "Bias tensor shape is invalid. Expect [\(self.numUnits)], given [\(self.bias!.shape.shapeArray[0])].")
            }
        }
        
        return (true, "")
    }
    
    /// Compute synclly.
    ///
    /// - Parameters:
    ///   - tensors: input tensors
    ///   - computationMode: cmputation mode. If choose `GPU` but haven't configued a GPU SerranoEngine, operator will use `CPU` to compute.
    /// - Returns: result tensors
    public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
        // check
        let (pass, msg) = self.inputOutputTensorsCheck()
        guard pass else {
            SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) aborts calculation cause given invalid data: \(msg)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
            fatalError()
        }
        
        
        // assign flattened shape
        for (inTensor, outTensor) in zip(self.inputTensors!, self.outputTensors!) {
            inTensor.shape = TensorShape(dataType: .float, shape: [1, inTensor.count])
            outTensor.shape = TensorShape(dataType: .float, shape: [1, outTensor.count])
        }
        
        // mult
        let matrixMultOp = MatrixMultOperator()
        matrixMultOp.inputTensors = self.inputTensors!
        matrixMultOp.inputTensors!.append(self.weight!)
        matrixMultOp.outputTensors = self.outputTensors!
        matrixMultOp.disableInputOutputCheck = true
        matrixMultOp.compute(computationMode)
        
        // add bias
        
        if self.biasEnabled {
            self.bias!.shape = TensorShape(dataType: .float, shape: [1, self.bias!.count])
            for tensor in self.outputTensors! {
                tensor &+ self.bias!
            }
            self.bias!.shape = TensorShape(dataType: .float, shape: [self.bias!.count])
        }
        
        // assign back shapes
        for (inTensor, outTensor) in zip(self.inputTensors!, self.outputTensors!) {
            inTensor.shape = TensorShape(dataType: .float, shape: [inTensor.count])
            outTensor.shape = TensorShape(dataType: .float, shape: [outTensor.count])
        }
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
            self.computationDelegate?.operatorWillBeginComputation(self)
            self.compute(computationMode)
            self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
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
    
    /// Bind according to labels.
    ///
    /// -Note: if cannot bind all needed parameters. `fatalError` will be raised.
    public func bindParamSymbols(_ symbols: [GraphSymbol]) {
        var paramsLabels = ["weight"]
        if self.biasEnabled {
            paramsLabels.append("bias")
        }
        
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
            if label == "weight" {
                guard let weightTensor = dataSymbol.bindedData! as? Tensor else {
                    SerranoLogging.errorLogging(message: "Fully connected operator \(self.operatorLabel) is trying to bind to data symbol \(dataSymbol) for weight. But seems this symbol is not a tensor symbol as expected.",
                        file: "\(#file)", function: "\(#function)", line: "\(#line)")
                    fatalError("Faltal error raised by Serrano. Check log for details.")
                }
                self.weight = weightTensor
            } else {
                guard let biasTensor = dataSymbol.bindedData! as? Tensor else {
                    SerranoLogging.errorLogging(message: "Fully connected operator \(self.operatorLabel) is trying to bind to data symbol \(dataSymbol) for bias. But seems this symbol is not a tensor symbol as expected.",
                        file: "\(#file)", function: "\(#function)", line: "\(#line)")
                    fatalError("Faltal error raised by Serrano. Check log for details.")
                }
                self.bias = biasTensor
            }
        }
    }
    
    /// Attribute `weight` as a `TensorSymbol`.
    /// Attribute `bias` as a `TensorSymbol`.
    ///
    /// - Returns:  Array  of GraphSymbol
    public func paramSymbols() -> [GraphSymbol] {
        // These labels are important for bindParamSymbols(:)
        let weight = SerranoTensorSymbol("weight", dataSource: .Parameter, shape: TensorShape(dataType: .float, shape: [self.inputDim, self.numUnits]))
        let bias = SerranoTensorSymbol("bias", dataSource: .Parameter, shape: TensorShape(dataType: .float, shape: [self.numUnits]))

        return [weight, bias]
    }
}
