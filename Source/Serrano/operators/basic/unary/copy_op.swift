//
//  copy_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 4/16/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Dispatch

/**
 Do copy operation on `inputTensors`.
 */
public class CopyOperator: ComputableOperator {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Attributes
    
    /// Operator label. Conforms to `ComputableOperator`
    public var operatorLabel: String
    
    
    /// This operator does not operator on GPU. Conforms to `ComputableOperator`
    public var metalKernelFuncLabel = ""
    
    /// Conforms to `ComputableOperator`
    public var computationDelegate: OperatorCalculationDelegate?
    
    /// Conforms to `ComputableOperator`
    public var inputTensors: [Tensor]?
    
    // Conforms to `ComputableOperator`
    public var outputTensors: [Tensor]?
    
    /// If `true`, operator will not check the `upGrads`'s shape.
    /// This is used inside framework to speed up in situation we know it will not be wrong.
    /// Cases like auto generated differentiation graph.
    public var disableUpGradShapeCheck: Bool = false
    
    /// If `true`, operator will not call `inputOutputTensorsCheck()` before doing calculation.
    /// This is used inside framework to speed up in situation we know it will not be wrong.
    public var disableInputOutputCheck: Bool = false
    
    /// Indicate if this operator would do paramter update.
    ///
    /// - Note: `CopyOperator` is not trainable.
    public var trainable: Bool = false
    
    /// The mapping type of this operator.
    /// `OneToOne` for this operator.
    public var mapType: OperatorMappingType {
        get {
            return OperatorMappingType.OneToOne
        }
    }
    
    /// Copy operator cannot do in-place calculation
    public var inPlaceble: Bool = false
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Initializers
    
    /// Designated init function
    ///
    /// - Parameters:
    ///   - label: label description
    ///   - delegate: delegate description
    public init(operatorLabel label: String, computationDelegate delegate: OperatorCalculationDelegate?,
                inputTensors: [Tensor]?, outputTensors: [Tensor]?) {
        self.operatorLabel = label
        self.computationDelegate = delegate
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors
    }
    
    /// Convenience init function
    ///
    /// - Parameter label: label description
    public convenience init(operatorLabel label: String = "CopyOperator") {
        self.init(operatorLabel: label, computationDelegate: nil,
                  inputTensors: nil, outputTensors: nil)
    }
    
    public convenience init(inputTensors tensors: [Tensor]) {
        self.init(operatorLabel: "CopyOperator", computationDelegate: nil,
                  inputTensors: tensors, outputTensors: nil)
    }
    
    
    public convenience init(inputTensors tensors: [Tensor], outputTensors: [Tensor]) {
        self.init(operatorLabel: "CopyOperator", computationDelegate: nil,
                  inputTensors: tensors, outputTensors: outputTensors)
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Methods
    
    /// For any input shapes, the output tensor's shapes will be exactly the same
    ///
    /// - Parameter shapes: shapes description
    /// - Returns: return value description
    public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
        return shapes
    }
    
    /// The `inputTensors` should not be `nil`.
    ///
    ///
    public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
        // inputTensors should not be `nil`
        guard self.inputTensors != nil else {
            return (false, "Operator \(self.operatorLabel) should have valid inputTensors.")
        }
        
        guard self.outputTensors != nil else {
            return (false, "Operator \(self.operatorLabel) should have valid outputTensors.")
        }
        
        // if assigned outputTensors, check match
        guard self.inputTensors!.count == self.outputTensors!.count else {
            return (false, "Operator \(self.operatorLabel) should have same amount of input tensors and output tensors. " +
                "Given \(self.inputTensors!.count) inputTensors and \(self.outputTensors!.count) outputTensors")
        }
        
        for i in 0..<self.inputTensors!.count {
            if !(self.inputTensors![i].shape == self.outputTensors![i].shape) {
                return (false, "Operator \(self.operatorLabel) should have same shape of input tensors and output tensors. " +
                    "Given \(self.inputTensors![i].shape) and \(self.outputTensors![i].shape).")
            }
        }
        
        
        return (true, "")
    }
    
    /// Compute sync
    ///
    /// - Note: This operator always runs on CPU.
    ///
    /// - Parameters:
    ///   - tensors: tensors description
    ///   - computationMode: computationMode description. Will be ignored.
    /// - Returns: return value description
    public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
        // check
        let (pass, msg) = self.inputOutputTensorsCheck()
        guard pass else {
            SerranoLogging.errorLogging(message: msg, file: "\(#file)", function: "\(#function)", line: "\(#line)")
            fatalError()
        }
        
        // no gpu mode
        if computationMode == .GPU {
            SerranoLogging.warningLogging(message: "Trying to do Copy operator on GPU which is not implemented.", file: "\(#file)", function: "\(#function)", line: "\(#line)")
        }
        
        // 0, return
        if self.inputTensors!.count == 0 {
            SerranoLogging.warningLogging(message: "Trying to do Copy on empty input.", file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return
        }
        
        // outputTensors is `nil`
        if self.outputTensors == nil {
            SerranoLogging.warningLogging(message: "Trying to do Copy, but output tensor is empty.", file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return
        }
        
        // copy
        for i in 0..<self.outputTensors!.count {
            let copyFromAddress = UnsafeRawPointer(self.inputTensors![i]._dataMemoryBaseAdrress)
            let copyToAddress  = UnsafeMutableRawPointer(self.outputTensors![i]._dataMemoryBaseAdrress)
            memcpy(copyToAddress, copyFromAddress, self.outputTensors![i].count * MemoryLayout<Float>.stride)
        }
    }
    
    
    /// Compute Async
    ///
    /// - Parameters:
    ///   - tensors: tensors description
    ///   - computationMode: computationMode description
    public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)  {
        OperatorUtils.delegateNilWarning(op: self, file: #file, function: #function, line: #line)
        self.computationDelegate?.operatorWillBeginComputation(self)
        DispatchQueue.global(qos: .userInitiated).async {
            self.compute(computationMode)
            self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
        }
    }
    
    /// Calulate grads sync.
    /// The gradient of copy is `1`.
    ///
    /// - Parameters:
    ///   - computationMode: computationMode
    /// - Returns: return `upGrads` if not nil. Else return an empty array.
    public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType] {
        var grads = [String: DataSymbolSupportedDataType]()
        for (i, input) in self.inputTensors!.enumerated() {
            let label = "input_\(i)"
            grads[label] = Tensor(repeatingValue: 1.0, tensorShape: input.shape)
        }
        return grads
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
}
