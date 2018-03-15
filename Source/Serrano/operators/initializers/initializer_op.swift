//
//  initializer_op.swift
//  Serrano
//
//  Created by ZHONGHAO LIU on 12/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Accelerate

//TODO: Implementations

/**
 Operator that do inilialization on input tensors.
 */
public class InitializerOperator: ComputableOperator {
    public var computationDelegate: OperatorCalculationDelegate?
    
    public var metalKernelFuncLabel: String
    
    public var operatorLabel: String
    
    public var inputTensors: [Tensor]?
    
    public var outputTensors: [Tensor]?
    
    public var disableInputOutputCheck: Bool
    
    public var trainable: Bool
    
    public var mapType: OperatorMappingType = OperatorMappingType.OneToOne
    
    public var inPlaceble: Bool = true
    
    public var forwadMode: GraphForwardMode
    
    public init(inputTensors: [Tensor]? = nil,
                outputTensors: [Tensor]? = nil,
                trainable: Bool = true,
                disableInputOutputCheck: Bool = false,
                metalKernelFuncLabel: String = "",
                operatorLabel: String = "InitializerOperator",
                computationDelegate: OperatorCalculationDelegate? = nil,
                forwadMode: GraphForwardMode = GraphForwardMode.training) {
        self.operatorLabel = operatorLabel
        self.metalKernelFuncLabel = metalKernelFuncLabel
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors
        self.trainable = trainable
        self.disableInputOutputCheck = disableInputOutputCheck
        self.computationDelegate = computationDelegate
        self.forwadMode = forwadMode
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Conforms to ComputableOperator, forward
    
    public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
        fatalError()
    }
    
    public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
        fatalError()
    }
    
    public func compute(_ computationMode: OperatorComputationMode) {
        fatalError()
    }
    
    public func computeAsync(_ computationMode: OperatorComputationMode) {
        fatalError()
    }
    
    public func gradCompute(_ computationMode: OperatorComputationMode) -> [String : DataSymbolSupportedDataType] {
        fatalError()
    }
    
    public func gradComputAsync(_ computationMode: OperatorComputationMode) {
        fatalError()
    }
    
    public func bindParamSymbols(_ symbols: [GraphSymbol]) {
        fatalError()
    }
    
    public func paramSymbols() -> [GraphSymbol] {
        fatalError()
    }
}

public class ZerosInitializerOperator {}
public class OnesInitializerOperator {}
public class ConstantInitializerOperator {}
public class RandomNormalInitializerOperator {}
public class RandomUniformInitializerOperator {}
