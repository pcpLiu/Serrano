//
//  nt1_op.swift
//  Serrano
//
//  Created by ZHONGHAO LIU on 12/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Accelerate


/**
 `StatOperaotr` analyze on all input tensors in element-wise way and store statistic result into one tensor.
 Like compute mean, variance of input tensors.
 */
public class StatOperator: ComputableOperator {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Attributes
    
    public var computationDelegate: OperatorCalculationDelegate?
    
    public var metalKernelFuncLabel: String = ""
    
    public var operatorLabel: String
    
    public var inputTensors: [Tensor]?
    
    public var outputTensors: [Tensor]?
    
    public var disableInputOutputCheck: Bool
    
    public var trainable: Bool
    
    public var mapType: OperatorMappingType = OperatorMappingType.Constant
    
    public var inPlaceble: Bool = false
    
    public var forwadMode: GraphForwardMode
    
    /// cpu computation block.
    /// 1st param -- input tensors. 2nd param -- output tensor.
    internal var cpuBlock: (([Tensor], Tensor) -> Void)?
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - init
    
    public init(inputTensors: [Tensor]? = nil,
                outputTensors: [Tensor]? = nil,
                trainable: Bool = true,
                disableInputOutputCheck: Bool = false,
                operatorLabel: String = "StatOperaotr",
                computationDelegate: OperatorCalculationDelegate? = nil,
                forwadMode: GraphForwardMode = GraphForwardMode.training) {
        self.operatorLabel = operatorLabel
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors
        self.trainable = trainable
        self.disableInputOutputCheck = disableInputOutputCheck
        self.computationDelegate = computationDelegate
        self.forwadMode = forwadMode
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Conforms to ComputableOperator, forward
    
    /// Same as input. All input shapes should be same
    ///
    /// - Parameter shapes: shapes
    /// - Returns: output shapes
    public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
        guard shapes.count > 0 else {
            SerranoLogging.errorLogging(message: "shapes are empty.",
                                        file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return nil
        }
        
        for shape in shapes {
            guard shape == shapes[0] else {
                SerranoLogging.errorLogging(message: "Shape in shapes should have same dimension.",
                                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
                return nil
            }
        }
        
        return [shapes[0]]
    }
    
    /// Check:
    /// - input, output not `nil`
    /// - input validation
    /// - output validation
    ///
    /// - Returns: checking result, error message
    public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
        guard self.inputTensors != nil else {
            return (false, "inputTensors is nil")
        }
        
        guard self.outputTensors != nil else {
            return (false, "outputTensors is nil")
        }
        
        guard let outputShapeCheck = self.outputShape(shapeArray: self.inputTensors!.map {$0.shape}) else {
            return (false, "inputTensors is not valid. Check log for details.")
        }
        
        guard self.outputTensors!.count == 1 else {
            return (false, "outputTensors contains \(self.outputTensors!.count) tensors. It should just has 1 tensor.")
        }
        guard self.outputTensors!.first!.shape == outputShapeCheck.first! else {
            return (false, "outputTensor shape is not valid. Expecting \(outputShapeCheck.first!.description), given \(self.outputTensors!.first!.shape).")
        }
        
        return (true, "")
    }
    
    
    /// Compute sync
    ///
    /// - Parameter computationMode: mode
    public func compute(_ computationMode: OperatorComputationMode = OperatorComputationMode.GPU) {
        if !self.disableInputOutputCheck {
            let (pass, msg) = self.inputOutputTensorsCheck()
            guard pass == true else {
                SerranoLogging.errorLogging(message: "Error: \(msg)",
                                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
                fatalError("Raised by serrano. Check log for detail.")
            }
        }
        
        switch computationMode {
        case .CPU:
            self.cpu()
        case .GPU,
             .Auto:
            if SerranoEngine.configuredEngine.hasAvailableGPU() {
                self.gpu()
            } else {
                self.cpu()
            }
        }
    }
    
    /// Compute async
    ///
    /// - Parameter computationMode: mode
    public func computeAsync(_ computationMode: OperatorComputationMode = OperatorComputationMode.GPU) {
        OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
        DispatchQueue.global(qos: .userInitiated).async {
            self.computationDelegate?.operatorWillBeginComputation(self)
            self.compute(computationMode)
            self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
        }
    }
    
    internal func cpu() {
        self.cpuBlock!(self.inputTensors!, self.outputTensors!.first!)
    }
    
    internal func gpu() {
        fatalError("Should be override by subclass")
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Conforms to ComputableOperator, backward
    
    public func gradCompute(_ computationMode: OperatorComputationMode = OperatorComputationMode.GPU) -> [String : DataSymbolSupportedDataType] {
        fatalError("Should be override by subclass")
    }
    
    public func gradComputAsync(_ computationMode: OperatorComputationMode = OperatorComputationMode.GPU) {
        OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
        DispatchQueue.global(qos: .userInitiated).async {
            self.computationDelegate?.operatorWillBeginGradsComputation(self)
            let grads = self.gradCompute(computationMode)
            self.computationDelegate?.operatorDidEndGradsComputation(self, grads: grads)
        }
    }
    
    /// No param to bind.
    ///
    /// - Parameter symbols: symbols
    public func bindParamSymbols(_ symbols: [GraphSymbol]) {
        return
    }
    
    /// No param symbol to return
    ///
    /// - Returns: empty list
    public func paramSymbols() -> [GraphSymbol] {
        return [GraphSymbol]()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

public class StatMedianOperator: StatOperator {
    public override init(inputTensors: [Tensor]? = nil,
                         outputTensors: [Tensor]? = nil,
                         trainable: Bool = true,
                         disableInputOutputCheck: Bool = false,
                         operatorLabel: String = "StatMedianOperaotr",
                         computationDelegate: OperatorCalculationDelegate? = nil,
                         forwadMode: GraphForwardMode = GraphForwardMode.training) {
        super.init(inputTensors: inputTensors, outputTensors: outputTensors, trainable: trainable,
                   disableInputOutputCheck: disableInputOutputCheck, operatorLabel: operatorLabel,
                   computationDelegate: computationDelegate, forwadMode: forwadMode)
        
        // cpu block
        self.cpuBlock = {(inputs: [Tensor], output: Tensor) -> Void in
            for i in 0..<output.count {
                let array = inputs.map { $0.floatValueReader[i] }.sorted()
                if array.count % 2 == 0 {
                    output.floatValueReader[i] = (array[array.count / 2] + array[array.count / 2 - 1]) / 2.0
                } else {
                    output.floatValueReader[i] = array[array.count / 2]
                }
            }
        }
        
         self.metalKernelFuncLabel = "stat_median"
   }
    
    override internal func gpu() {
        //TODO: implementation
        fatalError("not implemented")
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 Compute mean on all inputs and store result in output tensor.
 */
public class StatMeanOperator: StatOperator {
    public override init(inputTensors: [Tensor]? = nil,
                         outputTensors: [Tensor]? = nil,
                         trainable: Bool = true,
                         disableInputOutputCheck: Bool = false,
                         operatorLabel: String = "StatMeanOperator",
                         computationDelegate: OperatorCalculationDelegate? = nil,
                         forwadMode: GraphForwardMode = GraphForwardMode.training) {
        super.init(inputTensors: inputTensors, outputTensors: outputTensors, trainable: trainable,
                   disableInputOutputCheck: disableInputOutputCheck, operatorLabel: operatorLabel,
                   computationDelegate: computationDelegate, forwadMode: forwadMode)
        
        // cpu block
        self.cpuBlock = {(inputs: [Tensor], output: Tensor) -> Void in
            for i in 0..<output.count {
                let array = inputs.map { $0.floatValueReader[i] }
                output.floatValueReader[i] = array.reduce(0, +) / Float(array.count)
            }
        }
        
        self.metalKernelFuncLabel = "stat_mean"
    }
    
    override internal func gpu() {
        let resultTensor = self.outputTensors!.first!
        resultTensor.resetValues(0.0)
        let resultBuffer = resultTensor.gpuBufferResource()
        
        let engine = SerranoEngine.configuredEngine
        let (kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
        guard kernel != nil else {
            fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
        }
        
        var prevCount: MetalInt = 0
        for input in self.inputTensors! {
            let inputBuffer = input.gpuBufferResource()
            var boundary: MetalUInt = MetalUInt(input.count)
            
            let cmdBuffer = engine.serranoCommandQueue!.makeCommandBuffer()
            let encoder = cmdBuffer.makeComputeCommandEncoder()
            encoder.setComputePipelineState(kernel!)
            encoder.setBuffer(inputBuffer.buffer, offset: inputBuffer.offset, at: 0)
            encoder.setBuffer(resultBuffer.buffer, offset: resultBuffer.offset, at: 1)
            encoder.setBytes(&prevCount, length: MemoryLayout<MetalInt>.size, at: 2)
            encoder.setBytes(&boundary, length: MemoryLayout<MetalUInt>.size, at: 3)
            
            let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
                                                    1,
                                                    1)
            let threadgroupsPerGrid = MTLSizeMake((input.count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                                  1,
                                                  1)
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
            
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
            
            prevCount += 1
        }
    }
    
    override public func gradCompute(_ computationMode: OperatorComputationMode = OperatorComputationMode.GPU) -> [String : DataSymbolSupportedDataType] {
        fatalError("Should be override by subclass")
        //TODO: Implementation
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

public class StatVarianceOperator: StatOperator {
    //TODO: Implementation
}
