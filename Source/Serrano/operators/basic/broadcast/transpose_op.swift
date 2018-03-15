//
//  transpose_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/23/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Accelerate

public struct TransposeMatrixInfo {
    var M: MetalUInt
    var N: MetalUInt
    var stride: MetalUShort
}

/**
Transpose 2D matrix on all `inputTensors` and put transposed values in `outputTensors`.
*/
public class TransposeOperator: ComputableOperator {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Attributes
    
    /// Conforms to `ComputableOperator`
    public var computationDelegate: OperatorCalculationDelegate?
    
    /// Conforms to `ComputableOperator`
    public var metalKernelFuncLabel: String = "Transpose"
    
    /// Conforms to `ComputableOperator`
    public var operatorLabel: String
    
    /// Conforms to `ComputableOperator`
    public var inputTensors: [Tensor]?
    
    /// Conforms to `ComputableOperator`
    public var outputTensors: [Tensor]?
    
    /// If `true`, operator will not check the `upGrads`'s shape.
    /// This is used inside framework to speed up in situation we know it will not be wrong.
    /// Cases like auto generated differentiation graph.
    public var disableUpGradShapeCheck: Bool = false
    
    /// If `true`, operator will not call `inputOutputTensorsCheck()` before doing calculation.
    /// This is used inside framework to speed up in situation we know it will not be wrong.
    public var disableInputOutputCheck: Bool = false
    
    /// Indicate if this operator would do paramter update.
    public var trainable: Bool = false
    
    /// The mapping type of this operator.
    /// `OneToOne` for this operator.
    public var mapType: OperatorMappingType {
        get {
            return OperatorMappingType.OneToOne
        }
    }

    /// Transpose operator cannot do in-place calculation
    public var inPlaceble: Bool = false
    
    /// Default in training mode
    public var forwadMode: GraphForwardMode = GraphForwardMode.training
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Initializers
    
    public init(inputTensors: [Tensor]?, outputTensors: [Tensor]?,
                computationDelegate: OperatorCalculationDelegate?, operatorLabel: String) {
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors
        self.computationDelegate = computationDelegate
        self.operatorLabel = operatorLabel
    }
    
    public convenience init(computationDelegate:  OperatorCalculationDelegate?) {
        self.init(inputTensors: nil, outputTensors: nil, computationDelegate: computationDelegate, operatorLabel: "TransposeOp")
    }
    
    public convenience init(label: String = "TransposeOp") {
        self.init(inputTensors: nil, outputTensors: nil, computationDelegate: nil, operatorLabel: label)
    }
    
    public convenience init(inputTensors: [Tensor], outputTensors: [Tensor]) {
        self.init(inputTensors: inputTensors, outputTensors: outputTensors, computationDelegate: nil, operatorLabel: "TransposeOp")
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Conforms to `ComputableOperator`
    
    
    /// Only accepts 2D matrix.
    ///
    /// - Parameter shapes: input shapes
    /// - Returns: output shapes
    public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
        guard shapes.count != 0 else {
            SerranoLogging.errorLogging(message: "Input shapes contains no shape.",
                                          file: "\(#file)", function: "#\(#function)", line: "\(#line)")
            return nil
        }
        
        var returnShapes = [TensorShape]()
        for shape in shapes {
            guard shape.rank == 2 else {
                SerranoLogging.errorLogging(message: "Shape \(shape) should has rank values as 2, given \(shape.rank)",
                                            file: "\(#file)", function: "#\(#function)", line: "\(#line)")
                return nil
            }
            returnShapes.append(TensorShape(dataType: shape.dataType, shape: [shape.shapeArray[1], shape.shapeArray[0]]))
        }
        
        return returnShapes
    }
    
    
    /// Check if `inputTensors` and `outputTensors` have correct corresponding tensors.
    ///
    /// - Returns: passing and message
    public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
        // input tensors not nil
        guard self.inputTensors != nil else {
            return (false, "Input tensors are nil")
        }
        
        // output tensors not nil
        guard self.outputTensors != nil else {
            return (false, "Output tensors are nil")
        }
        
        // same count
        guard self.inputTensors!.count == self.outputTensors!.count else {
            return (false, "Input tensors and output tensors have different number of elements. Input tensors: \(self.inputTensors!.count). Output tensors: \(self.outputTensors!.count).")
        }
        
        // input shapes check
        let inputShapes = self.inputTensors!.map { $0.shape }
        let outputShapesCheck = self.outputShape(shapeArray: inputShapes)
        guard outputShapesCheck != nil else {
            return (false, "Input tensors' shapes are not valid. Check log for details.")
        }
        
        // comapre shapes
        let outputShapes = self.outputTensors!.map { $0.shape }
        for shapeIndex in 0..<outputShapes.count {
            guard outputShapes[shapeIndex] == outputShapesCheck![shapeIndex] else {
                return (false, "Output tensor \(self.outputTensors![shapeIndex]) does not have valid shape. Expect \(outputShapesCheck![shapeIndex]), given \(outputShapes[shapeIndex]).")
            }
        }
        
        return (true, "")
    }
    
    
    /// Do calcualtion.
    ///
    /// - Parameter computationMode: mode
    public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
        // check
        let (pass, msg) = self.inputOutputTensorsCheck()
        guard pass else {
            SerranoLogging.errorLogging(message: msg, file: "\(#file)", function: "\(#function)", line: "\(#line)")
            fatalError()
        }
        
        self.computationDelegate?.operatorWillBeginComputation(self)
        
        switch computationMode {
        case .GPU:
            if !SerranoEngine.configuredEngine.hasAvailableGPU() {
                SerranoLogging.warningLogging(message: "Serrano Engine has no available configured GPU device. Use CPU doing calculation instead.",
                                              file: "\(#file)", function: "\(#function)", line: "\(#line)")
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
    ///   - upGrds: upGrds
    /// - Returns: return grads tensor
    public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType] {
        //TODO: Implementation
        fatalError("Not implemented")
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
        for i in 0..<self.inputTensors!.count {
            let inputPointer = self.inputTensors![i].contentsAddress
            let outputPointer = self.outputTensors![i].contentsAddress
            vDSP_mtrans(inputPointer, 1, outputPointer, 1,
                        vDSP_Length(self.inputTensors![i].shape.shapeArray[1]),
                        vDSP_Length(self.inputTensors![i].shape.shapeArray[0]))
        }
    }
    
    internal func gpu() {
        // prepare resources
        let engine = SerranoEngine.configuredEngine
        
        // kernel
        let (kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
        guard kernel != nil else {
            fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
        }
        
        // command buffer
        let commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
        guard commandBuffer != nil else {
            fatalError("[Serrano] Failed to make new command buffer.")
        }
        
        // encoder for each computation
        for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
            let inputBufferResource = input.gpuBufferResource()
            let outputBufferResource = output.gpuBufferResource()
            
            // matrix info
            var matrixInfo = TransposeMatrixInfo(M: MetalUInt(output.shape.shapeArray[0]),
                                                 N: MetalUInt(output.shape.shapeArray[1]),
                                                 stride: MetalUShort(MemoryLayout<Float>.stride))
            
            // encoder
            let encoder = commandBuffer!.makeComputeCommandEncoder()
            encoder.setComputePipelineState(kernel!)
            encoder.setBuffer(inputBufferResource.buffer, offset: inputBufferResource.offset, at: 0)
            encoder.setBuffer(outputBufferResource.buffer, offset: outputBufferResource.offset, at: 1)
            encoder.setBytes(&matrixInfo, length: MemoryLayout<TransposeMatrixInfo>.stride, at: 2)
            
            // dispatch
            let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
                                                    kernel!.maxTotalThreadsPerThreadgroup / kernel!.threadExecutionWidth,
                                                    1)
            let threadgroupsPerGrid = MTLSizeMake((output.shape.shapeArray[0] + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                                  (output.shape.shapeArray[1] + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                                  1)
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        commandBuffer!.commit()
        commandBuffer!.waitUntilCompleted()
        
    }
    
}
