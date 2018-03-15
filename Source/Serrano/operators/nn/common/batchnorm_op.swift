//
//  batchnorm_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/14/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Dispatch
import Accelerate


/// Corresponding to `BatchNormInfo` in metal file
public struct BatchNormInfo {
    var channelPosition: MetalShort
    var channels: MetalInt
    var inputWidth: MetalInt
    var inputHeight: MetalInt
    var epsilon: MetalFloat
    
    public static func makeBatchNormInfo(channelOrder: TensorChannelOrder, epsilon: Float, input: Tensor) -> BatchNormInfo {
        let (channel, height, width) = parseImgChannelShapeInfo(channelOrder, shapeArray: input.shape.shapeArray)
        return BatchNormInfo(channelPosition: channelOrder.rawValue.metalShort,
                             channels: channel.metalInt,
                             inputWidth: width.metalInt,
                             inputHeight: height.metalInt,
                             epsilon: epsilon.metalFloat)
    }
}

/**
Batch normalization operator.

Normalize on input tensors by each tensor's mean and variance.

## All input tensors should have same dimentsion.

## output tensors  should have same channel order as input

## Currently, only support 2D tensor with channel information.
*/
public class BatchNormOperator: ComputableOperator {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Attributes
    
    public var computationDelegate: OperatorCalculationDelegate?
    
    public var metalKernelFuncLabel: String
    
    public var operatorLabel: String
    
    public var inputTensors: [Tensor]?
    
    public var outputTensors: [Tensor]?
    
    public var disableInputOutputCheck: Bool
    
    public var trainable: Bool
    
    public var mapType: OperatorMappingType
    
    public var inPlaceble: Bool = true
    
    /// Channel order of input
    public var channelOrder: TensorChannelOrder
    
    /// Moving mean.
    /// Should be same dimension with feature axis.
    public var movingMean: Tensor?
    
    /// Moving varience
    /// Should be same dimension with feature axis.
    public var movingVar: Tensor?
    
    /// A small float number to avoid dividing by 0. default is `0.001`
    public var epsilon: Float
    
    /// If use scale tensor
    public var useScale: Bool
    
    /// Scale tensor to output.
    /// Should be same dimension with feature axis.
    public var scale: Tensor?
    
    /// If use offset tensor
    public var useOffset: Bool
    
    /// momentum
    public var momentum: Float 
    
    /// offset tensor to output.
    /// Should be same dimension with feature axis.
    public var offset: Tensor?
    
    /// Input shape.
    /// Should not be `nil` if used in Graph API.
    public var inputShape: TensorShape?
    
    /// Default in training mode
    public var forwadMode: GraphForwardMode = GraphForwardMode.training
    
    internal var _movingMean: ExponentialMovingAverage?
    
    internal var _movingVar: ExponentialMovingAverage?
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - init
    
    public init(channelOrder: TensorChannelOrder = TensorChannelOrder.Last,
                movingMean: Tensor? = nil,
                movingVar: Tensor? = nil,
                useScale: Bool = true,
                scale: Tensor? = nil,
                useOffset: Bool = true,
                offset: Tensor? = nil,
                epsilon: Float = 0.001,
                momentum: Float = 0.99,
                metalKernelFuncLabel: String = "",
                operatorLabel: String = "BatchNormalizationOperator",
                inputTensors: [Tensor]? = nil,
                outputTensors: [Tensor]? = nil,
                disableInputOutputCheck: Bool = false,
                trainable: Bool = false,
                mapType: OperatorMappingType = OperatorMappingType.OneToOne,
                inputShape: TensorShape? = nil) {
        self.channelOrder = channelOrder
        self.movingMean = movingMean
        self.movingVar = movingVar
        self.useScale = useScale
        self.scale = scale
        self.useOffset = useOffset
        self.offset = offset
        self.epsilon = epsilon
        self.momentum = momentum
        self.metalKernelFuncLabel = metalKernelFuncLabel
        self.operatorLabel = operatorLabel
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors
        self.disableInputOutputCheck = disableInputOutputCheck
        self.trainable = trainable
        self.mapType = mapType
        self.inputShape = inputShape
        
        if self.movingMean == nil {
            self._movingMean = nil
        } else {
            self._movingMean = ExponentialMovingAverage(self.movingMean!)
        }
        
        if self.movingVar == nil {
            self._movingVar = nil
        } else {
            self._movingVar = ExponentialMovingAverage(self.movingVar!)
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Conforms to ComputableOperator, forward
    
    /// Outputshape are same as input shapes.
    /// - Note: All input shapes should have same dimension
    ///
    /// - Parameter shapes:
    /// - Returns:
    public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
        guard shapes.count > 0 else {
            SerranoLogging.errorLogging(message: "shapeArray empty",
                                        file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return nil
        }
        
        // first shape validation
        let firstShape = shapes.first!
        guard firstShape.rank == 3 && firstShape.shapeArray[0] > 0 && firstShape.shapeArray[1] > 0
            && firstShape.shapeArray[2] > 0 else {
                SerranoLogging.errorLogging(message: "inputShape \(firstShape.description) is not valid.",
                                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return nil
        }
        
        // all input shape same dimension
        for shape in shapes {
            guard firstShape == shape else {
                SerranoLogging.errorLogging(message: "All input shapes should have same dimension.",
                                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
                return nil
            }
        }
        
        return shapes
    }
    
    public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
        // input not nil
        guard self.inputTensors != nil else {
            return (false, "inputTensors should not be nil.")
        }
        
        // output not nil
        guard self.outputTensors != nil else {
            return (false, "outputTensors should not be nil.")
        }
        
        // input validation
        let inputShapes = self.inputTensors!.map {$0.shape}
        guard let outputShapesCheck = self.outputShape(shapeArray: inputShapes) else {
            return (false, "Input shapes are not valid. Check log for details.")
        }
        
        // output shame matching check
        let outputShapes = self.outputTensors!.map {$0.shape}
        for (outShape, outShapeCheck) in zip(outputShapes, outputShapesCheck) {
            guard outShape == outShapeCheck else {
                 return (false, "Output tensor has invalid shape \(outShape), expecting \(outShapeCheck).")
            }
        }
        
        let (channel, _, _) = parseImgChannelShapeInfo(self.channelOrder, shapeArray: inputShapes.first!.shapeArray)
        
        // validation of movingMean
        guard self.movingMean != nil else {
            return (false, "movingMean is nil.")
        }
        guard self.movingMean!.shape.rank == 1 && self.movingMean!.shape.shapeArray[0] == channel else {
            return (false, "movingMean is not valid. Expect dimension [\(channel)], given \(self.movingMean!.shape.shapeArray).")
        }
        
        // validation of movingVar
        guard self.movingVar != nil else {
            return (false, "movingVar is nil.")
        }
        guard self.movingVar!.shape.rank == 1 && self.movingVar!.shape.shapeArray[0] == channel else {
            return (false, "movingVar is not valid. Expect dimension [\(channel)], given \(self.movingVar!.shape.shapeArray).")
        }
        
        // validation of offset
        if self.useOffset {
            guard self.offset != nil else {
                return (false, "offset is nil.")
            }
            guard self.offset!.shape.rank == 1 && self.offset!.shape.shapeArray[0] == channel else {
                return (false, "offset is not valid. Expect dimension [\(channel)], given \(self.offset!.shape.shapeArray).")
            }
        }
        
        // validation of scale
        if self.useScale {
            guard self.scale != nil else {
                return (false, "scale is nil.")
            }
            guard self.scale!.shape.rank == 1 && self.scale!.shape.shapeArray[0] == channel else {
                return (false, "scale is not valid. Expect dimension [\(channel)], given \(self.scale!.shape.shapeArray).")
            }
        }
        
        return (true, "")
    }
    
    public func compute(_ computationMode: OperatorComputationMode) {
        // check
        if !self.disableInputOutputCheck {
            let (pass, msg) = self.inputOutputTensorsCheck()
            guard pass else {
                SerranoLogging.errorLogging(message: "Operator \(self.operatorLabel) calculation aborted cause invalid input tensors or output tensors: \(msg)", file: "\(#file)", function: "\(#function)", line: "\(#line)")
                fatalError()
            }
        }
        
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
    }
    
    public func computeAsync(_ computationMode: OperatorComputationMode) {
        OperatorUtils.delegateNilWarning(op: self, file: "\(#file)", function: "\(#function)", line: #line)
        self.computationDelegate?.operatorWillBeginComputation(self)
        DispatchQueue.global(qos: .userInitiated).async {
            self.compute(computationMode)
            self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
        }
    }
    
    internal func cpu() {
    if self.forwadMode == GraphForwardMode.training {
            self.trainPrepare()
            self.updateMeanAndVar()
        }
        self.cpu_inference()
    }
    
    internal func gpu() {
        if self.forwadMode == GraphForwardMode.training {
            self.trainPrepare()
            self.updateMeanAndVar()
        }
        self.gpu_inference()
    }
    
    internal func trainPrepare() {
        if self._movingMean == nil {
            self._movingMean = ExponentialMovingAverage(self.movingMean!)
        }
        
        if self._movingVar == nil {
            self._movingVar = ExponentialMovingAverage(self.movingVar!)
        }
    }

    /// Update movingMean and movingVar from all inputTensors
    internal func updateMeanAndVar() {
        //TODO: calcualte mean, var and then update
    }
    
    /// CPU in inference
    internal func cpu_inference() {
        // 1 / sqrt(movingVar + epsilon)
        let movingVarienceReciprocal = OperatorFuncs.copy(self.movingVar!)
        var count = Int32(self.movingVar!.count)
        // movingVar + epsilon
        vDSP_vsadd(movingVarienceReciprocal._dataMemoryBaseAdrress, 1,
                   &self.epsilon,
                   movingVarienceReciprocal._dataMemoryBaseAdrress, 1,
                   vDSP_Length(count))
        // inverse sqrt
        vvrsqrtf(movingVarienceReciprocal._dataMemoryBaseAdrress, movingVarienceReciprocal._dataMemoryBaseAdrress, &count)
        let movingVareReciprocalAddr = movingVarienceReciprocal._dataMemoryBaseAdrress
        
        let movingMeanAddr = self.movingMean!._dataMemoryBaseAdrress
        let featureDim = self.movingMean!.count
        
        // 1-value scale
        let value1Scale = Tensor(repeatingValue: 1.0, tensorShape: self.movingVar!.shape)
        var scaleAddress = value1Scale._dataMemoryBaseAdrress
        if self.useScale {
            scaleAddress = self.scale!._dataMemoryBaseAdrress
        }
        
        // 0-value offset
        let value0Offset = Tensor(repeatingValue: 0.0, tensorShape: self.movingVar!.shape)
        var offsetAddress = value0Offset._dataMemoryBaseAdrress
        if self.useOffset {
            offsetAddress = self.offset!._dataMemoryBaseAdrress
        }
        
        if self.channelOrder == TensorChannelOrder.Last {
            for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
                let featureCount = input.shape.shapeArray.prefix(upTo: input.rank - 1).reduce(1, *)
                let inAddress = input._dataMemoryBaseAdrress
                let outAddress = output._dataMemoryBaseAdrress
                // for each set of dim features, do the normalization
                for i in 0..<featureCount {
                    let thisInAddress = inAddress + i * featureDim
                    let thisOutAddress = outAddress + i * featureDim
                    vDSP_vsbm(thisInAddress, 1,
                              movingMeanAddr, 1,
                              movingVareReciprocalAddr, 1,
                              thisOutAddress, 1,
                              featureDim.vDSPLength)
                    
                    // apply scale and offset
                    vDSP_vma(thisOutAddress, 1,
                             scaleAddress, 1,
                             offsetAddress, 1,
                             thisOutAddress, 1,
                             featureDim.vDSPLength)
                }
            }
        } else {
            for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
                let featureCount = input.shape.shapeArray.suffix(from: 1).reduce(1, *)
                let inAddress = input._dataMemoryBaseAdrress
                let outAddress = output._dataMemoryBaseAdrress
                // for each channel of features, do the normalization
                for i in 0..<featureDim {
                    let thisInAddress = inAddress + i * featureCount
                    let thisOutAddress = outAddress + i * featureCount
                    vDSP_vsbm(thisInAddress, 1,
                              movingMeanAddr + i, 0,
                              movingVareReciprocalAddr + i, 0,
                              thisOutAddress, 1,
                              featureCount.vDSPLength)
                    
                    // apply scale and offset
                    vDSP_vma(thisOutAddress, 1,
                             scaleAddress + i, 0,
                             offsetAddress + i, 0,
                             thisOutAddress, 1,
                             featureCount.vDSPLength)
                }
            }
        }
    }
    
    /// GPU in inference
    internal func gpu_inference() {
        // prepare resource
        let engine = SerranoEngine.configuredEngine
        
        // kernel
        let (kernel, info) = engine.loadGPUKernel(kernelLabel: "batchNorm_inference")
        guard kernel != nil else {
            fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
        }
        
        let meanBuffer = self.movingMean!.gpuBufferResource()
        let varBuffer = self.movingVar!.gpuBufferResource()
        
        var offsetTensor = Tensor(repeatingValue: 0.0, tensorShape: self.movingMean!.shape)
        if self.useOffset {
            offsetTensor = self.offset!
        }
        let offsetBuffer = offsetTensor.gpuBufferResource()
        
        var scaleTensor = Tensor(repeatingValue: 1.0, tensorShape: self.movingMean!.shape)
        if self.useScale {
            scaleTensor = self.scale!
        }
        let scaleBuffer = scaleTensor.gpuBufferResource()
        
        var buffers = [MTLCommandBuffer]()
        for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
            //// command buffers
            let commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
            guard commandBuffer != nil else {
                fatalError("[Serrano] Failed to make new command buffer.")
            }
            
            let inputBufferResource = input.gpuBufferResource()
            let outputBufferResource = output.gpuBufferResource()
            var info = BatchNormInfo.makeBatchNormInfo(channelOrder: self.channelOrder, epsilon: self.epsilon, input: input)
            
            let encoder = commandBuffer!.makeComputeCommandEncoder()
            encoder.setComputePipelineState(kernel!)
            encoder.setBuffer(inputBufferResource.buffer, offset: inputBufferResource.offset, at: 0)
            encoder.setBuffer(outputBufferResource.buffer, offset: outputBufferResource.offset, at: 1)
            encoder.setBuffer(meanBuffer.buffer, offset: meanBuffer.offset, at: 2)
            encoder.setBuffer(varBuffer.buffer, offset: varBuffer.offset, at: 3)
            encoder.setBuffer(scaleBuffer.buffer, offset: scaleBuffer.offset, at: 4)
            encoder.setBuffer(offsetBuffer.buffer, offset: offsetBuffer.offset, at: 5)
            encoder.setBytes(&info, length: MemoryLayout<BatchNormInfo>.stride, at: 6)
            
            let (channels, height, width) = parseImgChannelShapeInfo(self.channelOrder, shapeArray: input.shape.shapeArray)
            let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
                                                    kernel!.maxTotalThreadsPerThreadgroup / kernel!.threadExecutionWidth,
                                                    1)
            let threadgroupsPerGrid = MTLSizeMake((width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                                  (height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                                  channels)
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
            commandBuffer!.commit()
            buffers.append(commandBuffer!)
        }
        
        for buffer in buffers {
            buffer.waitUntilCompleted()
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Conforms to ComputableOperator, backward
    
    public func gradCompute(_ computationMode: OperatorComputationMode) -> [String : DataSymbolSupportedDataType] {
        fatalError("Not implemented")
    }
    
    public func gradComputAsync(_ computationMode: OperatorComputationMode) {
        fatalError("Not implemented")
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Graph API support
    
    public func bindParamSymbols(_ symbols: [GraphSymbol]) {
        var paramsLabels = ["movingMean", "movingVar"]
        if self.useOffset {
            paramsLabels.append("offset")
        }
        if self.useScale {
            paramsLabels.append("scale")
        }
        
        for label in paramsLabels {
            guard let symbol = (symbols.filter {$0.symbolLabel == label}.first) else {
                SerranoLogging.errorLogging(message: "Trying to bind symbol to parameter \(label), but find no matched symbols",
                    file: "\(#file)", function: "\(#function)", line: "\(#line)")
                fatalError("Raised by Serrano. Check log for detail.")
            }
            let tensorSymbol = symbol as! TensorSymbol
            switch label {
            case "movingMean":
                self.movingMean = tensorSymbol.bindedData!.tensorValue
            case "movingVar":
                self.movingVar = tensorSymbol.bindedData!.tensorValue
            case "offset":
                self.offset = tensorSymbol.bindedData!.tensorValue
            case "scale":
                self.scale = tensorSymbol.bindedData!.tensorValue
            default:
                continue
            }
        }
    }
    
    public func paramSymbols() -> [GraphSymbol] {
        guard self.inputShape != nil else {
            SerranoLogging.errorLogging(message: "Fail to generate param symbols cause inputShape is nil",
                                        file: "\(#file)", function: "\(#function)", line: "\(#line)")
            fatalError("Raised by Serrano. Check log for details.")
        }
        
        var symbols = [GraphSymbol]()
        
        var dim = self.inputShape!.shapeArray.last!
        if self.channelOrder == TensorChannelOrder.First {
            dim = self.inputShape!.shapeArray.first!
        }
        
        // moving mean
        let movingMeanSymbol = SerranoTensorSymbol("movingMean",
                                                   dataSource: SymbolDataSource.Parameter,
                                                   shape: TensorShape(dataType: .float, shape: [dim]))
        symbols.append(movingMeanSymbol)
        
        // moving var
        let movingVarSymbol = SerranoTensorSymbol("movingVar",
                                                  dataSource: SymbolDataSource.Parameter,
                                                  shape: TensorShape(dataType: .float, shape: [dim]))
        symbols.append(movingVarSymbol)
        
        if self.useOffset {
            let offsetSymbol = SerranoTensorSymbol("offset",
                                                   dataSource: SymbolDataSource.Parameter,
                                                   shape: TensorShape(dataType: .float, shape: [dim]))
            symbols.append(offsetSymbol)
        }
        
        if self.useScale {
            let scaleSymbol = SerranoTensorSymbol("scale",
                                                   dataSource: SymbolDataSource.Parameter,
                                                   shape: TensorShape(dataType: .float, shape: [dim]))
            symbols.append(scaleSymbol)
        }
        
        return symbols
    }
    
    
}

