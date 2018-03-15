//
//  unary_op.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 6/2/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Dispatch
import Metal
import Accelerate

/**
 Abstract class define the standard unary operator working flow.
 This class should not be used directly.
 Any class inheritance this class is doing element-wise computation for input tensors.
 */
public class UnaryOperator: ComputableOperator {

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
    
    /// The element compuation block in CPU mode.
    /// In most cases, subclass should just override this part in `init` method instead overriding the whole `cpu()` method.
    /// The firat pointer is the input tensor,
    //// the second is the output tensor
    public var cpuElementComputationBlock: (Tensor, Tensor) -> Void
    
    
    /// The grad compuation block.
    /// parameter: inputTensors,
    public var gradComputationBlock: ([Tensor],  OperatorComputationMode) -> [DataSymbolSupportedDataType]
    
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
    
    /// Unary operator cannot do in-place calculation
    public var inPlaceble: Bool = true

    /// Default in training mode
    public var forwadMode: GraphForwardMode = GraphForwardMode.training
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: Init
    
    /// Designated init function
    ///
    /// - Parameters:
    ///   - label: label description
    ///   - delegate: delegate description
    init(operatorLabel label: String,
                cpuComputeBlock block: @escaping (Tensor, Tensor) -> Void ,
                gradComputationBlock gradBlock: @escaping  ([Tensor], OperatorComputationMode) -> [DataSymbolSupportedDataType],
                metalKernelFuncLabel kernelLabel: String,
                computationDelegate: OperatorCalculationDelegate?,
                inputTensors: [Tensor]?,
                outputTensors: [Tensor]?) {
        self.operatorLabel = label
        self.computationDelegate = computationDelegate
        self.metalKernelFuncLabel = kernelLabel
        self.cpuElementComputationBlock = block
        self.gradComputationBlock = gradBlock
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors
    }
    
    /// Convenience initializer
    /// Subclass should override this function to assign `cpuComputeBlock` and `metalKernelFuncLabel`
    ///
    /// - Parameter computationDelegate: computationDelegate
    public convenience required init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block = { (inputTensor: Tensor, oututTensor: Tensor) -> Void in
            fatalError()
        }
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            fatalError()
        }
        let defaultLabel = "NEED OVERRIDE"
        let kernelLabel = "NEED OVERRIDE"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
    
    
    /// Initial by assign input and output tensors
    ///
    /// - Parameters:
    ///   - inputTensors: inputTensors description
    ///   - outputTensors: outputTensors description
    public convenience init(inputTensors:[Tensor], outputTensors:[Tensor]) {
        self.init(computationDelegate: nil)
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: methods
    
    /// This operator would not do anything about shapes.
    /// Basically, it just return input shapes identically.
    ///
    /// - Note: If the `shapeArray` is empty, function returns `nil`.
    ///
    /// - Parameter shapes: input shapes
    /// - Returns: return shapes
    public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
        guard shapes.count != 0 else {
            SerranoLogging.warningLogging(message: "Input shapes are empty", file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return nil
        }
        return shapes
    }
    
    /// The `inputTensors` and `outputTensors` should be have same count of tensors and each tensor should be has same dimension.
    ///
    ///
    public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
        // input not nil
        guard self.inputTensors != nil else {
            return (false, "Input tensors are nil.")
        }
        
        // output not nil
        guard self.outputTensors != nil else {
            return (false, "Output tensors are nil.")
        }
        
        // same count
        guard self.outputTensors!.count == self.inputTensors!.count else {
            return (false, "Input tensors count is \(self.inputTensors!.count). " +
                           "Output tensors count is \(self.outputTensors!.count)." +
                           "Should be equal.")
        }
        
        // input shape check
        let inputShapes = self.inputTensors!.map { $0.shape }
        let outputShapeCheck = self.outputShape(shapeArray: inputShapes)
        guard outputShapeCheck != nil else {
            return (false, "Input tensors shapes are invalid. Check log for details.")
        }
        
        // output shape check
        let outputShapes = self.outputTensors!.map { $0.shape }
        for (i, t) in zip(outputShapes, outputShapeCheck!).enumerated() {
            guard t.0 == t.1 else {
                return (false, "Expect output tensor shape \(t.1.description), given \(t.0.description) at index \(i)")
            }
        }
        
        return (true, "")
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
    
    /// Compute synclly.
    ///
    /// - Parameters:
    ///   - tensors: input tensors
    ///   - computationMode: cmputation mode. If choose `GPU` but haven't configued a GPU SerranoEngine, operator will use `CPU` to compute.
    /// - Returns: result tensors
    public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) {
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
    
    /// Calulate grads sync.
    /// All unary operator return grads tensor with same number and shape as attribute `inputTensors`.
    ///
    /// - Parameters:
    ///   - computationMode: computationMode
    /// - Returns: return grads tensor
    public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType] {
        let grads =  self.gradComputationBlock(self.inputTensors!, computationMode)
        var result =  [String: DataSymbolSupportedDataType]()
        for (i, grad) in grads.enumerated() {
            result["input_\(i)"] = grad
        }
        return result
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
    
//    /// An unary operator's representation graph just has inputs tensor symbols, output tensor symbols
//    /// and an operator symbol.
//    ///
//    /// - Returns: graph object.
//    public func addedToGraph(with InputSymbols: [TensorSymbol]) -> Graph {
//        let graph = ComputationGraph()
//
//        let outputSymbols =
//
//        return graph
//    }
    
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
    
    /// Use cpu do the inplace computation.
    /// Default, `UnaryOperator` defines a workflow. Subclass just needs to override `cpuElementComputationBlock`.
    /// If subclass needs custom flow, it could just override this function.
    ///
    /// - Parameter tensors: the operation tensors
    internal func cpu() {
        let workGroup = DispatchGroup()
        for tensorIndex in 0..<self.inputTensors!.count {
            workGroup.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                self.cpuElementComputationBlock(self.inputTensors![tensorIndex], self.outputTensors![tensorIndex])
                workGroup.leave()
            }
        }
        
        workGroup.wait()
    }
    
    /// Let GPU call the Metal kernel to do the inplace computation.
    /// Default, `UnaryOperator` defines a workflow. Subclass just needs to override `metalKernelFuncLabel` attribute.
    /// If subclass needs custom flow, it could just override this function.
    ///
    /// - Parameter tensors: the operation tensors
    internal func gpu() {
        // prepare resource
        let engine = SerranoEngine.configuredEngine
        var kernel: MTLComputePipelineState?
        var commandBuffer: MTLCommandBuffer?
        
        // get kernel
           var info = ""
        (kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
        guard kernel != nil else {
            fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
        }
        
        // make command buffer
        commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
        guard commandBuffer != nil else {
            fatalError("[Serrano] Failed to make new command buffer.")
        }
        
        for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
            let inputBufferResource = input.gpuBufferResource()
            let outputBufferResource = output.gpuBufferResource()
            
            // dimension
            var count = MetalUInt(input.count)
            
            // encoder
            let encoder = commandBuffer!.makeComputeCommandEncoder()
            encoder.setComputePipelineState(kernel!)
            encoder.setBuffer(inputBufferResource.buffer, offset: inputBufferResource.offset, at: 0)
            encoder.setBuffer(outputBufferResource.buffer, offset: outputBufferResource.offset, at: 1)
            encoder.setBytes(&count, length: MemoryLayout<MetalUInt>.stride, at: 2)
            
            // dispatch
            let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
                                                    1,
                                                    1)
            let threadgroupsPerGrid = MTLSizeMake((input.count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                                  1,
                                                  1)
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        // commit command buffer
        commandBuffer!.commit()
        commandBuffer!.waitUntilCompleted()
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
Compute element-wise sine on input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class SinOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvsinf(outputAddress, inputAddress, &count)
        }
        
        // cos(x)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            let cosOp = CosOperator(inputTensors: inputs, outputTensors: grads)
            cosOp.disableInputOutputCheck = true
            cosOp.compute(mode)
            return grads
        }
        
        let defaultLabel = "SinOperator"
        let kernelLabel = "Sin"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise tangent on input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class TanOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvtanf(outputAddress, inputAddress, &count)
        }
        
        // 1 / cos(x)^2
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            let cosOp = CosOperator(inputTensors: inputs, outputTensors: grads)
            cosOp.disableInputOutputCheck = true
            cosOp.compute(mode)
            for grad in grads {
                grad &* grad
                1.0 &/ grad
            }
            return grads
        }
        
        let defaultLabel = "TanOperator"
        let kernelLabel = "Tan"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise cosine on input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class CosOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvcosf(outputAddress, inputAddress, &count)
        }
        
        // -sin(x)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            let sinOp = SinOperator(inputTensors: inputs, outputTensors: grads)
            sinOp.disableInputOutputCheck = true
            sinOp.compute(mode)
            for grad in grads {
                grad &* -1.0
            }
            return grads
        }
        
        let defaultLabel = "CosOperator"
        let kernelLabel = "Cos"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise arc sine on input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class ArcsinOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvasinf(outputAddress, inputAddress, &count)
        }
        
        // 1 / sqrt(1-x^2)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            let copyOp = CopyOperator(inputTensors: inputs, outputTensors: grads)
            copyOp.disableInputOutputCheck = true
            copyOp.compute(mode)
            for grad in grads {
                1.0 &- (grad &* grad)
                let sqrtOp = SqrtOperator(inputTensors: [grad], outputTensors: [grad])
                sqrtOp.disableInputOutputCheck = true
                sqrtOp.compute(mode)
                1 &/ grad
            }
            return grads
        }
        let defaultLabel = "ArcsinOperator"
        let kernelLabel = "Arcsin"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise arc cosine on input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class ArccosOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvacosf(outputAddress, inputAddress, &count)
        }
        
        // -1/sqrt(1-x^2)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            let copyOp = CopyOperator(inputTensors: inputs, outputTensors: grads)
            copyOp.disableInputOutputCheck = true
            copyOp.compute(mode)
            for grad in grads {
                1.0 &- (grad &* grad)
                let sqrtOp = SqrtOperator(inputTensors: [grad], outputTensors: [grad])
                sqrtOp.disableInputOutputCheck = true
                sqrtOp.compute(mode)
                -1.0 &/ grad
            }
            return grads
        }
        
        let defaultLabel = "ArccosOperator"
        let kernelLabel = "Arccos"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise arc tangent on input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class ArctanOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvatanf(outputAddress, inputAddress, &count)
        }
        
        // 1 / (1 + x^2)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            // First allocate as managed to speed up incase using GPU with reusing MTLBuffers
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            // copy
            let copyOp = CopyOperator(inputTensors: inputs, outputTensors: grads)
            copyOp.disableInputOutputCheck = true
            copyOp.compute(mode)
            
            for grad in grads {
                1.0 &/ (1.0 &+ (grad &* grad))
            }
            
            
            return grads
        }
        
        let defaultLabel = "ArctanOperator"
        let kernelLabel = "Arctan"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise of input tensors  radians to degrees.
 */
public class DegreeOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            let count = vDSP_Length(output.count)
            var convert:Float = 180.0 / 3.1415926
            vDSP_vsmul(inputAddress, 1, &convert, outputAddress, 1, count)
        }
        
        // 180.0 / 3.1415926
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            let val:Float = 180.0 / 3.1415926
            for input in inputs {
                grads.append(Tensor(repeatingValue: val, tensorShape: input.shape))
            }
            
            
            return grads
        }
        
        let defaultLabel = "DegreeOperator"
        let kernelLabel = "Degree"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise abs values of input tensors.
 */
public class AbsOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            let count = UInt(output.count)
            vDSP_vabs(inputAddress, 1, outputAddress, 1, count)
        }
        
        // x / |x|. Note x != 0. or calcualted value is NaN.
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            // abs
            let absOp = AbsOperator(inputTensors: inputs, outputTensors: grads)
            absOp.disableInputOutputCheck = true
            absOp.compute(mode)
            
            // div
            for (input, grad) in zip(inputs, grads) {
                let rdivOp = DivOperator(inputTensors: [input, grad], outputTensors: [grad])
                rdivOp.disableInputOutputCheck = true
                rdivOp.compute(mode)
            }
            
            
            return grads
        }
        
        let defaultLabel = "AbsOperator"
        let kernelLabel = "Abs"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise radien values of input tensors fro degrees.
 */
public class RadianOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            let count = vDSP_Length(output.count)
            var convert:Float = 3.1415926 / 180.0
            vDSP_vsmul(inputAddress, 1, &convert, outputAddress, 1, count)
        }
        
        // 3.1415926 / 180.0
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            let val:Float = 3.1415926 / 180.0
            for input in inputs {
                grads.append(Tensor(repeatingValue: val, tensorShape: input.shape))
            }
            
            
            return grads
        }
        
        let defaultLabel = "RadienOperator"
        let kernelLabel = "Radien"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise hyperbolic sine  of input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class SinhOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvsinhf(outputAddress, inputAddress, &count)
        }
        
        // (e^x + e^-x) / 2
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            // e^x
            let expOp = ExpOperator(inputTensors: inputs, outputTensors: grads)
            expOp.disableInputOutputCheck = true
            expOp.compute(mode)
            
            // e^-x
            var eNegative = [Tensor]()
            for input in inputs { eNegative.append(-1 * input) }
            expOp.inputTensors = eNegative
            expOp.outputTensors = eNegative
            expOp.compute(mode)
            
            for (grad, negtivate) in zip(grads, eNegative) {
                (grad &+ negtivate) &/ 2
            }
            
            
            return grads
        }
        
        let defaultLabel = "SinhOperator"
        let kernelLabel = "Sinh"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise hyperbolic cosine  of input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class CoshOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvcoshf(outputAddress, inputAddress, &count)
        }
        
        // (e^x - e^-x) / 2
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            // e^x
            let expOp = ExpOperator(inputTensors: inputs, outputTensors: grads)
            expOp.disableInputOutputCheck = true
            expOp.compute(mode)
            
            // e^-x
            var eNegative = [Tensor]()
            for input in inputs { eNegative.append(-1 * input) }
            expOp.inputTensors = eNegative
            expOp.outputTensors = eNegative
            expOp.compute(mode)
            
            for (grad, negtivate) in zip(grads, eNegative) {
                (grad &- negtivate) &/ 2
            }
            
            
            return grads
        }
        
        let defaultLabel = "CoshOperator"
        let kernelLabel = "Cosh"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise hyperbolic tangent  of input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class TanhOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvtanhf(outputAddress, inputAddress, &count)
        }
        
        // 1 / cosh(x) ^ 2
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            let coshOp = CoshOperator(inputTensors: inputs, outputTensors: grads)
            coshOp.disableInputOutputCheck = true
            coshOp.compute(mode)
            
            for grad in grads { 1.0 &/ (grad &* grad) }
            
            
            return grads
        }
        
        let defaultLabel = "TanhOperator"
        let kernelLabel = "Tanh"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise inverse hyperbolic tangent  on input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class ArctanhOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvatanhf(outputAddress, inputAddress, &count)
        }
        
        // 1 / (1 - x^2)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            let squareOp = SquareOperator(inputTensors: inputs, outputTensors: grads)
            squareOp.disableInputOutputCheck = true
            squareOp.compute(mode)
            
            for grad in grads { 1.0 &/ (1.0 &- grad) }
            
            
            
            // release from management pool
            DispatchQueue.global(qos: .userInitiated).async { SerranoResourceManager.globalManager.releaseTensors(grads) }
            return grads
        }
        
        let defaultLabel = "ArctanhOperator"
        let kernelLabel = "Arctanh"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise inverse hyperbolic cosine  on input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class ArccoshOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvacoshf(outputAddress, inputAddress, &count)
        }
        
        // 1 / sqrt(x^2 - 1)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            // square
            let squreOp = SquareOperator(inputTensors: inputs, outputTensors: grads)
            squreOp.disableInputOutputCheck = true
            squreOp.compute(mode)
            
            for grad in grads { grad &- 1.0 }
            let sqrtOp = SqrtOperator(inputTensors: inputs, outputTensors: grads)
            sqrtOp.disableInputOutputCheck = true
            sqrtOp.compute(mode)
            
            for grad in grads { 1.0 &/ grad }
            
            
            return grads
        }
        
        let defaultLabel = "ArccoshOperator"
        let kernelLabel = "Arccosh"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise inverse hyperbolic cosine  on input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class ArcsinhOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvasinhf(outputAddress, inputAddress, &count)
        }
        
        // 1 / sqrt(x^2 +1)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            let squreOp = SquareOperator(inputTensors: inputs, outputTensors: grads)
            squreOp.disableInputOutputCheck = true
            squreOp.compute(mode)
            
            for grad in grads {
                grad &+ 1.0
            }
            
            let sqrtOp = SqrtOperator(inputTensors: grads, outputTensors: grads)
            sqrtOp.disableInputOutputCheck = true
            sqrtOp.compute(mode)
            
            for grad in grads {
                1.0 &/ grad
            }
            
            return grads
        }
        
        let defaultLabel = "ArcsinhOperator"
        let kernelLabel = "Arcsinh"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Operator computes element-wise floor of the input tensors and returen result tensors.
 */
public class FloorOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvfloorf(outputAddress, inputAddress, &count)
        }
        
        // 0 for any inputs.
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            // 0
            for input in inputs { grads.append(Tensor(repeatingValue: 0.0, tensorShape: input.shape)) }
            
            
            return grads
        }
        
        let defaultLabel = "FloorOperator"
        let kernelLabel = "Floor"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Operator computes element-wise floor of the input tensors and returen result tensors.
 */
public class CeilOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvceilf(outputAddress, inputAddress, &count)
        }
        
        // 0 for any inputs.
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            // 0
            for input in inputs { grads.append(Tensor(repeatingValue: 0.0, tensorShape: input.shape)) }
            
            
            return grads
        }
        
        let defaultLabel = "CeilOperator"
        let kernelLabel = "Ceil"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Operator computes element-wise rounded value to the nearest integer of the input.
 */
public class RintOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvnintf(outputAddress, inputAddress, &count)
        }
        
        // 0 for any inputs.
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            // 0
            for input in inputs { grads.append(Tensor(repeatingValue: 0.0, tensorShape: input.shape)) }
            
            
            return grads
        }
        
        let defaultLabel = "RintOperator"
        let kernelLabel = "Rint"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Operator computes element-wise rounded value to the truncating integers of input values.
 */
public class RoundOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvintf(outputAddress, inputAddress, &count)
        }
        
        // 0 for any inputs.
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            // 0
            for input in inputs { grads.append(Tensor(repeatingValue: 0.0, tensorShape: input.shape)) }
            
            
            return grads
        }
        
        let defaultLabel = "RoundOperator"
        let kernelLabel = "Round"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise square values of input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class SquareOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            let count = UInt(output.count)
            vDSP_vsq(inputAddress, 1, outputAddress, 1, count)
        }
        
        // 2x
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            for input in inputs { grads.append( 2.0 * input ) }
            
            
            return grads
        }
        
        let defaultLabel = "SquareOperator"
        let kernelLabel = "Square"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
Compute element-wise reciprocal values of square-root of input tensors.
`1 / sqrt(x)`
 */
public class RsqrtOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvrsqrtf(outputAddress, inputAddress, &count)
        }
        
        // -0.5 * x^(-1.5)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            let powOp = PowOperator()
            powOp.disableInputOutputCheck = true
            for (grad, input) in zip(grads, inputs) {
                let const = Tensor(repeatingValue: -1.5, tensorShape: input.shape)
                powOp.inputTensors = [input, const]
                powOp.outputTensors = [grad]
                powOp.compute(mode)
                -0.5 &* grad
            }
            
            
            return grads
        }
        
        let defaultLabel = "RsqrtOperator"
        let kernelLabel = "Rsqrt"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise square-root values of input tensors.
 */
public class SqrtOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvsqrtf(outputAddress, inputAddress, &count)
        }
        
        // 0.5 * x^(-0.5)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            
            let powOp = PowOperator()
            powOp.disableInputOutputCheck = true
            for (grad, input) in zip(grads, inputs) {
                let const = Tensor(repeatingValue: -0.5, tensorShape: input.shape)
                powOp.inputTensors = [input, const]
                powOp.outputTensors = [grad]
                powOp.compute(mode)
                0.5 &* grad
            }
            
            
            return grads
        }
        
        let defaultLabel = "SqrtOperator"
        let kernelLabel = "Sqrt"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
Compute element-wise `log(1 + x)` values of input tensors.
Base is `e`.
 */
public class Log1pOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvlog1pf(outputAddress, inputAddress, &count)
        }
        
        // 1 / (1+x)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            for input in inputs {
                grads.append(1.0 &/ (1.0 + input))
            }
            
            
            return grads
        }
        
        let defaultLabel = "Log1pOperator"
        let kernelLabel = "Log1p"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise `log2(x)` values of input tensors.
 */
public class Log2Operator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvlog2f(outputAddress, inputAddress, &count)
        }
        
        // 1 / (ln(2) * x)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            let ln2: Float = log(2.0)
            for input in inputs {
                grads.append(1.0 &/ (ln2 * input))
            }
            
            
            return grads
        }
        
        let defaultLabel = "Log2Operator"
        let kernelLabel = "Log2"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}


/**
 Compute element-wise `log10(x)` values of input tensors.
 */
public class Log10Operator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvlog10f(outputAddress, inputAddress, &count)
        }
        
        // 1 / (ln(10) * x)
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            let ln10: Float = log(10.0)
            for input in inputs {
                grads.append(1.0 &/ (ln10 * input))
            }
            
            
            return grads
        }
        
        let defaultLabel = "Log10Operator"
        let kernelLabel = "Log10"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
Compute element-wise `log(x)` values of input tensors.
Base is `e`.
 */
public class LogOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvlogf(outputAddress, inputAddress, &count)
        }
        
        // 1 / x
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            var grads = [Tensor]()
            for input in inputs {
                grads.append(1.0 / input)
            }
            return grads
        }
        
        let defaultLabel = "LogOperator"
        let kernelLabel = "Log"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}

/**
 Compute element-wise `e^x - 1` values of input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class Expm1Operator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvexpm1f(outputAddress, inputAddress, &count)
        }
        
        // e^x
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            let expOp = ExpOperator(inputTensors: inputs, outputTensors: grads)
            expOp.disableInputOutputCheck = true
            expOp.compute(mode)
            
            return grads
        }
        
        let defaultLabel = "Expm1Operator"
        let kernelLabel = "Expm1"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}


/**
 Compute element-wise `exp(x)` values of input tensors.

- Note: This operator may output `NaN` or `Infinite` values and operator would not check these situations.
 */
public class ExpOperator: UnaryOperator {
    
    /// Override init
    ///
    /// - Parameter computationDelegate: delegate
    public required convenience init(computationDelegate: OperatorCalculationDelegate? = nil) {
        let block =  { (input: Tensor, output: Tensor) -> Void in
            let inputAddress = input.contentsAddress
            let outputAddress = output.contentsAddress
            var count = Int32(output.count)
            vvexpf(outputAddress, inputAddress, &count)
        }
        
        // e^x
        let gradBlock = { (inputs: [Tensor], mode: OperatorComputationMode) -> [DataSymbolSupportedDataType] in
            let grads = SerranoResourceManager.globalManager.allocateUnamangedTensors(inputs.map{$0.shape})
            let expOp = ExpOperator(inputTensors: inputs, outputTensors: grads)
            expOp.disableInputOutputCheck = true
            expOp.compute(mode)
            return grads
        }
        
        let defaultLabel = "ExpOperator"
        let kernelLabel = "Exp"
        self.init(operatorLabel: defaultLabel, cpuComputeBlock: block, gradComputationBlock: gradBlock,
                  metalKernelFuncLabel: kernelLabel, computationDelegate: computationDelegate,
                  inputTensors: nil, outputTensors: nil)
    }
}
