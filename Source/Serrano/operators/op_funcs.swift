//
//  op_funcs.swift
//  Serrano
//
//  Created by ZHONGHAO LIU on 12/20/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation


/**
 Convenient function APIs of operators.
*/
public class OperatorFuncs {
    
    internal static func calculate(_ op: inout ComputableOperator, input: [Tensor], output: Tensor?) -> Tensor {
        var outputTensor = output
        if outputTensor == nil {
            let inputShapes = input.map {$0.shape}
            guard let outputShape = op.outputShape(shapeArray: inputShapes)?.first else {
                SerranoLogging.errorLogging(message: "Input tensor \(input.description) is not a valid input for operator \(op).",
                    file: "\(#file)", function: "\(#function)", line: "\(#line)")
                fatalError("Raised by Serrano. Check log for detail")
            }
            outputTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(outputShape)
        }
        op.inputTensors = input
        op.outputTensors = [outputTensor!]
        op.compute(.GPU)
        return outputTensor!
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Unary operators
    
    /// Copy
    public static func copy(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = CopyOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }

    /// `sinh`
    public static func sinh(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = SinhOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `tan`
    public static func tan(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = TanOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `tanh`
    public static func tanh(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = TanhOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `sin`
    public static func sin(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = SinOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `cos`
    public static func cos(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = CosOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `cosh`
    public static func cosh(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = CoshOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `arcsin`
    public static func arcsin(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = ArcsinOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `arcsinh`
    public static func arcsinh(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = ArcsinhOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `arccos`
    public static func arccos(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = ArccosOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `arccosh`
    public static func arccosh(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = ArccoshOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `arctan`
    public static func arctan(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = ArctanOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `arctanh`
    public static func arctanh(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = ArctanhOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `degree`
    public static func degree(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = DegreeOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `radian`
    public static func radian(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = RadianOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `floor`
    public static func floor(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = FloorOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `ceil`
    public static func ceil(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = CeilOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `rint`
    public static func rint(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = RintOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `round`
    public static func round(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = RoundOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `abs`
    public static func abs(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = AbsOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `square`
    public static func square(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = SquareOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `sqrt`
    public static func sqrt(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = SqrtOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `exp`
    public static func exp(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = ExpOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `expm1`
    public static func expm1(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = Expm1Operator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `rsqrt`
    public static func rsqrt(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = RsqrtOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `log`
    public static func log(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = LogOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `log2`
    public static func log2(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = Log2Operator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `log10`
    public static func log10(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = Log10Operator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// `log1p`
    public static func log1p(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = Log1pOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Binary operators
    
    /// `add`
    public static func add(_ inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = AddOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    /// `sub`
    public static func sub(_ inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = SubOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    /// `mult`
    public static func mult(_ inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = MultOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    /// `div`
    public static func div(_ inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = DivOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    /// `rdiv`
    public static func rdiv(_ inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = RDivOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    /// `pow`
    public static func pow(_ inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = PowOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Matrix operators
    
    /// Matrix multiplication
    public static func matrixMult(_ inputA: Tensor, _ inputB: Tensor,
                                  transposeA:Bool = false, transeposeB: Bool = false,
                                  output: Tensor? = nil) -> Tensor {
        var op = MatrixMultOperator(transposeA: transposeA, transposeB: transeposeB) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    /// Matrix transpose
    public static func matrixTranspose(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = TransposeOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Broadcast operators
    
    /// `broadcast`
    public static func broadcast(_ shape: TensorShape, _ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = BroadcastOperator(targetShape: shape) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// Broadcasting add
    public static func add(broadcast inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = BroadcastAddOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    /// Broadcasting sub
    public static func sub(broadcast inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = BroadcastSubOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    /// Broadcasting mult
    public static func mult(broadcast inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = BroadcastMultOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    /// Broadcasting div
    public static func div(broadcast inputA: Tensor, _ inputB: Tensor, output: Tensor? = nil) -> Tensor {
        var op = BroadcastDivOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [inputA, inputB], output: output)
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Reduce operators
    
    /// Reduce sum
    public static func reduceSum(_ input: Tensor, axis: [Int], keepDim: Bool = false, output: Tensor? = nil) -> Tensor {
        var op = ReduceSumOperator(axis: axis, keepDim: keepDim) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// Reduce product
    public static func reduceProduct(_ input: Tensor, axis: [Int], keepDim: Bool = false, output: Tensor? = nil) -> Tensor {
        var op = ReduceProductOperator(axis: axis, keepDim: keepDim) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// Reduce max
    public static func reduceMax(_ input: Tensor, axis: [Int], keepDim: Bool = false, output: Tensor? = nil) -> Tensor {
        var op = ReduceMaxOperator(axis: axis, keepDim: keepDim) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// Reduce min
    public static func reduceMin(_ input: Tensor, axis: [Int], keepDim: Bool = false, output: Tensor? = nil) -> Tensor {
        var op = ReduceMinOperator(axis: axis, keepDim: keepDim) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// Reduce mean
    public static func reduceMean(_ input: Tensor, axis: [Int], keepDim: Bool = false, output: Tensor? = nil) -> Tensor {
        var op = ReduceMeanOperator(axis: axis, keepDim: keepDim) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Activiation operators
    
    /// ReLU
    public static func relu(_ input: Tensor, alpha: Float = 0.0, output: Tensor? = nil) -> Tensor {
        var op = ReLUOperator(alpha: alpha) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// sigmoid
    public static func sigmoid(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = SigmoidOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// softplus
    public static func softplus(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = SoftplusOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// softsign
    public static func softsign(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = SoftsignOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// linear
    public static func linear(_ input: Tensor, output: Tensor? = nil) -> Tensor {
        var op = LinearOperator() as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// ELU
    public static func elu(_ input: Tensor, alpha: Float = 1.0, output: Tensor? = nil) -> Tensor {
        var op = ELUOperator(alpha: alpha) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// SELU
    public static func selu(_ input: Tensor, alpha: Float = 1.673263, scale: Float = 1.050701, output: Tensor? = nil) -> Tensor {
        var op = SELUOperator(alpha: alpha, scale: scale) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// Softmax
    public static func softmax(_ input: Tensor, dim: Int = -1, output: Tensor? = nil) -> Tensor {
        var op = SoftmaxOperator(dim: dim) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// LeakyReLU
    public static func leakyReLU(_ input: Tensor, alpha: Float = 0.3, output: Tensor? = nil) -> Tensor {
        var op = LeakyReLUOperator(alpha: alpha) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// ThresholdedReLU
    public static func thresholdedReLU(_ input: Tensor, alpha: Float = 1.0, output: Tensor? = nil) -> Tensor {
        var op = ThresholdedReLUOperator(alpha: alpha) as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - nn operators
    
    /// FullyConnected
    public static func fc(_ input: Tensor, numUnits: Int,
                          weight: Tensor, bias: Tensor? = nil,
                          biasEnabled: Bool = true, output: Tensor? = nil) -> Tensor {
        let fcOp = FullyconnectedOperator(inputDim: input.count, numUnits: numUnits, weight: weight, bias: bias)
        fcOp.biasEnabled = biasEnabled
        var op = fcOp as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// batchnorm
    public static func batchnorm(_ input: Tensor, movingMean: Tensor, movingVar: Tensor,
                                 channelOrder: TensorChannelOrder = TensorChannelOrder.Last,
                                 useScale: Bool = true, scale: Tensor? = nil,
                                 useOffset: Bool = true, offset: Tensor? = nil,
                                 epsilon: Float = 0.001, output: Tensor? = nil) -> Tensor {
        let bnOp = BatchNormOperator(channelOrder: channelOrder, movingMean: movingMean, movingVar: movingVar,
                                     useScale: useScale, scale: scale, useOffset: useOffset, offset: offset, epsilon: epsilon)
        var op = bnOp as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// MaxPool2D
    public static func maxPool2D(_ input: Tensor, kernelSize: [Int], stride: [Int]? = nil,
                                 channelOrder: TensorChannelOrder = TensorChannelOrder.Last,
                                 paddingMode: PaddingMode = PaddingMode.Valid, output: Tensor? = nil) -> Tensor {
        let pool = MaxPool2DOperator(kernelSize: kernelSize, stride: stride, channelPosition: channelOrder, paddingMode: paddingMode)
        var op = pool as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// AvgPool2D
    public static func avgPool2D(_ input: Tensor, kernelSize: [Int], stride: [Int]? = nil,
                                 channelOrder: TensorChannelOrder = TensorChannelOrder.Last,
                                 paddingMode: PaddingMode = PaddingMode.Valid, output: Tensor? = nil) -> Tensor {
        let pool = AvgPool2DOperator(kernelSize: kernelSize, stride: stride, channelPosition: channelOrder, paddingMode: paddingMode)
        var op = pool as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// SumPool2D
    public static func sumPool2D(_ input: Tensor, kernelSize: [Int], stride: [Int]? = nil,
                                 channelOrder: TensorChannelOrder = TensorChannelOrder.Last,
                                 paddingMode: PaddingMode = PaddingMode.Valid, output: Tensor? = nil) -> Tensor {
        let pool = SumPool2DOperator(kernelSize: kernelSize, stride: stride, channelPosition: channelOrder, paddingMode: paddingMode)
        var op = pool as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
    
    /// convolution 2D
    public static func conv2D(_ input: Tensor, numFilters: Int, kernelSize: [Int],
                              weight: Tensor, bias: Tensor? = nil, biasEnabled: Bool = true,
                              stride: [Int] = [1, 1], padMode: PaddingMode = .Valid,
                              channelPosition: TensorChannelOrder = TensorChannelOrder.First,
                              output: Tensor? = nil) -> Tensor {
        let conv2d = ConvOperator2D(numFilters: numFilters, kernelSize: kernelSize, stride: stride, padMode: padMode,
                                    channelPosition: channelPosition, weight: weight, bias: bias, biasEnabled: biasEnabled)
        var op = conv2d as ComputableOperator
        return OperatorFuncs.calculate(&op, input: [input], output: output)
    }
}
