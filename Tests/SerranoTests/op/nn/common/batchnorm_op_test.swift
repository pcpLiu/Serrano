//
//  batchnorm_op_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 11/27/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

public class OperatorDelegateBatchNormOp: OperatorDelegateConv {
    
    public var batchnormOp: BatchNormOperator? = nil
    
    func getVerifyTensor(_ input: Tensor) -> Tensor {
        let channelOrder = self.batchnormOp!.channelOrder
        let mean = self.batchnormOp!.movingMean!
        let variance = self.batchnormOp!.movingVar!
        let offset = self.batchnormOp!.offset!
        let scale = self.batchnormOp!.scale!
        
        let outTensor = Tensor(repeatingValue: 0.0, tensorShape: input.shape)
        let (featureDim, height, width) = parseImgChannelShapeInfo(channelOrder, shapeArray: input.shape.shapeArray)
        for i in 0..<height {
            for j in 0..<width {
                for c in 0..<featureDim {
                    if channelOrder == .Last {
                        outTensor[i, j, c] = (input[i, j, c] - mean[c])/variance[c]
                    } else {
                        outTensor[c, i, j] = (input[c, i, j] - mean[c])/variance[c]
                    }
                    
                    if self.batchnormOp!.useScale {
                        if channelOrder == .Last {
                            outTensor[i, j, c] *= scale[c]
                        } else {
                            outTensor[c, i, j] *= scale[c]
                        }
                    }
                    
                    if self.batchnormOp!.useOffset {
                        if channelOrder == .Last {
                            outTensor[i, j, c] += offset[c]
                        } else {
                            outTensor[c, i, j] += offset[c]
                        }
                    }
                }
            }
        }
        
        
        return outTensor
    }
    
    override public func compare() {
        for (input, output) in zip(self.batchnormOp!.inputTensors!, self.batchnormOp!.outputTensors!) {
            let verifyOutput = self.getVerifyTensor(input)
            let verifyReader = verifyOutput.floatValueReader
            let outputReader = output.floatValueReader
            for i in 0..<outputReader.count {
                XCTAssertEqual(verifyReader[i], outputReader[i], accuracy: abs(verifyReader[i]*0.001))
            }
        }
    }
}

class BatchNormOpTest: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Target:
     init....
     */
    func testInit() {
        let numCase = 100
        for i in 0..<numCase {
            print("Test case \(i+1)...")
            
            // channel order
            var channelOrder = TensorChannelOrder.First
            if randomInt([0, 10]) % 3 == 0 {
                channelOrder = TensorChannelOrder.Last
            }
            
            // useoffset
            var useOffset = true
            if randomInt([0, 10]) % 2 == 0 {
                useOffset = false
            }
            
            // useScale
            var useScale = true
            if randomInt([0, 10]) % 2 == 0 {
                useScale = false
            }
            
            var inTraining = false
            if randomInt([0, 10]) % 2 == 0 {
                inTraining = false
            }
            
            let movingMean = Tensor.randomTensor(TensorShape(dataType: .int, shape: [3, 2, 3]))
            let movingVar = Tensor.randomTensor(TensorShape(dataType: .int, shape: [3, 2, 3]))
            let scale = Tensor.randomTensor(TensorShape(dataType: .int, shape: [ 3]))
            let offset = Tensor.randomTensor(TensorShape(dataType: .int, shape: [3]))
            let epsilon = randomFloat()
            let label = randomString(length: 6)
            let inputShape = randomShape(dimensions: 2, dimensionSizeRange: [1, 3], dataType: .int)
            
            let op = BatchNormOperator(channelOrder: channelOrder, movingMean: movingMean, movingVar: movingVar,
                                       useScale: useScale, scale: scale, useOffset: useOffset, offset: offset,
                                       epsilon: epsilon,operatorLabel: label,
                                      inputShape: inputShape, inTraining: inTraining)
            
            XCTAssertEqual(movingMean, op.movingMean!)
            XCTAssertEqual(movingVar, op.movingVar!)
            XCTAssertEqual(channelOrder, op.channelOrder)
            XCTAssertEqual(useScale, op.useScale)
            XCTAssertEqual(useOffset, op.useOffset)
            XCTAssertEqual(scale, op.scale!)
            XCTAssertEqual(offset, op.offset!)
            XCTAssertEqual(epsilon, op.epsilon)
            XCTAssertEqual(label, op.operatorLabel)
            XCTAssertEqual(inputShape, op.inputShape!)
            XCTAssertEqual(inTraining, op.inTraining)
            
            print("Finish Test case \(i+1)\n")
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Target:
     public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
     */
    func testOutputShape() {
        let numCase = 100
        
        let op = BatchNormOperator()
        
        for i in 0..<numCase {
            print("Test case \(i+1)...")
            
            // generate valid input shapes
            var inputShape = [TensorShape]()
            let shapeArray = [3, 5, 5]
            for _ in 0..<randomInt([2, 5]) {
                inputShape.append(TensorShape(dataType: .float, shape: shapeArray))
            }
            
            // setup invalid cases
            if i % 2 != 0 {
                let randCase = randomInt([0, 3])
                if randCase == 0 {
                    // input shape empty
                    inputShape.removeAll()
                    print("Invalid case: input shape empty")
                } else if randCase == 1 {
                    // input shape not valid
                    inputShape[0] = TensorShape(dataType: .float, shape: [1, 2])
                    print("Invalid case: input shape not valid")
                } else {
                    // input shape not same dimension
                    inputShape[1] = TensorShape(dataType: .float, shape: [23, 233,14])
                    print("Invalid case: input shape not same")
                }
            }
            
            let outputShape = op.outputShape(shapeArray: inputShape)
            if i % 2 == 0 {
                XCTAssertNotNil(outputShape)
            } else {
                XCTAssertNil(outputShape)
            }
            
            print("Finish Test case \(i+1)\n")
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Target:
     public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
     */
    func testInputOutputTensorsCheck() {
        let numCase = 100
        
        for i in 0..<numCase {
            print("Test case \(i+1)...")
            
            // channel order
            var channelOrder = TensorChannelOrder.First
            if randomInt([0, 10]) % 3 == 0 {
                channelOrder = TensorChannelOrder.Last
            }
            
            // generate valid input, output tensors
            var input: [Tensor]? = [Tensor]()
            var output: [Tensor]? = [Tensor]()
            let channel = randomInt([3, 6])
            var shape = TensorShape(dataType: .int, shape: [channel, randomInt([10, 20]), randomInt([10, 20])])
            if channelOrder == .Last {
                shape = TensorShape(dataType: .int, shape: [randomInt([10, 20]), randomInt([10, 20]), channel])
            }
            for _ in 0..<randomInt([1, 4]) {
                input!.append(randomTensor(fromShape: shape))
                output!.append(randomTensor(fromShape: shape))
            }
            
            // random mean
            var mean: Tensor? = randomTensor(fromShape: TensorShape(dataType: .float, shape: [channel]))
            
            // random variance
            var variance: Tensor? = randomTensor(fromShape: TensorShape(dataType: .float, shape: [channel]))
            
            // random scale
            var scale: Tensor? = randomTensor(fromShape: TensorShape(dataType: .float, shape: [channel]))
            
            // random offset
            var offset: Tensor? = randomTensor(fromShape: TensorShape(dataType: .float, shape: [channel]))
            
            // useoffset
            let useOffset = true
            
            // useScale
            let useScale = true
            
            // setup invalid cases
            if i % 2 != 0 {
                let randCase = randomInt([0, 12])
                if randCase == 0 {
                    // input nil
                    input = nil
                    print("Invalid case: input nil")
                } else if randCase == 1 {
                    // output nil
                    output = nil
                    print("Invalid case: output nil")
                } else if randCase == 2 {
                    // input not valid
                    input![0] = Tensor.randomTensor(TensorShape(dataType: .float, shape: [1]))
                    print("Invalid case: input not valid")
                } else if randCase == 3 {
                    // output not valid
                    output![0] = Tensor.randomTensor(TensorShape(dataType: .float, shape: [1]))
                    print("Invalid case: output not valid")
                } else if randCase == 4 {
                    // movingMean is nil
                    mean = nil
                    print("Invalid case: movingMean nil")
                } else if randCase == 5 {
                    // movingMean is invalid
                    mean = Tensor.randomTensor(TensorShape(dataType: .float, shape: [channel + 3]))
                    print("Invalid case: movingMean is invalid")
                } else if randCase == 6 {
                    // movingVar is nil
                    variance = nil
                    print("Invalid case: movingVar nil")
                } else if randCase == 7 {
                    // movingMean is invalid
                    variance = Tensor.randomTensor(TensorShape(dataType: .float, shape: [channel + 3]))
                    print("Invalid case: movingVar is invalid")
                } else if randCase == 8 {
                    // offset is nil
                    offset = nil
                    print("Invalid case: offset nil")
                } else if randCase == 9 {
                    // offset is invalid
                    offset = Tensor.randomTensor(TensorShape(dataType: .float, shape: [channel + 3]))
                    print("Invalid case: offset is invalid")
                } else if randCase == 10 {
                    // scale is nil
                    scale = nil
                    print("Invalid case: scale nil")
                } else if randCase == 11 {
                    // scale is invalid
                    scale = Tensor.randomTensor(TensorShape(dataType: .float, shape: [channel + 3]))
                    print("Invalid case: scale is invalid")
                }
            }
            
            let op = BatchNormOperator()
            op.inputTensors = input
            op.outputTensors = output
            op.offset = offset
            op.scale = scale
            op.movingVar = variance
            op.movingMean = mean
            op.useScale = useScale
            op.useOffset = useOffset
            op.channelOrder = channelOrder
            
            let (valid, msg) = op.inputOutputTensorsCheck()
            if i % 2 == 0 {
                XCTAssertTrue(valid)
            } else {
                XCTAssertFalse(valid)
                print(msg)
            }
            
            print("Finish Test case \(i+1)\n")
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     public func compute(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
     public func computeAsync(_ computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
     internal func cpu()
     internal func gpu()
     */
    func testCompute_inference() {
        let numCase = 20
        // gpu initial
        _ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU)
        
        let workGroup = DispatchGroup()
        let delegate = OperatorDelegateBatchNormOp()
        delegate.dispatchGroup = workGroup
        
        for i in 0..<numCase {
            print("Test case \(i+1)...")
            
            // channel order
            var channelOrder = TensorChannelOrder.First
            if randomInt([0, 10]) % 3 == 0 {
                channelOrder = TensorChannelOrder.Last
            }
            print("channelOrder: ",channelOrder)
            
            // input tensors
            var input = [Tensor]()
            var output = [Tensor]()
            let channel = randomInt([3, 6])
            if i < 16 {
                var shape = TensorShape(dataType: .int, shape: [channel, randomInt([10, 20]), randomInt([10, 20])])
                if channelOrder == .Last {
                    shape = TensorShape(dataType: .int, shape: [randomInt([10, 20]), randomInt([10, 20]), channel])
                }
                print("Tensor shape", shape)
                for _ in 0..<randomInt([1, 4]) {
                    input.append(randomTensor(fromShape: shape))
                    output.append(randomTensor(fromShape: shape))
                }
            } else {
                var shape = TensorShape(dataType: .int, shape: [channel, randomInt([100, 200]), randomInt([100, 200])])
                if channelOrder == .Last {
                    shape = TensorShape(dataType: .int, shape: [randomInt([100, 200]), randomInt([100, 200]), channel])
                }
                print("Tensor shape", shape)
                input.append(randomTensor(fromShape: shape))
                output.append(randomTensor(fromShape: shape))
            }
            
            // random mean
            let mean = randomTensor(fromShape: TensorShape(dataType: .float, shape: [channel]))
            let variance = randomTensor(fromShape: TensorShape(dataType: .float, shape: [channel]))
            let scale = randomTensor(fromShape: TensorShape(dataType: .float, shape: [channel]))
            print("scale: ", scale.flatArrayFloat())
            
            let offset = randomTensor(fromShape: TensorShape(dataType: .float, shape: [channel]))
            print("offset: ", offset.flatArrayFloat())
            
            // useoffset
            var useOffset = true
            if randomInt([0, 10]) % 2 == 0 {
                useOffset = false
            }
            print("useOffset: ",useOffset)
            
            // useScale
            var useScale = true
            if randomInt([0, 10]) % 2 == 0 {
                useScale = false
            }
            print("useScale: ",useScale)
            
            let batchnormOp = BatchNormOperator()
            batchnormOp.channelOrder = channelOrder
            batchnormOp.movingMean = mean
            batchnormOp.movingVar = variance
            batchnormOp.useScale = useScale
            batchnormOp.useOffset = useOffset
            batchnormOp.scale = scale
            batchnormOp.offset = offset
            batchnormOp.computationDelegate = delegate
            batchnormOp.inTraining = false
            batchnormOp.disableInputOutputCheck = true
            batchnormOp.inputTensors = input
            batchnormOp.outputTensors = output
            delegate.batchnormOp = batchnormOp
            
            if i % 2 == 0 {
                print("Run CPU")
                workGroup.enter()
                batchnormOp.computeAsync(.CPU)
            } else {
                if !SerranoEngine.configuredEngine.hasAvailableGPU() {
                    print("No gpu available, give up gpu test \n\n")
                    continue
                }
                workGroup.enter()
                batchnormOp.computeAsync(.GPU)
            }
            
            workGroup.wait()
            
            print("Finish Test case \(i+1)\n\n")
        }
    }
}
