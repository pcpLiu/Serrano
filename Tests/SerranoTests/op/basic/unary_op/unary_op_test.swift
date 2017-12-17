//
//  unary_op_test.swift
//  SerranoTests
//
//  Created by ZHONGHAO LIU on 6/5/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
import Dispatch
import Metal
@testable import Serrano


public class OperatorDelegateConvUnaryOp: OperatorDelegateConv {
    
    public var compareBlock: (Tensor, Tensor) -> Void
    
    public var gradVerifyBlock: (([String : DataSymbolSupportedDataType], [Tensor]) -> Void)?

    required public convenience init(compareBlock: ((Tensor, Tensor) -> Void)?) {
        let blcok =  {(rawTensor: Tensor, resultTensor: Tensor) -> Void in
            print()
        }
        self.init(block: blcok)
    }
    
    // override this func
    public init(block: @escaping (Tensor, Tensor) -> Void) {
        self.compareBlock = block
        super.init()
    }
    
    override public func compare() {
        XCTAssertTrue(self.resultTensors.first!.count == self.veryfyTensors.first!.count)
        
        for i in 0..<self.resultTensors.count {
            self.compareBlock(self.veryfyTensors[i], self.resultTensors[i])
        }
    }
    
    override public func compareGrads() {
        self.gradVerifyBlock!(self.resultGrads, self.veryfyTensors)
    }
}

public class UnarOpTest<OpDelegate: OperatorDelegateConvUnaryOp, UnaryOp: UnaryOperator>: XCTestCase {
    
    override public func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
        
    }
    
    override public func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
//    func testExample() {
//        // This is an example of a functional test case.
//        // Use XCTAssert and related functions to verify your tests produce the correct results.
//    }
//
//    func testPerformanceExample() {
//        // This is an example of a performance test case.
//        self.measure {
//            // Put the code you want to measure the time of here.
//        }
//    }
    
    public func  testAll() {
        self.testInit()
        self.testOuputShapesCheck()
        self.testInputOutputTensorsCheck()
        self.testCompute()
        self.testGradCompute()
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Test init functions
     */
    func testInit() {
        let numCase = 100
        let op = UnaryOp()
        for _ in 0..<numCase {
            let label = randomString(length: randomInt([2, 10]))
            op.operatorLabel = label
            XCTAssertEqual(label, op.operatorLabel)
            print("label: \(label), \(op.operatorLabel)")
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Test:
     public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?
     */
    func testOuputShapesCheck() {
        let numCase = 100
        for i in 0..<numCase {
            print("Test case \(i+1)")
            let op = UnaryOp()
            
            var shapes = [TensorShape]()
            for _ in 0..<randomInt([2, 5]) {
                shapes.append(randomShape(dimensions: 4, dimensionSizeRange: [1, 10], dataType: .float))
            }
            let outShapes = op.outputShape(shapeArray: shapes)
            XCTAssertNotNil(outShapes)
            XCTAssertEqual(outShapes!.count, shapes.count)
            for i in 0..<outShapes!.count {
                XCTAssertTrue(outShapes![i] == shapes[i])
            }
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
    Test:
    public func inputOutputTensorsCheck() -> (check: Bool, msg: String)
    */
    func testInputOutputTensorsCheck() {
        let numCase = 50
        let op = UnaryOp()
        for i in 0..<numCase {
            print("Test case \(i+1)...")
            
            var inputTensors: [Tensor]? = [Tensor]()
            var outputTensors: [Tensor]? = [Tensor]()
            
            // generate valid input and output tensors
            for _ in 0..<randomInt([1, 4]) {
                inputTensors!.append(randomTensor(dimensions: 2, dimensionSizeRange: [10, 20], dataType: .float))
                outputTensors!.append(randomTensor(fromShape: inputTensors!.last!.shape))
                print("Generate input tensor: \(inputTensors!.last!.description)")
                print("Generate output tensor: \(outputTensors!.last!.description)")
            }
            
            // setup invalid cases
            if i % 2 == 0 {
                let randCase = randomInt([0, 4])
                if randCase % 4 == 0 {
                    //input nil
                    inputTensors = nil
                    print("Set input tensors nil")
                } else if randCase % 4 == 1 {
                    // output nil
                    outputTensors = nil
                    print("Set output tensors nil")
                } else if randCase % 4 == 2 {
                    // count not equal
                    outputTensors!.removeLast()
                    print("Set output tensors num not equal")
                } else {
                    // output shape not valid
                    var shape = outputTensors!.last!.shape.shapeArray
                    shape.removeLast()
                    let invalidShape =  TensorShape(dataType: .float, shape: shape)
                    outputTensors!.removeLast()
                    outputTensors!.append(randomTensor(fromShape: invalidShape))
                    print("Set output tensors shape not valid")
                }
            }
            
            op.inputTensors = inputTensors
            op.outputTensors = outputTensors
            
            let (pass , msg) = op.inputOutputTensorsCheck()
            if i % 2 != 0 {
                XCTAssertTrue(pass)
            } else {
                XCTAssertFalse(pass)
                print(msg)
            }
            
            SerranoResourceManager.globalManager.releaseAllResources()
            print("Finish Test case \(i+1)\n\n")
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Test:
     func compute(withInputTensors tensors:[Tensor], computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode) -> [Tensor]
     func compute(asyncWithInputTensors tensors:[Tensor], computationMode: OperatorComputationMode = SerranoEngine.configuredEngine.defaultComputationMode)
     */
    func testCompute() {
        let caseNum = 5
        let op = UnaryOp()
        
        // configure engine
        let (_, msg) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
        
        // setup delegate
        let delegate = OpDelegate(compareBlock: nil)
        let workingGroup = DispatchGroup()
        delegate.dispatchGroup = workingGroup
        op.computationDelegate = delegate
        
        
        for i in 0..<caseNum {
            print("Test case \(i+1)...")
            
            // generate tensors
            var inputTensors = [Tensor]()
            var outputTensors = [Tensor]()
            
            for _ in 0..<randomInt([1, 3]) {
                let shape = randomShape(dimensions: 2, dimensionSizeRange: [100, 200], dataType: .float)
                inputTensors.append(randomTensor(fromShape: shape))
                outputTensors.append(randomTensor(fromShape: shape))
                print("Generate Input tensor: \(inputTensors.last!.description)")
                print("Generate Output tensor: \(outputTensors.last!.description)")
            }
            
            op.inputTensors = inputTensors
            op.outputTensors = outputTensors
            delegate.veryfyTensors = inputTensors
            
            let inputtensor_hash_values = op.inputTensors!.map { $0._dataMemoryBaseAdrress.hashValue }
            let outputtesnor_hash_values  = op.outputTensors!.map { $0._dataMemoryBaseAdrress.hashValue }
            print("inputtensor_hash_values: \(inputtensor_hash_values)")
            print("outputtesnor_hash_values: \(outputtesnor_hash_values)")
            
            if i % 2 == 0 {
                print("Run on CPU")
                workingGroup.enter()
                op.computeAsync( .CPU)
            } else {
                print("Run on GPU")
                if !SerranoEngine.configuredEngine.hasAvailableGPU() {
                    print("No gpu available, give up Test \(i+1)\n\n\n)")
                    continue
                }
                workingGroup.enter()
                op.computeAsync( .GPU)
            }
            
            workingGroup.wait()
            
            print("Finish Test \(i+1)\n\n\n")
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Test:
     public func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType]
     public func gradComputAsync(_ computationMode: OperatorComputationMode)
     */
     func testGradCompute() {
        let caseNum = 5
        let op = UnaryOp()
        
        // configure engine
        let (_, msg) = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
        
        // setup delegate
        let delegate = OpDelegate(compareBlock: nil)
        let workingGroup = DispatchGroup()
        delegate.dispatchGroup = workingGroup
        op.computationDelegate = delegate
        
        
        for i in 0..<caseNum {
            print("Test case \(i+1)...")
            
            var inputTensors = [Tensor]()
            var outputTensors = [Tensor]()
            
            for _ in 0..<randomInt([1, 3]) {
                let shape = randomShape(dimensions: 2, dimensionSizeRange: [100, 200], dataType: .float)
                inputTensors.append(randomTensor(fromShape: shape))
                outputTensors.append(randomTensor(fromShape: shape))
                print("Generate Input tensor: \(inputTensors.last!.description)")
                print("Generate Output tensor: \(outputTensors.last!.description)")
            }
            
            op.inputTensors = inputTensors
            op.outputTensors = outputTensors
            delegate.veryfyTensors = inputTensors
            
            let inputtensor_hash_values = op.inputTensors!.map { $0._dataMemoryBaseAdrress.hashValue }
            let outputtesnor_hash_values  = op.outputTensors!.map { $0._dataMemoryBaseAdrress.hashValue }
            print("inputtensor_hash_values: \(inputtensor_hash_values)")
            print("outputtesnor_hash_values: \(outputtesnor_hash_values)")
            
            if i % 2 == 0 {
                print("Run on CPU")
                op.compute( .CPU)
                workingGroup.enter()
                op.gradComputAsync(.CPU)
            } else {
                print("Run on GPU")
                if !SerranoEngine.configuredEngine.hasAvailableGPU() {
                    print("No gpu available, give up Test \(i+1)\n\n\n)")
                    continue
                }
                op.compute( .GPU)
                workingGroup.enter()
                op.gradComputAsync(.GPU)
            }
            workingGroup.wait()
            print("Finish Test \(i+1)\n\n\n")
        }
    }
}
