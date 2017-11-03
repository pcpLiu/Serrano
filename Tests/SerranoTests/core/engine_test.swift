//
//  engine_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 3/23/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
import Metal
@testable import Serrano

class engine_test: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
//    func testExample() {
//        // This is an example of a functional test case.
//        // Use XCTAssert and related functions to verify your tests produce the correct results.
//
////        test_configure_engine()
////        testLoadingKernel()
////        testLoadingKernel()
//    }
    
//    func testPerformanceExample() {
//        // This is an example of a performance test case.
//        self.measure {
//            // Put the code you want to measure the time of here.
//        }
//    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Test function: configureEngine()
     
     - Note: This test needs to be running on physical iOS device with METAL support.
     */
    func test_configure_GPUEngine_default() {
        let engine = SerranoEngine.configuredEngine
        engine.resetEngine()
        let (success, msg) = engine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
        if !success {
            print("No GPU available. GIVE up test")
            return
        }
        XCTAssertTrue(success, "Fail test:\(msg)")
        
        // attributes
        XCTAssertNotNil(engine.GPUDevice)
        XCTAssertNotNil(engine.metalLibrary)
        XCTAssertNotNil(engine.serranoCommandQueue)
    }
    
    func test_configure_GPUEngine_fromUserParams() {
        let engine = SerranoEngine.configuredEngine
        engine.resetEngine()
        let GPUDevice = MTLCreateSystemDefaultDevice()
        guard GPUDevice != nil else {
            print("NO gpu available. Give up test")
            return
        }
        
        let (success, msg) = engine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: GPUDevice!)
        XCTAssertTrue(success, "Fail test:\(msg)")
        
        // attributes
        XCTAssertNotNil(engine.GPUDevice)
        XCTAssertNotNil(engine.metalLibrary)
        XCTAssertNotNil(engine.serranoCommandQueue)
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     Target:
         public func loadGPUKernel(kernelLabel label: String) -> (result: MTLComputePipelineState?, message: String)
     */
    func test_loading_kernels() {
        let engine = SerranoEngine.configuredEngine
        engine.resetEngine()
        let (success, msg) = engine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
        if !success {
            print("No gpu available. Give up test")
            return
        }
        
        // precondition
        XCTAssertTrue(success, "Fail test:\(msg)")
        XCTAssertNotNil(engine.metalLibrary)
        
        // All functions in serrano's default metal library should be successfully load
        print("\n\nTesting function loading....")
        for funcName in engine.metalLibrary!.functionNames {
            let (kernel, msg) = engine.loadGPUKernel(kernelLabel: funcName)
            XCTAssertNotNil(kernel, "Fail test: \(msg)")
            print("Pass! Successfully Load kernel \(funcName)")
        }
        
        
        // Randomly generate function name
        print("\n\nTesting random function loading....")
        for i in 0..<10 {
            let randomFuncName = randomString(length: i+1)
            let (kernel, msg) = engine.loadGPUKernel(kernelLabel: randomFuncName)
            XCTAssertNil(kernel, "Fail test: \(msg)")
            print("Pass! Random function name: \(randomFuncName)")

        }

    }
    
    
}
