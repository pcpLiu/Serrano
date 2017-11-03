//
//  metal_hardwares_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 4/23/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
import Metal
@testable import Serrano

class metal_hardwares_test: XCTestCase {
    
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
//    }
//    
//    func testPerformanceExample() {
//        // This is an example of a performance test case.
//        self.measure {
//            // Put the code you want to measure the time of here.
//        }
//    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    /**
     Target:
        public static func tensorSizeFitCheck(tensor: Tensor) -> (result: Bool, info: String)
     */
//    func testSizeFitCheck() {
//        let caseNum = 5
//        
//        for i in 0..<caseNum {
//            var shouldSuccess = true
//            var tensor = randomTensor(dimensions: 2, dimensionSizeRange: [1000, 2000], dataType: .int)
//            if i % 3 == 0 {
//                tensor = randomTensorLargeFlat(dimensions: 2, dimensionSizeRange: [5600, 5700], dataType: .int)
//                shouldSuccess = false
//            }
//            print("Test with tensor of size \(tensor.allocatedBytes).")
//            let (result, _) = MetalHardwareChecker.tensorSizeFitCheck(tensor: tensor)
//            XCTAssertTrue(shouldSuccess == result)
//        }
//    }
}
