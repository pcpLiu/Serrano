//
//  utils_test.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 4/11/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import XCTest
@testable import Serrano

class utils_test: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
    }
    
    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }
    
    
    
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    /**
     Target:
        TensorUtils.flattenArrayFloat()
     */
    func testFlattenFunc() {
        let numCase = 100
        let maxDimension = 5
        let maxDimensionRowSize = 5
        
        for _ in 0..<numCase {
            // generate shape
            var shapeArray = [Int]()
            for _ in 0..<randomInt([2, maxDimension]) {
                shapeArray.append(randomInt([1, maxDimensionRowSize]))
            }
            print("Random shape", shapeArray)
            
            // generate random array
            let (randArray, verifyFlat) = generateArrayFromShape(shapeArray: shapeArray, dataType: .int)
            
            let flatArray = TensorUtils.flattenArray(array: randArray, dataType: .int)
            XCTAssertEqual(flatArray, verifyFlat)
            print("Pass!")
        }
        
    }
}
