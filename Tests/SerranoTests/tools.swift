//
//  test_utils.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 3/18/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import XCTest
import Accelerate
@testable import Serrano

public func randomInt(_ range: [Int]) -> Int {
    return Int(arc4random_uniform(UInt32(range[1] - range[0]))) + range[0]
}

public func randomFloat() -> Float {
    return Float(drand48() * 2.0 - 1.0)
}

public func randomDouble() -> Double {
    return drand48() * 2.0 - 1.0 // -1.0 to 1.0
}

public func randomString(length: Int) -> String {
    let letters : NSString = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    let len = UInt32(letters.length)
    
    var randomString = ""
    
    for _ in 0 ..< length {
        let rand = arc4random_uniform(len)
        var nextChar = letters.character(at: Int(rand))
        randomString += NSString(characters: &nextChar, length: 1) as String
    }
    
    return randomString
}



/// Generate random shape object
///
/// - Parameters:
///   - dimensions: # of dimension
///   - dimensionSizeRange: dimension size
/// - Returns: `TensorShape`
public func randomShape(dimensions: Int, dimensionSizeRange:[Int], dataType: TensorDataType) -> TensorShape {
    var shape = [Int]()
    for _ in 0..<dimensions {
        shape.append(randomInt(dimensionSizeRange))
    }
    
    return TensorShape(dataType: dataType, shape: shape)
}


/**
 Generate random array from shape
 */
public func generateArrayFromShape(shapeArray: [Int], dataType: TensorDataType) -> ([Any], [Float]) {
    var array = [Any]()
    var flatArray = [Float]()
    let currDimSize = shapeArray[0]
    
    // last dim
    if shapeArray.count == 1 {
        for _ in 0..<currDimSize {
            var val: Float = 0.0
            switch dataType {
                case .double:
                    let rand = randomDouble()
                    array.append(rand)
                    val = Float(rand)
                case .int:
                    let rand = randomInt([1, 1000])
                    array.append(rand)
                    val = Float(rand)
                case .float:
                    let rand = randomFloat()
                    array.append(rand)
                    val = Float(rand)
            }
            flatArray.append(val)
        }
    } else {
        let nextShape = Array(shapeArray[1..<shapeArray.count])
        for _ in 0..<currDimSize {
            let result = generateArrayFromShape(shapeArray: nextShape, dataType: dataType)
            array.append(result.0)
            flatArray.append(contentsOf: result.1)
        }
    }
    
    return (array, flatArray)
}


public func generateFlatArrayFromShape(shapeArray: [Int], dataType: TensorDataType) ->  [Float] {
    var flatArray = Array(repeating: Float(0.0), count: shapeArray.reduce(1, *))
	
	switch dataType {
	case .float:
		for i in 0..<flatArray.count { flatArray[i] = randomFloat() }
	case .double:
		for i in 0..<flatArray.count { flatArray[i] = Float(randomDouble()) }
	case .int:
		for i in 0..<flatArray.count { flatArray[i] = Float(randomInt([-100, 100])) }
	}
	
	
    return flatArray
}


/**
 Generate random tensor
 */
public func randomTensor(dimensions: Int, dimensionSizeRange:[Int], dataType: TensorDataType) -> Tensor {
    let shape = randomShape(dimensions: dimensions, dimensionSizeRange: dimensionSizeRange, dataType: dataType)
    let tensor =  Tensor(repeatingValue: 0.0, tensorShape: shape)
    tensor.reloadData(fromFlatArray: generateFlatArrayFromShape(shapeArray: shape.shapeArray, dataType: dataType), tensorShape: shape)
    return tensor
}

public func randomTensor(fromShape shape: TensorShape) -> Tensor {
    let tensor = Tensor(repeatingValue: 0.0, tensorShape: shape)
	tensor.reloadData(fromFlatArray: generateFlatArrayFromShape(shapeArray: shape.shapeArray, dataType: shape.dataType), tensorShape: shape)
    return tensor
}


public class OperatorDelegateConv: OperatorCalculationDelegate {
	public func operatorDidEndGradsComputation(_ op: ComputableOperator, grads: [String : DataSymbolSupportedDataType]) {
		
	}
	
	
	public func operatorDidEndGradsComputation(_ op: ComputableOperator, outputTensors tensors: [Tensor]) {
		fatalError("")
	}
	
	public func operatorWillBeginGradsComputation(_ op: ComputableOperator) {
		fatalError()
	}

    
    public var resultTensors: [Tensor] = [Tensor]()
    public var veryfyTensors: [Tensor] = [Tensor]()
    
    public var dispatchGroup: DispatchGroup? = nil
            
    public func operatorWillBeginComputation(_ op: ComputableOperator)  {
        print("Operator will begin computation \(op.operatorLabel)")
    }
    
    public func operatorDidEndComputation(_ op: ComputableOperator, outputTensors tensors: [Tensor]){
        print("Operator did end computation \(op.operatorLabel)")
        self.resultTensors = tensors
		
		print("start compare")
		self.compare()
		print("Finish compare")
		
        self.dispatchGroup!.leave()
    }
	
    
    public func compare() {
        fatalError("Need override")
    }
}










