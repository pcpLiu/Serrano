//
//  tensor.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 3/15/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Accelerate
import Metal 
#if  !((arch(i386)  || arch(x86_64)) && os(iOS)) // prevent build error on simulaor
	import MetalPerformanceShaders
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK:

/**
 Defined the supported data type stored in Tensor object.
 
 Should be matching with supported scalar data types in
 [Apple's Metal specification](https://developer.apple.com/metal/metal-shading-language-specification.pdf) (section 2.1)
 
 Currently compatible with v1.2 with types:
 
 - int32:   A signed two’s complement 32-bit integer
 - uint32:  An unsigned 32-bit integer
 - float16: A 16-bit floating-point
 - float32: A 32-bit floating-point
 */
public enum TensorDataType {
    ///A signed two’s complement 32-bit integer
    case int
    case float
    case double
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK:

public class TensorUtils {
    /**
     Randomlly generate a `Tensor` object.
     */
    public static func generateRandomTensor(targetTensorShape shape: TensorShape, maxVal: Float) -> Tensor{
        var array = [SupportedScalarDataType]()
        switch shape.dataType {
        case .double:
            for _ in 0..<shape.shapeArray.reduce(1, *) {
                array.append(Double(arc4random_uniform(UInt32(maxVal))) / Double(maxVal))
            }
        case .float:
            for _ in 0..<shape.shapeArray.reduce(1, *) {
                array.append(Float(arc4random_uniform(UInt32(maxVal))) / maxVal)
            }
        case .int:
            for _ in 0..<shape.shapeArray.reduce(1, *) {
                array.append(Int(arc4random_uniform(UInt32(maxVal))) as SupportedScalarDataType)
            }
        }
        return Tensor(dataArray: array, tensorShape: shape)
    }
    
    
    /**
     For a `swift` n-d array, flat it into a 1-d array.
     
     
     - Parameters:
     - array: The given raw array data. We don't preassume it's shape.
     
     - Returns: An array of `SupportedTensorInitialDataType` elements. It could be empty.
     
     - Note: The returned array could be empty if all elements of the passed in array are not supported types.
     */
    public static func flattenArray(array: [Any], dataType type: TensorDataType) -> [Float] {
        var flatten = [Float]()
        for element in array {
            if element is SupportedNestedType {
                flatten.append(contentsOf: TensorUtils.flattenArray(array: element as! [Any], dataType: type))
            } else if element is SupportedScalarDataType {
                switch type {
                case .int:
                    flatten.append(Float(element as! Int))
                case .float:
                    flatten.append(element as! Float)
                case .double:
                    flatten.append(Float(exactly: element as! Double)!)
                }
            } else {
                SerranoLogging.warningLogging(message: "Meet unsupported datatype: \(type(of: element))", file: "\(#file)", function: "\(#function)", line: "\(#line)")
            }
        }
        
        return flatten
    }
}


infix operator .==: ComparisonPrecedence


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK:

/**
 The tensor shape description. Specify the shape and data type of a `Tensor` data object.
 
 ## dataType
 Indicate the initial set data type.
 
 ## shapeArray
 The `shape` attrribute defines the dimension of a `Tensor` object. In `serrano`, we follow `row-marjor`
 order to store and access elements in a `Tensor` object and each __row__ is represented as an array.
 For a given shape array with `n` indices `[i_0, i_1, ..., i_(n-1)]`,  each index from `i_0` to `i_(n-2)` defines the number of rows in its
 previous dimension. The last index define the number of elements in its previous dimention.
 For example, a `TensorShape` object with `shpae` as `[2, 1, 3]`. 
 It's 1st dimension has `2` rows in which each row has `1` row with 3 elements.
 
 
 User should be clear and unserstanding what a `Tensor` object _looks_ like when they passing in a `TensorShape` argument.
 For example, a `Tensor` object with shape `[2, 3]`, it can be visulized as a nested array like below:
 ```
     // shape [2, 3]
     [
        [1.0, 0.5, 1.3],
        [2.0, 4.2, 6.7],
     ]
 ```
 And a typical real-world example, a 3-channel RGB image data could be represented with shape `[3, image_hight, image_width]`:
 ```
     [
        // R channel frame
        [
            [232, ..., 123], // (# of Int elements) = image_width
            .
            .
            .
            [113, ..., 225]
        ], // (# of Array elements) = image_hight
     
        // G channel frame
        [
            [232, ..., 123],
            .
            .
            .
            [113, ..., 225]
        ],
     
        // B channel frame
        [
            [232, ..., 123],
            .
            .
            .
            [113, ..., 225]
        ]
    ]
 ```

## Equatable
Two TensorShape objects are __equal__ (`==`)if they have the same `shape`.

Two TensorShape objects are __dot equal__ (`.==`) if they have the same `shape` and same `dataType`.

## Rank is `0`
If a `Tensor` object's shapeArray has `0` rank, it indicates that it just contains a __scalar__ value.
 */
public struct TensorShape: Equatable, Comparable {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Properties
    
    /// Data type
    var dataType:TensorDataType
    
    /// Shape array
    var shapeArray: [Int]
	
	/// Rank of dimensions
	var rank: Int {
		get {
			return self.shapeArray.count
		}
	}
	
	/// Element count
	var count: Int {
		get {
			return self.shapeArray.reduce(1, *)
		}
	}
	
	/// Description
	var description: String {
		get {
			return "TensorShape(rank: \(self.rank), shape: \(self.shapeArray), dataType:\(self.dataType))"
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializer
	
	
	/// Public init for out init
	///
	/// - Parameters:
	///   - dataType: dataType 
	///   - shape: shape
	public init(dataType: TensorDataType, shape: [Int]) {
		self.dataType = dataType
		self.shapeArray = shape
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
    /// Check if `shape` array is valid
    public func shapeVerify() -> Bool {
        if self.shapeArray.count == 0 {
            return false
        }
        
        for dimensionSize in self.shapeArray {
            if dimensionSize <= 0 {
                return false
            }
        }
        return true
    }
    
    /// Get `shape` array in reversed
    public func reversedShapArray() -> [Int] {
        return Array(self.shapeArray.reversed())
    }
	
	/// Get transposed shape of current shape.
	/// - Note: only works for shapes with rank values as `2`. Otherwise, `fatalError` will be throws.
	public func transposed() -> TensorShape {
		guard self.rank == 2 else {
			SerranoLogging.errorLogging(message: "Trying to get transposed shape from shape with rank \(self.rank)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		return TensorShape(dataType: self.dataType, shape: [self.shapeArray[1], self.shapeArray[0]])
	}
	
    
    /// Tow shapes dot equals when have same `dataType` and `==`.
    ///
    /// - Parameters:
    ///   - shapeA: shape A
    ///   - shapeB: shape B
    public static func .==(shapeA: TensorShape, shapeB: TensorShape) -> Bool {
        return shapeA.dataType == shapeB.dataType && shapeA == shapeB
    }
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Equable protocol
	
	/// Two shap equals when has same rank and same dimensions.
    ///
    /// - Parameters:
    ///   - shapeA: shapeA description
    ///   - shapeB: shapeB description
    /// - Returns: return value description
    public static func ==(shapeA: TensorShape, shapeB: TensorShape) -> Bool {
        return shapeA.rank == shapeB.rank && shapeA.shapeArray.elementsEqual(shapeB.shapeArray)
    }
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Comparable protocol
	
	
	/// `shapeA` larger than `shapeB` if:
	///	- `shapeA` has a larger `rank` or
	///	- `shapeA` and `shapeB` has same rank while `shapeA` has a larger element `count`
	///
	/// - Parameters:
	///   - shapeA: shapeA description
	///   - shapeB: shapeB description
	/// - Returns: return value description
	public static func >(shapeA: TensorShape, shapeB: TensorShape) -> Bool {
		if shapeA.rank > shapeB.rank {
			return true
		} else if shapeA.rank == shapeB.rank {
			return shapeA.count > shapeB.count
		} else {
			return false
		}
	}
	
	/// `shapeA` larger than or equal to `shapeB` if:
	///		- `shapeA` has larger `rank` or
	///		- `shapeA` and `shapeB` has same rank while `shapeA` has a larger or same`count`
	///
	/// - Parameters:
	///   - shapeA: shapeA description
	///   - shapeB: shapeB description
	/// - Returns: return value description
	public static func >=(shapeA: TensorShape, shapeB: TensorShape) -> Bool {
		if shapeA.rank > shapeB.rank {
			return true
		} else if shapeA.rank == shapeB.rank {
			return shapeA.count >= shapeB.count
		} else {
			return false
		}
	}
	
	/// `shapeA` less than `shapeB` if:
	///		- `shapeA` has smaller `rank` or
	///		- `shapeA` and `shapeB` has same rank while `shapeA` has a smaller `count`
	///
	/// - Parameters:
	///   - shapeA: shapeA description
	///   - shapeB: shapeB description
	/// - Returns: return value description
	public static func <(shapeA: TensorShape, shapeB: TensorShape) -> Bool {
		if shapeA.rank < shapeB.rank {
			return true
		} else if shapeA.rank == shapeB.rank {
			return shapeA.count < shapeB.count
		} else {
			return false
		}
	}
	
	/// `shapeA` less than or equal to `shapeB` if:
	///		- `shapeA` has smaller `rank` or
	///		- `shapeA` and `shapeB` has same rank while `shapeA` has a smaller or same `count`
	///
	/// - Parameters:
	///   - shapeA: shapeA description
	///   - shapeB: shapeB description
	/// - Returns: return value description
	public static func <=(shapeA: TensorShape, shapeB: TensorShape) -> Bool {
		if shapeA.rank < shapeB.rank {
			return true
		} else if shapeA.rank == shapeB.rank {
			return shapeA.count <= shapeB.count
		} else {
			return false
		}
	}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: Operators

/// Custom Operator: in-place addition of two tensors or tensor-scalar, result stored in left tensor.
infix operator &+: AdditionPrecedence

/// Custom Operator: in-place substraction of two tensors or tensor-scalar, result stored in left tensor.
infix operator &-: AdditionPrecedence

/// Custom Operator: in-place multiplication of two tensors or tensor-scalar, result stored in left tensor.
infix operator &*: MultiplicationPrecedence

/// Custom Operator: in-place division of two tensors or tensor-scalar, result stored in left tensor.
infix operator &/: MultiplicationPrecedence



/// Custom Operator: broadcast addition between two tensors.
infix operator .+: AdditionPrecedence

/// Custom Operator: broadcast substraction between two tensors.
infix operator .-: AdditionPrecedence

/// Custom Operator: broadcast multiplication between two tensors.
infix operator .*: MultiplicationPrecedence

/// Custom Operator: broadcast division between two tensors.
infix operator ./: MultiplicationPrecedence



/// Custom Operator: in-place broadcast addition of two tensors or tensor-scalar, result stored in left tensor.
infix operator .&+: AdditionPrecedence

/// Custom Operator: in-place broadcast substraction of two tensors or tensor-scalar, result stored in left tensor.
infix operator .&-: AdditionPrecedence

/// Custom Operator: in-place broadcast multiplication of two tensors or tensor-scalar, result stored in left tensor.
infix operator .&*: MultiplicationPrecedence

/// Custom Operator: in-place broadcast division of two tensors or tensor-scalar, result stored in left tensor.
infix operator .&/: MultiplicationPrecedence

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//MARK:

/**
A `Tensor` object is a n-dimension page-aligned data container with fixed-size memory space.
The attribute `shape` specifying the dimension information of a `Tensor` object.

## All elements stored as `Float` values
No matter what type of array user passing in: `Double` or `Float` or `Int`, inside a `Tensor` object
values will be converted and stored as `Float` (32-bit signle precision).
`Serrano` chose this because 32-bit `Float` is the maximum precise floating data type `Metal` could support (v1.2).


## Inside Memory Layout
Inside a `Tensor` object, it maintains a _virtual_ 1-d array as a contiguous memory space manually allocated.
Elements are stored following `row-major` order. Details can be found in `TensorShape` docs.


## Tensor-Tensor arithmetic operators
Besides [Operators]() which contains many math and NN operations, `Tensor` object itself implements/overloads common
arithmetic operators.
Tensor object support element-wise arithmetic operation with or without broadcasting.
Also it supports in-place operation choice.

### Element-wise operation without broadcasting:
- `+` Addition without broadcasting
- `-` Substraction without broadcasting
- `*` Multiplication without broadcasting
- `/` Division without broadcasting

### Element-wise in-place operation:
- `&+` Addition without broadcasting
- `&-` Substraction without broadcasting
- `&*` Multiplication without broadcasting
- `&/` Division without broadcasting

### Element-wise operation with broadcasting:
- `.+` Addition without broadcasting
- `.-` Substraction without broadcasting
- `.*` Multiplication without broadcasting
- `./` Division without broadcasting

### Element-wise in-place operation with broadcasting:
- `.&+` Addition without broadcasting
- `.&-` Substraction without broadcasting
- `.&*` Multiplication without broadcasting
- `.&/` Division without broadcasting

Example usage:
```swift
/// Element wise addition without broadcasting
let tensorA = Tensor(repeatingValue: 1.0,
					 tensorShape: TensorShape(dataType:.float, shape: [2, 5]))
let tensorB = Tensor(repeatingValue: 2.0,
					 tensorShape: TensorShape(dataType:.float, shape: [2, 5]))
let result1 = tensorA + tensorB

/// Element wise in-place addition without broadcasting
let result2 = tensorA &+ tensorB
print(result2 == tensorA) // true

/// Element wise addition with broadcasting
let tensorD = Tensor(repeatingValue: 2.0,
				     tensorShape: TensorShape(dataType:.float, shape: [1, 5]))
let result3 = tensorA .+ tensorD

/// Element wise in-place addition with broadcasting
let resunt4 = tensorA .&+ tensorD
print(result4 == tensorA) // true
```

## Directly used as TensorSymbol
`Tensor` conforms to `TensorSymbol` protocol. 
So a tensor object could be used as a symbol participating in graph computation.

 */
public class Tensor: Hashable, Equatable, TensorSymbol {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Properties
	
    /// shape
    internal var _shape: TensorShape
    
    /// Allocated aligned memory size in bytes
    internal var _allocatedSize: Int
    
    /// The base address of allocated memory.
    internal var _dataMemoryBaseAdrress: UnsafeMutablePointer<Float>
	
    /// Convenience `Float` reader
    internal var _elementReader: UnsafeMutableBufferPointer<Float>
    
    /// How many elements can store
    internal var _capacity: Int
	
	/// Slice marker.
	/// Indicate if this tensor object is a slice from another tensor.
	/// If `true`, this object is just a reference slice and should not own the memeory.
	internal var _sliceMarker: Bool = false
	
	/// The root tensor of a sliced tensor
	internal var _sliceRootTensor: Tensor? = nil
	
	/// The index array of this slice object in raw tensor.
	/// `nil` if not a sliced tensor
	internal var _sliceIndex: [Int]?
	
	/// `MTLBuffer` binded to this tensor.
	/// - Note: If the tensor is sliced, this is `nil`.
 	internal var _mtlbuffer: MTLBuffer?
    
    /// Count of data elements tored
    public var count: Int {
        get {
            return self._shape.shapeArray.reduce(1, *)
        }
    }
    
    
    /// Dimension of the array
    public var dimension: Int {
        get {
            return self._shape.shapeArray.count
        }
    }
    
    /// How many elements can store
    public var capacity: Int {
        get {
            return self._capacity
        }
    }
    
    
    /// Shape
    public var shape: TensorShape {
        get {
            return self._shape
        }
        set(newShape) {
            self._shape = newShape
        }
    }
    
    /// Allocated memeory bytes
    public var allocatedBytes: Int {
        get {
            return self._allocatedSize
        }
    }
    
    /// Readable description of a object
    public var description: String {
        get {
            return "TensorObject at \(self._dataMemoryBaseAdrress), Allocated bytes: \(self._allocatedSize), Element count: \(self._shape.shapeArray.reduce(1, *)), shape: \(self._shape.shapeArray)"
        }
    }
    
    /// A readable label for distinguishing different tensors. Serrano does not check label unique.
    public var label: String = "Tensor"
	
    /// Base address of allocated memeory space
    ///
    /// - Warning:
    /// User could reuse a `Tensor` object with this pointer by assigning new values.
    /// However, it must be very careful with the boundary.
    public var contentsAddress: UnsafeMutablePointer<Float> {
        get {
            return self._dataMemoryBaseAdrress
        }
    }
    
    /// Float value reader pointer
    public var floatValueReader: UnsafeMutableBufferPointer<Float> {
        get {
            return self._elementReader
        }
    }
	
	/// Rank
	public var rank: Int {
		get {
			return self._shape.shapeArray.count
		}
	}
	
	/// Slice marker.
	/// Indicate if this tensor object is a slice from another tensor.
	/// If `true`, this object is just a reference slice and should not own the memeory.
	public var isSliceTensor: Bool {
		get {
			return self._sliceMarker
		}
	}
	
	/// The root tensor of a sliced tensor
	/// - Note: `nil` if `_sliceMarker` is `false`
	public var sliceRootTensor: Tensor? {
		get {
			return self._sliceRootTensor
		}
	}
	
	/// The index of this slice object in raw tensor.
	/// `nil` if not a sliced tensor
	public var sliceIndex: [Int]? {
		get {
			return self._sliceIndex
		}
	}
	
	/// Conforms to protocol `TensorSymbol`.
	public var symbolType: SymbolType = SymbolType.Tensor
	
	/// Conforms to protocol `TensorSymbol`.
	public var UID: String = ""
	
	/// Conforms to protocol `TensorSymbol`.
	public var symbolLabel: String = ""
	
	/// Inbound symbols list. Conforms to `GraphSymbol`.
	/// To prevent from cycle reference, we dynamic constructing this attribute from `inBoundsWeak`.
	public var inBounds: [GraphSymbol] {
		get {
			return self.inBoundsWeak.filter {$0.value != nil}.map {$0.value!}
		}
		set(bounds) {
			for symbol in bounds {
				self.addToInBound(symbol)
			}
		}
	}
	
	/// Outbound symbols list. Conforms to `GraphSymbol`.
	/// To prevent from cycle reference, we dynamic constructing this attribute from `inBoundsWeak`.
	public var outBounds: [GraphSymbol] {
		get {
			return self.outBoundsWeak.filter {$0.value != nil}.map {$0.value!}
		}
		set(bounds) {
			for symbol in bounds {
				self.addToOutBound(symbol)
			}
		}
	}
	
	/// Weak reference array of inbounds objects
	internal var inBoundsWeak: [WeakSerranoGraphSymbol] = [WeakSerranoGraphSymbol]()
	
	/// Weak reference array of outbounds objects
	internal var outBoundsWeak: [WeakSerranoGraphSymbol] = [WeakSerranoGraphSymbol]()
		
	/// __Read-only__. Conforms to protocol `TensorSymbol`.
	///
	/// - Note: If a tensor object used as a `TensorSymbol`, atrribute `bindedData` could not be modifed.
	public var bindedData: DataSymbolSupportedDataType? {
		get {
			return self
		}
		set {
			SerranoLogging.errorLogging(message: "Could not modify the bindedData of a TensorSymbol represented from a tensor object.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
	}
	
	/// Conforms to protocol `TensorSymbol`.
	public var dataSource: SymbolDataSource = SymbolDataSource.Other
	
	/// If differentiable
	public var updatable = false
	
	/// Current grad
	public var currentGrad: DataSymbolSupportedDataType?
	
	/// If enabled history grads recording.
	/// Default is `false`.
	public var historyGradsEnabled = false
	
	/// history grads
	public var historyGrads: [DataSymbolSupportedDataType] = [Tensor]()
	
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - subscript
    
    
    /// Check if index is valid for fetching a single element
    ///
    /// - Parameter index: index description
    /// - Returns: return value description
    public func indexIsValid(_ index: [Int]) -> Bool {
        if index.count != self._shape.shapeArray.count {
            return false
        } else {
            for dimension in 0..<index.count {
                if index[dimension] >= self._shape.shapeArray[dimension] || index[dimension] < 0  {
                    return false
                }
            }
            return true
        }
    }
    
    
    
    /// Get offset in terms of element counting from valid index
    ///
    /// - Parameter index: valid index list
    /// - Returns: offset
    internal func offSetFromIndex(_ index: [Int]) -> Int {
        var offset = 0
        for i in 0..<(index.count-1) {
            offset += index[i] * self._shape.shapeArray.suffix(from: i+1).reduce(1, *)
        }
        offset += index.last!
        return offset
    }
	
	/// Custom subscript to fetch single `Float` element
	///
	/// - Parameter index: Indices list
	subscript(_ index: Int...) -> Float {
		set(newValue) {
			guard indexIsValid(index) else {
				fatalError("[serrano]Invalid index \(index) for tensor with shape \(self._shape.shapeArray)")
			}
			let offset = self.offSetFromIndex(index)
			self._elementReader[offset] = newValue
		}
		
		get {
			guard indexIsValid(index) else {
				fatalError("[serrano]Invalid index \(index) for tensor with shape \(self._shape.shapeArray)")
			}
			let offset = self.offSetFromIndex(index)
			return  (self._dataMemoryBaseAdrress + offset).pointee
		}
	}
	
    /// Custom subscript to fetch single `Float` element
    ///
    /// - Parameter index: Indices list
    subscript(_ index: [Int]) -> Float {
        set(newValue) {
            guard indexIsValid(index) else {
                fatalError("[serrano]Invalid index \(index) for tensor with shape \(self._shape.shapeArray)")
            }
            let offset = self.offSetFromIndex(index)
            self._elementReader[offset] = newValue
        }
        
        get {
            guard indexIsValid(index) else {
                fatalError("[serrano]Invalid index \(index) for tensor with shape \(self._shape.shapeArray)")
            }
            let offset = self.offSetFromIndex(index)
            return  (self._dataMemoryBaseAdrress + offset).pointee
        }
    }
	
	
	/// Internal use, already checked boundary before fetch or set
	///
	/// - Warning: May cause unexpected result or fatal error if `index` is not valid.
	///
	/// - Parameter index: index array
	public subscript(withoutChecking index:[Int]) -> Float {
		set(newValue) {
			let offset = self.offSetFromIndex(index)
			self._elementReader[offset] = newValue
		}
		
		get {
			let offset = self.offSetFromIndex(index)
			return  (self._dataMemoryBaseAdrress + offset).pointee
		}
	}
	
	
	/// Get element value from `index`.
	/// If input index is invalid, return `missingValue`.
	///
	/// - Parameters:
	///   - index: index array
	///   - missingValue: default value for missing elements. Default is `0.0`
	/// - Returns: value
	public func fetchValueOrDefault(_ index: [Int], missingValue: Float = 0.0) -> Float {
		if indexIsValid(index) {
			return self[withoutChecking:index]
		} else {
			return missingValue
		}
	}
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Initializers
	
	
	/// Designated init. 
	///
	/// - Parameters:
	///   - shape: shape
	///   - allocateSize: allocateSize
	///   - capacity: capacity
	///   - dataMemoryBaseAddres: dataMemoryBaseAddres
	///   - elementReader: elementReader
	internal init(shape: TensorShape, allocateSize: Int, capacity: Int,
	            dataMemoryBaseAddres: UnsafeMutablePointer<Float>, elementReader:UnsafeMutableBufferPointer<Float>) {
		self._shape = shape
		self._allocatedSize = allocateSize
		self._capacity = capacity
		self._dataMemoryBaseAdrress = dataMemoryBaseAddres
		self._elementReader = elementReader
	}
	
    /// Initial a tensor with repeating value
    ///
    /// - Parameters:
    ///   - repeatingValue: repeat value
    ///   - shape: shape
    public convenience init(repeatingValue value: Float, tensorShape shape: TensorShape) {
		let pageSize: Int = Int(getpagesize())
		let needSize = shape.count * MemoryLayout<Float>.stride
		let allocateSize = needSize.padded(alignmentSize: pageSize)!
		let capacity = allocateSize / MemoryLayout<Float>.stride
		var memory: UnsafeMutableRawPointer? = nil
		posix_memalign(&memory, pageSize, allocateSize) // use posix_memalign to make it work on MAC
		let dataMemoryBaseAddress = UnsafeMutablePointer<Float>(OpaquePointer(memory!))
		let elementReader = UnsafeMutableBufferPointer<Float>(start: dataMemoryBaseAddress, count: capacity)
		
		self.init(shape: shape,
		          allocateSize: allocateSize,
		          capacity: capacity,
		          dataMemoryBaseAddres: dataMemoryBaseAddress,
		          elementReader: elementReader)
		
		// initial
		self._dataMemoryBaseAdrress.initialize(to: value, count: capacity)
    }
	
	/// Construct a tensor from a flatten (1-D) swift array.
	///
	/// - Parameters:
	///   - array: 1-D array.
	///   - shape: The shape which the constructed tensor have. The count of this shape should be the same of count of passed in array.
	public convenience init(fromFlatArray array:[SupportedScalarDataType], tensorShape shape: TensorShape) {
		self.init(repeatingValue: 0.0, tensorShape: shape)
		
		// Initialize values
		// Note: initialize(from:count:) not used here cause we can not make sure passed in array is contiguous.
		for i in 0..<self.count { self._elementReader[i] = array[i].floatValue }
	}
	
	/// Construct a tensor from a nested (N-D) swift array.
	///
	/// - Parameters:
	///   - array: The nested array object. Note that this array should only contains objects of type `SupportedScalarDataType`.
	///            Also this dimension of this array should be matching with parameter `tensorShape`.
	///   - shape: The shape describing the passed in array.
	public convenience init(dataArray array:[Any], tensorShape shape: TensorShape) {
		// Allocate memory
		let flatArray = TensorUtils.flattenArray(array: array, dataType: shape.dataType)
		self.init(fromFlatArray: flatArray, tensorShape: shape)
	}
	
	
	/// Used internal only to generate slice tensor
	///
	/// - Parameters:
	///   - sliceTensorFrom: contentAddress
	///	  - sliceContentAddress: slice content address
	///   - count: count
	///   - shape: shape
	internal init(sliceTensorFrom rootTensor: Tensor,
	              sliceContentAddress: UnsafeMutablePointer<Float>,
	              count: Int, shape: TensorShape, index: [Int]) {
		self._dataMemoryBaseAdrress = sliceContentAddress
		self._elementReader = UnsafeMutableBufferPointer(start: self._dataMemoryBaseAdrress, count: count)
		self._shape = shape
		self._capacity = count
		self._sliceMarker = true
		self._sliceIndex = index
		self._allocatedSize = count * MemoryLayout<Float>.stride
		self._sliceRootTensor = rootTensor
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Deinitializers
    
    /// Need to free allcoated memory space manually.
    deinit {
		if !self._sliceMarker { // Slice tensor could not free memeory
			free(self._dataMemoryBaseAdrress)
			
		}
    }
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Make tensors randomly
	
	/// Generate a random tensor.
	///
	/// - Parameters:
	///   - shape:
	///   - min:
	///   - max:
	/// - Returns:
	public static func randomTensor(_ shape: TensorShape, min: Float = 0.0, max: Float = 1.0) -> Tensor {
		let tensor = Tensor(repeatingValue: 0.0, tensorShape: shape)
		for i in 0..<tensor.count {
			tensor.floatValueReader[i] = RandomValueGenerator.randomFloat(min: min, max: max)
		}
		return tensor
	}
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Util methods
	
    /// Reload data from a flat array.
    ///
    /// - Note: If `array` size < tensor's `count`
    ///
    /// - Parameter array:
    public func reloadData(fromFlatArray array:[SupportedScalarDataType], tensorShape shape: TensorShape) {
        guard self.capacity >= shape.count else {
            SerranoLogging.errorLogging(message: "Trying to load a data array larer than tensor's capacity size. Capacity \(self.capacity), shape count \(shape.count), array count: \(array.count)",
                                        file: "\(#file)", function: "\(#function)", line: "\(#line)")
            fatalError()
        }
        
        for i in 0..<self.capacity {
            if i < array.count {
                self._elementReader[i] = array[i].floatValue
            } else {
                self._elementReader[i] = 0.0
            }
        }
    }
	
	
	/// Set tensor's all selements to a same `value`.
	///
	/// - Warning: This operation will erase current values of tensor object.
	///
	/// - Parameter value: initial value
	public func resetValues(_ value:SupportedScalarDataType) {
		var val = value.floatValue
		vDSP_vfill(self.contentsAddress, &val, 1, vDSP_Length(self.count))
	}
	
	/// Set tensor's all elements to `0`.
	///
	/// - Warning: This operation will erase current values of tensor object.
	///
	/// - Parameter value: initial value
	public func clear() {
		vDSP_vclr(self.contentsAddress, 1, vDSP_Length(self.count))
	}
	
	
	/// Copy values from another tensor.
	///
	/// ## Dimension check
	/// This function just check cp from tensor's total element count.
	/// If two tensors has same number of elements, do the copy.
	/// If not, raise `fatalError`
	///
	/// - Parameter tensor: cp from tensor
	public func copyValues(_ tensor: Tensor) {
		let copyOp = CopyOperator(inputTensors: [tensor], outputTensors: [self])
		copyOp.compute()
	}
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Array representation
    
    /// Return flat array containing all elements stored in tensor with `Float` type.
    ///
    /// - Returns: Type of declared.
    public func flatArrayFloat() -> [Float] {
        var faltArray = [Float]()
        for offset in 0..<self.count {
            faltArray.append(self._elementReader[offset])
        }
        return faltArray
    }
    
    
    /// Return nested array following object's `shape`.
    ///
    /// - Returns: A nested array with same shape in `shape` attribute.
    public func nestedArrayFloat() -> [Any] {
		if self.rank == 0 {
			return [self.floatValueReader[0]]
		} else {
			return constructNestedArrayFrom(location: [])
		}
    }
    
    
    /// Recursively construct nested array
    ///
    /// - Parameter shape: shape needs to construct
    /// - Returns: result
    internal func constructNestedArrayFrom(location: [Int]) -> [Any] {
        let currDimSize = self._shape.shapeArray[location.count]
        var array = [Any]()
        
        if location.count == self._shape.shapeArray.count - 1 {
            var offset = 0
            for i in 0..<location.count {
                offset += location[i] * self._shape.shapeArray.suffix(from: i+1).reduce(1, *)
            }
			for _ in 0..<self._shape.shapeArray.last! {
                array.append(self._elementReader[offset])
                offset += 1
            }
        } else {
            for i in 0..<currDimSize {
                var newLocation = Array(location)
                newLocation.append(i)
                array.append(constructNestedArrayFrom(location: newLocation))
            }
        }
        
        return array
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Hashable protocol
	
    // Conforms to Hashable
    public var hashValue: Int {
        get {
            return self._dataMemoryBaseAdrress.hashValue
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Equatable protocol
	
    // conforms to Equatable
    public static func  ==(lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs._dataMemoryBaseAdrress == rhs._dataMemoryBaseAdrress
    }
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Metal Buffer Binding APIs
	
	
	/// Get a `MTLBuffer` associated with this tensor object.
	///
	/// - Returns:
	public func gpuBufferResource() -> MTLBufferResource {
		// check gpu available
		guard SerranoEngine.configuredEngine.hasAvailableGPU() else {
			SerranoLogging.errorLogging(message: "No available GPU device.",
										file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError("Fatal error raised by Serrano. Check log for details.")
		}
		
		if self.isSliceTensor {
			return MTLBufferResource(buffer: self.sliceRootTensor!.gpuBufferResource().buffer,
									 offset: self.sliceRootTensor!.slicedTensorOffset(self)!)
		} else {
			if self._mtlbuffer == nil {
				self._mtlbuffer = SerranoEngine.configuredEngine.GPUDevice!.makeBuffer(bytesNoCopy: self._dataMemoryBaseAdrress,
																					   length: self.allocatedBytes,
																					   options: MTLResourceOptions.storageModeShared)
			}
			return MTLBufferResource(buffer: self._mtlbuffer!, offset: 0)
		}
	}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Tensor manipulation
	
	/// Return a new tensor contains the absolute values of this tensor.
	/// Same effect as using `AbsOperator`.
	///
	/// - Returns: new tensor object with absed values.
	public func abs() -> Tensor {
		let newTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(self.shape)
		let absOp = AbsOperator(inputTensors: [self], outputTensors: [newTensor])
		absOp.compute()
		return newTensor
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Tensor-Tensor arithmetic operations (not support broadcasting)
	
	
	/// Element-wise addition. __Not support braodcasting__.
	/// Same effect as using `AddOperator`.
	///
	///	- Warning: If two tensors don't have same shape, program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: result tensor
	public static func +(left: Tensor, right:Tensor) -> Tensor {
		let outputTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let op = AddOperator(inputTensors: [left, right], outputTensors: [outputTensor])
		op.compute()
		return outputTensor
	}
	
	/// Element-wise substraction. __Not support braodcasting__.
	/// Same effect as using `SubOperator`.
	///
	///	- Warning: If two tensors don't have same shape, program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: result tensor
	public static func -(left: Tensor, right:Tensor) -> Tensor {
		let outputTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let op = SubOperator(inputTensors: [left, right], outputTensors: [outputTensor])
		op.compute()
		return outputTensor
	}
	
	/// Element-wise multiplication. __Not support braodcasting__.
	/// Same effect as using `AddOperator`.
	///
	///	- Warning: If two tensors don't have same shape, program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: result tensor
	public static func *(left: Tensor, right:Tensor) -> Tensor {
		let outputTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let op = MultOperator(inputTensors: [left, right], outputTensors: [outputTensor])
		op.compute()
		return outputTensor
	}
	
	/// Element-wise division. __Not support braodcasting__.
	/// Same effect as using `DivOperator`.
	///
	///	- Warning: If two tensors don't have same shape, program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: result tensor
	public static func /(left: Tensor, right:Tensor) -> Tensor {
		let outputTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let op = DivOperator(inputTensors: [left, right], outputTensors: [outputTensor])
		op.compute()
		return outputTensor
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Tensor-Tensor in-place arithmetic operations (not support broadcasting)
	
	/// Element-wise in-place addition. __Not support braodcasting__.
	/// Same effect as using `AddOperator`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	///	- Warning: If two tensors don't have same shape, program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: left tensor after calculated.
	@discardableResult public static func &+(left: Tensor, right:Tensor) -> Tensor {
		let op = AddOperator(inputTensors: [left, right], outputTensors: [left])
		op.compute(.GPU)
		return left
	}
	
	/// Element-wise in-place substraction. __Not support braodcasting__.
	/// Same effect as using `SubOperator`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	///	- Warning: If two tensors don't have same shape, program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: left tensor after calculated.
	@discardableResult public static func &-(left: Tensor, right:Tensor) -> Tensor {
		let op = SubOperator(inputTensors: [left, right], outputTensors: [left])
		op.compute()
		return left
	}
	
	/// Element-wise in-place multiplication. __Not support braodcasting__.
	/// Same effect as using `MultOperator`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	///	- Warning: If two tensors don't have same shape, program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: left tensor after calculated.
	@discardableResult public static func &*(left: Tensor, right:Tensor) -> Tensor {
		let op = MultOperator(inputTensors: [left, right], outputTensors: [left])
		op.compute()
		return left
	}

	/// Element-wise in-place division. __Not support braodcasting__.
	/// Same effect as using `DivOperator`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	///	- Warning: If two tensors don't have same shape, program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: left tensor after calculated.
	@discardableResult public static func &/(left: Tensor, right:Tensor) -> Tensor {
		let op = DivOperator(inputTensors: [left, right], outputTensors: [left])
		op.compute()
		return left
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Tensor-Tensor arithmetic operations (support broadcasting)
	
	/// Element-wise addition. __support braodcasting__.
	/// Same effect as using `BrodcastAddOperator`.
	///
	///	- Warning: If two tensors don't have same shape or one of them cannot be broadcasted to anoter one,
	///	           program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: result tensor
	public static func .+(left: Tensor, right:Tensor) -> Tensor {
		let outputTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let op = BroadcastAddOperator(inputTensors: [left, right], outputTensors: [outputTensor])
		op.compute()
		return outputTensor
	}
	
	/// Element-wise substraction. __support braodcasting__.
	/// Same effect as using `BroadcastSubOperator`.
	///
	///	- Warning: If two tensors don't have same shape or one of them cannot be broadcasted to anoter one,
	///	           program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: result tensor
	public static func .-(left: Tensor, right:Tensor) -> Tensor {
		let outputTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let op = BroadcastSubOperator(inputTensors: [left, right], outputTensors: [outputTensor])
		op.compute()
		return outputTensor
	}
	
	/// Element-wise multiplication. __support braodcasting__.
	/// Same effect as using `BroadcastMultOperator`.
	///
	///	- Warning: If two tensors don't have same shape or one of them cannot be broadcasted to anoter one,
	///	           program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: result tensor
	public static func .*(left: Tensor, right:Tensor) -> Tensor {
		let outputTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let op = BroadcastMultOperator(inputTensors: [left, right], outputTensors: [outputTensor])
		op.compute()
		return outputTensor
	}
	
	/// Element-wise division. __support braodcasting__.
	/// Same effect as using `BroadcastDivOperator`.
	///
	///	- Warning: If two tensors don't have same shape or one of them cannot be broadcasted to anoter one,
	///	           program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: result tensor, a new created tensor.
	public static func ./(left: Tensor, right:Tensor) -> Tensor {
		let outputTensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let op = BroadcastDivOperator(inputTensors: [left, right], outputTensors: [outputTensor])
		op.compute()
		return outputTensor
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Tensor-Tensor in-place arithmetic operations (support broadcasting)
	
	/// Element-wise in-place addition. __support braodcasting__.
	/// Same effect as using `BrodcastAddOperator`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	///	- Warning: If two tensors don't have same shape or one of them cannot be broadcasted to anoter one,
	///	           program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: left tensor after calcualted
	@discardableResult public static func .&+(left: Tensor, right:Tensor) -> Tensor {
		let op = BroadcastAddOperator(inputTensors: [left, right], outputTensors: [left])
		op.compute()
		return left
	}
	
	/// Element-wise in-place substraction. __support braodcasting__.
	/// Same effect as using `BroadcastSubOperator`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	///	- Warning: If two tensors don't have same shape or one of them cannot be broadcasted to anoter one,
	///	           program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: left tensor after calcualted
	@discardableResult public static func .&-(left: Tensor, right:Tensor) -> Tensor {
		let op = BroadcastSubOperator(inputTensors: [left, right], outputTensors: [left])
		op.compute()
		return left
	}
	
	/// Element-wise in-place multiplication. __support braodcasting__.
	/// Same effect as using `BroadcastMultOperator`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	///	- Warning: If two tensors don't have same shape or one of them cannot be broadcasted to anoter one,
	///	           program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: left tensor after calcualted
	@discardableResult public static func .&*(left: Tensor, right:Tensor) -> Tensor {
		let op = BroadcastMultOperator(inputTensors: [left, right], outputTensors: [left])
		op.compute()
		return left
	}
	
	/// Element-wise in-place division. __support braodcasting__.
	/// Same effect as using `BroadcastDivOperator`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	///	- Warning: If two tensors don't have same shape or one of them cannot be broadcasted to anoter one,
	///	           program would be aborted (`fatalError()` called).
	///
	/// - Parameters:
	///   - left: left tensor
	///   - right: right tensor
	/// - Returns: left tensor after calcualted
	@discardableResult public static func .&/(left: Tensor, right:Tensor) -> Tensor {
		let op = BroadcastDivOperator(inputTensors: [left, right], outputTensors: [left])
		op.compute()
		return left
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Tensor-Scalar arithmetic operations. 
	
	/// A tensor plus a scalar variable. Ex. `[2, 3] + 0.5 --> [2.5, 3.5]`.
	///
	/// - Parameters:
	///   - left: A tensor object
	///   - rightScalar: A scalar variable
	/// - Returns: Result tensor, new created.
	public static func + (left: Tensor,  rightScalar: SupportedScalarDataType) -> Tensor {
		let tensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let outAddress = tensor.contentsAddress
		let inAddress = left.contentsAddress
		var scalar = rightScalar.floatValue
		let count = vDSP_Length(left.count)
		vDSP_vsadd(inAddress, 1, &scalar, outAddress, 1, count)
		return tensor
	}
	
	/// A tensor substract a scalar variable. Ex. `[2, 3] - 0.5 --> [1.5, 2.5]`.
	///
	/// - Parameters:
	///   - left: A tensor object
	///   - rightScalar: A scalar variable
	/// - Returns: Result tensor, new created.
	public static func - (left: Tensor,  rightScalar: SupportedScalarDataType) -> Tensor {
		let tensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let outAddress = tensor.contentsAddress
		let inAddress = left.contentsAddress
		var scalar = -rightScalar.floatValue
		let count = vDSP_Length(left.count)
		vDSP_vsadd(inAddress, 1, &scalar, outAddress, 1, count)
		return tensor
	}

	/// A tensor multiply a scalar variable. Ex. `[2, 3] * 0.5 --> [1.0, 1.5]`.
	///
	/// - Parameters:
	///   - left: A tensor object
	///   - rightScalar: A scalar variable
	/// - Returns: Result tensor, new created.
	public static func * (left: Tensor,  rightScalar: SupportedScalarDataType) -> Tensor {
		let tensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let outAddress = tensor.contentsAddress
		let inAddress = left.contentsAddress
		var scalar:Float = rightScalar.floatValue
		let count = vDSP_Length(left.count)
		vDSP_vsmul(inAddress, 1, &scalar, outAddress, 1, count)
		return tensor
	}
	
	/// A tensor divide a scalar variable. Ex. `[2, 3] / 0.5 --> [4.0, 6.0]`.
	///
	/// - Parameters:
	///   - left: A tensor object
	///   - rightScalar: A scalar variable
	/// - Returns: Result tensor, new created.
	public static func / (left: Tensor,  rightScalar: SupportedScalarDataType) -> Tensor {
		let tensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(left.shape)
		let outAddress = tensor.contentsAddress
		let inAddress = left.contentsAddress
		var scalar:Float = rightScalar.floatValue
		let count = vDSP_Length(left.count)
		vDSP_vsdiv(inAddress, 1, &scalar, outAddress, 1, count)
		return tensor
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Tensor-Scalar in-place arithmetic operations.
	
	/// A tensor plus a scalar variable. Ex. `[2, 3] + 0.5 --> [2.5, 3.5]`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	/// - Parameters:
	///   - left: A tensor object
	///   - rightScalar: A scalar variable
	/// - Returns:
	@discardableResult public static func &+ (left: Tensor,  rightScalar: SupportedScalarDataType) -> Tensor {
		let outAddress = left.contentsAddress
		let inAddress = left.contentsAddress
		var scalar = rightScalar.floatValue
		let count = vDSP_Length(left.count)
		vDSP_vsadd(inAddress, 1, &scalar, outAddress, 1, count)
		return left
	}
	
	/// A tensor substract a scalar variable. Ex. `[2, 3] - 0.5 --> [1.5, 2.5]`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	/// - Parameters:
	///   - left: A tensor object
	///   - rightScalar: A scalar variable
	/// - Returns:
	@discardableResult public static func &- (left: Tensor,  rightScalar: SupportedScalarDataType) -> Tensor {
		let outAddress = left.contentsAddress
		let inAddress = left.contentsAddress
		var scalar = -rightScalar.floatValue
		let count = vDSP_Length(left.count)
		vDSP_vsadd(inAddress, 1, &scalar, outAddress, 1, count)
		return left
	}
	
	/// A tensor multiply a scalar variable. Ex. `[2, 3] * 0.5 --> [1.0, 1.5]`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	/// - Parameters:
	///   - left: A tensor object
	///   - rightScalar: A scalar variable
	/// - Returns: left tensor after calculated.
	@discardableResult public static func &* (left: Tensor,  rightScalar: SupportedScalarDataType) -> Tensor {
		let outAddress = left.contentsAddress
		let inAddress = left.contentsAddress
		var scalar:Float = rightScalar.floatValue
		let count = vDSP_Length(left.count)
		vDSP_vsmul(inAddress, 1, &scalar, outAddress, 1, count)
		return left
	}
	
	/// A tensor divide a scalar variable. Ex. `[2, 3] / 0.5 --> [4.0, 6.0]`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in left tensor and return the calcualted left tensor.
	///
	/// - Parameters:
	///   - left: A tensor object
	///   - rightScalar: A scalar variable
	/// - Returns:
	@discardableResult public static func &/ (left: Tensor,  rightScalar: SupportedScalarDataType) -> Tensor {
		let address = left.contentsAddress
		let count = vDSP_Length(left.count)
		var scalar: Float = rightScalar.floatValue
		vDSP_vsdiv(address, 1, &scalar, address, 1, count)
		return left
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Scalar-Tensor arithmetic operations.
	
	/// A scalar plus a tensor variable. Ex. ` 0.5 + [2, 3] --> [2.5, 3.5]`.
	///
	/// - Parameters:
	///   - leftScalar: A scalar variable
	///   - right: A tensor object
	/// - Returns: Result tensor, new created.
	public static func + (leftScalar: SupportedScalarDataType, right: Tensor) -> Tensor {
		let tensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(right.shape)
		let outAddress = tensor.contentsAddress
		let inAddress = right.contentsAddress
		var scalar = leftScalar.floatValue
		let count = vDSP_Length(right.count)
		vDSP_vsadd(inAddress, 1, &scalar, outAddress, 1, count)
		return tensor
	}
	
	
	/// A scalar substracts a tensor variable. Ex. ` 0.5 - [2, 3] --> [-1.5, -2.5]`.
	///
	/// - Parameters:
	///   - leftScalar: A scalar variable
	///   - right: A tensor object
	/// - Returns: Result tensor, new created.
	public static func - (leftScalar: SupportedScalarDataType, right: Tensor) -> Tensor {
		let tensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(right.shape)
		let outAddress = tensor.contentsAddress
		let inAddress = right.contentsAddress
		var scalar = leftScalar.floatValue
		var scale:Float = -1.0
		let count = vDSP_Length(right.count)
		vDSP_vsmsa(inAddress, 1, &scale, &scalar, outAddress, 1, count)
		return tensor
	}
	
	/// A scalar multiplies a tensor variable. Ex. ` 0.5 * [2, 3] --> [1.0, 1.5]`.
	///
	/// - Parameters:
	///   - leftScalar: A scalar variable
	///   - right: A tensor object
	/// - Returns: Result tensor, new created.
	public static func * (leftScalar: SupportedScalarDataType, right: Tensor) -> Tensor {
		let tensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(right.shape)
		let outAddress = tensor.contentsAddress
		let inAddress = right.contentsAddress
		var scalar:Float = leftScalar.floatValue
		let count = vDSP_Length(right.count)
		vDSP_vsmul(inAddress, 1, &scalar, outAddress, 1, count)
		return tensor
	}
	
	/// A scalar divides by a tensor variable. Ex. ` 0.5 / [2, 3] --> [0.25, 0.16]`.
	///
	/// - Parameters:
	///   - leftScalar: A scalar variable
	///   - right: A tensor object
	/// - Returns: Result tensor, new created.
	public static func / (leftScalar: SupportedScalarDataType, right: Tensor) -> Tensor {
		let tensor = SerranoResourceManager.globalManager.allocateUnamangedTensor(right.shape)
		let outAddress = tensor.contentsAddress
		let inAddress = right.contentsAddress
		var scalar = leftScalar.floatValue
		let count = vDSP_Length(right.count)
		vDSP_svdiv(&scalar, inAddress, 1, outAddress, 1, count)
		return tensor
	}
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Scalar-Tensor in-place arithmetic operations.
	
	/// A scalar plus a tensor variable. Ex. ` 0.5 + [2, 3] --> [2.5, 3.5]`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in right tensor and return the calcualted left tensor.
	///
	/// - Parameters:
	///   - leftScalar: A scalar variable
	///   - right: A tensor object
	/// - Returns: Result right tensor.
	@discardableResult public static func &+ (leftScalar: SupportedScalarDataType, right: Tensor) -> Tensor {
		let outAddress = right.contentsAddress
		let inAddress = right.contentsAddress
		var scalar = leftScalar.floatValue
		let count = vDSP_Length(right.count)
		vDSP_vsadd(inAddress, 1, &scalar, outAddress, 1, count)
		return right
	}
	
	/// A scalar substracts a tensor variable. Ex. ` 0.5 - [2, 3] --> [-1.5, -2.5]`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in right tensor and return the calcualted left tensor.
	///
	/// - Parameters:
	///   - leftScalar: A scalar variable
	///   - right: A tensor object
	/// - Returns: Result right tensor.
	@discardableResult public static func &- (leftScalar: SupportedScalarDataType, right: Tensor) -> Tensor {
		let outAddress = right.contentsAddress
		let inAddress = right.contentsAddress
		var scalar = leftScalar.floatValue
		var scale:Float = -1.0
		let count = vDSP_Length(right.count)
		vDSP_vsmsa(inAddress, 1, &scale, &scalar, outAddress, 1, count)
		return right
	}
	
	/// A scalar multiplies a tensor variable. Ex. ` 0.5 * [2, 3] --> [1.0, 1.5]`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in right tensor and return the calcualted left tensor.
	///
	/// - Parameters:
	///   - leftScalar: A scalar variable
	///   - right: A tensor object
	/// - Returns: Result tensor, new created.
	@discardableResult public static func &* (leftScalar: SupportedScalarDataType, right: Tensor) -> Tensor {
		let outAddress = right.contentsAddress
		let inAddress = right.contentsAddress
		var scalar:Float = leftScalar.floatValue
		let count = vDSP_Length(right.count)
		vDSP_vsmul(inAddress, 1, &scalar, outAddress, 1, count)
		return right
	}
	
	/// A scalar divides by a tensor variable. Ex. ` 0.5 / [2, 3] --> [0.25, 0.16]`.
	///
	/// - Note: This is an __in place__ operation.
	///			Result stored in right tensor and return the calcualted left tensor.
	///
	/// - Parameters:
	///   - leftScalar: A scalar variable
	///   - right: A tensor object
	/// - Returns: Result tensor, new created.
	@discardableResult public static func &/ (leftScalar: SupportedScalarDataType, right: Tensor) -> Tensor {
		let outAddress = right.contentsAddress
		let inAddress = right.contentsAddress
		var scalar = leftScalar.floatValue
		let count = vDSP_Length(right.count)
		vDSP_svdiv(&scalar, inAddress, 1, outAddress, 1, count)
		return right
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Slice tensor management
	
	/// Make a slice tensor from this object.
	///
	/// - Parameter sliceIndex: Slice Inde
	/// - Returns: A sliced tensor
	public func slice(sliceIndex:[Int]) -> Tensor {
		guard sliceIndex.count > 0 && sliceIndex.count <= self.shape.rank else {
			SerranoLogging.errorLogging(message: "Argument sliceIndex \(sliceIndex) is invalid. Trying to slice from Tensor \(self.description)",
				file: "\(#file)", function: "\(#function)", line: "\(#function)")
			fatalError("Fatal from Serrano.  Check log for detail.")
		}
		
		for (indexSlice, indexTensor) in zip(sliceIndex, self.shape.shapeArray) {
			guard indexSlice < indexTensor else { // <, not <=
				SerranoLogging.errorLogging(message: "Argument sliceIndex \(sliceIndex) is invalid. Trying to slice from Tensor \(self.description)",
					file: "\(#file)", function: "\(#function)", line: "\(#function)")
				fatalError("Fatal from Serrano.  Check log for detail.")
			}
		}
		
		var elementOffset = 0
		for (dimIndex, index) in sliceIndex.enumerated() {
			elementOffset += index * self.shape.shapeArray.suffix(from: dimIndex + 1).reduce(1, *)
		}
		let address = self._dataMemoryBaseAdrress + elementOffset
		let count = self._shape.shapeArray.suffix(from: sliceIndex.count).reduce(1, *)
		let sliceShape = TensorShape(dataType: self._shape.dataType, shape: Array( self._shape.shapeArray.suffix(from: sliceIndex.count)))
		let sliceTensor = Tensor(sliceTensorFrom: self, sliceContentAddress: address,
								 count: count, shape: sliceShape, index: sliceIndex)
		return sliceTensor
	}
	
	/// Check if a slice tensor object belong to this tensor.
	///
	/// - Note: If the passed in tensor is not a slice tensor, always returns `false`.
	///
	/// - Parameter slice: slice tensor
	/// - Returns: Bool. Result
	public func containsSlice(_ slice: Tensor) -> Bool {
		guard slice.isSliceTensor == true else {
			SerranoLogging.warningLogging(message: "Trying to check a non-silce tensor object.",
			                              file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return false
		}
		return slice.sliceRootTensor! == self
	}
	
	/// Get the bytes offset for a slice tensor spawned from this tensor.
	///
	/// - Note: The function will first check if contains this slice, if not it will return `nil`
	///
	/// - Parameter slice: slice object
	/// - Returns: offset in bytes
	public func slicedTensorOffset(_ slice: Tensor) -> Int? {
		guard self.containsSlice(slice) else {
			return nil
		}
		
		var elementOffset = 0
		for (dimIndex, index) in slice.sliceIndex!.enumerated() {
			elementOffset += index * self.shape.shapeArray[dimIndex]
		}
		return elementOffset * MemoryLayout<Float>.stride
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Protocol TensorSymbol
	
	/// Return self
	///
	/// - Returns: tensor
	public func evaluate() -> [DataSymbolSupportedDataType]? {
		return [self]
	}
	
	/// Add new symbol to `inBounds`.
	/// Should check duplicate.
	public func addToInBound(_ symbol: GraphSymbol) {
		let s = symbol as! SerranoGraphSymbol
		if !(self.inBoundsWeak.contains {$0.value == s}) {
			let weakSymbol = WeakSerranoGraphSymbol(value: s)
			self.inBoundsWeak.append(weakSymbol)
		}
	}
	
	/// Add new symbol to `outBounds`
	/// Should check duplicate.
	public func addToOutBound(_ symbol: GraphSymbol) {
		let s = symbol as! SerranoGraphSymbol
		if !(self.outBoundsWeak.contains {$0.value == s}) {
			let weakSymbol = WeakSerranoGraphSymbol(value: s)
			self.outBoundsWeak.append(weakSymbol)
		}
	}
	
	/// A tensor object could not be binded to another tensor.
	/// - Warning: This function will do nonthing and always return `false`.
	///			   A error logging will be gaven.
	///
	/// - Parameter data: data. Should be a `Tensor` object.
	/// - Returns: always `False`
	public func bindData(_ data:DataSymbolSupportedDataType) -> Bool {
		SerranoLogging.errorLogging(message: "Tensor symbol \(self.symbolLabel) is a tensor object conforms to TensorSymbol. " +
			                                 "It cannot bind to another tensor object.",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
		return false
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - MPS utils
	
	/// Get wrapped `MPSImage` from a 3D Tensor (including channel information)
	///
	/// - Note: If this is not a 3D tensor, `fatalError()` will be raised
	///
	/// - Parameter dataFormat: `TensorChannelOrder`
	/// - Returns: `MPSImage`
	#if  !((arch(i386)  || arch(x86_64)) && os(iOS))
	@available(OSX 10.13, iOS 11.0, *)
	public func getMPSImage(dataFormat: TensorChannelOrder) -> MPSImage {
			guard SerranoEngine.configuredEngine.hasAvailableGPU() else {
				SerranoLogging.errorLogging(message: "Trying to use MPS but no available GPU.",
											file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Raised by Serrano. Check log for detail")
			}
			
			guard self.rank == 3 else {
				SerranoLogging.errorLogging(message: "Trying to generate MPSImage from a tensor but with rank \(self.rank).",
											file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Raised by Serrano. Check log for detail")
			}
			
			let (c, h, w) = parseImgChannelShapeInfo(dataFormat, shapeArray: self.shape.shapeArray)
			let imgDescriptior = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float32,
													width: w,
													height: h,
													featureChannels: c)
			let mpsi = MPSImage(device: SerranoEngine.configuredEngine.GPUDevice!, imageDescriptor: imgDescriptior)
			mpsi.readBytes(self._dataMemoryBaseAdrress, dataLayout: dataFormat.MPSImageOrder, imageIndex: 0)
			return mpsi
	}
	#endif
<<<<<<< HEAD
=======
	
>>>>>>> naive_convolution
}


