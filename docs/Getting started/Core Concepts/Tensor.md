# Tensor
`Tensor`([API](http://serrano-lib.org/docs/latest/api/Classes/Tensor.html)) is the N-D array implementation in Serrano, just like NDArray in `NumPy`.

<hr>
<hr>


## Create a Tensor object
In the [API](http://serrano-lib.org/docs/latest/api/Classes/Tensor.html) of `Tensor`, there lists several initializations.
 
- Below code creates a tensor with shape `[255, 255, 3]` and set all elements' values to `0.3`:
	```swift
	// dimension info of tensor
	let shape = TensorShape(dataType: .float, 
	                        shape: [255, 255, 3]
	                        ) 
	let tensor = Tensor(repeatingValue: 0.3, tensorShape: shape)
	```
- Creating a tensor object from a Swift array:
	```swift
	let data = [
	            [1.0, 0.5, 1.3],
	            [2.0, 4.2, 6.7],
	           ]
	let shape = TensorShape(dataType: .float, 
	                        shape [2, 3]
	                        )
	let tensor = Tensor(dataArray: data, tensorShape: shape)
	```

!!! note "Loading Util APIs":
	Serrano plans to support more convenient data loading APIs and welcome contribution.

<hr>
<hr>

## Shape of a Tensor
A tensor object has an attribute `shape` indicating this tensor object's dimension information.
Struct `TensorShape` ([API](http://serrano-lib.org/docs/latest/api/Structs/TensorShape.html)) is defined in Serrano to represent shape information.

#### Creating TensorShape
`TensorShape` is defined as a `struct`. It can be created directly like:
```swift
let shapeA = TensorShape(dataType: .float, shape [255, 255, 3])

let shapeB = TensorShape(dataType: .int, shape [10, 3])

``` 


`TensorShape` has two attributes: 

- dataType: a [`TensorDataType`](http://serrano-lib.org/docs/latest/api/Enums/TensorDataType.html) variable
- shape `[Int]` array

__dataType__:

Enum [`TensorDataType`](http://serrano-lib.org/docs/latest/api/Enums/TensorDataType.html) has `int`, `float` and `double` cases.
__However, inside tensor objects all values are stored as `Float` values.__
The `dataType` of a `TensorShape` just help user understand what initial data type the shape represents for.

__shapeArray__:

In Serrano, we follow __row-marjor__ order to store and access elements in a Tensor object and each row is represented as an array. 

For example, a Tensor object with shape [2, 3], it can be visulized as a nested array like below:
```swift
// shape [2, 3]
[
  [1.0, 0.5, 1.3],
  [2.0, 4.2, 6.7],
]
```
And a typical example, a 3-channel RGB image data could be represented with shape `[3, image_hight, image_width]` (channel first):
```swift
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

####  Rank `0` as scalar
If a tensor shape object with `0` rank, i.e.: `shape.ShapeArray.count == 0`, it means the shape represent a scalar variable.

####  Equatable of TensorShapes:
- Two `TensorShape` objects are equal (`==`) if their `shapeArray` attributes are equal.
- Two `TensorShape` objects are dot equal (`.==`) if they have the same `shapeArray` and same `dataType`.
Example:
```swift
let shapeA = TensorShape(dataType: .float, shape [255, 255, 3])
let shapeB = TensorShape(dataType: .int, shape [255, 255, 3])
let shapeC = TensorShape(dataType: .float, shape [255, 255, 3])
shapeA == shapeB // true
shapeA .== shapeB // false
shapeA .== shapeC // true
```

<hr>
<hr>


## Memory layout
When user create a Tensor object, Serrano will manually allocate a continuous memory space for this Tensor object to store its values.
So basically, inside a Tensor object it stores the N-D array as a flattened vector.

#### Page-aligned allocation
Serrano uses `posix_memalign`([man](https://developer.apple.com/legacy/library/documentation/Darwin/Reference/ManPages/man3/posix_memalign.3.html)) to allocate the continuous memory.
We allocated page-alinged memory so that later when using Metal GPU, we do not need to copy values to construct a MTLBuffer. Details check [here](https://developer.apple.com/documentation/metal/mtldevice/1433382-makebuffer).

!!! Warning "Under Improvement":
	This memory allocation strategy is under improvement. May change in future.

#### Access memory 
`floatValueReader`([API](http://serrano-lib.org/docs/latest/api/Classes/Tensor.html#/s:7Serrano6TensorC16floatValueReaderSrySfGv)) allow users to access allocated memory.
It is a `UnsafeMutableBufferPointer<Float>` pointer. User can access and modify a single element like:
```swift
let t = Tensor(repeatingValue: 0.2, 
               tensorShape: TensorShape(dataType: .float, shape [3, 2])
               )

let reader = t.floatValueReader
print(reader[0]) // '0.2'

reader[0] = 1.0
print(reader[0]) // '1.0' 
```

<hr>
<hr>

## Tensor Slice
Sometimes, user may want to access part of a tensor object. Tensor class has __`slice(sliceIndex:[Int])`__ ([API](http://serrano-lib.org/docs/latest/api/Classes/Tensor.html)) function which could slice part of a tensor into a new tensor object.

For example we have a tensor with shape `[3, 2, 2]` and we think the first dimension as channel, next two dimensions are height and width. 
```swift

let dataArray = [
	// 1st channel
	[
		[0.5, 2.6],
		[1.1, 9.3],
	], 
	
	// 2nd channel
	[
		[2.7, 4.1],
		[6.3, 1.7],
	], 
	
	// 3rd channel
	[
		[5.6, 3.2],
		[1.9, 9.1],
	], 
]
let rootTensor = Tensor(dataArray: dataArray, 
                        tensorShape: TensorShape(dataType: .float, shape: [3, 2, 2])
                        )
```

If you want to get the 1st channel's information:
```swift
let sliceOne = rootTensor.slice([0])
print(slice.shape.shapeArray)
// [2, 2]
print(slice.nestedArrayFloat())
/** Print out:
[
		[0.5, 2.3],
		[1.1, 9.3],
]
*/

```

If you want to get the 1st row in 2nd channel:
```swift
let sliceTwo = rootTensor.slice([1, 1])
print(slice.shape.shapeArray)
// [2]
print(slice.nestedArrayFloat())
/** Print out:
[2.7, 4.1] 
*/
```

Sliced tensor can also be used to slice again:
```swift
let sliceThree = sliceOne.slice([0])
print(sliceThree.shape.shapeArray)
// [2]
print(sliceThree.nestedArrayFloat())
/** Print out:
[0.5, 2.3]
*/
```

!!! Danger "Sliced Tensor Holds a Strong Reference to root Tensor":
	A sliced Tensor share memory of its tensor object. And it holds a strong reference to root tensor. Keep an eye on this.


!!! Warning "Under Improvement":
	Slice related APIs are under improvement. More useful functions are adding up.

