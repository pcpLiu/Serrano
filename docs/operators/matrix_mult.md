`MatrixMultOperator`([API](http://serrano-lib.org/docs/latest/api/Classes/MatrixMultOperator.html)) do matrix multiplication on input tensors.

**Few notes:**

- Operator will use [Metal Performance Shaders
](https://developer.apple.com/documentation/metalperformanceshaders) if it's available on the platform.
- This operator can taken multiple pairs of inputs. 
	If `inputTensors` contains more than 2 tensor, operator will take the **last** tensor as input `B` and all previous tensors as input `A`s. Then for each tensor in `A`, do `AxB`.

**transposeA and transposeB**

`MatrixMultOperator` has two attributes:

- `transposeA`: If transposing input A before do computation. This applies to all input A.
- `transposeB`: If transposing input B before do computation.

## Initialization 
```swift
let op = MatrixMultOperator()
let op = MatrixMultOperator(transposeB: true)
let op = MatrixMultOperator(transposeA: true, transposeB: true)
```

