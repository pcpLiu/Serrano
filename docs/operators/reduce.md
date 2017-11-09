The reduce operator has a collection of basic reduce calculation, like reduce sum and reduce multiplication.
A reduce operator do reduce calculation along axis in attribute `axis`.

**Few notes:**

- The `inputTensors` and `outputTensors` should have same number of tensors. 

## Initialization 
Reduce operators requires user specifying the `axis` 
```swift
let op = SomeReduceOperator(axis:[1], keepDim: false)
```
In this initialization , `#!swift keepDim` is optional. Its default values is `#!swift false`.
Value of `#!swift keepDim` indicates if keep dimensions in result tensor and this just affects result tensor's `shape` attributes.
<hr>
<hr>

## List

#### [ReduceSumOperator](http://serrano-lib.org/docs/latest/api/Classes/ReduceSumOperator.html)
Computes the sum of array elements over given axes.

#### [ReduceProductOperator](http://serrano-lib.org/docs/latest/api/Classes/ReduceProductOperator.html)
Computes the product of array elements over given axes.

#### [ReduceMaxOperator](http://serrano-lib.org/docs/latest/api/Classes/ReduceMaxOperator.html)
Computes the max of array elements over given axes.

#### [ReduceMinOperator](http://serrano-lib.org/docs/latest/api/Classes/ReduceMinOperator.html)
Computes the min of array elements over given axes.

#### [ReduceMeanOperator](http://serrano-lib.org/docs/latest/api/Classes/ReduceMeanOperator.html)
Computes the mean of array elements over given axes.
