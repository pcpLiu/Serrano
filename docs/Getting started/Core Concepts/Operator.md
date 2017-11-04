# Operator

__Operator__ is the basic calculation unit in Serrano. Serrano defines a protocol `ComputableOperator` ([API](http://serrano-lib.org/docs/latest/api/Protocols/ComputableOperator.html)) to unify the computation actions of an Operator.
Any class conforms to `ComputableOperator` can be used in Serrano. 
Check [here](/Extension//Write your own operator.md) to know how to define your own operator.

<hr>
<hr>

## Basic
The action of a ComputableOperator is taking in some tensors and output some tensors. And it may also take in some parameters.
![ComputableOperator computation](/imgs/operator_intro.png)

For example the `AbsOperator`([API](http://serrano-lib.org/docs/latest/api/Classes/AbsOperator.html)), a unary operator, will do absolute computation on its `inputTensors` and store result in `outputTensors`.

#### `inputTensors` and `outputTensors`
A ComputableOperator has two attributes:

- `inputTensors`: an array of `Tensor` representing inputs
- `outputTensors`: an array of `Tensor` representing outputs

For different operators, the number of input tensors and number of output tensors may be same or <noscript></noscript>.

!!! note "Tensor allocation":
	In Serrano, operators do not take care of tensor allocation. So before doing calculation, `inputTensors` and `outputTensors` should be assigned.
	If user use `Graph` or `Model` APIs, they do not need to concern tensor allocations for operators.
	They just need to feed in the input tensors for the model.

#### Checking before computation
There two APIs defined in protocol `ComputableOperator`([API](http://serrano-lib.org/docs/latest/api/Protocols/ComputableOperator.html)):

- `outputShape(shapeArray:)`: Calculate the output tensors' shapes given a list of input shapes. If the operator cannot operate on the input shapes, this function return `nil`.
This function can be used to check if some input tensors are valid or not for an operator.
- `inputOutputTensorsCheck()`: This function actually check validation of `inputTensor`, `outputTensors` and other related parameter variables. If cannot pass validation, `fatalError()` will be raised and application will exit. And users can find error information in loggings (or console window if in development).

Usually, users do not need to call these two functions.

<hr>
<hr>