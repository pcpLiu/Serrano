## Protocol ComputableOperator
All operators conform to `ComputableOperator`([API](http://serrano-lib.org/docs/latest/api/Protocols/ComputableOperator.html)).

#### Initialization
For most operator, it just simply:
```swift
let addOp = AddOperator()
let relu = ReLUOperator()
```

For some operators with required initial arguments:
```swift
let poolOp = MaxPool2DOperator(kernelSize: [2, 2])
let convOp =  ConvOperator2D(numFilters: 64, kernelSize: [3, 3])
```

If you are not sure, you can check [API reference](http://serrano-lib.org/docs/latest/api/) for more details of each operator.

#### Attributes
Below are some useful attributes:

- `#!swift public var inputTensors: [Tensor]`: The list of input tensors of an operator.
- `#!swift public var outputTensors: [Tensor]`: The list of output tensors of an operator.
- `#!swift public var operatorLabel: String`: A readable label of this operator.
- `#!swift public var computationDelegate: OperatorCalculationDelegate?`: 
	A delegate which could monitor the computation activities of an operator.

#### Methods
Below are some useful methods:

- `#!swift func outputShape(shapeArray shapes:[TensorShape]) -> [TensorShape]?`:
	This function returns the output shapes for given input shapes.
- `#!swift func inputOutputTensorsCheck() -> (check: Bool, msg: String)`:
	Validate an operator's input tensors and output tensors. Return checking result (`#!swift Bool`)
	and error message (`#!swift String`) if invalid.
- `#!swift func compute(_ computationMode: OperatorComputationMode)`:
	Compute results and store result in output tensors in sync way.
- `#!swift func computeAsync(_ computationMode: OperatorComputationMode)`:
	Compute results and store result in output tensors in async way.
	This function will return immediately and operator will do computation in background.
	When begin and done computation, the operator's `computationDelegate` will be notified.
- `#!swift func gradCompute(_ computationMode: OperatorComputationMode) -> [String: DataSymbolSupportedDataType]`:
	Return calculated gradient in sync way.
	The key of returned dictionary is the label of corresponding input tensor or parameter.
- `#!swift func gradComputAsync(_ computationMode: OperatorComputationMode)`:
	Compute gradient in async way. When begin and done computation, the operator's `computationDelegate` will be notified.

#### Delegate
The attribute `computationDelegate` can be used to monitor the operator's activities.
User can just make their class conform to `OperatorCalculationDelegate`([API](http://serrano-lib.org/docs/latest/api/Protocols/OperatorCalculationDelegate.html)) by implementation these 4 methods:

- `#!swift func operatorWillBeginComputation(_ op: ComputableOperator)`:
	An operator will begin to do computation. `op` is the sender.
- `#!swift func operatorDidEndComputation(_:outputTensors:)`:
	An operator has completed computation. `op` is the sender.
- `#!swift func operatorWillBeginGradsComputation(_ op: ComputableOperator)`:
	An operator will begin to do gradient computation. `op` is the sender.
- `#!swift func operatorDidEndGradsComputation(_ op: ComputableOperator, grads: [String: DataSymbolSupportedDataType])`:
	An operator has completed gradient computation. `op` is the sender.

!!! warning "Expecting changes":
	Expecting changes in APIs since now Serrano is in alpha development.
	If you find any mismatching in guides and code, please make a PR.