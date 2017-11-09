FullyconnectedOperator ([API](http://serrano-lib.org/docs/latest/api/Classes/FullyconnectedOperator.html)) do regular fully connected calculation between input tensors and weights.

**Notes**

- __Auto flatten__.
	For input tensor with rank `>=2`, the operator will automatically flatten the tensor and then do calcualtion (shape kept). Actually, Serrano will do nothing to flatten it. Cause inside a tensor, all data elements are stored as a 1D array.
- If `inputTensors` has more than 1 tensor object,
	the operator applies calculation on each input tensor independently.
	Results will be stored corresponding tensor of `outputTensors`


## Initialization

```swift
let op = FullyconnectedOperator(inputDim: 100, numUnits: 20)

let op2 = FullyconnectedOperator(inputDim: 100, numUnits: 20, biasEnabled: False)
```

- `inputDim`: Input dimensions of input tensors
- `numUnits`: Number of hidden units.
- `biasEnabled`: Indicating whether use `bias`. Default is `true`.