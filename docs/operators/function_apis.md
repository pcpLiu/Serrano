## Class OperatorFuncs
Sometimes we may want to use Serrano as a math library to do quick tensor arithmetic calculation.
In this situation, declaring and configuring operator instances will be a little bit too much.
Serrano supports a class __OperatorFuncs__ which wraps all an operator's computation as a static function.

Example:
```swift
// log computation
let input = Tensor.randomTensor(TensorShape(dataType: .float, shape: [3, 3]))
let result = OperatorFuncs.log(input)

// add
let a = Tensor.randomTensor(TensorShape(dataType: .float, shape: [3, 3]))
let b = Tensor.randomTensor(TensorShape(dataType: .float, shape: [3, 3]))
let c = OperatorFuncs.add(a, b)

// fully connected
let input = Tensor.randomTensor(TensorShape(dataType: .float, shape: [50]))
let weight = Tensor.randomTensor(TensorShape(dataType: .float, shape: [50, 20]))
let bias = Tensor.randomTensor(TensorShape(dataType: .float, shape: [20]))
let result = OperatorFuncs.fc(input, numUnits: 20, weight: weight, bias: bias)
```

The full list of supported operator APIs can be found [here](http://serrano-lib.org/docs/latest/api/Classes/OperatorFuncs.html)
