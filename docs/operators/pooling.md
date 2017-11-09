Currently, Serrano supports 2D Pooling operators.

**Notes**

- The `inputTensors` and `outputTensors` should have same number of tensors. 


## Initialization
```swift
let op = MaxPool2DOperator(kernelSize: [2, 2],
                           channelPosition: TensorChannelOrder.Last,
                           paddingMode: PaddingMode.Valid)
```

- `kernelSize`. **Required**. 2D `Int` array.
- `paddingMode`: Default is `PaddingMode.Valid`.
- `channelPosition`. Default is `TensorChannelOrder.First`

<hr>
<hr>

## List

#### [MaxPool2DOperator](http://serrano-lib.org/docs/latest/api/Classes/MaxPool2DOperator.html)
Max pooling

#### [AvgPool2DOperator](http://serrano-lib.org/docs/latest/api/Classes/AvgPool2DOperator.html)
Average pooling

#### [SumPool2DOperator](http://serrano-lib.org/docs/latest/api/Classes/SumPool2DOperator.html)
Sum pooling