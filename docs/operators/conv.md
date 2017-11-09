Currently, Serrano supports 2D convolutional operator ([API](http://serrano-lib.org/docs/latest/api/Classes/ConvOperator2D.html)).

**Notes**

- The `inputTensors` and `outputTensors` should have same number of tensors. 
- There can be multiple tensors in `inputTensors` and all tensors in `inputTensors` should have same shapes. Operator will do calculation for each input tensor independently.
- __Dilation__. Currently, calculation with `dilation > 1` has not been implemented.

## Initialization

```swift
let convOp  = ConvOperator2D(numFilters: 64,
                             kernelSize: [3, 3],
                             stride: [1, 1],
                             padMode: PaddingMode.Same,
                             channelPosition: TensorChannelOrder.Last)
```

- `numFilters`. **Required**. Number of filters (feature maps)
- `kernelSize`. **Required**. 2D `Int` array.
- `stride`. 2D `Int` array. Default is `[1, 1]`.
- `padMode`: Default is `PaddingMode.Valid`.
- `channelPosition`. Default is `TensorChannelOrder.First`
