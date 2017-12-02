Currently, Serrano supports 2D BatchNormalization ([API](http://serrano-lib.org/docs/latest/api/Classes/BatchNormOperator.html)).

**Notes**

- The `inputTensors` and `outputTensors` should have same number of tensors. 
- There can be multiple tensors in `inputTensors` and all tensors in `inputTensors` should have same shapes. Operator will do calculation for each input tensor independently.

## Initialization

```swift
let bn  = BatchNormOperator(channelOrder = TensorChannelOrder.Last)
```

- `channelOrder`. The feature channel.
