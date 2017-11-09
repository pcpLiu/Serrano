The unary operator has a collection of basic unary calculation, like Absolute, trigonometric functions etc.
An unary operator do **element-wise** calculation on each input tensor and store result in corresponding output tensor.

**Few notes:**

- The `inputTensors` and `outputTensors` should have same number of tensors. 
- Corresponding input and output tensors should have same shapes.

All unary operators can be initialized:
```swift
let op = SomeUnaryOp()
```

<hr>
<hr>

## Trigonometric

#### [SinOperator](http://serrano-lib.org/docs/latest/api/Classes/SinOperator.html)
For each input tensor `x` and output tensor `y`: $y = sin(x)$

#### [SinhOperator](http://serrano-lib.org/docs/latest/api/Classes/SinhOperator.html)
For each input tensor `x` and output tensor `y`: $y = sinh(x)$

#### [TanOperator](http://serrano-lib.org/docs/latest/api/Classes/TanOperator.html)
For each input tensor `x` and output tensor `y`: $y = tan(x)$

#### [TanhOperator](http://serrano-lib.org/docs/latest/api/Classes/TanhOperator.html)
For each input tensor `x` and output tensor `y`: $y = tanh(x)$

#### [CosOperator](http://serrano-lib.org/docs/latest/api/Classes/CosOperator.html)
For each input tensor `x` and output tensor `y`: $y = cos(x)$

#### [CoshOperator](http://serrano-lib.org/docs/latest/api/Classes/CoshOperator.html)
For each input tensor `x` and output tensor `y`: $y = cosh(x)$

#### [ArcsinOperator](http://serrano-lib.org/docs/latest/api/Classes/ArcsinOperator.html)
For each input tensor `x` and output tensor `y`: $y = acsin(x)$

#### [ArcsinhOperator](http://serrano-lib.org/docs/latest/api/Classes/ArcsinhOperator.html)
For each input tensor `x` and output tensor `y`: $y = acsinh(x)$

#### [ArccosOperator](http://serrano-lib.org/docs/latest/api/Classes/ArccosOperator.html)
For each input tensor `x` and output tensor `y`: $y = arccos(x)$

#### [ArccosOperator](http://serrano-lib.org/docs/latest/api/Classes/ArccosOperator.html)
For each input tensor `x` and output tensor `y`: $y = arccos(x)$

#### [ArccoshOperator](http://serrano-lib.org/docs/latest/api/Classes/ArccoshOperator.html)
For each input tensor `x` and output tensor `y`: $y = arctanh(x)$

#### [ArctanhOperator](http://serrano-lib.org/docs/latest/api/Classes/ArctanhOperator.html)
For each input tensor `x` and output tensor `y`: $y = arctanh(x)$

#### [DegreeOperator](http://serrano-lib.org/docs/latest/api/Classes/DegreeOperator.html)
Convert radians to degrees.
For each input tensor `x` and output tensor `y`: $y = x \times 180/  \pi$

#### [RadianOperator](http://serrano-lib.org/docs/latest/api/Classes/RadianOperator.html)
Convert degree to radian.
For each input tensor `x` and output tensor `y`: $y = x \times \pi / 180$

## Round and abs

#### [FloorOperator](http://serrano-lib.org/docs/latest/api/Classes/FloorOperator.html)
For each input tensor `x` and output tensor `y`: $y = floor(x)$

#### [CeilOperator](http://serrano-lib.org/docs/latest/api/Classes/CeilOperator.html)
For each input tensor `x` and output tensor `y`: $y = ceil(x)$

#### [RintOperator](http://serrano-lib.org/docs/latest/api/Classes/RintOperator.html)
Operator computes element-wise rounded value to the nearest integer of the input.
For each input tensor `x` and output tensor `y`: $y = rint(x)$

#### [RoundOperator](http://serrano-lib.org/docs/latest/api/Classes/RoundOperator.html)
Operator computes element-wise rounded value to the truncating integers of input values.
For each input tensor `x` and output tensor `y`: $y = round(x)$

#### [AbsOperator](http://serrano-lib.org/docs/latest/api/Classes/AbsOperator.html)
Compute element-wise abs values of input tensors.
For each input tensor `x` and output tensor `y`: $y =abs(x)$

## Exponential, sqrt and logarithm

#### [SquareOperator](http://serrano-lib.org/docs/latest/api/Classes/SquareOperator.html)
For each input tensor `x` and output tensor `y`: $y = x \times x$

#### [SqrtOperator](http://serrano-lib.org/docs/latest/api/Classes/SqrtOperator.html)
For each input tensor `x` and output tensor `y`: $y = sqrt(x)$

#### [ExpOperator](http://serrano-lib.org/docs/latest/api/Classes/ExpOperator.html)
For each input tensor `x` and output tensor `y`: $y = e^{x}$

#### [Expm1Operator](http://serrano-lib.org/docs/latest/api/Classes/Expm1Operator.html)
For each input tensor `x` and output tensor `y`: $y = e^{x} - 1$

#### [RsqrtOperator](http://serrano-lib.org/docs/latest/api/Classes/RsqrtOperator.html)
For each input tensor `x` and output tensor `y`: $y = 1 / sqrt(x)$

#### [LogOperator](http://serrano-lib.org/docs/latest/api/Classes/LogOperator.html)
For each input tensor `x` and output tensor `y`: $y = log_{e}(x)$

#### [Log2Operator](http://serrano-lib.org/docs/latest/api/Classes/Log2Operator.html)
For each input tensor `x` and output tensor `y`: $y = log_{2}(x)$

#### [Log10Operator](http://serrano-lib.org/docs/latest/api/Classes/Log10Operator.html)
For each input tensor `x` and output tensor `y`: $y = log_{10}(x)$

#### [Log1pOperator](http://serrano-lib.org/docs/latest/api/Classes/Log1pOperator.html)
For each input tensor `x` and output tensor `y`: $y = log_{e}(1+x)$

