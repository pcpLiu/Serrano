Activation operators behavior like unary operator.

**Few notes:**

- The `inputTensors` and `outputTensors` should have same number of tensors. 
- Corresponding input and output tensors should have same shapes.

All unary operators can be initialized:
```swift
let op = SomeActivationOp()
```


<hr>
<hr>

## List


#### [ReLUOperator](http://serrano-lib.org/docs/latest/api/Classes/ReLUOperator.html)
$y=max(x,alpha)$

Initial: `#!swift ReLUOperator(alpha : 0.0)`

- `alpha`: `#!swift Float`, default values is `0.0`

#### [SigmoidOperator](http://serrano-lib.org/docs/latest/api/Classes/SigmoidOperator.html)
$y = 1 / (1 + e^{x})$

#### [SoftplusOperator](http://serrano-lib.org/docs/latest/api/Classes/SoftplusOperator.html)
$y = log_{e}(e^x + 1)$

#### [SoftsignOperator](http://serrano-lib.org/docs/latest/api/Classes/SoftsignOperator.html)
$y = x / (1 + abs(x))$

#### [LinearOperator](http://serrano-lib.org/docs/latest/api/Classes/LinearOperator.html)
$y = x$

#### [ELUOperator](http://serrano-lib.org/docs/latest/api/Classes/ELUOperator.html)
\begin{equation}
  y=
  \begin{cases}
    x, & \text{if}\ x>0 \\
    alpha \times (e^{x} - 1), & \text{otherwise}
  \end{cases}
\end{equation}

Initial: `#!swift ELUOperator(alpha : 1.0)`

- `alpha`: `#!swift Float`, default values is `1.0`

#### [SELUOperator](http://serrano-lib.org/docs/latest/api/Classes/SELUOperator.html)
$y = scale * ELU(x)$

Initial: `#!swift SELUOperator(alpha : 1.673263, scale: 1.050701)`

- `alpha`: `#!swift Float`, default values is `1.673263`
- `scale`: `#!swift Float`, default values is `1.050701`

#### [SoftmaxOperator](http://serrano-lib.org/docs/latest/api/Classes/SoftmaxOperator.html)
$y = exp(x) / \text{reduce_sum}(e^x, \text{dim})$

Initial: `#!swift ELUOperator(dim : = -1)`

- `dim`: `#!swift Int`,  Reduce summing dimension. 
	The value should be `>=0`. 
	Any negative value will be automatically making this attribute value to `-1`.
	`-1` is a special value indicating last dim.

#### [LeakyReLUOperator](http://serrano-lib.org/docs/latest/api/Classes/LeakyReLUOperator.html)
\begin{equation}
  y=
  \begin{cases}
    alpha \times x, & \text{if}\ x < 0 \\
    x, & \text{otherwise}
  \end{cases}
\end{equation}

Initial: `#!swift LeakyReLUOperator(alpha : 0.3)`

- `alpha`: `#!swift Float`, default values is `0.3`

#### [ThresholdedReLUOperator](http://serrano-lib.org/docs/latest/api/Classes/ThresholdedReLUOperator.html)
\begin{equation}
  y=
  \begin{cases}
    x, & \text{if}\ x > \text{alpha} \\
    0, & \text{otherwise}
  \end{cases}
\end{equation}

Initial: `#!swift ThresholdedReLUOperator(alpha : 1.0)`

- `alpha`: `#!swift Float`, default values is `1.0`