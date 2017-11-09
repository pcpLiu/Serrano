The binary operator has a collection of basic binary calculation, like add, substraction and times.
A binary operator do **element-wise** calculation on two input tensors and store result in  output tensor.

**Few notes:**

- The `inputTensors` should exactly have 2 tensors and `outputTensors` should have 1 tensor. 
- All input and output tensors should have same shape.

All binary operators can be initialized:
```swift
let op = SomeBinaryOp()
```

<hr>
<hr>

## List

#### [AddOperator](http://serrano-lib.org/docs/latest/api/Classes/AddOperator.html)
For input tensors `a` and `b`, output tensor `c`: $c = a+b$

#### [SubOperator](http://serrano-lib.org/docs/latest/api/Classes/SubOperator.html)
For input tensors `a` and `b`, output tensor `c`: $c = a-b$

#### [MultOperator](http://serrano-lib.org/docs/latest/api/Classes/MultOperator.html)
For input tensors `a` and `b`, output tensor `c`: $c = a \times b$

#### [DivOperator](http://serrano-lib.org/docs/latest/api/Classes/DivOperator.html)
For input tensors `a` and `b`, output tensor `c`: $c = a / b$

#### [RDivOperator](http://serrano-lib.org/docs/latest/api/Classes/RDivOperator.html)
For input tensors `a` and `b`, output tensor `c`: $c = b / a$

#### [RDivOperator](http://serrano-lib.org/docs/latest/api/Classes/RDivOperator.html)
For input tensors `a` and `b`, output tensor `c`: $c = b / a$

#### [PowOperator](http://serrano-lib.org/docs/latest/api/Classes/PowOperator.html)
For input tensors `a` and `b`, output tensor `c`: $c = a^{b}$
