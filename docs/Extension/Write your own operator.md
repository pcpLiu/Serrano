# Write your own operator

Like said in [here](/Getting started/Core concepts/Operator.md), to define your own operator you just need to create a class conforms to `ComputableOperator` ([API](http://serrano-lib.org/docs/latest/api/Protocols/ComputableOperator.html)). In this guide, we will go through a complete example introducing how to define your own operator.



## `x10` Operator
Here we are going to create an operator which gonna times `10` on its input tensors.
First, lets create a class named `x10Operator`:
```swift
public class x10Operator {

}
```

#### Conforms to `ComputableOperator`
Next, we make this class conforms to `ComputableOperator`:
```swift
import Serrano

public class x10Operator: ComputableOperator {

}
```
And obviously, there will be error information saying that we didn't implemented required attributes and functions in `ComputableOperator`.
So next we first meet all requirement and for functions we fill `fatalError("Not implemented")` in it.

```swift
import Serrano

public class x10Operator: ComputableOperator {
	public var computationDelegate: OperatorCalculationDelegate?
	
	public var metalKernelFuncLabel: String
	
	public var operatorLabel: String
	
	public var inputTensors: [Tensor]?
	
	public var outputTensors: [Tensor]?
	
	public var disableInputOutputCheck: Bool
	
	public var trainable: Bool
	
	public var mapType: OperatorMappingType

	/////////////////////////////////////////////////////////////////////////////
	// Forward related functions
	
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		fatalError("Not implemented")
	}
	
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
		fatalError("Not implemented")
	}
	
	public func compute(_ computationMode: OperatorComputationMode) {
		fatalError("Not implemented")
	}
	
	public func computeAsync(_ computationMode: OperatorComputationMode) {
		fatalError("Not implemented")
	}

	/////////////////////////////////////////////////////////////////////////////
	// Backward related functions
	
	public func gradCompute(_ computationMode: OperatorComputationMode) -> [String : DataSymbolSupportedDataType] {
		fatalError("Not implemented")
	}
	
	public func gradComputAsync(_ computationMode: OperatorComputationMode) {
		fatalError("Not implemented")
	}

	/////////////////////////////////////////////////////////////////////////////	
	// Graph construction related functions

	public func bindParamSymbols(_ symbols: [GraphSymbol]) {
		fatalError("Not implemented")
	}
	
	public func paramSymbols() -> [GraphSymbol] {
		fatalError("Not implemented")
	}
}
```

Right now it may looks like a lot of work needs to be done, but don't be afraid.
We can do it in easy way. :sunglasses:

The code comments separate the functions into 3 sections:

- Forward computation functions.
- Backward computation functions.
- Graph construction related functions.

We will talk about each part later. First, we need to add the init function to the class.
It's preferred that just define one `init` function and set all optional attributes with default values:
```swift
import Foundation
import Serrano

public class x10Operator: ComputableOperator {
	public var computationDelegate: OperatorCalculationDelegate?
	
	// forward related attributes
	
	public var metalKernelFuncLabel: String = "x10Operator"
	
	public var operatorLabel: String
	
	public var inputTensors: [Tensor]?
	
	public var outputTensors: [Tensor]?
	
	public var disableInputOutputCheck: Bool
	
	// backwar related attributes
	
	public var trainable: Bool
	
	public var mapType: OperatorMappingType = OperatorMappingType.OneToOne
	
	/////////////////////////////////////////////////////////////////////////////
	// init
	public init(operatorLabel: String = "x10Operator",
				computationDelegate: OperatorCalculationDelegate? = nil,
				inputTensors: [Tensor]? = nil,
				outputTensors: [Tensor]? = nil,
				disableInputOutputCheck: Bool = false,
				trainable: Bool = false) {
		self.operatorLabel = operatorLabel
		self.computationDelegate = computationDelegate
		self.inputTensors = inputTensors
		self.outputTensors = outputTensors
		self.disableInputOutputCheck = disableInputOutputCheck
		self.trainable = trainable
	}
	
	/////////////////////////////////////////////////////////////////////////////
	// Forward related functions
	
	public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
		fatalError("Not implemented")
	}
	
	public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
		fatalError("Not implemented")
	}
	
	public func compute(_ computationMode: OperatorComputationMode) {
		fatalError("Not implemented")
	}
	
	public func computeAsync(_ computationMode: OperatorComputationMode) {
		fatalError("Not implemented")
	}
	
	/////////////////////////////////////////////////////////////////////////////
	// Backward related functions
	
	public func gradCompute(_ computationMode: OperatorComputationMode) -> [String : DataSymbolSupportedDataType] {
		fatalError("Not implemented")
	}
	
	public func gradComputAsync(_ computationMode: OperatorComputationMode) {
		fatalError("Not implemented")
	}
	
	/////////////////////////////////////////////////////////////////////////////
	// Graph construction related functions
	
	public func bindParamSymbols(_ symbols: [GraphSymbol]) {
		fatalError("Not implemented")
	}
	
	public func paramSymbols() -> [GraphSymbol] {
		fatalError("Not implemented")
	}
	
	
}
```
Notice that we assign default values to `metalKernelFuncLabel` and `mapType`. Because, for a fixed operator, this attribute should be constant.

#### Forward computation

Now let's begin to do forward computation implementation.

__Output shapes from input shapes__

The 1st function `outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]?`. This function give output shapes for given input shapes calculated through an operator.
Since `x10` is an unary operator, the output shapes are same as input shapes:
```swift
public func outputShape(shapeArray shapes: [TensorShape]) -> [TensorShape]? {
	// You can add some error checking code here.
	// Like checking if input shapes is empty or not based on your condition.
	return shapes
}
```

__Input and output compatible validation__

Next function we will implement is `inputOutputTensorsCheck() -> (check: Bool, msg: String)`.
This function check if the operator's `inputTensors` and `outputTensors` are valid for computation. 
In our case, the corresponding tensors in `inputTensors` and `outputTensors` should have same shape:
```swift
public func inputOutputTensorsCheck() -> (check: Bool, msg: String) {
	// check input nil
	guard self.inputTensors != nil else {
		return (false, "Attribute inputTensors are nil")
	}
	
	// check output nil
	guard self.outputTensors != nil else {
		return (false, "Attribute outputTensors are nil")
	}
	
	// same #
	guard self.inputTensors!.count == self.outputTensors!.count else {
		return (false, "inputTensors count \(self.inputTensors!.count) while outputTensors count \(self.outputTensors!.count)")
	}
	
	// input and output has same shape for corrsponding tensors
	for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
		guard input.shape == output.shape else {
			return (false, "Input tensor with shape \(input.shape) is not compatible with output tensor with shape \(output.shape)")
		}
	}
	
	return (true, "")
}
```

!!! note "Attribute `disableInputOutputCheck`":
	When a operator's attribute `disableInputOutputCheck` is set to `true`. This operator should not call `inputOutputTensorsCheck` before calculation. This usually happens when the operator is created inside Serrano like constructing a `Graph` model. Serrano allocated the input and output tensors and can make sure they are compatible. So disable this function could speed up the forward computation. Later we will see how to use this attribute.

__Computation__

Now we can do the computation stuff.
An operator has two way to do the computation: sync or async.
In most cases, implementing async function is just calling sync in background.
```swift
public func computeAsync(_ computationMode: OperatorComputationMode) {
		DispatchQueue.global(qos: .userInitiated).async {
			self.computationDelegate?.operatorWillBeginComputation(self)
			self.compute(computationMode)
			self.computationDelegate?.operatorDidEndComputation(self, outputTensors: self.outputTensors!)
		}
	}
```

Then we can focus on computation method in sync function.
An operator should support both CPU and GPU calculation.
We declare two new functions `cpu()` and `gpu()` and declare them as `internal` functions cause we don't need to expose these details to other users.
However, it's up to you.
```swift
internal func cpu() {
	// implement later...
}

internal func gpu() {
	// implement later...
}

```
We will come back to these functions later.
Let's fill `compute(_ computationMode: OperatorComputationMode)`:
```swift
public func compute(_ computationMode: OperatorComputationMode) {
	if !self.disableInputOutputCheck {
		self.inputOutputTensorsCheck()
	}
	
	switch computationMode {
	case .CPU:
		self.cpu()
	case .GPU:
		if SerranoEngine.configuredEngine.hasAvailableGPU() {
			self.gpu()
		} else {
			self.cpu()
		}
	case .Auto:
		// This is very arbitraty. Implement it according to your needs.
		self.gpu()
	}
}
```
In this implementation, we dispatch the calculation to `cpu()` or `gpu()` according to parameter `computationMode`.
A notice is that, before we call `gpu()` we should check if there's available GPU device.

!!! Warning "Auto mode":
	`Auto` mode in `OperatorComputationMode` right now is being evaluated. It may be canceled in future if we found it is not practicable.

__CPU__

Now we can implement `cpu()` function:
```swift
internal func cpu() {
	let workGroup = DispatchGroup()
	var timeValue: Float = 10.0
	for (input, output) in zip(self.inputTensors!, self.outputTensors!) {
		workGroup.enter()
		DispatchQueue.global(qos: .userInitiated).async {
			vDSP_vsmul(input.contentsAddress, 1, &timeValue, output.contentsAddress, 1, vDSP_Length(input.count))
			workGroup.leave()
		}
	}
	workGroup.wait()
}
```
Here we use `DispatchGroup` to do the paralleling computing for each input.
[vDSP_vsmul](https://developer.apple.com/documentation/accelerate/1450020-vdsp_vsmul) is used as a speedup.

__GPU__

Here's GPU calculation code:
```swift
internal func gpu() {
	// prepare resource
	let engine = SerranoEngine.configuredEngine
	var kernel: MTLComputePipelineState?
	var commandBuffer: MTLCommandBuffer?
	var inputBuffers: [MTLBuffer] = [MTLBuffer]()
	var resultBuffers: [MTLBuffer] = [MTLBuffer]()
	
	//// kernel
	var info = ""
	(kernel, info) = engine.loadGPUKernel(kernelLabel: self.metalKernelFuncLabel)
	guard kernel != nil else {
		fatalError("[Serrano] Failed to load kernel \(self.metalKernelFuncLabel). Info: \(info)")
	}
		
	//// command buffer
	commandBuffer = engine.serranoCommandQueue?.makeCommandBuffer()
	guard commandBuffer != nil else {
		fatalError("[Serrano] Failed to make new command buffer.")
	}

	for input in self.inputTensors! {
		inputBuffers.append(engine.GPUDevice!.makeBuffer(bytesNoCopy: input.contentsAddress,
														 length: input.allocatedBytes,
														 options: MTLResourceOptions.storageModeShared)
		)
	}
	
	for output in self.outputTensors! {
		resultBuffers.append(engine.GPUDevice!.makeBuffer(bytesNoCopy: output.contentsAddress,
														 length: output.allocatedBytes,
														 options: MTLResourceOptions.storageModeShared)
		)
	}
	
	for index in 0..<inputBuffers.count {
		// encoder
		let encoder = commandBuffer!.makeComputeCommandEncoder()
		encoder.setComputePipelineState(kernel!)
		encoder.setBuffer(inputBuffers[index], offset: 0, at: 0)
		encoder.setBuffer(resultBuffers[index], offset: 0, at: 1)
		
		// dispatch
		let threadsPerThreadgroup = MTLSizeMake(kernel!.threadExecutionWidth,
												1,
												1)
		let threadgroupsPerGrid = MTLSizeMake((self.inputTensors![index].count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
											  1,
											  1)
		encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
		encoder.endEncoding()
	}
	
	// commit command buffer
	commandBuffer!.commit()
	commandBuffer!.waitUntilCompleted()
}
```

And we need a Metal file to store our kernel:
```c++
#include <metal_stdlib>
using namespace metal;


kernel void x10Operator(device float* in_tensor    [[ buffer(0) ]],
                        device float* out_tensor   [[ buffer(1) ]],
                        uint2 gid                  [[ thread_position_in_grid ]])
{
	out_tensor[gid.x] = in_tensor[gid.x] * 10.0f;
}

```

Now, we basically have everything needs to do forward computation.
