# Welcome to Serrano Guides

![Logo](https://github.com/pcpLiu/Serrano/raw/master/logo.png "Serrano")


This guide site introduces the basic concepts and usage of [Serrano](https://github.com/pcpLiu/Serrano).
Serrano was developed as a tool help researchers and developers quickly implementing deep learning models on iOS and macOS.

### Design
- __Like the deep learning library you used before__. Serrano aims to offer similar programing experience in Swift as popular deep learning frameworks.
- __Easy to customize.__ Adding a new Operator/Optimizer/Loss Function is simple. Create a Swift class and conform the protocol, that's it.
- __Flat architecture.__ In Serrano, we try best to reduce complex of architecture by less abstraction. You could be clear with the whole architecture by grasping few core concepts. Also in this way, we hope everyone could easily contribute to this project.

<hr>


#### Quick Start: Simple CNN

###### Low-level Graph API

In Serrano, a `Graph` represent your computation model. It allows you to construct arbitrary graphs of Operators.
`Operator` in a graph, can be viewed as layer in other frameworks, is the basic calculation unit which takes into some tensors and output some tensors.
The `Tensor` object is a N-D array implemented in Serrano, like array in NumPy.

First, let's construct a empty graph:
```swift
import Serrano
let g = ComputationGraph()
// In Serrano, class ComputationGraph conforms to protocol Graph and you should use ComputationGraph to initialize a graph model.
```

Then, we can add an input tensor for our constructed graph
```swift
let shape = TensorShape(dataType: .float, shape: [244, 244, 3]) // shape of the tensor
let input = g.tensor(shape: shape) // add an input tensor
```

Next, let's add a convolutional layer, max pooling layer and a fully connected layer into this graph.
```swift
// CNN
let convOp  = ConvOperator2D(numFilters: 96,
         kernelSize: [11,11],
         stride: [4, 4],
         padMode: PaddingMode.Valid,
         channelPosition: TensorChannelOrder.Last,
         inputShape: input.shape)
let (convOut, _, _) = g.operation(inputs: [input], op: convOp)

// MaxPooling
let maxPool = MaxPool2DOperator(kernelSize: [2, 2], stride: [2, 2],
        channelPosition: TensorChannelOrder.Last,
        paddingMode: PaddingMode.Valid)
let (poolOut, _, _) = g.operation(inputs: [convOut], op: maxPool)

// Fully connected
let fc = FullyconnectedOperator(inputDim: poolOut.first!.shape.count,
        numUnits: 200)
let _ = g.operation(inputs: [poolOut], op: fc)

```

Right now we have a computation graph. Next you can bind the weight of layers from your trained models. Here we can quickly initialize weights and do forward computation.
```swift
g.allocateAllTensors()
g.forwardPrepare()

let _ = SerranoEngine.configuredEngine.configureEngine(computationMode: .GPU) // prepare GPU device
let results = g.forward(mode: .GPU) // or g.forward(mode: .CPU)
```


The complete and executable code are in [Serrano/Examples/Graph/SimpleCNN.swift]()

Also, there's an VGG16 example listed in [Serrano/Examples/Graph/VGG16.swift](https://github.com/pcpLiu/Serrano/blob/master/Examples/Graph/VGG16.swift).


###### High-level Model API

- Note: Model is high-level API which allow user to construct graph more quickly and easily.
Currently, it is under developing.

<hr>


#### Imperative Computation with Serrano
All codes talked about before is focusing on constructing a static computation graph and then feed the data and get result.
During this process, Serrano will try best to reuse memory and do multiprocessing.
However, in case if you need imperative programming, i.e. get result immediately after each operation, Serrano is coming with this capability.

The way is simple, you just do not construct a Graph!
Directly play with operators and tensors.
Below is a code fragment using convolutional operator to do calcualtion:
```swift
let inputTensor = randomTensor(fromShape: TensorShape(dataType: .float, shape: [244, 244, 3]))

// conv
let convOp = ConvOperator2D(numFilters: 96,
       kernelSize: [11,11],
       stride: [4, 4],
       padMode: PaddingMode.Valid,
       channelPosition: TensorChannelOrder.Last)
// Initialize a tensor object to store convOp's result.
// In serrano, operator cannot allocate memeory for output tensors so that it can control memory allcoation precisely.
let convOutputs = SerranoResourceManager.globalManager.allocateUnamangedTensors(convOp.outputShape(shapeArray: [inputTensor.shape])!)
convOp.inputTensors = [inputTensor]
convOp.outputTensors = convOutputs
convOp.compute(.GPU)
```

Complete code can be found [Serrano/Examples/Imperative/SimpleCNN_imperative.swift]()

<hr>

#### Install Serrano

###### CocoaPods
The simplest way installing Serrano is via [CocoaPods](https://cocoapods.org/). 
Add below code into your `podfile`:
```ruby
pod 'Serrano', :git => 'https://github.com/pcpLiu/Serrano.git'
```

###### Download and integrate
Download Serrano from [https://github.com/pcpLiu/Serrano](https://github.com/pcpLiu/Serrano).
Drag `serrano.xcodeproj` into your current project or workspace:

![Install](https://78.media.tumblr.com/da8804b3b18d81c471dbae07d29bf957/tumblr_oyumqmb7CT1wghhyvo1_400.png "Install")

In your target's `Build Phases`, add `Serrano` framework into `Target Dependencies` settings:

![Install](https://78.media.tumblr.com/68426c9a5bc22a8ea75b17f244e7b312/tumblr_oyuneosA3n1wghhyvo1_1280.png "Install")

<hr>

#### Support

###### Bug reports
You can open bug-related and __only__ bug-related issues in [Serrano issues](https://github.com/pcpLiu/Serrano/issues).

###### Questions and General Discussion
For any other issues, questions and discussion, please go to [SerranoExplore](https://github.com/pcpLiu/SerranoExplore). You are free to open any issue there.

Also, there's [Gitter chatting room](https://gitter.im/SerranoFramework/Lobby) where you can ask questions. 

<hr>

#### Help Improve Guides
Since Serrano is under alpha-development, guides listed in this site may be outdated.
If you find any error or misleading information, you are welcome to make a PR.
For any guides improvement suggestions or better --- adding more guides --- please do not hesitate to make a PR.


