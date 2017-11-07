# Graph and Symbol

## Symbols

!!! note "Coming soon..":
	Under construction

## ComputationGraph
`ComputationGraph`([API](http://serrano-lib.org/docs/latest/api/Classes/ComputationGraph.html)) is the basic computation model class.
All other computation model are based on this class.

#### Forward 
A typical forward computation with `ComputationGraph` like below:
```swift
let g = ComputationGraph()

/// Add input and operators
/// ...

/// This only needs to be called before 1st forward
g.forwardPrepare()

let results = g.forward()
```

Below figure illustrates the forward process in `ComputationGraph`:

![Forward Process](/imgs/graph_forward_process.png)

!!! note "`ComputationGraph` and `ForwardGraph`":
	In `ComputationGraph` forward computation, Serrano does not do any memory optimization.
	Because a `ComputationGraph` stores all inner results for each operator's output.

	If you want to use less memory and only cares about a forward computation's final results,
	you can use `ForwardGraph`([API](http://serrano-lib.org/docs/latest/api/Classes/ForwardGraph.html)). Check guide [here](/Getting started/Forward Computation.md).