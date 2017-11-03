//
//  computation_graph.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/7/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation

/**
## Intro
The implementation of `Graph`.
A `ComputationGraph` just a set of symbols and connections between them.
The graph computes the output tensors stage by stage.

A typical usage:

```swift

let graph = ComputationGraph()

let a = graph.tensor("A", shape: TensorShape(dataType: .float, shape: [2,3]))
let b = graph.tensor("B", shape: TensorShape(dataType: .float, shape: [2,3]))
let (c, op) = graph.operation("", inputSymbols: [a, b], op: PowOperator()).first
```

## ComputationGraph V.S. Model
`ComputationGraph` is low-level abstraction of machine learning models.
It has two basic funtions:
 - forward. Compute results from input to output
 - backward. Compute and update data symbols that are differentiable use optimizer
User needs to call `forward(:)` and `backward(:)` manually to do training and `ComputationGraph`
is not aware of _loss function_.
If you want to use loss function if a graph, you need add it into this graph as an operator symbol.

`Model` is a higher level abstraction inherited from `ComputationGraph`.
It has all functions `ComputationGraph` has and beyond that:
 - Loss function. `Model` could setup loss function to do the backward training.
   'ComputationGraph'
 - High level training functions. `Model` could automatically repeat `forward(:)` and `backward(:)` util
   reaches conditions users setup like max number of epoch, early stop etc.
 - High level prediction functions.
*/
public class ComputationGraph: Graph {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Conforms to `Graph`
	public var symbols: [String: GraphSymbol]
	
	/// Conforms to `Graph`
	public var graphLabel: String
	
	/// Conforms to `Graph`.
	/// Default is `false`.
	public var trainable: Bool = false
	
	/// Description
	public var description: String {
		get {
			return "ComputationGraph(\(self.graphLabel))"
		}
	}
	
	/// Optimizer of this graph doing backward training.
	/// Could be `nil` if just do forward calcualtion.
	public var optimizer: Optimizer? = nil
	
	/// This attribute indicates if the graph has been sorted.
	/// Initial value is `false`.
	public var sorted: Bool = false
	
	/// A dictionary stores the sorted symbols after applying topology sorting.
	/// The key is the depth value, and the value is the list of symbols in this depth stage.
	public var symbolStages: [Int: [GraphSymbol]] = [Int: [GraphSymbol]]()
	
	/// Counter of backward training
	public var epoch: Int {
		get {
			return self._epoch
		}
	}

	/// Counter of backward training
	internal var _epoch: Int = 0
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Designated init
	///
	/// - Parameters:
	///   - symbols: symbols
	///   - graphLabel: graphLabel
	///   - trainable: trainable
	public init(symbols: [String: SerranoGraphSymbol], graphLabel: String, trainable: Bool) {
		self.symbols = symbols
		self.graphLabel = graphLabel
		self.trainable = trainable
	}
	
	/// Convenience init
	///
	/// - Parameter graphLabel: graphLabel
	public convenience init(_ graphLabel: String? = nil) {
		if graphLabel != nil {
			self.init(symbols: [String : SerranoGraphSymbol](), graphLabel: graphLabel!, trainable: false)
		} else {
			self.init(symbols: [String : SerranoGraphSymbol](), graphLabel: "Serrano Graph", trainable: false)
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Util
	
	/// Generate a default label for a symbol.
	///
	/// - Parameter type: SymbolType
	/// - Returns: return value
	public func defaultSymbolLabel(_ type: SymbolType) -> String {
		return "Serrano Graph"
	}
	
	/// Get this graph's all data symbols
	///
	/// - Returns: list of data symbol
	public func dataSymbols() -> [DataSymbol] {
		return self.symbols.filter {$0.value.symbolType.isDataSymbol()}.map {$0.value as! DataSymbol}
	}
	
	
	/// Get his graph's all operator symbols
	///
	/// - Returns: list of operator symbols
	public func opSymbols() -> [OperatorSymbol] {
		return self.symbols.filter {$0.value.symbolType == SymbolType.Operator}.map {$0.value as! OperatorSymbol}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods conforms to `Graph`
	
	/// Add a `TensorSymbol` to the graph.
	///
	/// - Parameter label: label
	/// - Returns: A `TensorSymbol`
	@discardableResult
	public func tensor(_ label: String? = nil, shape: TensorShape) -> TensorSymbol {
		// new symbol, graph unsorted
		self.sorted = false
		
		var symbolLabel = label
		if symbolLabel == nil { symbolLabel = self.defaultSymbolLabel(.Tensor) }
		let tensorSymbol = SerranoTensorSymbol(symbolLabel!, dataSource: .User, shape: shape)
		self.symbols[tensorSymbol.UID] = tensorSymbol
		// logging
		SerranoLogging.stdLogging(message: "New TensorSymbol added to graph [\(self.graphLabel)]",
								  file: "\(#file)", function: "\(#function)", line: "\(#line)",
		                          loggingLevel: SerranoLoggingType.Regular)
		
		return tensorSymbol
	}
	
	/// Add a `ScalarSymbol` to the graph.
	///
	/// - Parameter label: label
	/// - Returns: A `ScalarSymbol`
	@discardableResult
	public func scalar(_ label: String? = nil, dataType: TensorDataType) -> ScalarSymbol {
		// new symbol, graph unsorted
		self.sorted = false
		
		var symbolLabel = label
		if symbolLabel == nil { symbolLabel = self.defaultSymbolLabel(.Scalar) }
		let scalarSymbol = SerranoScalarSymbol(symbolLabel!, dataType: dataType, dataSource: .User)
		self.symbols[scalarSymbol.UID] = scalarSymbol
		// logging
		SerranoLogging.stdLogging(message: "New ScalarSymbol added to graph [\(self.graphLabel)]",
			file: "\(#file)", function: "\(#function)", line: "\(#line)",
			loggingLevel: SerranoLoggingType.Regular)
		
		return scalarSymbol
	}
	
	/// Add a `OperatorSymbol` to the graph.
	/// This function will also update inBounds and outBounds information for involving symbols.
	///
	/// - Parameters:
	///   - inputs: input array of `TensorSymbol`
	///   - operator: A `ComputableOperator` instance
	/// - Returns:
	///   - outputTensorSymbols: output tensor symbols
	///   - operatorSymbol: added operator symbol
	///   - paramSymbols: Parameter symbols attached to this operator. Empty array if not available.
	@discardableResult
	public func operation(_ label: String? = nil, inputs: [TensorSymbol], op: ComputableOperator) -> (outputTensorSymbols: [TensorSymbol], operatorSymbol: OperatorSymbol, paramSymbols: [GraphSymbol]) {
		// new symbol, graph unsorted
		self.sorted = false
		
		var symbolLabel = label
		if symbolLabel == nil { symbolLabel = self.defaultSymbolLabel(.Scalar) }
		let opSymbol = SerranoOperatorSymbol(symbolLabel!, serranoOperator: op, inputSymbols: inputs)
		self.addSymbols(opSymbol)
		
		// logging
		SerranoLogging.stdLogging(message: "New OperatorSymbol added to graph [\(self.graphLabel)]",
			file: "\(#file)", function: "\(#function)", line: "\(#line)",
			loggingLevel: SerranoLoggingType.Regular)
		
		// output symbols bounds process
		let outTensorSymbols = opSymbol.outputSymbols()
		for outputSymbol in outTensorSymbols {
			let symbol = outputSymbol as GraphSymbol?
			symbol!.addToInBound(opSymbol)
			
			opSymbol.addToOutBound(outputSymbol)
	
			self.addSymbols(outputSymbol)
		}
		
		// inputs symbols bounds process
		for inputSymbol in inputs {
			let symbol = inputSymbol as GraphSymbol?
			symbol!.addToOutBound(opSymbol)
			
			self.addSymbols(inputSymbol)
		}
		
		// param symbols
		let paramSymbols = op.paramSymbols()
		for paramSymbol in paramSymbols {
			let symbol = paramSymbol as GraphSymbol?
			symbol!.addToOutBound(opSymbol)
			
			// add to opSymbol
			opSymbol.addToInBound(paramSymbol)
			opSymbol.addToParamSymbols(paramSymbol)
			
			self.addSymbols(paramSymbol)
		}
		
		return (outTensorSymbols, opSymbol, paramSymbols)
	}
	
	/// Bind data to `TensorSymbol` or `ScalarSymbol` which need feeding data from users.
	/// This function will try to bind every entry in `data` to tensor or scalar symbol in `symbols`.
	/// If could find an entry has same UID in `symbols` for passed in data, it will be ignored.
	///
	/// - Note: This function does not verify any restrictions.
	///
	/// - Parameter data: A dictinary whose key is `UID` of a symbol
	///                   and the value is a `DataSymbolSupportedDataType` object.
	public func bindData(_ data: [String: DataSymbolSupportedDataType]) {
		for (UID, entry) in data {
			let symbol = self.symbols[UID]
			// not find
			if symbol == nil {
				SerranoLogging.warningLogging(message: "Could not find symbol with UID (\(UID)) in \(self.description).",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				continue
			}
			
			guard symbol! is DataSymbol else {
				SerranoLogging.errorLogging(message: "Trying to bind data to symbol \(symbol!), but this is not a data symbol",
				                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Error raised by Serrano. Check log for details.")
			}
			
			guard (symbol! as! DataSymbol).bindData(entry) else {
				SerranoLogging.errorLogging(message: "Trying to bind data to symbol \(symbol!), but failed. Check log for details.",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("Error raised by Serrano. Check log for details.")
			}
		}
	}
	
	/// Compute the whole graph from inputs to outputs.
	/// If there's any error during calculation or pre-checking, return `nil`.
	///
	/// - Parameter mode: computation mode
	/// - Returns: Array of tensors/Scalar or `nil` if error happens
	public func forward(mode: OperatorComputationMode) -> [DataSymbolSupportedDataType]? {
		// compute in stage order
		self.stageOrderCalculate(mode: mode)
		
		// group last stage tensors and return
		let lastStage = self.symbolStages.keys.max()!
		let outputDataSymbols = self.symbolStages[lastStage]?.filter { $0.symbolType.isDataSymbol()}
		
		guard outputDataSymbols != nil else {
			SerranoLogging.errorLogging(message: "Could not gather last stage tensors. Maybe graph defined is not valid.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return nil
		}
		
		return (outputDataSymbols as! [DataSymbol]).map {$0.bindedData!}
	}
	
	/// Backward computing the grads for updatable data symbols.
	///
	/// This function supposes that graph already been verifed and done at least onece forward computation.
	/// If not, it may cause unexpected error.
	///
	/// - Note: If this graph is not `trainable`, `fatalError` will be raised.
	///
	/// - Parameters:
	///   - mode: computation mode
	public func backward(mode: OperatorComputationMode) {
		// update parameters
		self.dataSymbolsUpdate(mode)
		
		// windup
		self.windup()
		
		// increase epoch
		self._epoch += 1
		SerranoLogging.stdLogging(message: "Finish epoch \(self._epoch).",
			file: "\(#file)", function: "\(#function)", line: "\(#line)", loggingLevel: SerranoLoggingType.LowLevel)
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Forawrd related methods
	
	
	/// Prepare before forwarding.
	/// Usually should be called just before 1st fowarding.
	/// However, if user chagne graph strcture, this should be called again.
	public func forwardPrepare() {
		// sort
		self.sortGraph()
		
		// user input bind check
		var (valid, msg) = self.userInputBindCheck()
		guard valid else {
			SerranoLogging.errorLogging(message: "Some symbols havn't been binded. Detail:\(msg)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		// allocate tensors for needs
		self.allocateTensors()
		
		// verify
		(valid, msg) = self.verifyGraph()
		guard valid else {
			SerranoLogging.errorLogging(message: "Graph is not valid. Detail:\(msg)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
	}
	
	/// Use topology sorting to sort all symbols into different stages.
	/// Store results in `symbolStages`.
	/// Currently, only supports __directed acyclic graph__ (i.e. no directed cycles).
	///
	/// - Note: If there is no input data symbols, `fatalError` will be raised.
	///
	/// # Algorithm
	/// Use DFS(Depth-First Search) to construct the staged information.
	///
	/// 1. For each data symbol whose `dataSource` is `SymbolDataSource.User`:
	///
	///	    I. Add this symbol to a list with depth 0.
	///
	///		II. Use DFS to starting from this symbol following outBounds path,
	///         and add the visiting symbol and corresponding depth into list.
	///         Each time gos deeper, depth plus 1.
	///
	///     III. If the visiting symbol is already in the statck, set depth to max(this visiting depth, existing depth).
	/// 2. Then according the list information, add symbols in same stage into `symbolStages`.
	public func sortGraph() {
		// get input data symbols
		let inputDataSymbols = self.symbols.filter { (UID, symbol) -> Bool in
			return symbol.symbolType.isDataSymbol() && symbol.inBounds.count == 0
		}
		
		guard inputDataSymbols.count > 0 else {
			SerranoLogging.errorLogging(message: "Could not found input data symbols in this graph (\(self.graphLabel))." +
				"Thus cannot generate computation graph.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		// DFS
		var symbolDepthList = [SerranoGraphSymbol: Int]() // key is UID
		for (_, symbol) in inputDataSymbols {
			self.visitSymbol(symbol, symbolDepthList: &symbolDepthList, currentDepth: 0)
		}
		
		// add stage information
		for (symbol, depth) in symbolDepthList {
			if self.symbolStages[depth] == nil { self.symbolStages[depth] = [SerranoGraphSymbol]() }
			self.symbolStages[depth]!.append(symbol)
		}
		
		// set attribute
		self.sorted = true
	}
	
	/// Visist a symbol and its outBounds symbols in DFS manner.
	///
	/// - Parameters:
	///   - symbol: visiting symbol
	///   - symbolDepthList: symbolDepthList. The passed in dictionary will be mutated.
	///   - currentDepth: current visiting depth
	public func visitSymbol(_ symbol: GraphSymbol, symbolDepthList: inout [SerranoGraphSymbol: Int], currentDepth: Int) {
		let serranoSymbol = symbol as! SerranoGraphSymbol
		if symbolDepthList[serranoSymbol] == nil {
			symbolDepthList[serranoSymbol] = currentDepth
		} else {
			symbolDepthList[serranoSymbol] = max(symbolDepthList[serranoSymbol]!, currentDepth)
		}
		
		for outSymbol in symbol.outBounds {
			self.visitSymbol(outSymbol, symbolDepthList: &symbolDepthList, currentDepth: currentDepth + 1)
		}
	}
	
	/// This functions verifies:
	///	- All symbols should have been binded to data
	/// - Shape compatibility between connected operators and tensors
	///
	/// If any incorrectness found, return `false` and associated message.
	///
	/// - Note: This function assumes graph already been sorted and attribute `symbolStages` already has stage info.
	///
	/// - Returns: `valid` represent if passing the verification and error msg if has
	public func verifyGraph() -> (valid: Bool, msg: String) {
		// check all data symbols have been binded
		for (_, symbol) in self.symbols {
			if symbol.symbolType == .Scalar {
				let scalarSymbol = symbol as! ScalarSymbol
				guard scalarSymbol.bindedData != nil else {
					return (false, "ScalarSymbol(\(scalarSymbol.symbolLabel)) needs user feeded data but bindedData is nil.")
				}
			} else if symbol.symbolType == .Tensor {
				let tensorSymbol = symbol as! TensorSymbol
				guard tensorSymbol.bindedData != nil else {
					return (false, "TensorSymbol(\(tensorSymbol.symbolLabel)) needs user feeded data but bindedData is nil.")
				}
			}
		}
		
		// check if sorted
		guard self.sorted else { return (false, "Graph has not been sorted.") }
		
		// check shape matching
		let (valid, msg) = self.checkShapeChain()
		guard valid else { return (false, "\(msg)") }
		
		return (true, "")
	}
	
	/// This function check a sorted graph's every path to see if all connected operators
	/// and symbols could match each other considering `TensorShape`.
	///
	/// ## Checking
	///	```
	///	For each operator symbol in this stage:
	///		I.   Get input tensors from inBounds and output tensors from outBounds.
	///          Assign tensors to operator's inputTensors and outputTensors.
	///		II.  Call inputOutputTensorCheck().
	///			a). If return true, mark operator's disableInputOutputCheck to true.
	///				Continue.
	///			b). If return false, return false and related errror msg.
	/// ```
	///
	/// - Note: This internal function assumes the graph's all symbols have been binded.
	///
	/// - Returns: validation result and error message if has
	public func checkShapeChain() -> (valid: Bool, msg: String) {
		let opSymbols = self.symbols.filter { (UID, symbol) -> Bool in
			return symbol.symbolType == SymbolType.Operator
		}
		
		for (_, symbol) in opSymbols {
			var opSymbol = symbol as! OperatorSymbol
			
			// setup input tensors
			// Note here should use `inputSymbols` not `inBounds`.
			// Cause `inBounds` also includes `paramSymbols`.
			let inputTensors = opSymbol.inputSymbols.map { $0.bindedData! }
			opSymbol.serranoOperator.inputTensors = (inputTensors as! [Tensor])
			
			// setup output tensors
			let outputTensors = opSymbol.outBounds.map { ($0 as! TensorSymbol).bindedData! }
			opSymbol.serranoOperator.outputTensors = (outputTensors as! [Tensor])
			
			// bind parameter
			let paramSymbols = opSymbol.inBounds.filter({ (symbol) -> Bool in
				// not in input
				return !opSymbol.inputSymbols.contains { $0.UID == symbol.UID}
			})
			opSymbol.serranoOperator.bindParamSymbols(paramSymbols)
			
			let (pass, msg) = opSymbol.serranoOperator.inputOutputTensorsCheck()
			guard pass else {
				return (false, "Operator symbol \(opSymbol.symbolLabel) failed to pass inputOutputTensorsCheck(). " +
							   "Details: \(msg)")
			}
		}
		
		return (true, "")
	}
	
	/// Check if all user data source symbols have been binded.
	///
	/// - Returns: if pass checking and related error msg if has.
	public func userInputBindCheck() -> (valid: Bool, msg: String) {
		let dataSymbols = self.symbols.filter { $0.value.symbolType.isDataSymbol() }
		for (_, symbol) in dataSymbols {
			switch symbol.symbolType {
			case .Tensor:
				let tensorSymbol = symbol as! TensorSymbol
				if tensorSymbol.dataSource == .User {
					guard tensorSymbol.bindedData != nil else {
						return (false, "Tensor symbol (\(tensorSymbol.symbolLabel)) should bind data feeding from user but is nil.")
					}
				}
			case .Scalar:
				let scarlarSymbol = symbol as! ScalarSymbol
				if scarlarSymbol.dataSource == .User {
					guard scarlarSymbol.bindedData != nil else {
						return (false, "Scalar symbol (\(scarlarSymbol.symbolLabel)) should bind data feeding from user but is nil.")
					}
				}
			default:
				continue
			}
		}
		
		return (true, "")
	}
	
	
	/// Allocate tensors for all tensor symbols.
	/// This is used when training a network from zero.
	///
	/// - Note: This method ignore existing binding, allocate new tensors for all tensor symbols.
	public func allocateAllTensors() {
		for (_, symbol) in self.symbols {
			if symbol.symbolType == .Tensor {
				var tensorSymbol = symbol as! TensorSymbol
				tensorSymbol.bindedData = SerranoResourceManager.globalManager.allocateUnamangedTensor(tensorSymbol.shape)
			}
		}
	}
	
	/// Allocate tensors for all tensor symbols not feeding from users.
	/// Currently, user `SerranoResourceManager.globalManager` to allocate unmanaged resources.
	///
	/// TODO: Do memory dependency analize, reuser tensors.
	public func allocateTensors() {
		for (_, symbol) in self.symbols {
			if symbol.symbolType == .Tensor {
				var tensorSymbol = symbol as! TensorSymbol
				
				if tensorSymbol.dataSource == .User || tensorSymbol.bindedData != nil {
					continue
				} else {
					tensorSymbol.bindedData = SerranoResourceManager.globalManager.allocateUnamangedTensor(tensorSymbol.shape)
				}
			}
		}
	}
	
	
	/// Stage by stage, run all operators.
	/// 
	/// ## Algorithm
	/// ````
	/// for stage i in [0, n]:
	///		run all operators in stage i simutaneously
	/// ```
	///
	/// - Note: Graph should have been sorted. Else `fatalError` will be raised.
	internal func stageOrderCalculate(mode: OperatorComputationMode) {
		// check sorted
		guard self.sorted else {
			SerranoLogging.errorLogging(message: "Graph not sorted. Abort calculation.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		let stageWorkGroup = DispatchGroup()
		let begginTime = CFAbsoluteTimeGetCurrent()
		for stage in self.symbolStages.keys.sorted() {
			for symbol in self.symbolStages[stage]! {
				if symbol.symbolType == .Operator {
					var opSymbol = symbol as! OperatorSymbol
					stageWorkGroup.enter()
					DispatchQueue.global(qos: .userInitiated).async {
						opSymbol.serranoOperator.disableInputOutputCheck = true
						opSymbol.serranoOperator.compute(mode)
						stageWorkGroup.leave()
					}
				}
			}
			// wait all complete in this stage
			stageWorkGroup.wait()
		}
		
		SerranoLogging.stdLogging(message: "Finish forward for graph \(self.graphLabel) in \(CFAbsoluteTimeGetCurrent() - begginTime) seconds ",
			file: "\(#file)", function: "\(#function)", line: "\(#line)", loggingLevel: SerranoLoggingType.LowLevel)
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Backward related methods
	
	
	/// Prepare workd.
	internal func backwardPrepare() {
		guard self.trainable else {
			SerranoLogging.errorLogging(message: "Graph is not trainable.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError("")
		}
		
		// check opmizer not nil
		guard self.optimizer != nil else {
			SerranoLogging.errorLogging(message: "Optimizer of graph is nil. Cannot do backward training.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError("")
		}
		
		// initialize grad value tensors at the very first training
		if self._epoch == 0 {
			self.allocateTensorsForGrads()
		}
		
		// let optimizer do prepare work if need
		self.optimizer!.prepare(self)
	}
	
	/// Allocate tensors and initialize scalar values for grads of data symbol.
	///
	/// ## Initialization strategy
	/// We will only look at operators with `true` values for attribute `enabledParameterUpdate`,
	/// and will initialize currentGrad of these operator symbols' inbound data symbols with
	/// `true` value for attribute `updateble`.
	internal func allocateTensorsForGrads() {
		for opSymbol in self.opSymbols() {
			if opSymbol.enabledParameterUpdate {
				for dataSymbol in opSymbol.inBounds {
					if dataSymbol.symbolType.isDataSymbol() {
						switch(dataSymbol.symbolType) {
						case SymbolType.Scalar:
							var scalarSymbol = dataSymbol as! ScalarSymbol
							scalarSymbol.currentGrad = Float(0.0)
						case SymbolType.Tensor:
							var tensorSymbol = dataSymbol as! TensorSymbol
							tensorSymbol.currentGrad = SerranoResourceManager.globalManager.allocateUnamangedTensor(tensorSymbol.shape)
						default:
							continue
						}
					}
				}
			}
		}
	}
	
	/// This function do updating during backward training.
	///
	/// For operator symbols at each stage starting from last to first,
	/// calculate the grads for all inbound data symbols.
	/// Then update values for those are `updateble`.
	internal func dataSymbolsUpdate(_ mode: OperatorComputationMode) {
		let workGroup = DispatchGroup()
		for (_, symbols) in (self.symbolStages.sorted {$0.0.key > $0.1.key}) { // desending
			for symbol in (symbols.filter {$0.symbolType == SymbolType.Operator}) {
				workGroup.enter()
				DispatchQueue.global(qos: .userInitiated).async {
					let opSymbol = symbol as! OperatorSymbol
					// TODO: Improve API design.
					// In this function operator will allocate new tensors/scalars holding grads, which is not good.
					// The idea is that graph itself do all memory allocation, so later we can do optimization.
					let gradsInfo = opSymbol.serranoOperator.gradCompute(mode)
					for (label, gradValue) in gradsInfo {
						// get corresponding inpbound data symbol
						var inboundSymbol = opSymbol.inboundSymbolForGradLabel(label)
						guard inboundSymbol != nil else {
							SerranoLogging.errorLogging(message: "Unexpexted nil object.",
							                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
							fatalError("")
						}
						
						if inboundSymbol!.updatable {
							//  up grads
							let gradRelatedOutSymbols = opSymbol.gradRelatedOutputSymbols(onInboundSymbol: inboundSymbol!)
							let upGrads = gradRelatedOutSymbols.flatMap {$0.currentGrad} // take out nil
							
							inboundSymbol!.backwardUpdate(self.optimizer!, gradValue: gradValue, upGrads: upGrads)
						}
					}
					workGroup.leave()
				}
			}
		}
		workGroup.wait()
	}
	
	/// This function does windup work after updating all grads in this graph.
	internal func windup() {
		
	}
	
}

