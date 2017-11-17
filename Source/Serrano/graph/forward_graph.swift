//
//  forward_graph.swift
//  Serrano
//
//  Created by ZHONGHAO LIU on 11/6/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation


//TODO: Allocate a whole tensor for param and calcualtion to save memeory

/**
`ForwardGraph` is a subclass of `ComputationGraph` that can only do forward computation.
Thus, it has mooptimization during forward computation and the internal results are not accessable.
*/
public class ForwardGraph: ComputationGraph {
	
	
//	/// Get shared calcualtion tensor size lsit for an operator
//	///
//	/// - Parameter opSymbol:
//	/// - Returns:
//	internal func shareCalTensorSizeList(_ opSymbol: OperatorSymbol) -> [Int] {
//		var thisOpsSizeList = [Int]()
//
//		let inputCalSymbols = opSymbol.inputSymbols.filter {$0.dataSource == SymbolDataSource.Calculation}
//		let inputCalSymbolsSizeList = inputCalSymbols.map {$0.shape.count}
//
//		let outputCalSymbols = (opSymbol.outBounds as! [TensorSymbol]).filter {$0.dataSource == SymbolDataSource.Calculation}
//		let outputCalSymbolsSizeList = outputCalSymbols.map {$0.shape.count}
//
//		// share size list of this stage
//		thisOpsSizeList.append(contentsOf: inputCalSymbolsSizeList)
//		thisOpsSizeList.append(contentsOf: outputCalSymbolsSizeList)
//
//		// sorted from biggest to smallest
//		thisOpsSizeList = thisOpsSizeList.sorted(by: <)
//
//		// If operator can do inplace operation, using same tensors for input and output
//		if opSymbol.serranoOperator.inPlaceble {
//			// get tensors needed
//			let count = max(outputCalSymbolsSizeList.count, inputCalSymbolsSizeList.count)
//			// get max size list
//			thisOpsSizeList = Array(thisOpsSizeList.prefix(count))
//		}
//
//		return thisOpsSizeList
//	}
//
//	/**
//	When allocate tensors for graph, this function will try best to reuse tensors for `TensorSymbol` whose
//	`SymbolDataSource` is `Calculation`. And for input and output of same
//	*/
//	override public func allocateAllTensors() {
//		// Calculation tensor of all
//		var allCalTensorShareSizeList = [Int]()
//
//		// This list contains share size list for each stage
//		var stagedCalTensorShareSizeList = [[Int]]()
//
//		// max calculation tensors needed among all stage
//		var maxCalTensorsNeeded = 0
//
//		// Analyze each stage, decide how many share tensors needed
//		// According to inPlaceble decides how manay calculation tensors we need for each stage
//		for stage_index in self.symbolStages.keys.sorted() {
//			// shared tensors size considering in-place situation
//			var calShareSizeList = [Int]()
//			let stageOps = self.symbolStages[stage_index]!.filter {$0.symbolType == SymbolType.Operator} as! [OperatorSymbol]
//			for opSymbol in stageOps {
//				let thisOpsSizeList = self.shareCalTensorSizeList(opSymbol)
//				calShareSizeList.append(contentsOf: thisOpsSizeList)
//			}
//
//			maxCalTensorsNeeded = max(maxCalTensorsNeeded, calShareSizeList.count)
//
//			allCalTensorShareSizeList.append(contentsOf: calShareSizeList)
//
//			stagedCalTensorShareSizeList.append(calShareSizeList)
//
//			print("################################")
//			print("Stage ", stage_index)
//			print("size list ", calShareSizeList)
//			print("################################")
//		}
//
//		////////////////////////////////////////////////////////
//		////////////////////////////////////////////////////////
//
//		// Sort cal tensor list so that we can decid what size to allocate
//		allCalTensorShareSizeList = Array(Set(allCalTensorShareSizeList.sorted(by: <)))
//
//
//		// allocate biggest maxCalTensorsNeeded tensors with max sizes
//		var sharedTensors = [Tensor]()
//		for size in allCalTensorShareSizeList.suffix(maxCalTensorsNeeded) {
//			sharedTensors.append(Tensor(repeatingValue: 0.0,
//										tensorShape: TensorShape(dataType: .float, shape: [size])))
//		}
//		let sharedTensorSizeList = sharedTensors.map {$0.shape.count}
//
//		print("################################")
//		for tensor in sharedTensors {
//			print("Share tensor: ", tensor.description)
//		}
//		print("################################")
//
//
//		////////////////////////////////////////////////////////
//		////////////////////////////////////////////////////////
//
//		// asssing to cal symbols
//		for stage_index in self.symbolStages.keys.sorted() {
//			var occupiedIndex = [Int]()
//			let stageOps = self.symbolStages[stage_index]!.filter {$0.symbolType == SymbolType.Operator} as! [OperatorSymbol]
//			for opSymbol in stageOps {
//				let inputCalSymbols = opSymbol.inputSymbols.filter {$0.dataSource == SymbolDataSource.Calculation}.sorted {(a,b) -> Bool in
//					return a.shape.count > b.shape.count
//				}
//
//				let outputCalSymbols = (opSymbol.outBounds as! [TensorSymbol]).filter {$0.dataSource == SymbolDataSource.Calculation}.sorted {(a,b) -> Bool in
//					return a.shape.count > b.shape.count
//				}
//
//
//				// assign input
//				var inputOccupiedIndex = [Int]()
//				for symbol in inputCalSymbols {
//					if symbol.bindedData != nil {
//						let index = sharedTensorSizeList.index {$0 >= symbol.shape.count}!
//						inputOccupiedIndex.append(index)
//						continue
//					}
//
//					var tensorSymbol = symbol
//					var index = sharedTensorSizeList.index {$0 >= symbol.shape.count}!
//					while occupiedIndex.contains(index) {
//						index += 1
//					}
//					tensorSymbol.bindedData = sharedTensors[index]
//					occupiedIndex.append(index)
//					inputOccupiedIndex.append(index)
//				}
//
//				// assign output
//				for (i, symbol) in outputCalSymbols.enumerated() {
//					if symbol.bindedData != nil {
//						continue
//					}
//
//					var tensorSymbol = symbol
//					var index = sharedTensorSizeList.index {$0 >= symbol.shape.count}!
//					if opSymbol.serranoOperator.inPlaceble &&  i < inputOccupiedIndex.count {
//						tensorSymbol.bindedData = sharedTensors[inputOccupiedIndex[i]]
//					} else {
//						while occupiedIndex.contains(index) || (inputCalSymbols.filter {$0.bindedData!.tensorValue == sharedTensors[index]}.count > 0) {
//							index += 1
//						}
//						tensorSymbol.bindedData = sharedTensors[index]
//						occupiedIndex.append(index)
//					}
//				}
//			}
//		}
//
//		// allocate param symbols .!!super!!
//		super.allocateAllTensors()
//
//		for stage_index in self.symbolStages.keys.sorted() {
//			let stageOps = self.symbolStages[stage_index]!.filter {$0.symbolType == SymbolType.Operator} as! [OperatorSymbol]
//			print("Stage: ", stage_index)
//			for opSymbol in stageOps {
//				print("Op:", opSymbol.serranoOperator.operatorLabel)
//				for inputSymbol in (opSymbol.inputSymbols.filter {$0.dataSource == SymbolDataSource.Calculation}) {
//					print("Cal inputSymbol, ", inputSymbol.UID,", shape:", inputSymbol.shape, ", binded", inputSymbol.bindedData!.description)
//				}
//				for outputSymbol in ((opSymbol.outBounds as! [TensorSymbol]).filter {$0.dataSource == SymbolDataSource.Calculation}) {
//					print("Cal outputSymbol, ", outputSymbol.UID,", shape:", outputSymbol.shape, ", binded", outputSymbol.bindedData!.description)
//				}
//
//			}
//			print("################################")
//		}
//	}
	
	/**
	When allocate tensors for graph, this function will try best to reuse tensors for `TensorSymbol` whose
	`SymbolDataSource` is `Calculation`
	*/
	override public func allocateAllTensors() {
		// get max number tensors
		var allCalculationTensorSymbolList = [TensorSymbol]()
		var stageCalculationTensorSymbol = [[TensorSymbol]]()
		var maxNumCalculationTensors = 0
		for stage_index in self.symbolStages.keys.sorted() {
			let symbolList = self.symbolStages[stage_index]!
			let tensorSymbolList = symbolList.filter {$0.symbolType == SymbolType.Tensor} as! [TensorSymbol]
			let calculationTensorSymbolList = tensorSymbolList.filter {$0.dataSource == SymbolDataSource.Calculation}
			maxNumCalculationTensors = max(maxNumCalculationTensors, calculationTensorSymbolList.count)
			// append all cal tensor symbol
			allCalculationTensorSymbolList.append(contentsOf: calculationTensorSymbolList)
			// append to stage info
			stageCalculationTensorSymbol.append(calculationTensorSymbolList)
		}
		
		// decide size of each tensor and initialize
		// First biggest tensor among all stages
		allCalculationTensorSymbolList = allCalculationTensorSymbolList.sorted {(a, b) -> Bool in
			return a.shape.count > b.shape.count
		}
	
		// allocate shared tensors
		var sharedTensors = [Tensor]()
		for shape in (allCalculationTensorSymbolList.prefix(upTo: maxNumCalculationTensors).map {$0.shape}) {
			sharedTensors.append(Tensor.randomTensor(shape))
		}
		
		// asssing to cal symbols
		for stageCalSymbols in stageCalculationTensorSymbol {
			var assignedTensorIndex = 0
			for (i,symbol) in stageCalSymbols.enumerated() {
				if symbol.bindedData == nil {
					/// should not assigned same tensor of previous assigned symbols within same stage
					while (stageCalSymbols[0..<i].map {$0.bindedData!.tensorValue == sharedTensors[assignedTensorIndex]}.contains { $0 == true}) {
						assignedTensorIndex += 1
					}
					var calSymbol = symbol
					calSymbol.bindedData = sharedTensors[assignedTensorIndex]
					
					// move to next available
					assignedTensorIndex += 1
					
				}
			}
		}
		
		// allocate param symbols .!!super!!
		super.allocateAllTensors()
	}
	
	/// Since calcualtion tensor symbols shares tensor.
	/// We need to correct its shape before verify.
	override public func checkShapeChain() -> (valid: Bool, msg: String) {
		for symbol in self.opSymbols() {
			var opSymbol = symbol
			// input calculation tensor symbol shape correction
			for symbol in (opSymbol.inBounds.filter {$0.symbolType == SymbolType.Tensor}) {
				let tensorSymbol = symbol as! TensorSymbol
				if (tensorSymbol.dataSource == SymbolDataSource.Calculation) {
					tensorSymbol.bindedData!.tensorValue.shape = tensorSymbol.shape
				}
			}

			// output calculation tensor symbol shape correction
			for symbol in (opSymbol.outBounds.filter {$0.symbolType == SymbolType.Tensor}) {
				let tensorSymbol = symbol as! TensorSymbol
				if (tensorSymbol.dataSource == SymbolDataSource.Calculation) {
					tensorSymbol.bindedData!.tensorValue.shape = tensorSymbol.shape
				}
			}

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
				return (false, "Operator symbol \(opSymbol.serranoOperator.operatorLabel) failed to pass inputOutputTensorsCheck(). " +
					"Details: \(msg)")
			}

		}
		
		return (true, "")
	}
	
	
	/// Since calcualtion tensor symbols shares tensor.
	/// We need to assign its shape before calcultaion.
	///
	/// - Parameter mode: mode
	override internal func stageOrderCalculate(mode: OperatorComputationMode) {
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
						
						// input calculation tensor symbol shape correction
						for symbol in (opSymbol.inBounds.filter {$0.symbolType == SymbolType.Tensor}) {
							let tensorSymbol = symbol as! TensorSymbol
							if (tensorSymbol.dataSource == SymbolDataSource.Calculation) {
								tensorSymbol.bindedData!.tensorValue.shape = tensorSymbol.shape
							}
						}
						
						// output calculation tensor symbol shape correction
						for symbol in (opSymbol.outBounds.filter {$0.symbolType == SymbolType.Tensor}) {
							let tensorSymbol = symbol as! TensorSymbol
							if (tensorSymbol.dataSource == SymbolDataSource.Calculation) {
								tensorSymbol.bindedData!.tensorValue.shape = tensorSymbol.shape
							}
						}
						
//						let start = CFAbsoluteTimeGetCurrent()
						opSymbol.serranoOperator.compute(mode)
//						let calTime = CFAbsoluteTimeGetCurrent() - start
//						print("Op \(opSymbol.serranoOperator.operatorLabel) Execution Time : \(calTime * 100) ms")
//						print("====================================")
						stageWorkGroup.leave()
					}
				}
			}
			// wait all complete in this stage
			stageWorkGroup.wait()
		}
	}
}
