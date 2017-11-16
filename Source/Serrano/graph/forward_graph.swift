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
}
