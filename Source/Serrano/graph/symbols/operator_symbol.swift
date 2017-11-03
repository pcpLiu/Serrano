//
//  operator_symbol.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/7/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation



/**
Implementaion of `OperatorSymbol`
*/
public class SerranoOperatorSymbol: SerranoGraphSymbol, OperatorSymbol {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Operator instance
	public var serranoOperator: ComputableOperator
	
	public var inputTensorShapes: [TensorShape]
	
	/// Input tensor symbols
	public var inputSymbols: [TensorSymbol]
	
	/// Params tensor symbols
	public var paramSymbols: [DataSymbol]
	
	/// Control if update asscoiated operator's parameter
	/// Default is `false`
	public var enabledParameterUpdate = false
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Designated initalizers
	///
	/// - Parameters:
	///   - serranoOperator: serranoOperator description
	///   - label: label description
	///   - inBounds: inBounds description
	///   - outBounds: outBounds description
	public init(_ label: String = "Operator symbol",
	            serranoOperator: ComputableOperator,
	            inputSymbols: [TensorSymbol],
	            paramSymbols: [DataSymbol]? = nil,
	            inBounds:  [GraphSymbol] = [GraphSymbol](),
	            outBounds:  [GraphSymbol] = [GraphSymbol]()) {
		self.serranoOperator = serranoOperator
		self.inputSymbols = inputSymbols
		
		if paramSymbols == nil {
			self.paramSymbols = [DataSymbol]()
		} else {
			self.paramSymbols = paramSymbols!
		}
		
		self.inputTensorShapes = inputSymbols.map {$0.shape}
		super.init(label: label, symbolType: .Operator, inBounds: inputSymbols, outBounds: [GraphSymbol]())
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Get output symbols of this operator.
	/// The output shapes are decided by operaotr's function `outputShape(shapeArray: inputShapes)`
	///
	/// - Note: This function will automatically add return symbols in `outBounds`.
	///
	/// - Returns: Array of TensorSymbols.
	public func outputSymbols() -> [TensorSymbol] {
		var output = [TensorSymbol]()
		
		let inputShapes = self.inputSymbols.map { $0.shape }
		let outputShapes = self.serranoOperator.outputShape(shapeArray: inputShapes)
		guard outputShapes != nil else {
			SerranoLogging.errorLogging(message: "Operator symbol (\(self.symbolLabel)) has invalid tensor symbols. It could not generate output tensor symbols. Check previous log for detail error info.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		for outShape in outputShapes! {
			output.append(SerranoTensorSymbol(dataSource: .Calculation, shape: outShape))
			self.addToOutBound(output.last! as GraphSymbol)
		}
		
		return output
	}
}

