//
//  tensor_symbol.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/7/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation


/**
Implementation of `TensorSymbol`
*/
public class SerranoTensorSymbol: SerranoGraphSymbol, TensorSymbol {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Data source. Conforms to `GraphSymbol`.
	public var dataSource: SymbolDataSource
	
	/// Shape. Conforms to `GraphSymbol`.
	public var shape: TensorShape
	
	/// Binded data. Conforms to `GraphSymbol`.
	public var bindedData: DataSymbolSupportedDataType? {
		get {
			return self._bindedData
		}
		set(newValue) {
			if newValue is Tensor? {
				self._bindedData = newValue as! Tensor?
			} else {
				SerranoLogging.errorLogging(message: "Unexpexted data type. Expect Tensor object",
				                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("")
			}
		}
	}
	
	/// If differentiable
	public var updatable = false
	
	/// Current grad
	public var currentGrad: DataSymbolSupportedDataType?
	
	/// If enabled history grads recording.
	/// Default is `false`.
	public var historyGradsEnabled = false
	
	/// grads
	public var historyGrads:[DataSymbolSupportedDataType] = [Tensor]()
	
	/// Binded tensor
	internal var _bindedData: Tensor?
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Designated init
	///
	/// - Parameters:
	///   - dataSource: dataSource
	///   - bindedData: bindedData
	///   - label: label
	///   - inBounds: inBounds
	///   - outBounds: outBounds
	public init(dataSource: SymbolDataSource, bindedData: Tensor?, shape: TensorShape,
	            label: String, inBounds:  [GraphSymbol], outBounds:  [GraphSymbol]) {
		self.dataSource = dataSource
		self._bindedData = bindedData
		self.shape = shape
		super.init(label: label, symbolType: .Tensor, inBounds: inBounds, outBounds: outBounds)
	}
	
	/// Covenient init
	///
	/// - Parameters:
	///   - label: label 
	///   - dataSource: dataSource
	public convenience init(_ label: String = "Tensor Symbol", dataSource: SymbolDataSource, shape: TensorShape) {
		self.init(dataSource: dataSource, bindedData: nil, shape: shape,
		          label: label, inBounds: [GraphSymbol](), outBounds: [GraphSymbol]())
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Bind to a tensor.
	/// There two cases could not bind successfull:
	///		1. Passed in `data` is not a tensor object;
	///		2. Passed in tensor object has not compatible shape with attribute `shape`.
	///
	/// - Parameter data: data. Should be a `Tensor` object.
	/// - Returns: bind result
	@discardableResult
	public func bindData(_ data:DataSymbolSupportedDataType) -> Bool {
		// check data type
		guard data is Tensor else {
			SerranoLogging.errorLogging(message: "Tensor symbol (\(self.symbolLabel)) expects a tensor object to bind. Given \(type(of: data)).",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return false
		}
		
		// check shape
		let tensor = data as! Tensor
		guard tensor.shape .== self.shape else {
			SerranoLogging.errorLogging(message: "Tensor symbol (\(self.symbolLabel)) expects a tensor object with shape \(self.shape.description). Given \(tensor.shape.description)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return false
		}
		self._bindedData = tensor
		
		return true
	}
}
