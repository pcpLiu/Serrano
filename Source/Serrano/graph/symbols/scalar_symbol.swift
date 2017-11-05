//
//  scalar_symbol.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/7/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation


/**
Implementation of `ScalarSymbol`
*/
public class SerranoScalarSymbol: SerranoGraphSymbol, ScalarSymbol {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Binded scalar value
	public var bindedData: DataSymbolSupportedDataType? {
		get {
			return self._bindedData as? DataSymbolSupportedDataType
		}
		set(newValue) {
			if newValue is SupportedScalarDataType? {
				self._bindedData = newValue as! SupportedScalarDataType?
			} else {
				SerranoLogging.errorLogging(message: "Unexpexted data type. Expecte scalar type.",
				                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
				fatalError("")
			}
		}
	}
	
	/// The data type
	public var dataType: TensorDataType
	
	/// Data source
	public var dataSource: SymbolDataSource
	
	/// If differentiable
	public var updatable = false
	
	/// Current grad
	public var currentGrad: DataSymbolSupportedDataType?
	
	/// If enabled history grads recording.
	/// Default is `false`.
	public var historyGradsEnabled = false
	
	/// grads
	public var historyGrads: [DataSymbolSupportedDataType] = [SupportedScalarDataType]() as! [DataSymbolSupportedDataType]
	
	/// Scalar value
	internal var _bindedData: SupportedScalarDataType?
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Designated init
	///
	/// - Parameters:
	///   - dataType: dataType
	///   - dataSource: dataSource
	///   - bindedData: bindedData
	///   - label: label
	///   - inBounds: inBounds
	///   - outBounds: outBounds
	public init(dataType: TensorDataType, dataSource: SymbolDataSource, bindedData: SupportedScalarDataType? = nil,
	            label: String, inBounds:  [GraphSymbol], outBounds:  [GraphSymbol]) {
		self.dataType = dataType
		self._bindedData = bindedData
		self.dataSource = dataSource
		super.init(label: label, symbolType: .Scalar, inBounds: inBounds, outBounds: outBounds)
	}
	
	/// Convenience init
	///
	/// - Parameter dataType: dataType
	public convenience init(_ label: String = "Scalar Symbol", dataType: TensorDataType, dataSource: SymbolDataSource) {
		self.init(dataType: dataType, dataSource: dataSource, bindedData: nil, label: label, inBounds: [GraphSymbol](), outBounds: [GraphSymbol]())
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Bind to a scalar variable
	///
	/// - Note: If passed in `data`'s type is uncompatible with attribute `dataType`, no further action will be taken.
	///			Since when doing calculation, all data converted to `Float`.
	///
	///
	/// - Parameter data: data. Should be a scalar variable (`Int`, 'Float' or `Double`).
	/// - Returns: bind result
	@discardableResult
	public func bindData(_ data:DataSymbolSupportedDataType) -> Bool {
		// check data type
		guard (data is SupportedScalarDataType) else {
			SerranoLogging.errorLogging(message: "Scalar symbol (\(self.symbolLabel)) expects a scalar object to bind. Given \(type(of: data)).",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return false
		}
		
		guard let bindedData = data as? SupportedScalarDataType else {
			SerranoLogging.errorLogging(message: "Scalar symbol \(self.symbolLabel) was trying to bind to data \(data). But seems this data is not a scalar.",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError("Faltal error raised by Serrano. Check log for details.")
		}
		self._bindedData = bindedData
		return true
	}
}
