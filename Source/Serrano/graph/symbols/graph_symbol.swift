//
//  graph_symbol.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/7/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation

typealias WeakSerranoGraphSymbol = WeakRef<SerranoGraphSymbol>

/**
Implementation of `GraphSymbol`
*/
public class SerranoGraphSymbol: GraphSymbol, Hashable {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK:  - Attributes
	
	/// Symbol type. Conforms to `GraphSymbol`.
	public var symbolType: SymbolType
	
	/// Unique symbol ID. Conforms to `GraphSymbol`.
	/// - Note: A 6-length `String` consists of `[a-zA-Z0-9]`
	public var UID: String
	
	/// Readable label. Conforms to `GraphSymbol`.
	public var symbolLabel: String
	
	/// Inbound symbols list. Conforms to `GraphSymbol`.
	/// To prevent from cycle reference, we dynamic constructing this attribute from `inBoundsWeak`.
	public var inBounds: [GraphSymbol] {
		get {
			return self.inBoundsWeak.filter {$0.value != nil}.map {$0.value!}
		}
		set(bounds) {
			for symbol in bounds {
				self.addToInBound(symbol)
			}
		}
	}
	
	/// Outbound symbols list. Conforms to `GraphSymbol`.
	/// To prevent from cycle reference, we dynamic constructing this attribute from `inBoundsWeak`.
	public var outBounds: [GraphSymbol] {
		get {
			return self.outBoundsWeak.filter {$0.value != nil}.map {$0.value!}
		}
		set(bounds) {
			for symbol in bounds {
				self.addToOutBound(symbol)
			}
		}
	}
	
	/// Hash value.
	/// Conforms to `equatable` protocol.
	public var hashValue: Int {
		get {
			return self.UID.hashValue
		}
	}
	
	/// Weak reference array of inbounds objects
	internal var inBoundsWeak: [WeakSerranoGraphSymbol]
	
	/// Weak reference array of outbounds objects
	internal var outBoundsWeak: [WeakSerranoGraphSymbol]
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializers
	
	/// Designated init
	///
	/// - Parameters:
	///   - symbolType: symbolType
	///   - label: label
	///   - inBounds: inBounds
	///   - outBounds: outBounds
	public init(label: String, symbolType: SymbolType,
	            inBounds:  [GraphSymbol], outBounds:  [GraphSymbol]) {
		self.symbolType = symbolType
		self.symbolLabel = label
		self.inBoundsWeak = [WeakSerranoGraphSymbol]()
		self.outBoundsWeak = [WeakSerranoGraphSymbol]()
		self.UID = serranoSymbolUIDGenerate()
		
		// add
		for symbolIn in inBounds {
			self.addToInBound(symbolIn)
		}
		for symbolOut in outBounds {
			self.addToOutBound(symbolOut)
		}
	}
	
	/// Convenience init
	///
	/// - Parameters:
	///   - symbolType: symbolType
	///   - label: label
	public convenience init(_ label: String = "", symbolType: SymbolType) {
		self.init(label: label, symbolType: symbolType, inBounds: [GraphSymbol](), outBounds: [GraphSymbol]())
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	public func evaluate() -> [DataSymbolSupportedDataType]? {
		fatalError("Should not be called.")
	}
	
	/// Add new symbol to `inBounds`.
	/// Should check duplicate.
	public func addToInBound(_ symbol: GraphSymbol) {
		let s = symbol as! SerranoGraphSymbol
		if !(self.inBoundsWeak.contains {$0.value == s}) {
			let weakSymbol = WeakSerranoGraphSymbol(value: s)
			self.inBoundsWeak.append(weakSymbol)
		}
	}
	
	/// Add new symbol to `outBounds`
	/// Should check duplicate.
	public func addToOutBound(_ symbol: GraphSymbol) {
		let s = symbol as! SerranoGraphSymbol
		if !(self.outBoundsWeak.contains {$0.value == s}) {
			let weakSymbol = WeakSerranoGraphSymbol(value: s)
			self.outBoundsWeak.append(weakSymbol)
		}
	}
	
	/// Conforms to `equatable` protocol
	///
	/// - Parameters:
	///   - lhs: left compare
	///   - rhs: right compare
	/// - Returns: return value
	public static func == (lhs: SerranoGraphSymbol, rhs: SerranoGraphSymbol) -> Bool {
		return lhs.UID == rhs.UID
	}
}
