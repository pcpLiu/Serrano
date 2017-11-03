//
//  model.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 8/4/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation

/**
ModelCallback defines a serires of APIs a callback object for models.
*/
public protocol ModelCallback {
	// TODO: IMPLEMENT
}

/**
This protocol defines higher-level APIs for creating, training and prediction of a model.
*/
public protocol Model: Graph {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Operator symbols of this model
	var operators: [OperatorSymbol] {get}
	
	/// List of input tensor symbols
	var inputs: [TensorSymbol] {get}
	
	/// List of output tensor symbol
	var outputs: [TensorSymbol] {get}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Constructing models
	
	/// Add an input entry for moedel
	///
	/// ## inputShape
	/// `inputShape` indicates the shape of a sample without batch index.
	/// For example if we have some 128x128 RGB picture as input, the `inputShape` should be `TensorShape`
	/// object with `shapeArray`: `[128, 128, 3]` if `channelOrder` is 'TensorChannelOrder.last'
	///
	/// - Parameter inputShape: input shape of sample. Not include batch index.
	/// - Returns: tensor symbol representation
	func addInput(inputShape: TensorShape, channelOrder: TensorChannelOrder) -> TensorSymbol
	
	/// Add layer to the model.
	///
	/// - Parameters:
	///   - inputs: list of input tensor symbols to this layer
	///   - op: operator
	/// - Returns: list of output tensor symbols from this layer
	func addLayer(_ inputs: [TensorSymbol], op: ComputableOperator) -> [TensorSymbol]
	
	func configure()
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Training
	
	// TODO: IMPLEMENT
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Prediction
	
	// TODO: IMPLEMENT
	
}
