//
//  compute_graph.swift
//  serrano
//
//
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


public protocol GraphSupportedBindingDataType {}

extension Array: GraphSupportedBindingDataType {}
extension Tensor: GraphSupportedBindingDataType {}
extension Float: GraphSupportedBindingDataType {}
extension Double: GraphSupportedBindingDataType {}
extension Int: GraphSupportedBindingDataType {}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/// The forwarding mode of a graph
///
/// - training: training
/// - inference: inference
public enum GraphForwardMode {
    case training
    case inference
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
Basic graph protocol
*/
public protocol Graph {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Attributes

    /// List of `GraphSymbol`.
    /// Key is the `UID` of symbol object in value field.
    var symbols: [String: GraphSymbol] {get set}
    
    /// The readable label of this graph
    var graphLabel: String {get set}
    
    /// If this graph is trainable.
    var trainable: Bool {get set}
    
    /// Description of this graph
    var description: String {get}
    
    /// Optimizer of this graph doing backward training.
    /// Could be `nil` if just do forward calcualtion.
    var optimizer: Optimizer? {get set}
    
    /// Counter of backward training
    var epoch: Int {get}
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Create symbols
    
    
    /// Add a `TensorSymbol` to the graph.
    /// - Parameters:
    ///   - label: label description
    ///   - shape: shape description
    /// - Returns: return value description
    func tensor(_ label: String?, shape: TensorShape) -> TensorSymbol
    
    /// Add a `ScalarSymbol` to the graph.
    ///
    /// - Parameter label: label
    /// - Returns: A `ScalarSymbol`
    func scalar(_ label: String?, dataType: TensorDataType) -> ScalarSymbol
    
    /// Add a `OperatorSymbol` to the graph.
    ///
    /// - Parameters:
    ///   - inputs: input array of `TensorSymbol`
    ///   - op: A `ComputableOperator` instance
    /// - Returns: Output `TensorSymbol` from `operator` calculation, and constructed `OperatorSymbol`
    func operation(_ label: String?, inputs: [TensorSymbol], op: ComputableOperator) -> (outputTensorSymbols: [TensorSymbol], operatorSymbol: OperatorSymbol, paramSymbols: [GraphSymbol])
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - compute
    
    /// Forward computing from inputs to outputs.
    ///
    /// - Parameter mode: computation mode
    func forward(mode: OperatorComputationMode)
    
    /// Backward computing the grads for updatable data symbols.
    ///
    /// - Parameters:
    ///   - mode: computation mode
    func backward(mode: OperatorComputationMode)
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - other
    
    /// Bind data to `TensorSymbol`.
    ///
    /// - Parameter data: A dictinary whose key is `label` of a `TensorSymbol`
    ///                   and value is an array of `DataSymbolSupportedDataType` objects.
    func bindData(_ data: [String: DataSymbolSupportedDataType])
    
    /// Add symbol to `symbols`.
    /// Should check duplicate
    ///
    /// - Parameter symbol: new symbol
    func addSymbols(_ symbol: GraphSymbol)
}

extension Graph {
    /// Add symbol to `symbols`.
    /// Should check duplicate
    ///
    /// - Parameter symbol: new symbol
    public func addSymbols(_ symbol: GraphSymbol) {
        if self.symbols[symbol.UID] == nil {
            var g = self as Graph
            g.symbols[symbol.UID] = symbol
        }
    }
}
