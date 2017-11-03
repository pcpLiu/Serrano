//
//  sgd_opt.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 10/13/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation

/**
Stochastic gradient descent.
*/
public class SGDOptimizer: Optimizer {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Learning rate.
	/// Should `>= 0`.
	public var learningRate: Float
	
	/// Initial leanring rate
	public var initLearningRate: Float {
		get {
			return self.initialLR
		}
	}
	
	/// LR Decay method
	public var decayMethod: LearningRateDecayMethod
	
	/// Learning rate decay per epoch.
	/// Before each epoch's parmaeter updating, do `learningRate -= learningRate * decay`.
	/// Should `>= 0`.
	public var decay: Float
	
	/// Momentum.
	/// Should `>= 0`.
	public var momentum: Float
	
	/// Whether to turn on Nesterov momentum.
	/// Default is `false`.
	public var nesterov: Bool
	
	/// Initial leanring rate
	internal var initialLR: Float
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Init
	
	public init(learningRate: Float = 0.001, momentum: Float = 0.0,
	            decayMethod: LearningRateDecayMethod = LearningRateDecayMethod.Step, decay: Float = 0.0,
	            nesterov: Bool = false) {
		self.initialLR = learningRate
		self.learningRate = learningRate
		self.momentum = momentum
		self.decayMethod = decayMethod
		self.decay = decay
		self.nesterov = false
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods conforming Optimizer protocol
	
	/// Do some preparing work before each epoch training
	///
	/// - Parameter graph: target graph
	public func prepare(_ graph: Graph) {
		// decay learning rate
		self.learningRate = self.decayMethod.decayLR(initialLR: self.initialLR, decay: self.decay, epoch: graph.epoch)
	}
	
	/// Update a data symbol's updated value
	///
	//// - Parameters:
	///   - dataSymbol: target symbol
	///   - gradValue: gradvalue fot this time updating
	public func updateParameter(_ dataSymbol: DataSymbol, gradValue: DataSymbolSupportedDataType) {
		// momentum
		if dataSymbol.symbolType == SymbolType.Scalar {
			var scalarSymbol = dataSymbol as! ScalarSymbol
			let value: Float = scalarSymbol.bindedData! as! Float
			let v: Float = self.momentum * scalarSymbol.currentGrad!.scarlarValue - self.learningRate * gradValue.scarlarValue
			if self.nesterov {
				scalarSymbol.bindedData! = value + self.momentum * v - self.learningRate * gradValue.scarlarValue
			} else {
				scalarSymbol.bindedData! = value + v
			}
		} else {
			var tensorSymbol = dataSymbol as! TensorSymbol
			let grad = tensorSymbol.currentGrad as! Tensor
			let v: Tensor = self.momentum * grad &- self.learningRate * gradValue.tensorValue
			if self.nesterov {
				// self.learningRate * gradValue.tensorValue cannot use inplace operation. will effect passed-in argument
				tensorSymbol.bindedData!.tensorValue &+ (self.momentum &* v) &- self.learningRate * gradValue.tensorValue
			} else {
				tensorSymbol.bindedData!.tensorValue &+ v
			}
			print("\(tensorSymbol.symbolLabel)",grad.flatArrayFloat())
		}
	}
	

}
