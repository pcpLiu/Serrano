//
//  optimizer.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 9/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation

/// Learning rate decay method
public enum LearningRateDecayMethod {
	/// Each epoch update following: `lr = lr_initial - decay * t` where `t` is current epoch..
	case Step
	
	/// Each epoch udpate following: `lr = lr_initial * e^(-decay * t)` where `t` is current epoch.
	case Exponential
	
	/// Each epoch udpate following: `lr = lr_initial /(1 + decay * t)` where `t` is current epoch.
	case Inverse
	
	
	/// Calculate decayed lr for current epoch.
	///
	/// - Parameters:
	///   - initialLR: initial lr
	///   - decay: decay hyperparameter
	/// - Returns: decayed lr
	func decayLR(initialLR: Float, decay: Float, epoch: Int) -> Float {
		var lr: Float = 0.0
		if self == LearningRateDecayMethod.Step {
			lr = initialLR - decay * Float(epoch)
		} else if self == LearningRateDecayMethod.Exponential {
			lr = initialLR * exp(-decay*Float(epoch))
		} else {
			lr = initialLR / (1 + decay * Float(epoch))
		}
		return lr
	}
}



/**
This protocol defines the API and behavior of an optimizer.
*/
public protocol Optimizer {
	//// Initial set learning reate
	var initLearningRate: Float {get}
	
	/// Learning rate of current epoch
	var learningRate: Float {set get}
	
	/// Decay method
	var decayMethod: LearningRateDecayMethod {get set}
	
	/// Do preapre work before 1st backward if needed.
	func prepare(_ graph: Graph)
	
	/// Update a data symbol's updated value
	///
	//// - Parameters:
	///   - dataSymbol: target symbol
	///   - gradValue: gradvalue fot this time updating
	func updateParameter(_ dataSymbol: DataSymbol, gradValue: DataSymbolSupportedDataType)
}

extension Optimizer {
}
