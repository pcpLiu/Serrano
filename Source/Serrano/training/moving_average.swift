//
//  moving_average.swift
//  Serrano
//
//  Created by ZHONGHAO LIU on 12/26/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation

/**
 `ExponentialMovingAverage` represents a variable that can update its values via moving average way with decay.
 */
public class ExponentialMovingAverage {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Attributes
    
    /// Count of updates. By default, starting from `0`.
    public var numUpdate: Int
    
    /// Momemtum. Default value is `0.99`
    public var momentum: Float
    
    /// Tensor object to update.
    ///
    /// - Note:
    ///     Here this ExponentialMovingAverage object `unowned` this tensor object
    unowned public var tensor: Tensor
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - Init
    
    /// Initializer
    ///
    /// - Parameters:
    ///   - tensor: target tensor
    ///   - momentum: momentum
    ///   - numUpdate: numUpdate
    public init(_ tensor: Tensor, momentum: Float = 0.99, numUpdate: Int = 0) {
        self.tensor = tensor
        self.momentum = momentum
        self.numUpdate = numUpdate
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MARK: - methods
    
    /// Update value of `tensor`
    ///
    /// - Parameter newValue: newest value
    public func update(_ newValue: Tensor) {
        (self.tensor &* (1 - self.momentum)) &+ (newValue * self.momentum)
        self.numUpdate += 1
    }
}
