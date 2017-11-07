//
//  utils.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 3/9/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal
import Accelerate

public class RandomValueGenerator {
	public static func randomInt(min: Int = 0, max: Int = 10) -> Int {
		return Int(arc4random_uniform(UInt32(max - min))) + min
	}

	public static func randomFloat(min: Float = 1.0, max: Float = 10.0) -> Float {
		return Float(drand48()) * max - min
	}
}

/**
 Get `CBLAS_TRANSPOSE` enum value from Bool marker
 */
public func cblasTrans(_ mark: Bool) -> CBLAS_TRANSPOSE {
    if mark {
		return CblasTrans
	}
    else {
		return CblasNoTrans
	}
}


/**
Weak reference object. Used in container of weak reference
*/
public class WeakRef<T: AnyObject> {
	weak var value : T?
	init (value: T) {
		self.value = value
	}
}




