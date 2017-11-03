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




