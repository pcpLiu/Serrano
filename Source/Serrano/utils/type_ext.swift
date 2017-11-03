//
//  type_ext.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 4/7/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///

extension Int {
    
    /// Description
    ///
    /// - Parameter size: size description
    /// - Returns: return value description
    func padded(alignmentSize size: Int) -> Int? {
        guard self >= 0 && size > 0 else {
            SerranoLogging.warningLogging(message: "Undefined padding action from \(self) to \(size).", file: "\(#file)", function: "\(#function)", line: "\(#line)")
            return nil
        }
        
        let remainder = self % size
        if remainder == 0 {
            return self
        } else {
            return self + size - remainder
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


public protocol SupportedScalarDataType {
    
    var floatValue: Float {get}
    
}

extension Int: SupportedScalarDataType {
    
    
    public var floatValue: Float {
        return Float(self)
    }
  
}

extension Double: SupportedScalarDataType {
    
    public var floatValue: Float {
        return Float(self)
    }
}

extension Float: SupportedScalarDataType {
    
    
    public var floatValue: Float {
        return self
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


public protocol SupportedNestedType {}
extension Array: SupportedNestedType {}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// In Metal, UInt is 32 bits. In iOS, it is 64 bits on 64 bit system.
typealias MetalUInt = UInt32

typealias MetalUShort = UInt16

typealias MetalShort = Int16

typealias MetalInt = Int32

typealias MetalFloat = Float32

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

public protocol DataSymbolSupportedDataType {
	var description: String {get}
	var tensorValue: Tensor {get}
	var scarlarValue: Float {get}
}
extension Int: DataSymbolSupportedDataType {
	public var description: String {
		get {
			return "Int value: \(self)"
		}
	}
	
	public var tensorValue: Tensor {
		get {
			fatalError("This DataSymbolSupportedDataType is a scarlar.")
		}
	}
	
	public var scarlarValue: Float {
		get {
			return Float(self)
		}
	}
}
extension Double: DataSymbolSupportedDataType {
	public var description:String {
		get {
			return "Double value: \(self)"
		}
	}
	
	public var tensorValue: Tensor {
		get {
			fatalError("This DataSymbolSupportedDataType is a scarlar.")
		}
	}
	
	public var scarlarValue: Float {
		get {
			return Float(self)
		}
	}
}
extension Float: DataSymbolSupportedDataType {
	public var description:String {
		get {
			return "Float value: \(self)"
		}
	}
	
	public var tensorValue: Tensor {
		get {
			fatalError("This DataSymbolSupportedDataType is a scarlar.")
		}
	}
	
	public var scarlarValue: Float {
		get {
			return self
		}
	}
}
extension Tensor: DataSymbolSupportedDataType {
	public var scarlarValue: Float {
		get {
			fatalError("This DataSymbolSupportedDataType is a tensor.")
		}
	}
	
	public var tensorValue: Tensor {
		get {
			return self
		}
	}
}


