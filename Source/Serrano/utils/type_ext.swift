//
//  type_ext.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 4/7/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Accelerate
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

// Following the same descripition in Metal Language Specification
public typealias MetalUInt = UInt32

public typealias MetalUShort = UInt16

public typealias MetalShort = Int16

public typealias MetalInt = Int32

public typealias MetalFloat = Float32

public protocol SerranoValueConverter {
	var metalUInt: MetalUInt {get}
	var metalUShort: MetalUShort {get}
	var metalShort: MetalShort {get}
	var metalInt: MetalInt {get}
	var metalFloat: MetalFloat {get}
	
	var vDSPLength: vDSP_Length {get}
}

extension Int: SerranoValueConverter {
	public var metalUInt: MetalUInt {
		return MetalUInt(self)
	}
	
	public var metalUShort: MetalUShort {
		return MetalUShort(self)
	}
	
	public var metalShort: MetalShort {
		return MetalShort(self)
	}
	
	public var metalInt: MetalInt {
		return MetalInt(self)
	}
	
	public var metalFloat: MetalFloat {
		return MetalFloat(self)
	}
	
	public var vDSPLength: vDSP_Length {
		return vDSP_Length(self)
	}
}

extension Float: SerranoValueConverter {
	public var metalUInt: MetalUInt {
		return MetalUInt(self)
	}
	
	public var metalUShort: MetalUShort {
		return MetalUShort(self)
	}
	
	public var metalShort: MetalShort {
		return MetalShort(self)
	}
	
	public var metalInt: MetalInt {
		return MetalInt(self)
	}
	
	public var metalFloat: MetalFloat {
		return MetalFloat(self)
	}
	
	public var vDSPLength: vDSP_Length {
		return vDSP_Length(self)
	}
}

extension Double: SerranoValueConverter {
	public var metalUInt: MetalUInt {
		return MetalUInt(self)
	}
	
	public var metalUShort: MetalUShort {
		return MetalUShort(self)
	}
	
	public var metalShort: MetalShort {
		return MetalShort(self)
	}
	
	public var metalInt: MetalInt {
		return MetalInt(self)
	}
	
	public var metalFloat: MetalFloat {
		return MetalFloat(self)
	}
	
	public var vDSPLength: vDSP_Length {
		return vDSP_Length(self)
	}
}

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


