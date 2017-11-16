//
//  metal_hardwares.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 4/22/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal

/// GPU familly not support MetalPerformanceShader
/// Ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
#if os(iOS)
public let GPU_FAMILTY_NOT_SUPPORT_MPS = [
		MTLFeatureSet.iOS_GPUFamily1_v3,
]
#endif

/**
 This util class contains APIs that check the device's hardware capabilities and limits accroding to [Apple's official docs](https://developer.apple.com/metal/availability).
 
 - Note: The limits values of each type of hardware device should be checked and updated with Apple's [doc](https://developer.apple.com/metal/limits/) frequently. If you find any mismatching with Apple's official doc, please make a PR at github.
 */
public class MetalHardwareChecker {
    /// Maximum length of a data block for a function, per render or compute command encode
    public static let MAX_BYTES_OF_DATA_BLOCK_PER_KERNEL = 4000
    
    // Maximum buffer length
    public static let MAX_BUFFER_SIZE_IN_BYTES = Int(2.56e+8)
    
    
    
    /// Check if tesor's data size largser thant MAX_BUFFER_SIZE_IN_BYTES
    ///
    /// - Parameter tensor: tensor
    ///
    /// - Returns:
    ///     - result: If fitting
    ///     - info:   Checking infomation
    public static func tensorSizeFitCheck(tensor: Tensor) -> (result: Bool, info: String) {
        if tensor.allocatedBytes >= MetalHardwareChecker.MAX_BUFFER_SIZE_IN_BYTES {
            return (false, "Trying to allocatea MTLBuffer with \(tensor.allocatedBytes) bytes. Metal could not create a MTLBuffer larger than \(MetalHardwareChecker.MAX_BUFFER_SIZE_IN_BYTES) bytes.")
        } else {
            return (true, "")
        }
    }
	
	
	/// If current GPU support MetalPerformanceShaser
	///
	/// - Returns: result bool
	public static func supportMPS() -> Bool {
		// macos test
		#if os(OSX)
			if #available(OSX 10.13, *) {
				return true
			} else {
				return false
			}
		#endif
		
		// ios Test
		#if os(iOS)
			let device = SerranoEngine.configuredEngine.GPUDevice
			guard device != nil else {
				return false
			}
			
			for feature in GPU_FAMILTY_NOT_SUPPORT_MPS {
				if device!.supportsFeatureSet(feature) {
					return false
				}
			}
			
			return true
		#endif
	}
}
