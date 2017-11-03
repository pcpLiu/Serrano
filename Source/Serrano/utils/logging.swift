//
//  logging.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 3/17/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation


public enum SerranoLoggingType{
    case Regular,
         MediumLevel,
         LowLevel
}


public class SerranoLogging {
    
    public static var enableWarnning = true
	
	/// If `release` is `true`, all std and warning logging will be omiited.
	/// Default is `false`.
	///
	public static var release = false
	
	public static func stdLogging(message: String, file: String, function: String, line: String, loggingLevel: SerranoLoggingType) {
		if release { return }
		
		let fileName = file.components(separatedBy: "/").last
		if fileName != nil {
			NSLog("[Serrano](File: %@, Function: %@, Line: %@) ==> %@", fileName!.description, function, line, message)
		} else {
			NSLog("[Serrano](File: %@, Function: %@, Line: %@) ==> %@", file, function, line, message)
		}
	}
	
	public static func warningLogging(message: String, file: String, function: String, line: String) {
		if release { return }
		
		let fileName = file.components(separatedBy: "/").last
		if fileName != nil {
			NSLog("[Serrano, Warning⚠️](File: %@, Function: %@, Line: %@) ==> %@", fileName!.description, function, line, message)
		} else {
			NSLog("[Serrano, Warning⚠️](File: %@, Function: %@, Line: %@) ==> %@", file, function, line, message)
		}
	}
	
	public static func errorLogging(message: String, file: String, function: String, line: String) {
		let fileName = file.components(separatedBy: "/").last
		if fileName != nil {
			NSLog("[Serrano, ERROR‼️](File: %@, Function: %@, Line: %@) ==> %@", fileName!.description, function, line, message)
		} else {
			NSLog("[Serrano, ERROR‼️](File: %@, Function: %@, Line: %@) ==> %@", file, function, line, message)
		}
	}
}

