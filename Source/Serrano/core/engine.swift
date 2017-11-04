//
//  engine.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 3/3/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal

//public enum SerranoBackend {
//    case Serrano
//    //case CoreML
//}


/**
 This class is supposed be initialized and configured at the very beggining of your app.
 It is responsible for setting up computation envirionment involving with iOS's GPU device.
 It will initialize serveral instances which will involve with heavy GPU evaluation and are reconmmended reusing by [Apple documentation](https://developer.apple.com/library/content/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html).
 
 The `configuredEngine` is a _singleton_ instance and should be used everywhere you need to access `GPUDevice` (an instance of `MTLDevice`),\
 `serranoCommandQueue` (an instance of `MTLCommandQueue`).
 
 Initial and configure a GPU engine:
 
 ````swift
    let configuredEngine = SerranoEngine.configuredEngine
    let (success, message) = configuredEngine.configureEngine(computationMode: .GPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
    if !success {
        // Failed to congiure GPU device
        print(message)
    }
 ````
 
 Initial and configure a CPU engine:
 ````swift
     let configuredEngine = SerranoEngine.configuredEngine
     configuredEngine.configureEngine(computationMode: .CPU, serranoCommandQueue: nil, serranoMTLLibrary: nil, systemDefaultGPUDevice: nil)
 ````
 */
public class SerranoEngine {
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Instance
    public static let configuredEngine = SerranoEngine()
    
    /// GPU device.
    public var  GPUDevice: MTLDevice?
    
    /// Metal Command Queue for serrano.
    public var serranoCommandQueue: MTLCommandQueue?

    /// Engine computation mode
    public var computationMode: OperatorComputationMode
    
    /// Loaded MTLCompute​Pipeline​States.
    /// A dictionary with `label` as key and corresponding `MTLComputePipelineState` as value.
    public var loadedGPUKernels: [String : MTLComputePipelineState]
    
    /// METAL library
    public var metalLibrary: MTLLibrary?
	
	/// User defined Metal libarary
	public var userMetalLibarary: [MTLLibrary] = [MTLLibrary]()
    
//    /// Backend compuation 
//    public var backend: SerranoBackend = SerranoBackend.Serrano
	
	/// Default operator computation Mode
	public var defaultComputationMode: OperatorComputationMode = .Auto
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Mark: - Initializer
    
    private init() {
        self.GPUDevice = nil
        self.serranoCommandQueue = nil
        self.computationMode = .CPU
        self.loadedGPUKernels = [:]
        self.metalLibrary = nil
    }

    /**
     Setup `GPUDevice`, `serranoCommandQueue` and `computationMode` of engine.
     
     - Parameters:
        - computationMode: one of choies in enum `OperatorComputationMode` (`GPU` or `CPU`).
        - serranoCommandQueue: Optional. If it is `nil`, method will initialize a command queue in `GPU` mode.
        - serranoMTLLibrary: Optional. If it is `nil`, method will initialize a default `MTLLibrary`.
        - systemDefaultGPUDevice: Optional. If this is nil and `computationMode` is `GPU`,\
                                 method will try to create a instance calling `MTLCreate​System​Default​Device()`.\
                                 If failed to initialize a GPU device instance, will return `false`.
     
     - Returns:
        - result: If configure engine successfully.
        - message: Message information of configure.
     
     - Note: When user gives no device instance and method fails to initialized a GPU device instance, \
             `Serrano` will automatically set up `computationMode` to `OperatorComputationMode.CPU`.
     
     - Warning: This method must be called before doing any GPU related computation.
     
     */
    public func configureEngine(computationMode mode: OperatorComputationMode,
                                serranoCommandQueue commandQueue: MTLCommandQueue? = nil,
                                serranoMTLLibrary library: MTLLibrary? = nil,
                                systemDefaultGPUDevice gpu: MTLDevice? = nil) -> (result: Bool, message: String){
        if mode == .GPU {
            // device
            if gpu == nil {
                self.GPUDevice = MTLCreateSystemDefaultDevice()
                
                guard (self.GPUDevice != nil) else {
                    self.computationMode = .CPU
                    SerranoLogging.warningLogging(message: "Failed to create a MTLDevice instance from MTLCreateSystemDefaultDevice().",
                                                  file: "\(#file)", function: "\(#function)", line: "\(#line)")
                    return (false, "Failed to create a MTLDevice instance from MTLCreateSystemDefaultDevice().")
                }
            } else {
                self.GPUDevice = gpu
            }
            
            // computeCommandQueue
            if commandQueue == nil {
                self.serranoCommandQueue = self.GPUDevice!.makeCommandQueue()
            } else {
                self.serranoCommandQueue = commandQueue
            }
            
            // default library
            if library == nil {
                // build directly
                var libpath = Bundle(for: type(of: self)).path(forResource: "default", ofType: "metallib")
				if libpath == nil {
					// build through cocoapod
					libpath = Bundle(identifier: "SerranoMetalLib")?.path(forResource: "default", ofType: "metallib")
				}
				guard libpath != nil else {
					return (false, "Failed to locate default.metlib")
				}
                do {
                    try self.metalLibrary = self.GPUDevice!.makeLibrary(filepath: libpath!)
                } catch {
                    return (false, "Failed to create the default metal library. Erro:\n\(error)")
                }
            } else {
                library!.label = "serranoMetalLibrary"
                self.metalLibrary = library!
            }
            
            self.computationMode = .GPU
            return (true, "Setup engine computation mode to \(self.computationMode) with device: \(self.GPUDevice!.description).")
        } else {
            self.computationMode = .CPU
            return (true, "Setup engine computation mode to \(self.computationMode).")
        }
        
        
    }
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Mark: - Methods
    
    /**
     Reset all attributes of engine to default values.
     */
    public func resetEngine() {
        self.GPUDevice = nil
        self.computationMode = .CPU
        self.loadedGPUKernels = [:]
        self.metalLibrary = nil
        self.serranoCommandQueue = nil
    }
    
    /**
     Check if current configured engine has available GPU device.
     
     - Returns: `true` if has available device.
     */
    public func hasAvailableGPU() -> Bool {
        return self.GPUDevice != nil
    }
    
    /**
     Get a kernel in `loadedGPUKernels`, if not in loaded kernels, return `nil`
     
     - Parameters
        - label: label of target `MTLComputePipelineState`
     */
    internal func getGPUKernelFromLoadedKernels(kenelLabel label: String) -> MTLComputePipelineState? {
       return self.loadedGPUKernels[label]
    }
    
    /**
     Load GPU compute kernel from Serrano's default Metal library.
     Before loading method will check if already loaded this kernel, if loaded just return the kernel
     If not found in `loadedGPUKernels`, method will create a new `MTLCompute​Pipeline​State` instance for function with target `label` and return the kernel.
     When failed to find the function, will return a `nil` kernel with error information.
     
     - Parameters:
        - label: Kernel function name in Metal file.
     
     - Returns: `kernel`: Optional. Target kernel. Will be `nil` if fails to load; `message`: message information.
     
     - Note: User must configure the engine first before loading any GPU kernels. If method could find `GPUDevice` or 'metalLibrary' is `nil`, it will
            raise a fatal error.
     */
    public func loadGPUKernel(kernelLabel label: String) -> (result: MTLComputePipelineState?, message: String){
        guard self.GPUDevice != nil && self.metalLibrary != nil else {
            fatalError("[Serrano]Trying to load GPU kernel without initializing a GPU device or Metal Library.")
        }
        
        var kernel: MTLComputePipelineState? = self.getGPUKernelFromLoadedKernels(kenelLabel: label)
        if kernel != nil {
            return (kernel!, "Successfully loaded GPU kernel \(label).")
        }
        
        var function = self.metalLibrary!.makeFunction(name: label)
        if function == nil {
			for mtlLib in self.userMetalLibarary {
				function =  mtlLib.makeFunction(name: label)
				if function != nil {
					break
				}
			}
		}
		if function == nil {
			return (nil, "Cannot load kernel \(label)")
		}
        
        do {
            try kernel = self.GPUDevice!.makeComputePipelineState(function: function!)
        }
        catch {
            return (nil, "Error catched when tring to load kernel \(label). Details:\n\(error)")
        }
        
        
        self.loadedGPUKernels[label] = kernel!
        return (kernel!, "Successfully loaded GPU kernel \(label).")
    }
}



