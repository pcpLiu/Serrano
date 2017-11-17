//
//  resource_manager.swift
//  serrano
//
//  Created by ZHONGHAO LIU on 7/15/17.
//  Copyright © 2017 ZHONGHAO LIU. All rights reserved.
//

import Foundation
import Metal


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK:

/**
A `MTLBufferResource` contains the allocated `MTLBuffer` related information for a `Tensor` object managed by a resource manager.
*/
public struct MTLBufferResource {
	/// `MTLBuffer` object
	var buffer: MTLBuffer
	
	/// Offset from the buffer base address.
	/// Used by sliced tensor object.
	var offset: Int
	
	public init(buffer: MTLBuffer, offset: Int) {
		self.buffer = buffer
		self.offset = offset
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK:

public enum SerranoTensorStatus {
	case Idle
	case Occupy
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK:

/**
This is a framework level class
*/
public class SerranoResourceManager {
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Attributes
	
	/// Table tracking allocated Tensor and its corresponding MTLBuffer
	public var tensorBufferTable: [Tensor: MTLBuffer]
	
	/// The dictionary tracking the usage status of tensor
	public var tensorStatusTable: [Tensor: SerranoTensorStatus]
	
	/// The operation queue when operate on `tensorBufferTable` and `tensorStatus`
	public var operationQueue: DispatchQueue
	
	/// Readable label
	public var label: String
	
	/// Description
	public var description: String {
		get {
			return "SerranoResourceManager(label: \(self.label))"
		}
	}
	
	// Global resource manager
	public static let globalManager = SerranoResourceManager(label: "gloabal_resource_manager")
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Initializer
	
	public init(label: String = "Resource Manager") {
		self.tensorBufferTable = [Tensor: MTLBuffer]()
		self.tensorStatusTable = [Tensor: SerranoTensorStatus]()
		self.operationQueue = DispatchQueue(label: "serrano.resourceQueue")
		self.label = label
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Allocate unmanaged tensors
	
	
	/// Allocate unmanaged tensors for shapes.
	/// `Unmanaged` means that the alloated tensors will not be hold via strong refrence by manager.
	///
	/// - Parameter shapes: shapes description
	/// - Returns: return value description
	public func allocateUnamangedTensors(_ shapes: [TensorShape]) -> [Tensor] {
		var tensors = [Tensor]()
		for shape in shapes {
			tensors.append(Tensor(repeatingValue: 0.0, tensorShape: shape))
		}
		return tensors
	}
	
	
	/// Allocate unmanaged tensors for shapes.
	/// `Unmanaged` means that the alloated tensors will not be hold via strong refrence by manager.
	///
	/// - Parameter shapes: shapes description
	/// - Returns: return value description
	public func allocateUnamangedTensor(_ shape: TensorShape) -> Tensor {
		let tensor = Tensor(repeatingValue: 0.0, tensorShape: shape)
		SerranoLogging.stdLogging(message: "Allocate unmanged tensor \(tensor.description)",
			file: "\(#file)", function: "\(#function)", line: "\(#line)",
			loggingLevel: SerranoLoggingType.LowLevel)
		return tensor
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Allocate unmanaged MTLBuffers
	
	/// Allocate unamanged `MTLBuffers`.
	///
	/// - Warning: If engine has no available GPU device, `fatalError` raised.
	///
	/// - Parameter tensors: target tensors
	/// - Returns: Array of `MTLBuffer`
	public func allocateUnmanagedMTLBuffers(_ tensors: [Tensor]) -> [MTLBuffer] {
		// check gpu available
		guard SerranoEngine.configuredEngine.hasAvailableGPU() else {
			SerranoLogging.errorLogging(message: "No available GPU device.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		var buffers = [MTLBuffer]()
		for tensor in tensors {
			let newBuffer = SerranoEngine.configuredEngine.GPUDevice!.makeBuffer(bytesNoCopy: tensor.contentsAddress,
			                                                                     length: tensor.allocatedBytes.padded(alignmentSize: Int(getpagesize()))!,
			                                                                     options: MTLResourceOptions.storageModeShared)
			buffers.append(newBuffer)
		}
		return buffers
	}
	
	/// Allocate unamanged `MTLBuffer`.
	///
	/// - Warning: If engine has no available GPU device, `fatalError` raised.
	///
	/// - Parameter tensor: target tensor
	/// - Returns: `MTLBuffer`
	public func allocateUnmanagedMTLBuffe(_ tensor: Tensor) -> MTLBuffer {
		return self.allocateUnmanagedMTLBuffers([tensor]).first!
	}

	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Allocate managed tensors
	
	
	/// Request managed tensors for target shapes.
	///
	/// - Note: All tensors requested by this method should be __return__ manually by calling
	///		    `returnTensors(_ tensors: [Tensor])` or `returnTensor(_ tensor: Tensor)`
	///
	/// - Warning: This method is desgined for framework internal usage. Usually user should not
	///		       call this method.
	///
	/// - Parameter shapes: target shapes
	/// - Returns: tensors
	public func allocateTensors(_ shapes: [TensorShape]) -> [Tensor] {
		var tensors = [Tensor]()
		for shape in shapes {
			let candidate = self.tensorStatusTable.index(where: { (element) -> Bool in
				return element.key.capacity >= shape.count && element.value == SerranoTensorStatus.Idle
			})
			
			if candidate == nil {
				// didn't find a reuseable tensor, spawn a new tensor
				let newTensor = Tensor(repeatingValue: 0.0, tensorShape: shape)
				self.tensorStatusTable[newTensor] = SerranoTensorStatus.Occupy
				tensors.append(newTensor)
				
				// initial a new MTLBuffer for this new Tensor in advance if has available GPU
				if SerranoEngine.configuredEngine.hasAvailableGPU() {
					let buffer = SerranoEngine.configuredEngine.GPUDevice!.makeBuffer(bytesNoCopy: newTensor.contentsAddress,
																					  length: newTensor.allocatedBytes,
																					  options: MTLResourceOptions.storageModeShared)
					self.tensorBufferTable[newTensor] = buffer
				}
				
				SerranoLogging.stdLogging(message: "\(self.description) allocated a new tensor [\(newTensor.description)] to target shape \(shape.shapeArray).",
					file: "\(#file)", function: "\(#function)", line: "\(#line)",
					loggingLevel: .LowLevel)
			} else {
				// reuse
				let tensor = self.tensorStatusTable[candidate!].key
				self.tensorStatusTable[tensor] = SerranoTensorStatus.Occupy
				tensor.shape = shape
				tensors.append(tensor)
				SerranoLogging.stdLogging(message: "\(self.description) reused existing tensor [\(tensor.description)] to target shape \(shape.shapeArray).",
					file: "\(#file)", function: "\(#function)", line: "\(#line)",
					loggingLevel: .LowLevel)
			}
			
		}
		
		return tensors
	}
	
	
	/// Allocate single tensor object. This function actually call `allocateTensors(forShapes shapes: [TensorShape])`.
	///
	/// - Note: All tensors requested by this method should be __return__ manually by calling
	///		    `returnTensors(_ tensors: [Tensor])` or `returnTensor(_ tensor: Tensor)`
	///
	/// - Warning: This method is desgined for framework internal usage. Usually user should not
	///		       call this method.
	///
	/// - Parameter tensorShape: shape
	/// - Returns: tensor
	public func allocateTensor(_ tensorShape: TensorShape) -> Tensor {
		return self.allocateTensors([tensorShape]).first!
	}
	
	
	/// Return managed tensors.
	///
	/// - Note: Tensor not managed by this manager or is a slice tensor will be ignored.
	///
	/// - Parameter tensors: returned tensors
	public func returnTensors(_ tensors: [Tensor]) {
		for tensor in tensors {
			guard self.isManagingTensor(tensor) else {
				SerranoLogging.warningLogging(message: "Return a tensor NOT allocated by resource manager.",
											  file: "\(#file)", function: "\(#function)", line: "\(#line)")
				continue
			}
			
			if tensor.isSliceTensor {
				SerranoLogging.warningLogging(message: "Received a slice tensor. Only root tensor can be returned.",
											  file: "\(#file)", function: "\(#function)", line: "\(#line)")
				continue
			}
			
			self.operationQueue.sync {
				self.tensorStatusTable[tensor] = SerranoTensorStatus.Idle
			}
			
			SerranoLogging.stdLogging(message: "Return tensor \(tensor.description) to \(self.description)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)",
				loggingLevel: .LowLevel)
		}
		
	}
		
	/// Return single tensor to resource manager
	///
	/// - Parameter tensor: tensor
	public func returnTensor(_ tensor: Tensor) {
		self.returnTensors([tensor])
	}
	
	/// Release target tensors.
	/// Actually clear corresponding entries in `tensorStatusTable` and `tensorBufferTable`.
	///
	/// - Note: Tensor not managed by this manager or is a slice tensor will be ignored.
	///
	/// - Parameter tensors: target tensors
	public func releaseTensors(_ tensors: [Tensor]) {
		for tensor in tensors {
			// not managing, ignore
			if !self.isManagingTensor(tensor) {
				SerranoLogging.warningLogging(message: "Trying to release a tensor \(tensor.description) not managed by this resource manager: [\(self.description)]",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				continue
			}
			
			// slice tensor,  ignore
			if tensor.isSliceTensor {
				SerranoLogging.warningLogging(message: "Trying to release a slice tensor \(tensor.description).",
					file: "\(#file)", function: "\(#function)", line: "\(#line)")
				continue
			}
			
			// remove entry
			let _ = self.tensorStatusTable.remove(at: self.tensorStatusTable.index(forKey: tensor)!)
		
			// also release attached MTLBuffer if has
			if self.tensorBufferTable[tensor] != nil{
				self.tensorBufferTable.remove(at: self.tensorBufferTable.index(forKey: tensor)!)
				SerranoLogging.stdLogging(message: "Remove tensor \(tensor.description) from resource manager: \(self.description)",
					file: "\(#file)", function: "\(#function)", line: "\(#line)",
					loggingLevel: SerranoLoggingType.LowLevel)
			}
		}
		
	}
	
	/// Check if a managed tensor is availabel for reuse
	/// If the passed in tensor is a sliced tensor, we check the status of its root tensor.
	///
	/// - Note: `false` will be returned if `tensor` is not managed by this manager.
	public func isTensorAvailable(_ tensor: Tensor) -> Bool {
		guard self.isManagingTensor(tensor) else {
			SerranoLogging.warningLogging(message: "Trying to check status of tensor [\(tensor.description)], but tensor was not managed by \(self.description)",
				file: "\(#file)", function: "\(#function)", line: "\(#line)")
			return false
		}

		var result: Bool = true
		if tensor.isSliceTensor {
			result = self.tensorStatusTable[tensor.sliceRootTensor!]! == SerranoTensorStatus.Idle
		} else {
			result = self.tensorStatusTable[tensor]! == SerranoTensorStatus.Idle
		}
		
		return result
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Allocate MTLBufferResource
	
	/// Request `MTLBufferResource` for tensors.
	///
	/// - Note: If the passed in tensors are not mamaged by this resource manager, 
	///         it will just call `allocateUnmanagedMTLBuffers(_ tensors: [Tensor])`.
	///
	/// - Parameter tensors: target tensors
	/// - Returns: Array of `MTLBufferResource`
	public func allocateMTLBufferResources(_ tensors: [Tensor]) -> [MTLBufferResource] {
		// check gpu available
		guard SerranoEngine.configuredEngine.hasAvailableGPU() else {
			SerranoLogging.errorLogging(message: "No available GPU device.",
			                            file: "\(#file)", function: "\(#function)", line: "\(#line)")
			fatalError()
		}
		
		var bufferResources = [MTLBufferResource]()
		for tensor in tensors {
			if !self.isManagingTensor(tensor){
				let buffer = self.allocateUnmanagedMTLBuffe(tensor)
				bufferResources.append(MTLBufferResource(buffer: buffer, offset: 0))
			} else {
				var buffer: MTLBuffer?
				var offset = 0
				var targetTensor = tensor
			
				// slice tensor just uses root tensor's buffer with offset
				if tensor.isSliceTensor {
					targetTensor = tensor.sliceRootTensor!
					offset = tensor.sliceRootTensor!.slicedTensorOffset(tensor)!
				}
				
				// get MTLBuffer
				self.operationQueue.sync {
					if self.tensorBufferTable[targetTensor] == nil {
						// new buffer
						buffer = SerranoEngine.configuredEngine.GPUDevice!.makeBuffer(bytesNoCopy: targetTensor.contentsAddress,
																					  length: targetTensor.allocatedBytes,
																					  options: MTLResourceOptions.storageModeShared)
						// add entry
						self.tensorBufferTable[targetTensor] = buffer
					} else {
						buffer = self.tensorBufferTable[targetTensor]!
					}
				}
	
				bufferResources.append(MTLBufferResource(buffer: buffer!, offset: offset))
			}
		}
		return bufferResources
	}
	
	/// Request `MTLBufferResource` for a tensor.
	///
	/// - Note: If the passed in tensor is not mamaged by this resource manager,
	///         it will just call `allocateUnmanagedMTLBuffer(_ tensor: Tensor)`.
	///
	/// - Parameter tensors: target tensor
	/// - Returns: `MTLBufferResource`
	public func allocateMTLBufferResource(_ tensor: Tensor) -> MTLBufferResource {
		return self.allocateMTLBufferResources([tensor]).first!
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Release managed resources
	
	/// Release all managed resources
	///
	/// - Warning: This function __does not__ guranteen that all managed tensors and buffers would be released in memory,
	///			   since if tensor or buffer objects are still used in other places, the ARC will nor clear it.
	///			   Before calling this function, the caller must be clear that all managed resources are in idle states.
	public func releaseAllResources() {
		self.operationQueue.sync {
			self.tensorBufferTable.removeAll()
			self.tensorStatusTable.removeAll()
		}
		SerranoLogging.stdLogging(message: "Resource manager: \(self.description) release all managed tensors and buffers",
			file: "\(#file)", function: "\(#function)", line: "\(#line)",
			loggingLevel: SerranoLoggingType.LowLevel)
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// MARK: - Methods
	
	/// Check if tensor is managed by this manager.
	/// If a tensor is managed by this manager, all its sliced tensors are managed by this manager.
	///
	/// - Note: If the tensor is a sliced tensor, we will check if managing its root tensor.
	///
	/// - Parameter tensor: tensor description
	/// - Returns: return value description
	public func isManagingTensor(_ tensor: Tensor) -> Bool {
		var result:Bool = true
		var checkTensor: Tensor = tensor
		if tensor.isSliceTensor { checkTensor = tensor.sliceRootTensor! }
		// check
		self.operationQueue.sync {
			result = self.tensorStatusTable[checkTensor] != nil
		}
		return result
	}
	
	/// Check if a MTLBuffer is managed by this manager
	///
	/// - Parameter buffer: buffer description
	/// - Returns: return value description
	public func isManagingBufferr(_ buffer: MTLBuffer) -> Bool {
		var result: Bool = false
		self.operationQueue.sync {
			result =  self.tensorBufferTable.index(where: { (element) -> Bool in
				return element.value.contents() == buffer.contents()
			}) != nil
		}
		return result
	}
}
