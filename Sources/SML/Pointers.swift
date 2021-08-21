//
//  Pointers.swift
//  MachineLearning
//
//  Created by Sahil Srivastava on 8/8/21.
//

import Foundation

public class Pointers {
    
    // MARK: 2D Alloc
    public static func alloc<T>(copy: UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>>) -> UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>> {
        // FIRST ALLOCATE SPACE
        // Allocate the space for this double pointer
        let doublePtr = UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>>.allocate(capacity: copy.count)
        // Iterate up till our capacity
        for i in 0..<copy.count {
            // Allocate the space for this pointer
            doublePtr[i] = UnsafeMutableBufferPointer<T>.allocate(capacity: copy[i].count)
        }
        // NOW POPULATE
        for i in 0..<copy.count {
            for j in 0..<copy[i].count {
                // Deep copy each pointee from copy[i]s jth offset to this jth pointer offset
                doublePtr[i][j] = copy[i][j]
            }
        }
        return doublePtr
    }
    
    public static func alloc<T>(rows: Int, cols: Int, copy: UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>>, randomSample: Bool = false) -> UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>> {
        // FIRST ALLOCATE SPACE
        // Allocate the space for this double pointer
        let doublePtr = UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>>.allocate(capacity: rows)
        // Iterate up till our capacity
        for i in 0..<rows {
            // Allocate the space for this pointer
            doublePtr[i] = UnsafeMutableBufferPointer<T>.allocate(capacity: cols)
        }
        // NOW POPULATE
        for i in 0..<rows {
            let rand = randomSample ? copy.randomElement()! : copy[i]
            for j in 0..<cols {
                // Deep copy each pointee from rands jth offset to this jth pointer offset
                doublePtr[i][j] = rand[j]
            }
        }
        return doublePtr
    }
    
    public static func alloc<T>(rows: Int, cols: Int, repeated: T) -> UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>> {
        // FIRST ALLOCATE SPACE
        // Allocate the space for this double pointer
        let doublePtr = UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>>.allocate(capacity: rows)
        // Iterate up till our capacity
        for i in 0..<rows {
            // Allocate the space for this pointer
            doublePtr[i] = UnsafeMutableBufferPointer<T>.allocate(capacity: cols)
        }
        // NOW POPULATE
        for i in 0..<rows {
            for j in 0..<cols {
                // Deep copy each pointee from rands jth offset to this jth pointer offset
                doublePtr[i][j] = repeated
            }
        }
        return doublePtr
    }
    
    // MARK: 2D Reset
    public static func reset<T>(_ ptr: UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>>, repeated: T) {
        // Reset all values
        for i in 0..<ptr.count {
            for j in 0..<ptr[i].count {
                ptr[i][j] = repeated
            }
        }
    }
    
    public static func reset<T>(_ ptr: UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>>, copy: UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>>, randomSample: Bool) {
        // Reset all values
        for i in 0..<ptr.count {
            let rand = randomSample ? copy.randomElement()! : copy[i]
            for j in 0..<ptr[i].count {
                ptr[i][j] = rand[j]
            }
        }
    }
    
    // MARK: 2D Free
    public static func free<T>(_ ptr: UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<T>>) {
        for i in 0..<ptr.count {
            ptr[i].deallocate()
        }
        ptr.deallocate()
    }
    
    // MARK: 1D Alloc
    public static func alloc<T>(copy: UnsafeMutableBufferPointer<T>) -> UnsafeMutableBufferPointer<T> {
        // FIRST ALLOCATE SPACE
        // Allocate the space for this pointer
        let ptr = UnsafeMutableBufferPointer<T>.allocate(capacity: copy.count)
        // NOW POPULATE
        for i in 0..<copy.count {
            ptr[i] = copy[i]
        }
        return ptr
    }
    
    public static func alloc<T>(cols: Int, copy: UnsafeMutableBufferPointer<T>, randomSample: Bool = false) -> UnsafeMutableBufferPointer<T> {
        // FIRST ALLOCATE SPACE
        // Allocate the space for this pointer
        let ptr = UnsafeMutableBufferPointer<T>.allocate(capacity: cols)
        // NOW POPULATE
        for i in 0..<cols {
            ptr[i] = randomSample ? copy.randomElement()! : copy[i]
        }
        return ptr
    }
    
    public static func alloc<T>(cols: Int, repeated: T) -> UnsafeMutableBufferPointer<T> {
        // FIRST ALLOCATE SPACE
        // Allocate the space for this pointer
        let ptr = UnsafeMutableBufferPointer<T>.allocate(capacity: cols)
        // NOW POPULATE
        for i in 0..<cols {
            ptr[i] = repeated
        }
        return ptr
    }
    
    // MARK: 1D Reset
    public static func reset<T>(_ ptr: UnsafeMutableBufferPointer<T>, repeated: T) {
        // Reset all values
        for i in 0..<ptr.count {
            ptr[i] = repeated
        }
    }
    
    public static func reset<T>(_ ptr: UnsafeMutableBufferPointer<T>, copy: UnsafeMutableBufferPointer<T>, randomSample: Bool) {
        // Reset all values
        for i in 0..<ptr.count {
            ptr[i] = randomSample ? copy.randomElement()! : copy[i]
        }
    }
    
    // MARK: 1D Free
    public static func free<T>(_ ptr: UnsafeMutableBufferPointer<T>) {
        ptr.deallocate()
    }
}
