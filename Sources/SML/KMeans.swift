//
//  KMeans.swift
//  MachineLearning
//
//  Created by Sahil Srivastava on 8/2/21.
//

import Foundation
import SwiftUI
import Accelerate

// MARK: KMeans
public class KMeans {
    
    private var data: UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<Double>>!
    
    private var training = false
    
    // MARK: Init
    public init(data: [[Double]]) {
        setup(data: data)
    }
    
    public init() {}
    
    // MARK: Setup
    public func setup(data: [[Double]]) {
        // Set up double pointer data (freed in deinit)
        self.data = Pointers.alloc(rows: data.count, cols: data[0].count, repeated: 0.0)
        // Set the double pointer data from our 2d data array
        for i in 0..<data.count {
            for j in 0..<data[i].count {
                // Deep copy each feature to our double pointer
                self.data[i][j] = data[i][j]
            }
        }
    }
    
    // MARK: Centers
    public func centers(K: Int, cap: Int = 50, batches: Int = 20) -> [[Double]] {
        let start = CFAbsoluteTimeGetCurrent()
        // Start training
        training = true
        // Set up cost and deviation
        var cost = 0.0
        var deviation = 0.0
        // Set up iterations
        var iterations = 0
        
        // Setup random subset of the data double pointer
        let dataMini = mini(batches: batches)
        // Mark pointer to be freed upon end of scope
        defer {
            Pointers.free(dataMini)
        }
        // Set up a random-ish centroids double pointer in the range of the values using kmeans++
        let centroids = betterCentroids(K: K, dataMini: dataMini)
        // Mark pointer to be freed upon end of scope
        defer {
            Pointers.free(centroids)
        }
        // Set up pointer of centroid indexs for each example in mini batch (to -1 since no Centroids chosen yet)
        let centers = Pointers.alloc(cols: dataMini.count, repeated: -1)
        // Mark pointer to be freed upon end of scope
        defer {
            Pointers.free(centers)
        }
        repeat {
            // Concurrently iterate through our mini batch
            DispatchQueue.concurrentPerform(iterations: dataMini.count) { i in
                // First set distance to be the max possible distance
                var dist = Double.greatestFiniteMagnitude
                // Iterate through each centroid to find the closest centroid to this mini batch training example
                for k in 0..<centroids.count {
                    // Set up pointer for calculation of distance
                    let power = Pointers.alloc(cols: centroids[k].count, repeated: 0.0)
                    // Mark pointer to be freed upon end of scope
                    defer {
                        Pointers.free(power)
                    }
                    // Calculate first part of distance
                    for j in 0..<power.count {
                        power[j] = pow(centroids[k][j] - dataMini[i][j], 2)
                    }
                    // Calculate second part of distance
                    let reduce = power.reduce(0.0, +)
                    let dist_k = reduce.squareRoot()
                    // Check if this computed distance is less than our current distance
                    if dist_k < dist {
                        // Assign this to be our new distance
                        dist = dist_k
                        // Also set this examples center to be this kth centroid
                        centers[i] = k
                    }
                }
            }
            
            // Set up centroid counts pointer for averaging
            let counts = Pointers.alloc(cols: K, repeated: 0.0)
            // Mark pointer to be freed upon end of scope
            defer {
                Pointers.free(counts)
            }
            // Save the old centroids as a copy of the centroids double pointer
            let oldCentroids = Pointers.alloc(rows: centroids.count, cols: data[0].count, copy: centroids)
            // Mark pointer to be freed upon end of scope
            defer {
                Pointers.free(oldCentroids)
            }
            // Now zero out centroids double pointerfor accumalation
            Pointers.reset(centroids, repeated: 0.0)
            
            // Average our training examples in each centroid to get our new centroid center
            for k in 0..<centroids.count {
                for i in 0..<centers.count {
                    // If this example is in our centroids cluster then add to it
                    if centers[i] == k {
                        // Add this example to the centroid k it has as a center
                        for j in 0..<centroids[k].count {
                            // Deep addition for each pointee for this example
                            centroids[k][j] += dataMini[i][j]
                        }
                        // Increment the count of this centroid k (used for averaging)
                        counts[k] += 1.0
                    }
                }
            }
            // Now divide each centroid by the number of examples added to it for averaging
            for k in 0..<centroids.count {
                // Check for empty clusters
                if counts[k] == 0 {
                    // Just reset to random centroid if empty
                    let rand = dataMini.randomElement()!
                    for j in 0..<centroids[k].count {
                        // Deep copy for each pointee for this example
                        centroids[k][j] = rand[j]
                    }
                } else {
                    // Average out the centroid by its count
                    for j in 0..<centroids[k].count {
                        // Deep division for each pointee for this example
                        centroids[k][j] /= counts[k]
                    }
                }
            }
            
            // Calculate the cost
            cost = 0.0
            for i in 0..<dataMini.count {
                // Get the centroid k for this example
                let k = centers[i]
                // Set up pointer for calculation of cost
                let power = Pointers.alloc(cols: centroids[k].count, repeated: 0.0)
                // Mark pointer to be freed upon end of scope
                defer {
                    Pointers.free(power)
                }
                // Calculate the first part of our cost
                for j in 0..<power.count {
                    power[j] = pow(centroids[k][j] - dataMini[i][j], 2)
                }
                // Calculate the second part of our cost
                let reduce = power.reduce(0.0, +)
                cost += reduce
            }
            // Finally average to find cost
            cost /= Double(dataMini.count)
            
            // Calculate the deviation
            deviation = 0.0
            for k in 0..<centroids.count {
                // Set up pointer for calculation of cost
                let power = Pointers.alloc(cols: centroids[k].count, repeated: 0.0)
                // Mark pointer to be freed upon end of scope
                defer {
                    Pointers.free(power)
                }
                // Calculate the first part of our deviation
                for j in 0..<power.count {
                    power[j] = pow(centroids[k][j] - oldCentroids[k][j], 2)
                }
                // Calculate the second part of our deviation
                let reduce = power.reduce(0.0, +)
                deviation += reduce.squareRoot()
            }
            // Finally avergae to find deviation
            deviation /= Double(centroids.count)
            
            // Increment the number of iterations
            iterations += 1
            // Re randomize mini subset of data if iterations < cap
            if iterations < cap {
                Pointers.reset(dataMini, copy: data, randomSample: true)
            }
            print("KMeans.predict: Cost \(cost) Deviation \(deviation) Iterations \(iterations)")
        } while iterations < cap && training
        
        // Get array from centroids
        var centroidsArr = [[Double]]()
        centroidsArr.reserveCapacity(K)
        for k in 0..<centroids.count {
            centroidsArr.append(Array(centroids[k]))
        }
        
        // End training
        training = false
        let diff = CFAbsoluteTimeGetCurrent() - start
        print("KMeans.predict: Took \(diff)")
        return centroidsArr
    }
    
    // MARK: KMeans++ Initialization
    private func betterCentroids(K: Int, dataMini: UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<Double>>) -> UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<Double>> {
        // Create double pointer (freed in centers)
        let centroids = Pointers.alloc(rows: K, cols: data[0].count, copy: dataMini, randomSample: true)
        // We don't touch the first centroid since it will always be random but the rest will be reinitialized to the farthest centroid from the previous centroids
        for k in 1..<K {
            // First set distance to be the min possible distance
            var dist = Double.leastNonzeroMagnitude
            // Iterate through each example
            for i in 0..<dataMini.count {
                // For each example iterate through all the currently initialized centroids
                for k0 in 0..<k {
                    // Set up pointer for calculation of cost
                    let power = Pointers.alloc(cols: centroids[k0].count, repeated: 0.0)
                    // Mark pointer to be freed upon end of scope
                    defer {
                        Pointers.free(power)
                    }
                    // Calculate first part of distance
                    for j in 0..<power.count {
                        power[j] = pow(centroids[k0][j] - dataMini[i][j], 2)
                    }
                    // Calculate second part of distance
                    let reduce = power.reduce(0.0, +)
                    let dist_k = reduce.squareRoot()
                    // Check if this computed distance is greater than our current distance
                    if dist_k > dist {
                        // Assign this to be our new distance
                        dist = dist_k
                        // Also deep copy this example to be our tentaively new kth centroid
                        for j in 0..<centroids[k].count {
                            // Deep copy each pointee from dataMini[i]s jth offset to this jth pointer offset
                            centroids[k][j] = dataMini[i][j]
                        }
                    }
                }
            }
        }
        return centroids
    }
    
    // MARK: KMeans Mini Batch
    private func mini(batches: Int) -> UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<Double>> {
        // Get the size we want to truncate to
        let b = data.count / batches
        // Create random mini batch double pointer (freed in centers)
        let dataMini = Pointers.alloc(rows: b, cols: data[0].count, copy: data, randomSample: true)
        return dataMini
    }
    
    deinit {
        Pointers.free(self.data)
    }
}
