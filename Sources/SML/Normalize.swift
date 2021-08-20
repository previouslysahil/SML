//
//  Normalize.swift
//  MachineLearning
//
//  Created by Sahil Srivastava on 7/26/21.
//

import Foundation
import SwiftUI

public class Normalize {
    
    private var maxs: Vector?
    private var mins: Vector?
    private var means: Vector?
    private var stds: Vector?
    
    private let type: NormalizeType
    
    public init(type: NormalizeType) {
        self.type = type
    }
    
    public func fit(X: Matrix) -> Matrix {
        switch self.type {
        case .linear:
            return normalize(X: X, maxs: &maxs, mins: &mins)
        case .zscore:
            return normalize(X: X, means: &means, stds: &stds)
        }
    }
    
    public func fit(x: Vector) -> Vector? {
        switch self.type {
        case .linear:
            if let mins = mins, let maxs = maxs {
                return (x - mins) / (maxs - mins)
            } else {
                return nil
            }
        case .zscore:
            if let means = means, let stds = stds {
                return (x - means) / stds
            } else {
                return nil
            }
        }
    }
    
    private func normalize(X: Matrix, maxs: inout Vector?, mins: inout Vector?) -> Matrix {
        // Init maxs and mins for the number of features plus theta_0
        maxs = Matrix(0..<X.columns, isColumnVector: true)
        mins = Matrix(0..<X.columns, isColumnVector: true)
        // Set up normalized X
        var Xn = X
        // Iterate through each feature and normalize
        for j in 0..<Xn.columns {
            // Get max an min for this feature
            var max = Xn.max(column: j).0
            var min = Xn.min(column: j).0
            // If feature has no range just set max to one and min to 0 so they don't affect normalizing
            // This if statement will always trigget for feature_0 added for theta_0 since all x_0 = 1
            // for training sets i
            if max - min == 0 {
                max = 1
                min = 0
            }
            // Add the range and min to our ranges and mins at the appropriate index for future normalizations
            maxs![0, j] = max
            mins![0, j] = min
            // Now normalize this feature set for our matrix
            Xn[column: j] = (Xn[column: j] - min) / (max - min)
        }
        return Xn
    }
    
    private func normalize(X: Matrix, means: inout Vector?, stds: inout Vector?) -> Matrix {
        // Init means and stds for the number of features plus theta_0
        means = X.mean()
        stds = X.std()
        // Set up normalized X
        var Xn = X
        // Iterate through each feature and normalize
        for j in 0..<Xn.columns {
            // Get mean and std
            let mean = means![j]
            let std = stds![j]
            // Now normalize this feature set for our matrix
            Xn[column: j] = (Xn[column: j] - mean) / std
        }
        return Xn
    }
}

public enum NormalizeType {
    case linear
    case zscore
}
