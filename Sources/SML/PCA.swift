//
//  PCA.swift
//  MachineLearning
//
//  Created by Sahil Srivastava on 7/23/21.
//

import Foundation
import SwiftUI

public class PCA {
    
    private var Ureduce: Matrix?
    
    private let normalize = Normalize(type: .zscore)
    
    public func fit(X: Matrix, k: Int? = nil) -> Matrix {
        // Mean normalize data
        let Xn = normalize.fit(X: X)
        // Compute covariance matrix sigma
        let SIGMA = 1 / Double(Xn.rows) * (Xn.transpose() <*> Xn)
        // Decompose sigma
        let (U, S, _) = SIGMA.svd()
        // Set up ks
        var K = [(index: Int, val: Double)](repeating: (0, 0.0), count: min(S.rows, S.columns))
        // Set the index for each k
        for i in 0..<K.count {
            K[i].index = i
        }
        // Iterate through and set up our k values
        for i in 0..<K.count {
            // Set sums to be 0
            var sum_k = 0.0
            var sum_m = 0.0
            // Solve for 99% variance
            for j in 0..<min(S.rows, S.columns) {
                // Sum up till k
                if j < K[i].index + 1 {
                    sum_k += S[i, i]
                }
                // Sum up till m
                sum_m += S[i, i]
            }
            // Find the variance error for this subset
            K[i].val = 1 - (sum_k / sum_m)
        }
        print("PCA.fit: All ks \(K)")
        // See if user chose a K
        if let k = k {
            precondition(k > 0 && k <= U.count, "Defined k is not acceptable for reduction of matrix X")
            print("PCA.fit: Chosen k \(k)")
            // Get our eigenspaces
            let Ureduce = U[columns: 0...k - 1]
            self.Ureduce = Ureduce
            // Get dimensionally reduced matrix
            let Z = Xn <*> Ureduce
            return Z
        } else {
            // Find our k given smallest variance
            let k = K.first(where: { $0.val < 0.01 })!
            print("PCA.fit: Chosen k \(k)")
            // Get our eigenspaces
            let Ureduce = U[columns: 0...k.index]
            self.Ureduce = Ureduce
            // Get dimensionally reduced matrix
            let Z = Xn <*> Ureduce
            return Z
        }
    }
    
    public func fit(x: Vector) -> Vector? {
        if let Ureduce = Ureduce {
            guard let xn = normalize.fit(x: x) else { return nil }
            let z = Ureduce.transpose() <*> xn
            return z
        } else {
            return nil
        }
    }
}
