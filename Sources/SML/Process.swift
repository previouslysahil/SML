//
//  Process.swift
//  MachineLearning
//
//  Created by Sahil Srivastava on 7/27/21.
//

import Foundation
import SwiftUI

public class Process {
    
    private let type: ProcessType
    
    private let normalize = Normalize(type: .zscore)
    private let pca = PCA()
    
    private var processed: Bool = false
    
    public init(type: ProcessType) {
        self.type = type
    }
    
    public func fit(X: Matrix) -> Matrix {
        processed = true
        switch type {
        case .normalize:
            return normalize.fit(X: X)
        case .pca:
            return pca.fit(X: X)
        case .none:
            return X
        }
    }
    
    public func fit(x: Vector) -> Vector {
        precondition(processed, "Data was not processed first")
        switch type {
        case .normalize:
            return normalize.fit(x: x)!
        case .pca:
            return pca.fit(x: x)!
        case .none:
            return x
        }
    }
}

public enum ProcessType {
    case normalize
    case pca
    case none
}
