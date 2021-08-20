//
//  NeuralNetwork.swift
//  MachineLearning
//
//  Created by Sahil Srivastava on 7/6/21.
//

import Foundation
import SwiftUI
import Accelerate

// Use for strictly vectors only to differentiate
public typealias Vector = Matrix

// MARK: NeuralNetwork
public class NeuralNetwork {
    
    private var X: Matrix!
    private var y: Matrix!
    private var hyper: NeuralNetworkHyper!
    private var THETA: [Matrix]?
    private var costs: [Double]!
    
    private var training: Bool = false
    private var ready = false
    
    private var data: [[Double]]!
    private var labels: [[Double]]!
    
    private let process = Process(type: .normalize)
    
    private static let keysTag = "NeuralNetwork.keys"
    
    // MARK: Init
    public init(data: [[Double]], labels: [[Double]], hyper: NeuralNetworkHyper) {
        // Init all necessary params
        setup(data: data, labels: labels, hyper: hyper)
    }
    
    public init(key: String) {
        // Init all necessary params
        load(key: key)
    }
    
    public init() {}
    
    // MARK: Setup
    public func setup(data: [[Double]], labels: [[Double]], hyper: NeuralNetworkHyper) {
        // Store raw
        self.data = data
        self.labels = labels
        // Save hyper params
        self.hyper = hyper
        // Use this to create X
        self.X = process.fit(X: Matrix(data))
        // Reshape X to batches
        self.X = self.X[rows: 0..<(X.rows / hyper.batches * hyper.batches)]
        // Use this to create y
        self.y = Matrix(labels)
        // Reshape y to batches
        self.y = self.y[rows: 0..<(y.rows / hyper.batches * hyper.batches)]
        // Set up empty costs
        self.costs = []
        // Make sure that X and y have the same number of training examples AKA rows
        precondition(X.rows == y.rows, "NeuralNetwork.setup: Data and labels have unequal number of training examples")
        // If we have hidden layers make sure each layers has one or more units
        precondition(hyper.hidden.allSatisfy { $0.units >= 1 }, "NeuralNetwork.setup: One or more hidden layer has 0 units")
        // Network is ready to be trained
        self.ready = true
    }
    
    // MARK: Load
    public func load(key: String) {
        // Get our json data for this key's neural network
        do {
            let info = try JSONDecoder().decode(NeuralNetworkInfo.self, from: UserDefaults.standard.data(forKey: key) ?? Data())
            // Store raw
            self.data = info.data
            self.labels = info.labels
            // Save hyper params
            self.hyper = info.hyper
            // Use this to create X
            self.X = process.fit(X: Matrix(data))
            // Reshape X to batches
            self.X = self.X[rows: 0..<(X.rows / info.hyper.batches * info.hyper.batches)]
            // Use this to create y
            self.y = Matrix(labels)
            // Reshape y to batches
            self.y = self.y[rows: 0..<(y.rows / info.hyper.batches * info.hyper.batches)]
            // Set up empty costs
            self.costs = []
            // Set up THETA
            self.THETA = info.THETA
            // Network is ready to be trained
            self.ready = true
        } catch {
            print("NeuralNetwork.load: Unable to find a saved network for this key")
        }
    }
    
    // MARK: Train
    public func train() {
        // Make sure network was setup
        guard ready else {
            precondition(ready, "NeuralNetwork.train: Network was not given data, labels, and hyper params")
            return
        }
        // Set up blank a vectors
        var a: [Vector] = blank_network(hidden: hyper.hidden, features: X.columns, classes: y.columns)
        var z: [Vector] = blank_network(hidden: hyper.hidden, features: X.columns, classes: y.columns)
        // Set up random bounded THETA
        var THETA: [Matrix] = rand_THETA(a: a, L: hyper.hidden.count + 2)
        // Train
        training = true
        train(X: X, y: y, a: &a, z: &z, THETA: &THETA, hyper: hyper, checking: false)
        // Set our theta
        self.THETA = THETA
    }
    
    // MARK: Stop
    public func stop() {
        training = false
    }
    
    // MARK: Predict
    public func predict(for example: [Double]) -> [Double]? {
        // Make sure the network was trained before saving
        guard let THETA = THETA, ready else {
            print("NeuralNetwork.predict: Network must be trained at least once before predictiing")
            return nil
        }
        // Set up blank net for forward prop
        var a: [Vector] = blank_network(hidden: hyper.hidden, features: X.columns, classes: y.columns)
        var z: [Vector] = blank_network(hidden: hyper.hidden, features: X.columns, classes: y.columns)
        // Get vector x
        let x = Vector(example)
        // Forward prop
        let h = forward(a: &a, z: &z, THETA: THETA, x: process.fit(x: x), L: hyper.hidden.count + 2)
        // Return predicted class vector
        return h.grid
    }
    
    // MARK: Save
    public func save(key: String) {
        // Make sure the network was trained before saving
        guard let THETA = THETA, ready else {
            print("NeuralNetwork.save: Network must be trained at least once before saving")
            return
        }
        // Set up struct containing NeuralNetwork information
        let info = NeuralNetworkInfo(data: data, labels: labels, hyper: hyper, THETA: THETA)
        // Attempt to encode
        do {
            // Convert neural net info to json and save
            UserDefaults.standard.setValue(try JSONEncoder().encode(info), forKey: key)
            
            // Save the key as well, first get previously saved keys
            var keys = UserDefaults.standard.array(forKey: NeuralNetwork.keysTag) as? [String] ?? [String]()
            // If this key is already saved, do nothing
            if !keys.contains(key) {
                // This key is not saved so add it and save the keys again
                keys.append(key)
                // Now convert keys to json and save
                UserDefaults.standard.setValue(keys, forKey: NeuralNetwork.keysTag)
            }
            print("NeuralNetwork.save: Successfully saved network with key \(key)")
        } catch {
            // Unable to encode either NeuralNet or keys, either way confirm we have cleared the k/v pair in UserDefaults for neural net
            UserDefaults.standard.removeObject(forKey: key)
            print("NeuralNetwork.save: Unable to save neural network")
        }
    }
    
    // MARK: Saved
    public static func keys() -> [String] {
        // Get the saved keys from UserDefaults
        let keys = UserDefaults.standard.array(forKey: NeuralNetwork.keysTag) as? [String] ?? [String]()
        return keys
    }
    
    // MARK: Remove Saved
    public static func clear() {
        // Remove stored infos first
        for key in NeuralNetwork.keys() {
            UserDefaults.standard.removeObject(forKey: key)
        }
        // Now remove keys
        UserDefaults.standard.removeObject(forKey: NeuralNetwork.keysTag)
        print("NeuralNetwork.clear: Cleared all stored neural networks")
    }
    
    public static func clear(key: String) {
        // Remove stored info first
        UserDefaults.standard.removeObject(forKey: key)
        // Get keys
        var keys = UserDefaults.standard.array(forKey: NeuralNetwork.keysTag) as? [String] ?? [String]()
        // Remove the key from our keys
        keys.removeAll(where: { $0 == key })
        // Resave the keys
        UserDefaults.standard.setValue(keys, forKey: NeuralNetwork.keysTag)
        print("NeuralNetwork.clear: Cleared neural network for key \(key)")
    }
    
    // MARK: Diagnostic
    public func diagnostic() {
        // Make sure network was setup
        precondition(ready, "Neural Network was not given data, labels, and hyper params")
        // Partition X and y into train and cv sets
        let (X_train, y_train, X_cv, y_cv) = diagnostic_partition(X: X, y: y)
        
        // Train the model
        let model_train = NeuralNetwork(data: X_train.array, labels: y_train.array, hyper: hyper)
        model_train.train()
        
        // Make the blank network
        var a: [Vector] = model_train.blank_network(hidden: model_train.hyper.hidden, features: X_train.columns, classes: y_train.columns)
        var z: [Vector] = model_train.blank_network(hidden: model_train.hyper.hidden, features: X_train.columns, classes: y_train.columns)
        // Set up cost_train to be 0 at first
        var cost_train: Double = 0.0
        for i in 0..<X_train.rows {
            // Propagate out hypothesis
            let h = model_train.forward(a: &a, z: &z, THETA: model_train.THETA!, x: X_train[row: i], L: model_train.hyper.hidden.count + 2)
            // Add to the cost
            let cost_i = model_train.costNested(h: h, y: y_train[row: i].transpose())
            // Make sure cost_i is a number
            if cost_i.count != 1 { fatalError("9") }
            // Add cost of this training set to total cost
            cost_train += cost_i.sum()
        }
        // Averages out cost and regularize for the specifies loss function
        cost_train = model_train.costOuter(cost: cost_train, m: X_train.rows, lm: model_train.hyper.lm, THETA: model_train.THETA!)
        
        // Make the blank network
        a = model_train.blank_network(hidden: model_train.hyper.hidden, features: X_cv.columns, classes: y_cv.columns)
        z = model_train.blank_network(hidden: model_train.hyper.hidden, features: X_cv.columns, classes: y_cv.columns)
        // Set up cost to be 0 at first
        var cost_cv: Double = 0.0
        for i in 0..<X_cv.rows {
            // Propagate out hypothesis
            let h = model_train.forward(a: &a, z: &z, THETA: model_train.THETA!, x: X_cv[row: i], L: model_train.hyper.hidden.count + 2)
            // Add to the cost
            let cost_i = model_train.costNested(h: h, y: y_cv[row: i].transpose())
            // Make sure cost_i is a number
            if cost_i.count != 1 { fatalError("9") }
            // Add cost of this training set to total cost
            cost_cv += cost_i.sum()
        }
        // Averages out cost and regularize for the specifies loss function
        cost_cv = model_train.costOuter(cost: cost_cv, m: X_cv.rows, lm: model_train.hyper.lm, THETA: model_train.THETA!)
        
        print("NeuralNetwork.diagnostic: Training cost: \(cost_train) Cross Validation cost: \(cost_cv)")
    }

    // MARK: Diagnostic (Partition)
    private func diagnostic_partition(X: Matrix, y: Matrix) -> (X_train: Matrix, y_train: Matrix, X_cv: Matrix, y_cv: Matrix) {
        // Concatentate X and y together for reshuffling
        var Xy_arr = X.array
        for i in 0..<Xy_arr.count {
            Xy_arr[i].append(contentsOf: y.array[i])
        }
        // Now shuffle the combined X and y
        Xy_arr.shuffle()
        
        // Now seperate them again
        var X_arr = X.array
        var y_arr = y.array
        for i in 0..<Xy_arr.count {
            X_arr[i] = Array(Xy_arr[i][0..<X.columns])
            y_arr[i] = Array(Xy_arr[i][X.columns..<Xy_arr[i].count])
        }
        
        // Now seperate our shuffled data into training and cv sets
        let training_range: Range<Int> = 0..<Int(Double(X.rows) * 0.75)
        let cv_range: Range<Int> = (training_range.upperBound)..<X.rows
        
        // Use training_range to create processed X_train_m
        var X_train = process.fit(X: Matrix(Array(X_arr[training_range])))
        // Reshape X_train_m to batches
        X_train = X_train[rows: 0..<(X_train.rows / hyper.batches * hyper.batches)]
        // Use training_range to create processed y
        var y_train = Matrix(Array(y_arr[training_range]))
        // Reshape y_train_m to batches
        y_train = y_train[rows: 0..<(y_train.rows / hyper.batches * hyper.batches)]
        
        // Use cv_range to create processed X_cv_m
        var X_cv = process.fit(X: Matrix(Array(X_arr[cv_range])))
        // Reshape X_train_m to batches
        X_cv = X_cv[rows: 0..<(X_cv.rows / hyper.batches * hyper.batches)]
        // Use cv_range to create processed y
        var y_cv = Matrix(Array(y_arr[cv_range]))
        // Reshape y_train_m to batches
        y_cv = y_cv[rows: 0..<(y_cv.rows / hyper.batches * hyper.batches)]
        
        return (X_train, y_train, X_cv, y_cv)
    }
    
    // MARK: Blank Neural Network
    private func blank_network(hidden: [NeuralNetworkHidden], features: Int, classes: Int) -> [Vector] {
        // Start off by setting up a vectors excluding bias units
        var a = [Vector](repeating: Vector(), count: hidden.count + 2)
        // Set first a vector to dimension of x_is in X
        a[0] = Vector(0..<features)
        // Iterate through our hidden layers if we have them
        for l in 0..<hidden.count {
            // Set lth a vector to dimension of # of units
            a[l + 1] = Vector(0..<hidden[l].units)
        }
        // Set up last a vector (hypothesis layer) to dimension of y_is in y, AKA number of classes
        a[a.count - 1] = Vector(0..<classes)
        return a
    }
    
    // MARK: Randomly Init Weights
    private func rand_THETA(a: [Vector], L: Int) -> [Matrix] {
        // Start off by making L optional THETA matrices to ease future ops
        var THETA = [Matrix](repeating: Matrix(), count: L - 1)
        // We only populate the first L-1 matrices since there are only L-1 THETA matrices
        for l in 0..<L - 1 {
            // THETA_L will always be a_l+1 by a_l + 1 in dimension
            let THETA_l: Matrix
            // Set up our random initializers
            switch hyper.activH {
            case .Sigmoid:
                THETA_l = Matrix.random_xavier(rows: a[l + 1].rows, columns: a[l].rows + 1, ni: a[l].rows, no: a[l + 1].rows)
            case .ReLU:
                THETA_l = Matrix.random_kaiming(rows: a[l + 1].rows, columns: a[l].rows + 1, ni: a[l].rows)
            case .LeakyReLU:
                THETA_l = Matrix.random_kaiming(rows: a[l + 1].rows, columns: a[l].rows + 1, ni: a[l].rows)
            }
            // Assign to our array of THETAs
            THETA[l] = THETA_l
        }
        return THETA
    }
    
    // MARK: - Neural Network Algorithm
    private func train(X: Matrix, y: Matrix, a: inout [Vector], z: inout [Vector], THETA: inout [Matrix], hyper: NeuralNetworkHyper, checking: Bool) {
        // Start training
        training = true
        // Mini batch size
        let b = X.rows / hyper.batches
        // Set up cost to be 0 at first
        var cost: Double = 0.0
        // Make sure costs is empty
        costs = []
        // Reserve capacity for the number of epocs unless max
        if hyper.epochs != Int.max { costs.reserveCapacity(hyper.epochs) }
        // Set up empty array of gradients (or all zeros)
        var m = [Matrix]()
        var v = [Matrix]()
        repeat {
            // Set up our iterations for this epoch
            var t = 0
            // Reset cost for this epoch
            cost = 0.0
            // Iterate through our training set by the batch size (all batches are even beacuse of clipping in setup)
            for i in stride(from: 0, to: X.rows, by: b) {
                // Add to our iterations
                t += 1
                // Init DELTA to empty (or all zeros)
                var DELTA = [Matrix]()
                // Solve forward and backward propagation for all training sets in this batch
                for k in i..<i + b {
                    // Compute forward propagation
                    let h = forward(a: &a, z: &z, THETA: THETA, x: X[row: k], L: hyper.hidden.count + 2)
                    // Compute backward propagation
                    backward(a: a, z: z, THETA: THETA, y: y[row: k].transpose(), DELTA: &DELTA, L: hyper.hidden.count + 2)
                    // Add to the cost
                    let cost_k = costNested(h: h, y: y[row: k].transpose())
                    // Make sure cost_k is a number
                    if cost_k.count != 1 { fatalError("5") }
                    // Add cost of this training set to total cost
                    cost += cost_k.sum()
                }
                // Get our derivative vectors of the cost function
                let D: [Matrix] = zip(DELTA, THETA).map { DELTA_l, THETA_l in
                    if (DELTA_l.rows != THETA_l.rows) || (DELTA_l.columns != THETA_l.columns) { fatalError("6") }
                    // Use coefficient to average out the training sets
                    return (1.0 / Double(b)) * (DELTA_l + hyper.lm * THETA_l)
                }
                // Use gradient checking to find the difference between the actual derivatives and gradient checked derivatives
                if checking { _ = gradCheck(X: X[rows: i..<i + b], y: y[rows: i..<i + b], THETA: THETA, D: D, hidden: hyper.hidden, epsilon: 0.000001) }
                
                // Run gradient descent with adam optimizer
                gradAdam1(D: D, m: &m, b1: 0.9)
                gradAdam2(D: D, v: &v, b2: 0.999)
                // Combine our adam1 and adam2 functions to get our gradient
                let gradient: [Matrix] = zip(m, v).map { m_l, v_l in
                    // Get corrected v and s
                    let mc_l: Matrix = m_l / (1 - pow(0.9, Double(t)))
                    let vc_l: Matrix = v_l / (1 - pow(0.999, Double(t)))
                    // Define our gradient
                    return hyper.lr * (mc_l / (vc_l.sqrt() + 0.00000001))
                }
                // Subtract our gradient for each layer from its respective theta
                THETA = zip(THETA, gradient).map { THETA_l, gradient_l in
                    THETA_l - gradient_l
                }
            }
            // Averages out cost and regularize for the specified loss function
            // NOTE: this isn't technically the cost since every batch we added to
            // the cost with a new, slightly modified THETA but it should be close,
            // for the true final cost run final cost
            // NOTE: The regularization is done with the THETA calculated after
            // every batch so it is completely correct
            cost = costOuter(cost: cost, m: X.rows, lm: hyper.lm, THETA: THETA)
            // Determine the direction of our descent
            let direction = costs.last ?? Double.greatestFiniteMagnitude > cost ? "DOWN" : "UP"
            costs.append(cost)
            // Print necessary info
            print("NeuralNetwork.train: Cost \(cost) Epoch \(costs.count) Dir \(direction)")
        } while costs.count < hyper.epochs && training
        // Display final cost
        finalCost(THETA: THETA)
        // Set training to false
        training = false
    }
    
    // MARK: Forwardprop
    private func forward(a: inout [Vector], z: inout [Vector], THETA: [Matrix], x: Vector, L: Int) -> Vector {
        // Start of by setting a(0) to be x AKA this training set
        a[0] = x.transpose() // Make n by 1
        z[0] = x.transpose()
        // Now iterate through the following layers and solve each a(l)
        for l in 1..<L {
            // First make a(l - 1) with bias unit so we can propagate the bias
            var a_l1 = a[l - 1].grid
            a_l1.insert(1.0, at: 0)
            // Then get z from previous layers a (with bias) and THETA
            let z_l = THETA[l - 1] <*> Vector(a_l1).transpose()
            // Set this for our z(l)
            z[l] = z_l
            // Pass z to activation function gH for hidden and gO for output
            let g = l < L - 1 ? activH(z_l: z_l) : activO(z_l: z_l)
            // Now create a vector from this array and set it to a(l)
            a[l] = g // Make n by 1
            // Make sure that a(l) is indeed a vector with only 1 column and x_m.count columns
            if a[l].columns != 1 { fatalError("7") }
        }
        // Note the last a[l] with be our hypothesis vector
        return a.last!
    }
    
    // MARK: Backprop
    private func backward(a: [Vector], z: [Vector], THETA: [Matrix], y: Vector, DELTA: inout [Matrix], L: Int) {
        // Use backward propagation to compute our Dv vectors "derivative of cost function"
        let delta = deltas(a: a, z: z, THETA: THETA, y: y, L: L)
        // Use delta vectors to compute DELTA
        DELTAS(DELTA: &DELTA, delta: delta, a: a, L: L)
    }
    
    // MARK: Backprop (Deltas)
    private func deltas(a: [Vector], z: [Vector], THETA: [Matrix], y: Vector, L: Int) -> [Vector] {
        // Start off by making a blank array of L-1 delta vectors
        var delta = [Vector](repeating: Vector(), count: L - 1)
        // Check loss function for derivative mult. or not, last d(l) always the difference between a(L) and y
        switch hyper.loss {
        case .CrossEntropy:
            // Partial derivative wrt to THETA doesn't mult activation derivative
            delta[delta.count - 1] = (a[delta.count] - y)
        case .MSE:
            // Partial derivative wrt to THETA does mult activation derivative
            let dg = dActivO(z_l: z[delta.count])
            delta[delta.count - 1] = (a[delta.count] - y) * dg
        }
        // Now iterate down until our second layer to get our final d(l), starting at the second to last delta
        for l in (0..<delta.count - 1).reversed() {
            // Find the derivative of our activation function
            let dg = dActivH(z_l: z[l + 1])
            // Calculate delta by multiplying THETA without bias column and previous delta and dz
            delta[l] = (THETA[l + 1][columns: 1..<THETA[l + 1].columns].transpose() <*> delta[l + 1]) * dg
        }
        return delta
    }
    
    // MARK: Backprop (Derivatives)
    private func DELTAS(DELTA: inout [Matrix], delta: [Vector], a: [Vector], L: Int) {
        if DELTA.isEmpty {
            // Delta is empty we set up our first DELTAs for each layer
            for l in 0..<L - 1 {
                // Add our bias node so we can calculate the gradient for the bias (which is just delta * 1)
                var a_l1 = a[l].grid
                a_l1.insert(1.0, at: 0)
                // Update the total DELTA vector
                let DELTA_l = delta[l] <*> Vector(a_l1)
                DELTA.append(DELTA_l)
            }
        } else {
            // Delta is not empty we add to each DELTA for each layer
            for l in 0..<L - 1 {
                // Add our bias node so we can calculate the gradient for the bias (which is just delta * 1)
                var a_l1 = a[l].grid
                a_l1.insert(1.0, at: 0)
                // Update the total DELTA vector
                let DELTA_l = delta[l] <*> Vector(a_l1)
                DELTA[l] = DELTA[l] + DELTA_l
            }
        }
    }
    
    // MARK: Activation (Hidden)
    private func activH(z_l: Vector) -> Vector {
        // Run the vector through the given activation
        let g: Matrix
        switch hyper.activH {
        case .Sigmoid:
            g = z_l.Sigmoid()
        case .ReLU:
            g = z_l.ReLU()
        case .LeakyReLU:
            g = z_l.LeakyReLU(a: 0.2)
        }
        return g
    }
    
    private func dActivH(z_l: Vector) -> Vector {
        // Run the vector through the derivative of given activation
        let dg: Vector
        switch hyper.activH {
        case .Sigmoid:
            dg = z_l.dSigmoid()
        case .ReLU:
            dg = z_l.dReLU()
        case .LeakyReLU:
            dg = z_l.dLeakyReLU(a: 0.2)
        }
        return dg
    }
    
    // MARK: Activation (Output)
    private func activO(z_l: Vector) -> Vector {
        // Run the vector through the given activation
        let g: Matrix
        switch hyper.activO {
        case .Sigmoid:
            g = z_l.Sigmoid()
        case .ReLU:
            g = z_l.ReLU()
        case .LeakyReLU:
            g = z_l.LeakyReLU(a: 0.2)
        }
        return g
    }
    
    private func dActivO(z_l: Vector) -> Vector {
        // Run the vector through the derivative of given activation
        let dg: Vector
        switch hyper.activO {
        case .Sigmoid:
            dg = z_l.dSigmoid()
        case .ReLU:
            dg = z_l.dReLU()
        case .LeakyReLU:
            dg = z_l.dLeakyReLU(a: 0.2)
        }
        return dg
    }
    
    // MARK: Gradient Checking
    private func gradCheck(X: Matrix, y: Matrix, THETA: [Matrix], D: [Matrix], hidden: [NeuralNetworkHidden], epsilon: Double) -> [Double] {
        // Set up blank a vectors
        var a: [Vector] = blank_network(hidden: hidden, features: X.columns, classes: y.columns)
        var z: [Vector] = blank_network(hidden: hidden, features: X.columns, classes: y.columns)

        // Set up empty fake derivative matrix
        var D_f = [Double]()
        for l in 0..<(hidden.count + 2) - 1 {
            // Set up initial fake layer derivative
            var D_fl = 0.0
            for j in 0..<THETA[l].columns {
                // Our plus epsilon THETA_l array
                var THETA_p = THETA
                // Add to column j of layer ls THETA
                THETA_p[l][column: j] = THETA_p[l][column: j] + epsilon
                // Our minus epsilon THETA_l array
                var THETA_m = THETA
                // Subtract from column j of layer ls THETA
                THETA_m[l][column: j] = THETA_m[l][column: j] - epsilon
                
                D_fl += (gradCheckCost(X: X, y: y, THETA: THETA_p, a: &a, z: &z, L: hidden.count + 2) - gradCheckCost(X: X, y: y, THETA: THETA_m, a: &a, z: &z, L: hidden.count + 2)) / (2.0 * epsilon)
            }
            // Average out the layer derivative
            D_fl /= Double(THETA[l].columns)
            // Add this layers fake derivative * -1 to the fake derivaative matrix
            D_f.append(D_fl)
        }
        // Get the difference between the fake and real derivatives and return
        var D_diffs = [Double]()
        for l in 0..<(hidden.count + 2) - 1 {
            D_diffs.append(abs((D[l].sum() / Double(D[l].columns)) - D_f[l]))
            print("NeuralNetwork.gradient_checking: Gradient difference for layer \(l) is \(D_diffs[l])")
        }
        return D_diffs
    }
    
    // MARK: Gradient Checking (Cost)
    private func gradCheckCost(X: Matrix, y: Matrix, THETA: [Matrix], a: inout [Vector], z: inout [Vector], L: Int) -> Double {
        // Set up cost to be 0 at first
        var cost: Double = 0.0
        for i in 0..<X.rows {
            // Compute forward propagation
            let h = forward(a: &a, z: &z, THETA: THETA, x: X[row: i], L: L)
            // Add to the cost
            let cost_i = costNested(h: h, y: y[row: i].transpose())
            // Make sure costNested is a number
            if cost_i.count != 1 { fatalError("8") }
            // Add cost of this training set to total cost
            cost += cost_i.sum()
        }
        // Averages out cost and regularize for the specifies loss function
        cost = costOuter(cost: cost, m: X.rows, lm: hyper.lm, THETA: THETA)
        
        return cost
    }
    
    // MARK: Cost (Final Cost)
    private func finalCost(THETA: [Matrix]? = nil) {
        // Make sure we have THETAs
        guard let THETA = THETA else {
            print("NeuralNetwork.finalCost: Not trained")
            return
        }
        // Set up blank a vectors
        var a: [Vector] = blank_network(hidden: hyper.hidden, features: X.columns, classes: y.columns)
        var z: [Vector] = blank_network(hidden: hyper.hidden, features: X.columns, classes: y.columns)
        // Set up empty cost
        var cost = 0.0
        // Solve forward propagation to get summed nested cost
        for k in 0..<X.rows {
            // Compute forward propagation
            let h = forward(a: &a, z: &z, THETA: THETA, x: X[row: k], L: hyper.hidden.count + 2)
            // Add to the cost
            let cost_k = costNested(h: h, y: y[row: k].transpose())
            // Make sure cost_k is a number
            if cost_k.count != 1 { fatalError("5") }
            // Add cost of this training set to total cost
            cost += cost_k.sum()
        }
        // Averages out cost and regularize for the specified loss function
        cost = costOuter(cost: cost, m: X.rows, lm: hyper.lm, THETA: THETA)
        
        print("NeuralNetwork.finalCost: Cost: \(cost)")
    }
    
    // MARK: Cost (Outer)
    private func costOuter(cost: Double, m: Int, lm: Double, THETA: [Matrix]) -> Double {
        switch hyper.loss {
        case .CrossEntropy:
            return (-1.0 / Double(m)) * cost + (lm / (2.0 * Double(m))) * THETA.reduce(0.0, { $0 + $1.pow(2).sum() })
        case .MSE:
            return (1.0 / (2.0 * Double(m))) * (cost + lm * THETA.reduce(0.0, { $0 + $1.pow(2).sum() }))
        }
    }
    
    // MARK: Cost (Nested)
    private func costNested(h: Vector, y: Vector) -> Vector {
        // Calculate the cost for each example given our loss function
        switch hyper.loss {
        case .CrossEntropy:
            let h_t = h.transpose()
            // Add epsilon to handle log(0) -inf
            let epsilon = 0.00000001
            return (h_t + epsilon).log() <*> y + ((1 - h_t + epsilon).log() <*> (1 - y))
        case .MSE:
            return Vector(arrayLiteral: (h - y).pow(2).sum())
        }
    }
    
    // MARK: GD Adam (1)
    private func gradAdam1(D: [Matrix], m: inout [Matrix], b1: Double) {
        // Check if this is iteration of gradient descent by seeing if our array of gradients is empty
        if m.isEmpty {
            // First set up empty gradient array to store each layers gradient matrix
            var gradient = [Matrix]()
            // On nth iteration get the gradient with momentum for each layer and add it to our gradient matrix
            for l in 0..<D.count {
                // Make our gradient for this layer which is just our derivative * lr this time
                let gradient_l = (1 - b1) * D[l]
                // Add this layers gradient to the gradient array
                gradient.append(gradient_l)
            }
            // At this point gradient[l] is a matrix of the rolling averages of the previous derivatives and this iterations derivative so add it to our gradients to be used for our next gradient calculation
            m = gradient
        } else {
            // First set up empty gradient array to store each layers gradient matrix
            var gradient = [Matrix]()
            // On nth iteration get the gradient with momentum for each layer and add it to our gradient matrix
            for l in 0..<D.count {
                // Make our gradient for this layer using our last gradient and the derivative * lr
                let gradient_l = b1 * m[l] + (1 - b1) * D[l]
                // Add this layers gradient to the gradient array
                gradient.append(gradient_l)
            }
            // At this point gradient[l] is a matrix of the rolling averages of the previous derivatives and this iterations derivative so add it to our gradients to be used for our next gradient calculation
            m = gradient
        }
    }
    
    // MARK: GD Adam (2)
    private func gradAdam2(D: [Matrix], v: inout [Matrix], b2: Double) {
        // Check if this is iteration of gradient descent by seeing if our array of gradients is empty
        if v.isEmpty {
            // First set up empty gradient array to store each layers gradient matrix
            var gradient = [Matrix]()
            // On nth iteration get the gradient with momentum for each layer and add it to our gradient matrix
            for l in 0..<D.count {
                // Make our gradient for this layer which is just our derivative * lr this time
                let gradient_l = (1 - b2) * D[l].pow(2)
                // Add this layers gradient to the gradient array
                gradient.append(gradient_l)
            }
            // At this point gradient[l] is a matrix of the rolling averages of the previous derivatives and this iterations derivative so add it to our gradients to be used for our next gradient calculation
            v = gradient
        } else {
            // First set up empty gradient array to store each layers gradient matrix
            var gradient = [Matrix]()
            // On nth iteration get the gradient with momentum for each layer and add it to our gradient matrix
            for l in 0..<D.count {
                // Make our gradient for this layer using our last gradient and the derivative * lr
                let gradient_l = b2 * v[l] + (1 - b2) * D[l].pow(2)
                // Add this layers gradient to the gradient array
                gradient.append(gradient_l)
            }
            // At this point gradient[l] is a matrix of the rolling averages of the previous derivatives and this iterations derivative so add it to our gradients to be used for our next gradient calculation
            v = gradient
        }
    }
}

// MARK: NeuralNetworkHyper
public struct NeuralNetworkHyper: Codable {
    var hidden: [NeuralNetworkHidden]
    var epochs: Int
    var batches: Int
    var lr: Double
    var lm: Double
    var activH: NeuralNetworkActivation
    var activO: NeuralNetworkActivation
    var loss: NeuralNetworkLoss
}

// MARK: NeuralNetworkHiddenLayer
public struct NeuralNetworkHidden: Codable {
    let units: Int
}

// MARK: NeuralNetworkInfo
public struct NeuralNetworkInfo: Codable {
    let data: [[Double]]
    let labels: [[Double]]
    let hyper: NeuralNetworkHyper
    let THETA: [Matrix]
}

// MARK: NeuralNetworkActivation
public enum NeuralNetworkActivation: Int, Codable {
    case Sigmoid
    case ReLU
    case LeakyReLU
}

// MARK: NeuralNetworkLoss
public enum NeuralNetworkLoss: Int, Codable {
    case CrossEntropy
    case MSE
}
