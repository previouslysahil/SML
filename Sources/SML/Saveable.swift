//
//  Saveable.swift
//  
//
//  Created by Sahil Srivastava on 8/20/21.
//

import Foundation

public protocol Saveable {
    associatedtype Info: Codable
    
    static var keysTag: String { get }
}
extension Saveable {
    
    public func load(key: String) -> Info? {
        // Get our decoded json data for this key
        guard let info = try? JSONDecoder().decode(Info.self, from: UserDefaults.standard.data(forKey: key) ?? Data()) else {
            return nil
        }
        return info
    }
    
    @discardableResult
    public func save(key: String, info: Info) -> Bool {
        // Attempt to encode
        do {
            // Convert info to json and save
            UserDefaults.standard.setValue(try JSONEncoder().encode(info), forKey: key)
            
            // Save the key as well, first get previously saved keys
            var keys = UserDefaults.standard.array(forKey: Self.keysTag) as? [String] ?? [String]()
            // If this key is already saved, do nothing
            if !keys.contains(key) {
                // This key is not saved so add it and save the keys again
                keys.append(key)
                // Now convert keys to json and save
                UserDefaults.standard.setValue(keys, forKey: Self.keysTag)
            }
            print("Saveable.save: Successfully saved info with key \(key)")
            return true
        } catch {
            // Unable to encode either info or keys, either way confirm we have cleared the k/v pair in UserDefaults
            UserDefaults.standard.removeObject(forKey: key)
            print("Saveable.save: Unable to save info")
            return false
        }
    }
    
    public static func keys() -> [String] {
        // Get the saved keys from UserDefaults
        let keys = UserDefaults.standard.array(forKey: Self.keysTag) as? [String] ?? [String]()
        return keys
    }
    
    @discardableResult
    public static func clear() -> Bool {
        if !Self.keys().isEmpty {
            // Remove stored infos first
            for key in Self.keys() {
                UserDefaults.standard.removeObject(forKey: key)
            }
            // Now remove keys
            UserDefaults.standard.removeObject(forKey: Self.keysTag)
            print("Saveable.clear: Cleared all stored infos")
            return true
        } else {
            print("Saveable.clear: No infos to clear")
            return false
        }
    }
    
    @discardableResult
    public static func clear(key: String) -> Bool {
        // Get keys
        var keys = UserDefaults.standard.array(forKey: Self.keysTag) as? [String] ?? [String]()
        if keys.contains(key) {
            // Remove stored info first
            UserDefaults.standard.removeObject(forKey: key)
            // Remove the key from our keys
            keys.removeAll(where: { $0 == key })
            // Resave the keys
            UserDefaults.standard.setValue(keys, forKey: Self.keysTag)
            print("Saveable.clear: Cleared info for key \(key)")
            return true
        } else {
            print("Saveable.clear: No info for key \(key)")
            return false
        }
    }
}
