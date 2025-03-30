import Foundation

extension TensorDataType {
    public var dataType: dataType {
        switch self {
        case .float32: return .float32
        case .float16: return .float16
        case .float64: fatalError("f64 not supported")
        case .int64: return .int64
        case .int32: return .int32
        case .int16: return .int16
        case .int8: return .int8
        case .uint8: return .uint8
        case .bool: fatalError("bool not supported")
        }
    }
}

public class ScopeManager {
    private var scopes: [String] = []

    private var sequentialCounters: [String: Int] = [:]

    private var registeredParameters: Set<String> = []

    private let strictNameChecking: Bool

    public init(strictNameChecking: Bool = true) {
        self.strictNameChecking = strictNameChecking
    }

    public func currentPath() -> String {
        return scopes.filter { !$0.isEmpty }.joined(separator: ".")
    }

    public func fullPath(for name: String, registerParam: Bool = true) -> String {
        guard !name.isEmpty else {
            return currentPath()
        }

        let currentScopePath = currentPath()
        let fullName = currentScopePath.isEmpty ? name : "\(currentScopePath).\(name)"

        if registerParam {
            registerParameter(fullName)
        }

        return fullName
    }

    public func registerParameter(_ name: String) {
        if registeredParameters.contains(name) {
            let errorMessage = "Parameter '\(name)' is being redefined!"

            if strictNameChecking {
                fatalError(errorMessage)
            } else {
                print("⚠️ WARNING: \(errorMessage)")
            }
        } else {
            registeredParameters.insert(name)
        }
    }

    public func nextSequentialIndex(for containerPath: String? = nil) -> Int {
        let path = containerPath ?? currentPath()
        let currentIndex = sequentialCounters[path, default: 0]
        sequentialCounters[path] = currentIndex + 1
        return currentIndex
    }

    public func push(_ scope: String) {
        if !scope.isEmpty {
            scopes.append(scope)
        }
    }

    public func pop() {
        if !scopes.isEmpty {
            scopes.removeLast()
        }
    }

    public var parameters: Set<String> {
        return registeredParameters
    }

    public func reset() {
        scopes.removeAll()
        sequentialCounters.removeAll()
        registeredParameters.removeAll()
    }
}

public func withScope<T>(_ scope: String, _ manager: ScopeManager, _ body: () -> T) -> T {
    manager.push(scope)
    let result = body()
    manager.pop()
    return result
}
