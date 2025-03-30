import Foundation

public protocol graphDtypes: Sendable {}

extension graphDtypes {
    public var stride: Int {
        MemoryLayout.stride(ofValue: self)
    }

    public var align: Int {
        MemoryLayout.alignment(ofValue: self)
    }
}

public enum padStyle: Sendable {
    case explicit, valid, same
}

public enum padMode: Sendable {
    case zero, reflect, symmetric, clamp, const
}

public enum convDataLayout: Sendable {
    case NCHW, NHWC
}

public enum convWeightLayout: Sendable {
    case OIHW, HWIO
}

public enum ExecutionError: Error {
    case invalidOperation
    case memoryAllocationFailed
    case invalidShape
    case parameterNotFound
}

enum GraphError: Error {
    case cycleDetected
}

public enum dataType {
    case float16, float32, bfloat16, float8, float64
    case int8, int16, int32, int64
    case uint8, uint16, uint32, uint64
}

public enum computeType {
    case metal, cpu, accelerate, cuda, mpsGraph
}

public protocol TensorType {
    associatedtype tensorStorage
}

public struct Tensor<T: TensorType> {
    public let id: UUID
    public var shape: [Int]
    public let name: String?
    public let backend: computeType
    public let dataType: dataType
    public var storage: T.tensorStorage?

    public init(
        id: UUID = UUID(),
        shape: [Int],
        name: String? = nil,
        backend: computeType,
        dataType: dataType,
        storage: T.tensorStorage? = nil
    ) {
        self.id = id
        self.shape = shape
        self.name = name
        self.backend = backend
        self.dataType = dataType
        self.storage = storage
    }


}

extension Tensor {

    public init(fromPlaceholder node: Node, backend: computeType) {
        precondition(node.isPlaceholder, "Node need to be placehodler \(node.op)")
        self.id = UUID()
        self.name = node.name
        self.shape = node.shape!
        self.backend = backend
        self.dataType = node.dataType!
        self.storage = nil
    }

    public init(placeholderShape: [Int], dataType: dataType, backend: computeType, name: String? = nil) {
        self.id = UUID()
        self.name = name
        self.shape = placeholderShape
        self.backend = backend
        self.dataType = dataType
        self.storage = nil
    }
}

public final class Node: Equatable, Hashable {
    public static func == (lhs: Node, rhs: Node) -> Bool {
        lhs.id == rhs.id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(self.id)
    }

    public let op: graphOp
    public let id: UUID
    public let inputs: [Node]

    init(op: consuming graphOp, inputs: [Node] = [], outputs: [Node] = []) {
        let id = UUID()
        self.id = id
        self.op = op
        self.inputs = inputs
    }
}

public extension Node {
    convenience init(op: graphOp, inputs: [Node]) {
        self.init(op: op, inputs: inputs, outputs: [])
    }

    convenience init(op: graphOp) {
        self.init(op: op, inputs: [], outputs: [])
    }
}

public enum graphOp {
    case placeholder(_ name: String, _ shape: [Int], dataType: dataType)
    case constant(_ name: String, _ shape: [Int], dataType: dataType)
    case constantScalar(_ value: Float, shape: [Int] = [1], dataType: dataType)
    case variable(_ name: String, _ shape: [Int], dataType: dataType)
    case conv2d(Conv2DParams)
    case relu, tanh, gelu, sigmoid
    case leakyRelu(_ slope: Float)
    case softmax(_ dim: Int)
    case split(_ numSplits: Int, dim: Int)
    case transpose(_ dim: Int, _ with: Int)
    case permute(_ dims: [Int])
    case tile(_ dims: [Int])
    case expandDim(_ dim: Int)
    case cat(_ dim: Int)
    case catWith(_ with: Node, dim: Int)
    case splitOutput(parentNode: UUID, index: Int)
    case add, subtract, mul
    case to(_ dataType: dataType)
    case matMul
    case interpolateNearest(scaleFactor: Int, dataLayout: convDataLayout)
    case pixelShuffle(_ scale: Int, dataLayout: convDataLayout = .NCHW)
    case pixelUnshuffle(_ scale: Int, dataLayout: convDataLayout = .NCHW)
    case multiHeadAttention(MultiHeadAttentionParams)

}

public struct Conv2DParams {
    public let inChannels: Int
    public let outChannels: Int
    public let kernelSize: (Int, Int)
    public let stride: (Int, Int)
    public let padding: (Int, Int)
    public let dilation: (Int, Int)
    public let groups: Int
    public let useBias: Bool
    public let dataLayout: convDataLayout
    public let dataType: dataType
    public let name: String
    public var weightName: String { "\(name).weight" }
    public var biasName: String { "\(name).bias" }
}

public struct MultiHeadAttentionParams {
    public let numHeads: Int
    public let headDim: Int
    public let hiddenSize: Int
    public let dataType: dataType
    public let scalingFactor: Float?
    public let dropoutProb: Float?
    public let useAttentionMask: Bool
}

extension Node {
    public var name: String? {
        switch self.op {
        case .placeholder(let name, _, _):
            return name
        case .constant(let name, _, _):
            return name
        case .variable(let name, _, _):
            return name
        case .conv2d(let params):
            return params.name
        default:
            return nil
        }
    }

    public var shape: [Int]? {
        switch self.op {
        case .placeholder(_, let shape, _):
            return shape
        case .constant(_, let shape, _):
            return shape
        case .variable(_, let shape, _):
            return shape
        default:
            return nil
        }
    }

    public var dataType: dataType? {
        switch self.op {
        case .placeholder(_, _, let dataType):
            return dataType
        case .constant(_, _, let dataType):
            return dataType
        case .variable(_, _, let dataType):
            return dataType
        case .conv2d(let params):
            return params.dataType
        default:
            return nil
        }
    }

}
public struct Executor {
}

extension Node {
    public static func + (lhs: Node, rhs: Node) -> Node {
        return .init(op: .add, inputs: [lhs, rhs])
    }

    public static func - (lhs: Node, rhs: Node) -> Node {
        return .init(op: .subtract, inputs: [lhs, rhs])
    }

    public static func * (lhs: Node, rhs: Node) -> Node {
        return .init(op: .mul, inputs: [lhs, rhs])
    }

    public static func relu(_ input: Node) -> Node {
        return .init(op: .relu, inputs: [input])
    }

    public func relu() -> Node {
        return .init(op: .relu, inputs: [self])
    }

    public static func placeholder(_ name: String, shape: [Int], _ dataType: dataType, _ scopeManager: ScopeManager) -> Node {
        let fullName = scopeManager.fullPath(for: name, registerParam: true)
        return Node(op: .placeholder(fullName, shape, dataType: dataType))
    }

    public static func variable(_ name: String, shape: [Int], _ dataType: dataType, _ scopeManager: ScopeManager) -> Node {
        let fullName = scopeManager.fullPath(for: name)
        return Node(op: .variable(fullName, shape, dataType: dataType))
    }

    public static func constant(_ name: String, shape: [Int], _ dataType: dataType, _ scopeManager: ScopeManager) -> Node {
        let fullName = scopeManager.fullPath(for: name)
        return Node(op: .constant(fullName, shape, dataType: dataType))
    }

    public static func conv2d(
        input: Node,
        inChannels: Int,
        outChannels: Int,
        kernelSize: (Int, Int) = (1, 1),
        stride: (Int, Int) = (1, 1),
        padding: (Int, Int) = (0, 0),
        dilation: (Int, Int) = (1, 1),
        groups: Int = 1,
        dataLayout: convDataLayout = .NCHW,
        dataType: dataType = .float32,
        name: String? = nil,
        useBias: Bool = true,
        scopeManager: ScopeManager
    ) -> Node {
        let fullName = scopeManager.fullPath(for: name ?? "conv")
        let params = Conv2DParams(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups,
            useBias: useBias,
            dataLayout: dataLayout,
            dataType: dataType,
            name: fullName
        )
        return Node(op: .conv2d(params), inputs: [input])
    }

    public static func transpose(input: Node, dim: Int, with: Int) -> Node {
        return Node(op: .transpose(dim, with), inputs: [input])
    }

    public static func split(
        _ input: Node,
        numSplits: Int,
        axis: Int
    ) -> [Node] {
        let splitNode = Node(op: .split(numSplits, dim: axis), inputs: [input])

        var outputNodes = [Node]()
        for i in 0..<numSplits {
            let outputNode = Node(op: .splitOutput(parentNode: splitNode.id, index: i), inputs: [splitNode])
            outputNodes.append(outputNode)
        }

        return outputNodes
    }

    public static func cat(_ tensors: [Node], dim: Int) -> Node {
        return Node(op: .cat(dim), inputs: tensors)
    }

}

extension Node {
    public func transpose(dim: Int, with: Int) -> Node {
        return Node(op: .transpose(dim, with), inputs: [self])
    }

    public func permute(dims: [Int]) -> Node {
        return Node(op: .permute(dims), inputs: [self])
    }

    public func matMul(_ with: Node) -> Node {
        return Node(op: .matMul, inputs: [self, with])
    }

    public func split(
        numSplits: Int,
        axis: Int
    ) -> [Node] {
        let splitNode = Node(op: .split(numSplits, dim: axis), inputs: [self])

        var outputNodes = [Node]()
        for i in 0..<numSplits {
            let outputNode = Node(op: .splitOutput(parentNode: splitNode.id, index: i), inputs: [splitNode])
            outputNodes.append(outputNode)
        }

        return outputNodes
    }

    public func softMax(dim: Int) -> Node {
        return Node.init(op: .softmax(dim), inputs: [self])
    }

    public func leakyReLU(_ slope: Float) -> Node {
        return Node.init(op: .leakyRelu(slope), inputs: [self])
    }

    public func pixelUnshuffle(_ scale: Int, dataLayout: convDataLayout) -> Node {
        return Node.init(op: .pixelUnshuffle(scale, dataLayout: dataLayout), inputs: [self])
    }

    public func catWith(_ with: Node, dim: Int) -> Node {
        return Node(op: .catWith(with, dim: dim), inputs: [self, with])
    }

    public func interpolateNearest(scaleFactor: Int, dataLayout: convDataLayout) -> Node {
        return Node(op: .interpolateNearest(scaleFactor: scaleFactor, dataLayout: dataLayout), inputs: [self])
    }
}

extension Node {
    public func generateTopologicalOrder() -> [Node] {
        var visited = Set<UUID>()
        var result = [Node]()

        func visit(_ node: Node) {
            if visited.contains(node.id) {
                return
            }

            for input in node.inputs {
                visit(input)
            }

            visited.insert(node.id)
            result.append(node)
        }

        visit(self)
        return result
    }
}

public struct StateDict<T: TensorType> {
    public var pendingParameters: Set<String> = []
    public var parameters: [String: Tensor<T>] = [:]

    public init() {}

    public var parameterNames: [String] {
        return Array(parameters.keys) + Array(pendingParameters)
    }

    public mutating func registerParameter(name: String) {
        pendingParameters.insert(name)
    }

    public mutating func updateParameter(name: String, tensor: Tensor<T>) {
        parameters[name] = tensor
        pendingParameters.remove(name)
    }

    public func hasParameter(name: String) -> Bool {
        return parameters.keys.contains(name) || pendingParameters.contains(name)
    }
}

extension Node {
    public func findNodeById(_ id: UUID, in node: Node) -> Node? {
        if node.id == id {
            return node
        }

        for input in node.inputs {
            if let found = findNodeById(id, in: input) {
                return found
            }
        }
        return nil
    }

    public func emptyGraphStateDict<T>() throws -> StateDict<T> {
        var stateDict = StateDict<T>()
        let orderedNodes = try self.generateTopologicalOrder()

        for node in orderedNodes {
            switch node.op {
            case .variable(let name, _, _):
                stateDict.registerParameter(name: name)
            case .constant(let name, _, _):
                stateDict.registerParameter(name: name)
            case .conv2d(let params):
                stateDict.registerParameter(name: params.weightName)
                if params.useBias {
                    stateDict.registerParameter(name: params.biasName)
                }
            default:
                continue
            }
        }
        return stateDict
    }
}

extension Node {
    public var isPlaceholder: Bool {
        if case .placeholder = self.op {
            return true
        }
        return false
    }
}

extension Conv2DParams {
    /// Format: [outChannels, inChannels/groups, kernelHeight, kernelWidth]
    public var weightShape: [Int] {
        let inChannelsPerGroup = inChannels / groups
        return [outChannels, inChannelsPerGroup, kernelSize.0, kernelSize.1]
    }

    /// Format: [outChannels]
    public var biasShape: [Int]? {
        if useBias {
            return self.dataLayout == .NCHW ? [1, outChannels, 1, 1] : [1, 1, 1, outChannels]
        }
        return nil
    }
}

extension Node {
    public var convWeightShape: [Int]? {
        switch self.op {
        case .conv2d(let params):
            return params.weightShape
        default:
            return nil
        }
    }

    public var convBiasShape: [Int]? {
        switch self.op {
        case .conv2d(let params):
            return params.biasShape
        default:
            return nil
        }
    }
}
