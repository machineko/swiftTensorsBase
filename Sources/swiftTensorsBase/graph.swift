import Foundation

#if arch(x86_64)
    public typealias Float16 = UInt16
#endif

public protocol graphDtypes: Sendable & Numeric {}

extension Float32: graphDtypes {}
extension Float16: graphDtypes {}
extension Int64: graphDtypes {}
extension Int32: graphDtypes {}
extension Int16: graphDtypes {}
extension Int8: graphDtypes {}
extension UInt64: graphDtypes {}
extension UInt32: graphDtypes {}
#if !arch(x86_64)
    extension UInt16: graphDtypes {}
#endif

extension UInt8: graphDtypes {}

extension graphDtypes {
    public var stride: Int {
        MemoryLayout.stride(ofValue: self)
    }

    public var align: Int {
        MemoryLayout.alignment(ofValue: self)
    }
}

public enum padStyle: Sendable, Codable {
    case explicit, valid, same
}

public enum padMode: Sendable, Codable {
    case zero, reflect, symmetric, clamp, const
}

public enum convDataLayout: Sendable, Codable  {
    case NCHW, NHWC
}

public enum convWeightLayout: Sendable, Codable  {
    case OIHW, HWIO
}

public enum ExecutionError: Sendable, Codable, Error {
    case invalidOperation
    case memoryAllocationFailed
    case invalidShape
    case parameterNotFound
}

enum GraphError: Sendable, Codable, Error {
    case cycleDetected
}

public enum dataType: Sendable, Codable {
    case float16, float32, bfloat16, float8, float64
    case int8, int16, int32, int64
    case uint8, uint16, uint32, uint64
}

public enum computeType: Sendable, Codable  {
    case metal, cpu, accelerate, cuda, mpsGraph, cudnn
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

public final class Node: Equatable, Hashable, Sendable {
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

extension Node {
    public convenience init(op: graphOp, inputs: [Node]) {
        self.init(op: op, inputs: inputs, outputs: [])
    }

    public convenience init(op: graphOp) {
        self.init(op: op, inputs: [], outputs: [])
    }
}

public enum graphOp: Sendable {
    case placeholder(_ name: String, _ shape: [Int], dataType: dataType)
    case constant(_ name: String, _ shape: [Int], dataType: dataType)
    case constantScalar(_ value: Float, shape: [Int] = [1], dataType: dataType)
    case variable(_ name: String, _ shape: [Int], dataType: dataType)
    case matmul
    case conv2d(Conv2DParams)
    case rsqrt, sqrt
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
    case add, subtract, mul, division
    case to(_ dataType: dataType)
    case matMul
    case clamp(_ min: Node, _ max: Node)
    case mean(_ dim: Int)
    case sum(_ dim: Int)
    case sliceDim(_ dim: Int, upTo: Node)
    case interpolateNearest(scaleFactor: Int, dataLayout: convDataLayout)
    case pixelShuffle(_ scale: Int, dataLayout: convDataLayout = .NCHW)
    case pixelUnshuffle(_ scale: Int, dataLayout: convDataLayout = .NCHW)
    case multiHeadAttention(MultiHeadAttentionParams)
    case gather(dim: Int)
    case argMax(dim: Int)
    case power(_ exponent: Float)
    case squeeze(_ dim: Int)
    case constPad(_ padding: [(Int, Int)], _ value: Float)
    case log
    case reduceMaximum(_ dim: Int)
    case quantize(scale: Float, zeroPoint: Float, targetType: dataType)
    case dequantize(scale: Float, zeroPoint: Float, targetType: dataType)
    case dynamicQuantize(targetType: dataType)
    case quantizePerChannel(scales: [Float], zeroPoints: [Float], axis: Int, targetType: dataType)
    case dequantizePerChannel(scales: [Float], zeroPoints: [Float], axis: Int, targetType: dataType)
    case conv2dEncrypted(Conv2DParams, encryptionAlgorithm: encryptionAlgorithm)
//    case linear(weights: Node, bias: Node)
//    case linearLora(weights: Node, bias: Node, α: Node, r: Node)

}

public struct Conv2DParams: Sendable {
    public let inChannels: Int
    public let outChannels: Int
    public let kernelSize: (Int, Int)
    public let stride: (Int, Int)
    public let padding: (Int, Int)
    public let padStyle: padStyle
    public let dilation: (Int, Int)
    public let groups: Int
    public let useBias: Bool
    public let dataLayout: convDataLayout
    public let dataType: dataType
    public let name: String
//    public let quantParams: (any QuantizationStats)? = nil
    public let encryptionParams: EncryptionInfo? = nil
    public var weightName: String { "\(name).weight" }
    public var biasName: String { "\(name).bias" }
}

public struct MultiHeadAttentionParams: Sendable, Codable {
    public let numHeads: Int
    public let headDim: Int
    public let hiddenSize: Int
    public let dataType: dataType
    public let scalingFactor: Float?
    public let dropoutProb: Float?
    public let useAttentionMask: Bool

    public init(
        numHeads: Int, headDim: Int, hiddenSize: Int, dataType: dataType, scalingFactor: Float? = nil, dropoutProb: Float? = nil,
        useAttentionMask: Bool = false
    ) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.dataType = dataType
        self.scalingFactor = scalingFactor
        self.dropoutProb = dropoutProb
        self.useAttentionMask = useAttentionMask
    }
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

    public static func / (lhs: Node, rhs: Node) -> Node {
        return .init(op: .division, inputs: [lhs, rhs])
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

    public static func constantScalar(_ value: Float, shape: [Int], _ dataType: dataType) -> Node {
        return Node(op: .constantScalar(value, shape: shape, dataType: dataType))
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
            padStyle: .explicit,
            dilation: dilation,
            groups: groups,
            useBias: useBias,
            dataLayout: dataLayout,
            dataType: dataType,
            name: fullName
        )
        return Node(op: .conv2d(params), inputs: [input])
    }
    
    public static func conv2dEncrypted(
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
        mode: encryptionAlgorithm, key: Node? = nil,
        a: Node? = nil, b: Node? = nil, c: Node? = nil,
        scopeManager: ScopeManager
    ) -> Node {
        let fullName = scopeManager.fullPath(for: name ?? "conv")
        let params = Conv2DParams(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            padStyle: .explicit,
            dilation: dilation,
            groups: groups,
            useBias: useBias,
            dataLayout: dataLayout,
            dataType: dataType,
            name: fullName
        )
        switch mode {
        case .offset:
            guard let key = key else {
                fatalError("Node key need to be provided for encryption with offset")
            }
            return Node(op: .conv2dEncrypted(params, encryptionAlgorithm: mode), inputs: [input, key])

        case .poly:
            guard let a = a,
                  let b = b,
                  let c = c
            else {
                fatalError("Node a,b,c need to get inside poly encryption")
            }
            return Node(op: .conv2dEncrypted(params, encryptionAlgorithm: mode), inputs: [input, a,b,c])

        }
    }
    
    

    public static func conv2d(
        input: Node,
        inChannels: Int,
        outChannels: Int,
        kernelSize: (Int, Int) = (1, 1),
        stride: (Int, Int) = (1, 1),
        dilation: (Int, Int) = (1, 1),
        groups: Int = 1,
        padStyle: padStyle,
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
            padding: (0, 0),
            padStyle: padStyle,
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

    public func GELU() -> Node {
        return Node.init(op: .gelu, inputs: [self])
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

    public func expandDim(_ dim: Int) -> Node {
        return Node(op: .expandDim(dim), inputs: [self])
    }

    public func clamp(_ min: Node, _ max: Node) -> Node {
        return Node(op: .clamp(min, max), inputs: [self, min, max])
    }

    public func mean(_ dim: Int) -> Node {
        return Node(op: .mean(dim), inputs: [self])
    }

    public func rsqrt() -> Node {
        return Node(op: .rsqrt, inputs: [self])
    }

    public func sqrt() -> Node {
        return Node(op: .sqrt, inputs: [self])
    }

    public func sliceDim(dim: Int, length: Node) -> Node {
        return Node(op: .sliceDim(dim, upTo: length), inputs: [self, length])
    }

    public func gather(indexTensor: Node, dim: Int) -> Node {
        return Node(op: .gather(dim: dim), inputs: [self, indexTensor])
    }

    public func to(_ dataType: dataType) -> Node {
        return Node(op: .to(dataType), inputs: [self])
    }

    public func argMax(dim: Int) -> Node {
        return Node(op: .argMax(dim: dim), inputs: [self])
    }

    public func sum(dim: Int) -> Node {
        return Node(op: .sum(dim), inputs: [self])
    }

    public func power(_ exponent: Float) -> Node {
        return Node(op: .power(exponent), inputs: [self])
    }

    public func squeeze(dim: Int) -> Node {
        return Node(op: .squeeze(dim), inputs: [self])
    }

    public func zeroPad(_ padding: [(Int, Int)]) -> Node {
        return Node(op: .constPad(padding, 0.0), inputs: [self])
    }

    public func log() -> Node {
        return Node(op: .log, inputs: [self])
    }

    public func reduceMaximum(dim: Int) -> Node {
        return Node(op: .reduceMaximum(dim), inputs: [self])
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
    public var quantParameters: [String: QuantizationStats] = [:]
    public var encryptionParameters: [String: EncryptionInfo] = [:]

    public init() {}

    public var parameterNames: [String] {
        return Array(parameters.keys) + Array(pendingParameters)
    }

    public mutating func registerParameter(name: String) {
        pendingParameters.insert(name)
    }

    public mutating func updateParameterWithQuant(name: String, tensor: Tensor<T>, quantStats: some QuantizationStats) {
        parameters[name] = tensor
        quantParameters[name] = quantStats
        pendingParameters.remove(name)
    }
    
    public mutating func updateEncryption(name: String, encryption: EncryptionInfo) {
        encryptionParameters[name] = encryption
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
        let orderedNodes = self.generateTopologicalOrder()

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
            case .conv2dEncrypted(let params, _):
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

    public var weightShapeOHWI: [Int] {
        let inChannelsPerGroup = inChannels / groups
        return [outChannels, kernelSize.0, kernelSize.1, inChannelsPerGroup]
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
