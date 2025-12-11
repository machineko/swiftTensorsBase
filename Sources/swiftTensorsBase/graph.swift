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
    case NCHW, NHWC, HWC, CHW
}

public enum convWeightLayout: Sendable, Codable  {
    case OIHW, HWIO
}

public enum resizeMode: Sendable, Codable {
    case nearest, bilinear
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
        self.shape = node.shape
        self.backend = backend
        self.dataType = node.dataType
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
    
    private let _cachedShape: [Int]
    private let _cachedDataType: dataType

    init(op: graphOp, inputs: [Node] = [], outputs: [Node] = []) {
        let id = UUID()
        self.id = id
        self.op = op
        self.inputs = inputs
        self._cachedShape = computeShapeForInit(op: op, inputs: inputs)
        self._cachedDataType = computeDataTypeForInit(op: op, inputs: inputs)
    }
}

public enum graphOp: Sendable {
    case placeholder(_ name: String, _ shape: [Int], dataType: dataType)
    case constant(_ name: String, _ shape: [Int], dataType: dataType)
    case constantScalar(_ value: Float, shape: [Int] = [1], dataType: dataType)
    case variable(_ name: String, _ shape: [Int], dataType: dataType)
    case matmul
    case sin, cos
    case rsqrt, sqrt
    case relu, tanh, gelu, sigmoid, silu, tan
    case leakyRelu(_ slope: Float)
    case softmax(_ dim: Int)
    case split(_ numSplits: Int, dim: Int)
    case transpose(_ dim: Int, _ with: Int)
    case permute(_ dims: [Int])
    case tile(_ dims: [Int]), repeatTensor(_ times: Int, _ dim: Int)
    case reshape(shape: [Int]), reshapeWith(withShape: Node)
    case expandDim(_ dim: Int)
    case cat(_ dim: Int)
    case catWith(_ with: Node, dim: Int)
    case splitOutput(parentNode: UUID, index: Int)
    case add, subtract, mul, division
    case greater(_ lhs: Node, _ rhs: Node), greaterEqual(_ lhs: Node, _ rhs: Node), less(_ lhs: Node, _ rhs: Node), lessEqual(_ lhs: Node, _ rhs: Node)
    case to(_ dataType: dataType)
    case matMul
    case clamp(_ min: Node, _ max: Node)
    case mean(_ dim: Int)
    case sum(_ dim: Int)
    case arange(start: Float, end: Float, step: Float, dataType: dataType), linspace(start: Float, end: Float, steps: Int, dataType: dataType)
    case sliceDim(_ dim: Int, upTo: Node)
    case sliceStatic(from: [Int], upTo: [Int], stride: [Int])
    case sliceStaticDim(_ dim: Int, start: Int, upTo: Int)
    case interpolateNearest(scaleFactor: Float, dataLayout: convDataLayout, alignCorners: Bool), interpolateBilinear(scaleFactor: Float, dataLayout: convDataLayout, alignCorners: Bool)
    case resize(outShape: (Int, Int), dataLayout: convDataLayout, alignCorners: Bool, mode: resizeMode)
    case pixelShuffle(_ scale: Int, dataLayout: convDataLayout = .NCHW)
    case pixelUnshuffle(_ scale: Int, dataLayout: convDataLayout = .NCHW)
    case tril(diagonal: Int, value: Float = 0.0), triu(diagonal: Int, value: Float = 0.0)
    case scaledDotProductAttention(query: Node, key: Node, value: Node, attnMask: Node?, params: attentionParams)
    case gather(dim: Int)
    case argMax(dim: Int)
    case power(_ exponent: Float)
    case squeeze(_ dim: Int)
    case constPad(_ padding: [(Int, Int)], _ value: Float)
    case log, exp, exp2
    case reduceMaximum(_ dim: Int)
    case shapeOf(of: Node)
    case brodcast(_ shape: [Int])
    case groupNorm2d(groups: Int, channels: Int, eps: Float, weights: Node? = nil, bias: Node? = nil, affine: Bool = true, dataLayout: convDataLayout = .NCHW)
//    case normalize(mean: Node, std: Node, variance: Node, gamma: Node?, beta: Node?, eps: Float)
    case randomUniform(shape: [Int], seed: Int, dataType: dataType), randomNormal(shape: [Int], mean: Float?, std: Float?, seed: Int, dataType: dataType)
    case conv2d(Conv2DParams), conv2dTranspose(Conv2DParams)
    case quantize(scale: Float, zeroPoint: Float, targetType: dataType)
    case dequantize(scale: Float, zeroPoint: Float, targetType: dataType)
    case dynamicQuantize(targetType: dataType)
    case quantizePerChannel(scales: [Float], zeroPoints: [Float], axis: Int, targetType: dataType)
    case dequantizePerChannel(scales: [Float], zeroPoints: [Float], axis: Int, targetType: dataType)
    case degree2radians
    case conv2dEncrypted(Conv2DParams, encryptionAlgorithm: encryptionAlgorithm)
    case linear(weights: Node, bias: Node? = nil)
    case linearLora(weights: Node, bias: Node, loraA: Node, loraB: Node, loraAlpha: Float, rank: Int)

}

public struct Conv2DParams: Sendable {
    public var inChannels: Int
    public var outChannels: Int
    public var kernelSize: (Int, Int)
    public var stride: (Int, Int)
    public var padding: (Int, Int, Int, Int)  // (left, right, top, bottom) - PyTorch convention
    public var padStyle: padStyle
    public var dilation: (Int, Int)
    public var groups: Int
    public var useBias: Bool
    public var dataLayout: convDataLayout
    public var dataType: dataType
    public var name: String
//    public let quantParams: (any QuantizationStats)? = nil
    public var encryptionParams: EncryptionInfo? = nil
    public var weightName: String { "\(name).weight" }
    public var biasName: String { "\(name).bias" }
    
    // Main initializer with explicit 4-value padding
    public init(inChannels: Int, outChannels: Int, kernelSize: (Int, Int), stride: (Int, Int), padding: (Int, Int, Int, Int), padStyle: padStyle, dilation: (Int, Int), groups: Int, useBias: Bool, dataLayout: convDataLayout, dataType: dataType, name: String, encryptionParams: EncryptionInfo? = nil) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.padStyle = padStyle
        self.dilation = dilation
        self.groups = groups
        self.useBias = useBias
        self.dataLayout = dataLayout
        self.dataType = dataType
        self.name = name
        self.encryptionParams = encryptionParams
    }
}

public extension Conv2DParams {
    // Convenience initializer with symmetric 2-value padding (expands to 4-value)
    init(inChannels: Int, outChannels: Int, kernelSize: (Int, Int), stride: (Int, Int), padding: (Int, Int), padStyle: padStyle = .explicit, dilation: (Int, Int) = (1, 1), groups: Int = 1, useBias: Bool = true, dataLayout: convDataLayout = .NCHW, dataType: dataType, name: String, encryptionParams: EncryptionInfo? = nil) {
        self.init(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: (padding.1, padding.1, padding.0, padding.0),  // Expand (pad_h, pad_w) -> (left, right, top, bottom)
            padStyle: padStyle,
            dilation: dilation,
            groups: groups,
            useBias: useBias,
            dataLayout: dataLayout,
            dataType: dataType,
            name: name,
            encryptionParams: encryptionParams
        )
    }
    
    init(inChannels: Int, outChannels: Int, kernelSize: (Int, Int), dilation: (Int, Int) = (1, 1), groups: Int = 1, padStyle: padStyle, useBias: Bool = true, dataLayout: convDataLayout = .NCHW, dataType: dataType, name: String, encryptionParams: EncryptionInfo? = nil) {
        self.init(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: (1, 1),
            padding: (0,0,0,0),
            padStyle: padStyle,
            dilation: dilation,
            groups: groups,
            useBias: useBias,
            dataLayout: dataLayout,
            dataType: dataType,
            name: name,
            encryptionParams: encryptionParams
        )
    }
}

public struct attentionParams: Sendable, Codable {
    public let numHeads: Int
    public let headDim: Int
    public let scalingFactor: Float?
    public let dropoutProb: Float?
    public let isCausal: Bool

    public init(
        numHeads: Int, headDim: Int, scalingFactor: Float? = nil, dropoutProb: Float? = nil,
        isCausal: Bool = false
    ) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scalingFactor = scalingFactor
        self.dropoutProb = dropoutProb
        self.isCausal = isCausal
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

    public var shape: [Int] {
        return _cachedShape
    }

    public var dataType: dataType {
        return _cachedDataType
    }
}


public extension Node {
    
    static func arange(_ start: Float, _ end: Float, step: Float = 1, dataType: dataType = .float32) -> Node {
        return .init(op: .arange(start: start, end: end, step: step, dataType: dataType))
    }
    
    static func linspace(_ start: Float, _ end: Float, steps: Int, dataType: dataType = .float32) -> Node {
        return .init(op: .linspace(start: start, end: end, steps: steps, dataType: dataType))
    }
    
    static func + (lhs: Node, rhs: Node) -> Node {
        return .init(op: .add, inputs: [lhs, rhs])
    }

    static func - (lhs: Node, rhs: Node) -> Node {
        return .init(op: .subtract, inputs: [lhs, rhs])
    }

    static func * (lhs: Node, rhs: Node) -> Node {
        return .init(op: .mul, inputs: [lhs, rhs])
    }

    static func / (lhs: Node, rhs: Node) -> Node {
        return .init(op: .division, inputs: [lhs, rhs])
    }

    static func > (lhs: Node, rhs: Node) -> Node {
        return .init(op: .greater(lhs, rhs), inputs: [lhs, rhs])
    }
    
    static func >= (lhs: Node, rhs: Node) -> Node {
        return .init(op: .greaterEqual(lhs, rhs), inputs: [lhs, rhs])
    }
    
    static func < (lhs: Node, rhs: Node) -> Node {
        return .init(op: .less(lhs, rhs), inputs: [lhs, rhs])
    }
    
    static func <= (lhs: Node, rhs: Node) -> Node {
        return .init(op: .lessEqual(lhs, rhs), inputs: [lhs, rhs])
    }
    
    
    static func relu(_ input: Node) -> Node {
        return .init(op: .relu, inputs: [input])
    }

    func relu() -> Node {
        return .init(op: .relu, inputs: [self])
    }
    
    static func tan(_ input: Node) -> Node {
        return .init(op: .tan, inputs: [input])
    }

    func tan() -> Node {
        return .init(op: .tan, inputs: [self])
    }
    
    func SiLU() -> Node {
        return .init(op: .silu, inputs: [self])
    }

    static func placeholder(_ name: String, shape: [Int], _ dataType: dataType, _ scopeManager: ScopeManager) -> Node {
        let fullName = scopeManager.fullPath(for: name, registerParam: true)
        return Node(op: .placeholder(fullName, shape, dataType: dataType))
    }

    static func variable(_ name: String, shape: [Int], _ dataType: dataType, _ scopeManager: ScopeManager) -> Node {
        let fullName = scopeManager.fullPath(for: name)
        return Node(op: .variable(fullName, shape, dataType: dataType))
    }

    static func constant(_ name: String, shape: [Int], _ dataType: dataType, _ scopeManager: ScopeManager) -> Node {
        let fullName = scopeManager.fullPath(for: name)
        return Node(op: .constant(fullName, shape, dataType: dataType))
    }
    
    static func constant(fullName: String, shape: [Int], _ dataType: dataType) -> Node {
//        let fullName = scopeManager.fullPath(for: name)
        return Node(op: .constant(fullName, shape, dataType: dataType))
    }
    
    static func constantScalar(_ value: Float, shape: [Int], _ dataType: dataType) -> Node {
        return Node(op: .constantScalar(value, shape: shape, dataType: dataType))
    }

    static func conv2d(
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
    
    static func conv2dTranspose(
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
        return Node(op: .conv2dTranspose(params), inputs: [input])
    }
    
    static func conv2dEncrypted(
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
    
    

    static func conv2d(
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
    
    static func conv2dTranspose(
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
        return Node(op: .conv2dTranspose(params), inputs: [input])
    }

    static func transpose(input: Node, dim: Int, with: Int) -> Node {
        return Node(op: .transpose(dim, with), inputs: [input])
    }

    static func split(
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

    static func cat(_ tensors: [Node], dim: Int) -> Node {
        return Node(op: .cat(dim), inputs: tensors)
    }
    
    static func randomUniform(_ shape: [Int], seed: Int = 0, dataType: dataType = .float32) -> Node {
        return Node(op: .randomUniform(shape: shape, seed: seed, dataType: dataType))
    }
    
    static func randomNormal(_ shape: [Int], mean: Float? = nil, std: Float? = nil, seed: Int = 0, dataType: dataType = .float32) -> Node {
        return Node(op: .randomNormal(shape: shape, mean: mean, std: std, seed: seed, dataType: dataType))
    }
    
    static func randomNormal(_ shape: [Int], seed: Int = 0, dataType: dataType = .float32) -> Node {
        return Node(op: .randomNormal(shape: shape, mean: nil, std: nil, seed: seed, dataType: dataType))
    }

}
public extension Node {
    func linear(weights: Node, bias: Node?) -> Node {
        var inputs = [self, weights]
        if let bias = bias {
            inputs.append(bias)
        }
        return Node(op: .linear(weights: weights, bias: bias), inputs: inputs)
    }
    
    func scaledDotProductAttention(query: Node, key: Node, value: Node, attnMask: Node?, params: attentionParams) -> Node {
        var inputs = [self, query, key, value]
        if let attnMask = attnMask {
            inputs.append(attnMask)
        }
        return Node(op: .scaledDotProductAttention(query: query, key: key, value: value, attnMask: attnMask, params: params), inputs: inputs)
    }
}



public extension Node {
    func transpose(dim: Int, with: Int) -> Node {
        return Node(op: .transpose(dim, with), inputs: [self])
    }
    

    func permute(dims: [Int]) -> Node {
        return Node(op: .permute(dims), inputs: [self])
    }
    
    func reshape(_ shape: [Int]) -> Node {
        return Node(op: .reshape(shape: shape), inputs: [self])
    }
    
    func reshapeWith(withShape: Node) -> Node {
        return Node(op: .reshapeWith(withShape: withShape), inputs: [self, withShape])
    }

    func matMul(_ with: Node) -> Node {
        return Node(op: .matMul, inputs: [self, with])
    }
    
    func tril(_ diagonal: Int = 0, value: Float = 0) -> Node {
        return Node(op: .tril(diagonal: diagonal, value: value), inputs: [self])
    }
    
    func triu(_ diagonal: Int = 0, value: Float = 0) -> Node {
        return Node(op: .triu(diagonal: diagonal, value: value), inputs: [self])
    }
    
    func tile(_ dims: [Int]) -> Node {
        return Node(op: .tile(dims), inputs: [self])
    }
    
    func repeatTensor(_ times: Int, dim: Int) -> Node {
        return Node(op: .repeatTensor(times, dim), inputs: [self])

    }
    
    func brodcast(_ shape: [Int]) -> Node {
        return .init(op: .brodcast(shape), inputs: [self])
    }

    func split(
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

    func softMax(dim: Int) -> Node {
        return Node.init(op: .softmax(dim), inputs: [self])
    }

    func leakyReLU(_ slope: Float) -> Node {
        return Node.init(op: .leakyRelu(slope), inputs: [self])
    }

    func GELU() -> Node {
        return Node.init(op: .gelu, inputs: [self])
    }

    func pixelUnshuffle(_ scale: Int, dataLayout: convDataLayout) -> Node {
        return Node.init(op: .pixelUnshuffle(scale, dataLayout: dataLayout), inputs: [self])
    }

    func catWith(_ with: Node, dim: Int) -> Node {
        return Node(op: .catWith(with, dim: dim), inputs: [self, with])
    }

    func interpolateNearest(scaleFactor: Float, dataLayout: convDataLayout, alignCorners: Bool = false) -> Node {
        return Node(op: .interpolateNearest(scaleFactor: scaleFactor, dataLayout: dataLayout, alignCorners: alignCorners), inputs: [self])
    }
    
    func resize(_ outShape: (Int, Int), dataLayout: convDataLayout, mode: resizeMode = .nearest, alignCorners: Bool = false) -> Node {
        return Node(op: .resize(outShape: outShape, dataLayout: dataLayout, alignCorners: alignCorners, mode: mode), inputs: [self])
    }

    func interpolateBilinear(scaleFactor: Float, dataLayout: convDataLayout, alignCorners: Bool = false) -> Node {
        return Node(op: .interpolateBilinear(scaleFactor: scaleFactor, dataLayout: dataLayout, alignCorners: alignCorners), inputs: [self])
    }
    
    func expandDim(_ dim: Int) -> Node {
        return Node(op: .expandDim(dim), inputs: [self])
    }

    func clamp(_ min: Node, _ max: Node) -> Node {
        return Node(op: .clamp(min, max), inputs: [self, min, max])
    }

    func mean(_ dim: Int) -> Node {
        return Node(op: .mean(dim), inputs: [self])
    }

    func rsqrt() -> Node {
        return Node(op: .rsqrt, inputs: [self])
    }

    func sqrt() -> Node {
        return Node(op: .sqrt, inputs: [self])
    }

    func sliceDim(dim: Int, length: Node) -> Node {
        return Node(op: .sliceDim(dim, upTo: length), inputs: [self, length])
    }
    func sliceDimStatic(dim: Int, start: Int, upTo: Int) -> Node {
        return Node(op: .sliceStaticDim(dim, start: start, upTo: upTo), inputs: [self])
    }
    
    func sliceStatic(_ from: [Int], upTo: [Int], stride: [Int]) -> Node {
        let fromCount = from.count
        precondition(fromCount == upTo.count && fromCount == stride.count, "Shape of from, upTo and stride need to be the same")
        return Node(op: .sliceStatic(from: from, upTo: upTo, stride: stride), inputs: [self])
    }
    
    func shapeNode() -> Node {
        return Node(op: .shapeOf(of: self), inputs: [self])
    }

    func gather(indexTensor: Node, dim: Int) -> Node {
        return Node(op: .gather(dim: dim), inputs: [self, indexTensor])
    }

    func to(_ dataType: dataType) -> Node {
        return Node(op: .to(dataType), inputs: [self])
    }

    func argMax(dim: Int) -> Node {
        return Node(op: .argMax(dim: dim), inputs: [self])
    }

    func sum(dim: Int) -> Node {
        return Node(op: .sum(dim), inputs: [self])
    }

    func power(_ exponent: Float) -> Node {
        return Node(op: .power(exponent), inputs: [self])
    }

    func squeeze(dim: Int) -> Node {
        return Node(op: .squeeze(dim), inputs: [self])
    }

    func zeroPad(_ padding: [(Int, Int)]) -> Node {
        return Node(op: .constPad(padding, 0.0), inputs: [self])
    }

    func log() -> Node {
        return Node(op: .log, inputs: [self])
    }
    
    func exp() -> Node {
        return Node(op: .exp, inputs: [self])
    }
    
    func exp2() -> Node {
        return Node(op: .exp2, inputs: [self])
    }
    
    func sin() -> Node {
        return Node(op: .sin, inputs: [self])
    }
    
    func cos() -> Node {
        return Node(op: .cos, inputs: [self])
    }
    
    func degree2radians() -> Node {
        return Node(op: .degree2radians, inputs: [self])
    }
    
    func reduceMaximum(dim: Int) -> Node {
        return Node(op: .reduceMaximum(dim), inputs: [self])
    }

}

public extension Node {
    func groupNorm2d(groups: Int, channels: Int, eps: Float, weights: Node? = nil, bias: Node? = nil, affine: Bool, dataLayout: convDataLayout = .NCHW) -> Node {
        var inputs = [self]
        if let weights = weights {
            inputs.append(weights)
        }
        if let bias = bias {
            inputs.append(bias)
        }
        return Node(op: .groupNorm2d(groups: groups, channels: channels, eps: eps, weights: weights, bias: bias, affine: affine, dataLayout: dataLayout), inputs: inputs)
    }
}

public extension Node {
    func generateTopologicalOrder() -> [Node] {
        var visited = Set<UUID>()
        var result = [Node]()
        var stack: [Node] = [self]
        
        // To avoid recursion, we need to simulate the call stack.
        // We can track if we have expanded a node.
        // Or simpler: use two stacks or one stack with state.
        
        // State: 0 = pre-visit (expand children), 1 = post-visit (add to result)
        var workStack: [(Node, Bool)] = [(self, false)] // (node, visitedChildren)
        
        while !workStack.isEmpty {
            let (node, childrenVisited) = workStack.last!
            
            if visited.contains(node.id) {
                workStack.removeLast()
                continue
            }
            
            if childrenVisited {
                workStack.removeLast()
                visited.insert(node.id)
                result.append(node)
            } else {
                // Mark as children visited for next time
                workStack[workStack.count - 1] = (node, true)
                
                // Add inputs
                for input in node.inputs {
                    if !visited.contains(input.id) {
                        workStack.append((input, false))
                    }
                }
            }
        }
        
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

public extension Node {
    func findNodeById(_ id: UUID, in node: Node) -> Node? {
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

    func emptyGraphStateDict<T>() throws -> StateDict<T> {
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
            case .conv2dTranspose(let params):
                stateDict.registerParameter(name: params.weightName)
                if params.useBias {
                    stateDict.registerParameter(name: params.biasName)
                }
            case .linear(let weights, let bias):
                stateDict.registerParameter(name: weights.name!)
                if let bias = bias {
                    stateDict.registerParameter(name: bias.name!)
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
    
    static func emptyGraphStateDict<T>(from nodes: [Node]) throws -> StateDict<T> {
        var stateDict = StateDict<T>()
        
        var allNodes: [Node] = []
        var visitedIds = Set<UUID>()
        
        for node in nodes {
            let orderedNodes = node.generateTopologicalOrder()
            for orderedNode in orderedNodes {
                if !visitedIds.contains(orderedNode.id) {
                    visitedIds.insert(orderedNode.id)
                    allNodes.append(orderedNode)
                }
            }
        }
        
        for node in allNodes {
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
            case .conv2dTranspose(let params):
                stateDict.registerParameter(name: params.weightName)
                if params.useBias {
                    stateDict.registerParameter(name: params.biasName)
                }
            case .linear(let weights, let bias):
                stateDict.registerParameter(name: weights.name!)
                if let bias = bias {
                    stateDict.registerParameter(name: bias.name!)
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
    
    static func emptyGraphStateDict<T>(from nodes: Node...) throws -> StateDict<T> {
        return try emptyGraphStateDict(from: nodes)
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

public extension Conv2DParams {
    /// Format: [outChannels, inChannels/groups, kernelHeight, kernelWidth]
    public var weightShape: [Int] {
        let inChannelsPerGroup = inChannels / groups
        return [outChannels, inChannelsPerGroup, kernelSize.0, kernelSize.1]
    }
    
    public var weightShapeTranspose: [Int] {
        let inChannelsPerGroup = inChannels / groups
        return [inChannelsPerGroup, outChannels, kernelSize.0, kernelSize.1]
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

//extension Node {
//    public var convWeightShape: [Int]? {
//        switch self.op {
//        case .conv2d(let params):
//            return params.weightShape
//        default:
//            return nil
//        }
//    }
//
//    public var convBiasShape: [Int]? {
//        switch self.op {
//        case .conv2d(let params):
//            return params.biasShape
//        default:
//            return nil
//        }
//    }
//}
