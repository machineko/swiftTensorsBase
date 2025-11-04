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
    
    // Cached shape - computed once during initialization
    private let _cachedShape: [Int]

    init(op: graphOp, inputs: [Node] = [], outputs: [Node] = []) {
        let id = UUID()
        self.id = id
        self.op = op
        self.inputs = inputs
        // Compute shape once using a helper that doesn't need self
        self._cachedShape = computeShapeForInit(op: op, inputs: inputs)
    }
}

// Free function to compute shape during Node initialization
// This allows shape caching without modifying existing inferShape methods
fileprivate func computeShapeForInit(op: graphOp, inputs: [Node]) -> [Int] {
    // Create a minimal temporary wrapper to reuse existing shape inference
    struct TempNodeWrapper {
        let op: graphOp
        let inputs: [Node]
    }
    let temp = TempNodeWrapper(op: op, inputs: inputs)
    
    // Inline the inferShape logic here - this is the key to caching
    // We duplicate the switch statement but it's computed only once per node
    switch temp.op {
    case .placeholder(_, let shape, _):
        return shape
    case .constant(_, let shape, _):
        return shape
    case .variable(_, let shape, _):
        return shape
    case .constantScalar(_, let shape, _):
        return shape
    case .arange(let start, let end, let step, _):
        let count = (end - start + step - 1) / step
        return [count]
    case .add, .subtract, .mul, .division:
        guard temp.inputs.count == 2 else { return [] }
        return broadcastShapesHelper(temp.inputs[0].shape, temp.inputs[1].shape)
    case .relu, .tanh, .gelu, .sigmoid, .silu, .sin, .cos:
        return temp.inputs.first?.shape ?? []
    case .leakyRelu:
        return temp.inputs.first?.shape ?? []
    case .rsqrt, .sqrt, .log, .exp, .exp2:
        return temp.inputs.first?.shape ?? []
    case .power:
        return temp.inputs.first?.shape ?? []
    case .matmul, .matMul:
        return inferMatMulShapeHelper(temp.inputs)
    case .transpose(let dim1, let dim2):
        return inferTransposeShapeHelper(temp.inputs, dim1: dim1, dim2: dim2)
    case .permute(let dims):
        return inferPermuteShapeHelper(temp.inputs, dims: dims)
    case .reshape(let shape):
        return shape
    case .reshapeWith(let shapeNode):
        return shapeNode.shape
    case .expandDim(let dim):
        return inferExpandDimShapeHelper(temp.inputs, dim: dim)
    case .squeeze(let dim):
        return inferSqueezeShapeHelper(temp.inputs, dim: dim)
    case .softmax(_):
        return temp.inputs.first?.shape ?? []
    case .mean(let dim), .sum(let dim), .reduceMaximum(let dim):
        return inferReductionShapeHelper(temp.inputs, dim: dim)
    case .argMax(let dim):
        return inferReductionShapeHelper(temp.inputs, dim: dim)
    case .cat(let dim):
        return inferCatShapeHelper(temp.inputs, dim: dim)
    case .catWith(_, let dim):
        return inferCatWithShapeHelper(temp.inputs, dim: dim)
    case .split(let numSplits, let dim):
        return inferSplitShapeHelper(temp.inputs, numSplits: numSplits, dim: dim)
    case .splitOutput(let parentId, let index):
        return temp.inputs.first?.shape ?? []
    case .sliceDim(let dim, let upToNode):
        return inferSliceDimShapeHelper(temp.inputs, dim: dim)
    case .sliceStaticDim(let dim, let start, let upTo):
        return inferSliceStaticDimShapeHelper(temp.inputs, dim: dim, start: start, upTo: upTo)
    case .sliceStatic(let from, let upTo, let stride):
        return inferSliceStaticShapeHelper(temp.inputs, from: from, upTo: upTo, stride: stride)
    case .conv2d(let params):
        return inferConv2DShapeHelper(temp.inputs, params: params)
    case .conv2dTranspose(let params):
        return inferConv2DTransposeShape(temp.inputs, params: params)
    case .conv2dEncrypted(let params, _):
        return inferConv2DShapeHelper(temp.inputs, params: params)
    case .linear(let weights, _):
        return inferLinearShapeHelper(temp.inputs, weights: weights)
    case .linearLora(let weights, _, _, _, _, _):
        return inferLinearShapeHelper(temp.inputs, weights: weights)
    case .gather(let dim):
        return inferGatherShapeHelper(temp.inputs, dim: dim)
    case .clamp:
        return temp.inputs.first?.shape ?? []
    case .tile(let dims):
        return inferTileShapeHelper(temp.inputs, dims: dims)
    case .interpolateNearest(let scaleFactor, let dataLayout, _):
        return inferInterpolateShapeHelper(temp.inputs, scaleFactor: scaleFactor, dataLayout: dataLayout)
    case .interpolateBilinear(let scaleFactor, let dataLayout, _):
        return inferInterpolateShapeHelper(temp.inputs, scaleFactor: scaleFactor, dataLayout: dataLayout)
    case .pixelShuffle(let scale, let dataLayout):
        return inferPixelShuffleShapeHelper(temp.inputs, scale: scale, dataLayout: dataLayout)
    case .pixelUnshuffle(let scale, let dataLayout):
        return inferPixelUnshuffleShapeHelper(temp.inputs, scale: scale, dataLayout: dataLayout)
    case .constPad(let padding, _):
        return inferPadShapeHelper(temp.inputs, padding: padding)
    case .tril, .triu:
        return temp.inputs.first?.shape ?? []
    case .scaledDotProductAttention(_, _, let value, _, _):
        return value.shape
    case .groupNorm2d:
        return temp.inputs.first?.shape ?? []
    case .to:
        return temp.inputs.first?.shape ?? []
    case .quantize, .dequantize, .dynamicQuantize:
        return temp.inputs.first?.shape ?? []
    case .quantizePerChannel, .dequantizePerChannel:
        return temp.inputs.first?.shape ?? []
    case .shapeOf(let ofNode):
        let inputShape = ofNode.shape
        return [inputShape.count]
    case .randomUniform(shape: let shape, seed: _, _):
        return shape
    }
}

// Helper functions for shape computation (these will be defined below)
fileprivate func broadcastShapesHelper(_ shape1: [Int], _ shape2: [Int]) -> [Int] {
    let maxLen = max(shape1.count, shape2.count)
    var result = [Int]()
    for i in 0..<maxLen {
        let idx1 = shape1.count - maxLen + i
        let idx2 = shape2.count - maxLen + i
        let dim1 = (idx1 >= 0 && idx1 < shape1.count) ? shape1[idx1] : 1
        let dim2 = (idx2 >= 0 && idx2 < shape2.count) ? shape2[idx2] : 1
        if dim1 == -1 || dim2 == -1 {
            result.append(-1)
        } else if dim1 == dim2 || dim1 == 1 {
            result.append(dim2)
        } else if dim2 == 1 {
            result.append(dim1)
        } else {
            result.append(max(dim1, dim2))
        }
    }
    return result
}

fileprivate func normalizeDimHelper(_ dim: Int, for shape: [Int]) -> Int {
    return dim >= 0 ? dim : shape.count + dim
}

fileprivate func inferMatMulShapeHelper(_ inputs: [Node]) -> [Int] {
    guard inputs.count >= 2 else { return [] }
    let lhs = inputs[0].shape
    let rhs = inputs[1].shape
    guard lhs.count >= 2, rhs.count >= 2 else { return [] }
    var result = [Int]()
    let maxBatch = max(lhs.count - 2, rhs.count - 2)
    for i in 0..<maxBatch {
        let lhsIdx = i < lhs.count - 2 ? lhs[i] : 1
        let rhsIdx = i < rhs.count - 2 ? rhs[i] : 1
        result.append((lhsIdx == -1 || rhsIdx == -1) ? -1 : max(lhsIdx, rhsIdx))
    }
    result.append(lhs[lhs.count - 2])
    result.append(rhs[rhs.count - 1])
    return result
}

fileprivate func inferTransposeShapeHelper(_ inputs: [Node], dim1: Int, dim2: Int) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    let d1 = normalizeDimHelper(dim1, for: shape)
    let d2 = normalizeDimHelper(dim2, for: shape)
    if d1 >= 0 && d1 < shape.count && d2 >= 0 && d2 < shape.count {
        shape.swapAt(d1, d2)
    }
    return shape
}

fileprivate func inferPermuteShapeHelper(_ inputs: [Node], dims: [Int]) -> [Int] {
    guard let input = inputs.first else { return [] }
    return dims.map { $0 < input.shape.count ? input.shape[$0] : 1 }
}

fileprivate func inferExpandDimShapeHelper(_ inputs: [Node], dim: Int) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    let insertIdx = dim < 0 ? shape.count + dim + 1 : dim
    if insertIdx >= 0 && insertIdx <= shape.count {
        shape.insert(1, at: insertIdx)
    }
    return shape
}

fileprivate func inferSqueezeShapeHelper(_ inputs: [Node], dim: Int) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    let d = normalizeDimHelper(dim, for: shape)
    if d >= 0 && d < shape.count && shape[d] == 1 {
        shape.remove(at: d)
    }
    return shape
}

fileprivate func inferReductionShapeHelper(_ inputs: [Node], dim: Int) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    let d = normalizeDimHelper(dim, for: shape)
    if d >= 0 && d < shape.count {
        shape[d] = 1
    }
    return shape
}

fileprivate func inferCatShapeHelper(_ inputs: [Node], dim: Int) -> [Int] {
    guard !inputs.isEmpty else { return [] }
    var result = inputs[0].shape
    let d = normalizeDimHelper(dim, for: result)
    if d >= 0 && d < result.count {
        let catDim = inputs.dropFirst().reduce(result[d]) { sum, node in
            let nodeShape = node.shape
            let nodeDim = normalizeDimHelper(dim, for: nodeShape)
            if nodeDim >= 0 && nodeDim < nodeShape.count {
                return (sum == -1 || nodeShape[nodeDim] == -1) ? -1 : sum + nodeShape[nodeDim]
            }
            return sum
        }
        result[d] = catDim
    }
    return result
}

fileprivate func inferCatWithShapeHelper(_ inputs: [Node], dim: Int) -> [Int] {
    guard inputs.count >= 2 else { return [] }
    let shape1 = inputs[0].shape
    let shape2 = inputs[1].shape
    var result = shape1
    let d1 = normalizeDimHelper(dim, for: shape1)
    let d2 = normalizeDimHelper(dim, for: shape2)
    if d1 >= 0 && d1 < result.count && d2 >= 0 && d2 < shape2.count {
        result[d1] = (result[d1] == -1 || shape2[d2] == -1) ? -1 : shape1[d1] + shape2[d2]
    }
    return result
}

fileprivate func inferSplitShapeHelper(_ inputs: [Node], numSplits: Int, dim: Int) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    let d = normalizeDimHelper(dim, for: shape)
    if d >= 0 && d < shape.count {
        shape[d] = (shape[d] == -1) ? -1 : shape[d] / numSplits
    }
    return shape
}

fileprivate func inferSliceDimShapeHelper(_ inputs: [Node], dim: Int) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    let d = normalizeDimHelper(dim, for: shape)
    if d >= 0 && d < shape.count {
        shape[d] = -1
    }
    return shape
}

fileprivate func inferSliceStaticDimShapeHelper(_ inputs: [Node], dim: Int, start: Int, upTo: Int) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    let d = normalizeDimHelper(dim, for: shape)
    if d >= 0 && d < shape.count {
        shape[d] = upTo - start
    }
    return shape
}

fileprivate func inferSliceStaticShapeHelper(_ inputs: [Node], from: [Int], upTo: [Int], stride: [Int]) -> [Int] {
    guard let input = inputs.first else { return [] }
    let inputShape = input.shape
    var result = [Int]()
    for i in 0..<inputShape.count {
        if i < from.count && i < upTo.count && i < stride.count {
            result.append((upTo[i] - from[i] + stride[i] - 1) / stride[i])
        } else {
            result.append(inputShape[i])
        }
    }
    return result
}

fileprivate func inferConv2DTransposeShape(_ inputs: [Node], params: Conv2DParams) -> [Int] {
    guard let input = inputs.first else { return [] }

    let inputShape = input.shape
    
    let (padLeft, padRight, padTop, padBottom) = params.padding
    let totalHeightPad = padTop + padBottom
    let totalWidthPad = padLeft + padRight
    
    if params.dataLayout == .NCHW {
        guard inputShape.count == 4 else { return [] }
        let batch = inputShape[0]
        let outChannels = params.outChannels
        let h = inputShape[2]
        let w = inputShape[3]
        
        let outH: Int
        let outW: Int
        
        if h == -1 {
            outH = -1
        } else {
            // Transpose convolution formula: output = (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1
            let effectiveKernelH = params.dilation.0 * (params.kernelSize.0 - 1) + 1
            
            switch params.padStyle {
            case .explicit:
                outH = (h - 1) * params.stride.0 - totalHeightPad + effectiveKernelH
            case .same:
                outH = h * params.stride.0
            case .valid:
                outH = (h - 1) * params.stride.0 + params.kernelSize.0
            }
        }
        
        if w == -1 {
            outW = -1
        } else {
            let effectiveKernelW = params.dilation.1 * (params.kernelSize.1 - 1) + 1
            
            switch params.padStyle {
            case .explicit:
                outW = (w - 1) * params.stride.1 - totalWidthPad + effectiveKernelW
            case .same:
                outW = w * params.stride.1
            case .valid:
                outW = (w - 1) * params.stride.1 + params.kernelSize.1
            }
        }
        return [batch, outChannels, outH, outW]
    } else { // NHWC
        guard inputShape.count == 4 else { return [] }
        let batch = inputShape[0]
        let h = inputShape[1]
        let w = inputShape[2]
        let outChannels = params.outChannels
        
        let outH: Int
        let outW: Int
        
        if h == -1 {
            outH = -1
        } else {
            let effectiveKernelH = params.dilation.0 * (params.kernelSize.0 - 1) + 1
            
            switch params.padStyle {
            case .explicit:
                outH = (h - 1) * params.stride.0 - totalHeightPad + effectiveKernelH
            case .same:
                outH = h * params.stride.0
            case .valid:
                outH = (h - 1) * params.stride.0 + params.kernelSize.0
            }
        }
        
        if w == -1 {
            outW = -1
        } else {
            let effectiveKernelW = params.dilation.1 * (params.kernelSize.1 - 1) + 1
            
            switch params.padStyle {
            case .explicit:
                outW = (w - 1) * params.stride.1 - totalWidthPad + effectiveKernelW
            case .same:
                outW = w * params.stride.1
            case .valid:
                outW = (w - 1) * params.stride.1 + params.kernelSize.1
            }
        }
        return [batch, outH, outW, outChannels]
    }
}

fileprivate func inferConv2DShapeHelper(_ inputs: [Node], params: Conv2DParams) -> [Int] {
    guard let input = inputs.first else { return [] }
    let inputShape = input.shape
    let (padLeft, padRight, padTop, padBottom) = params.padding
    let totalHeightPad = padTop + padBottom
    let totalWidthPad = padLeft + padRight
    
    if params.dataLayout == .NCHW {
        guard inputShape.count == 4 else { return [] }
        let batch = inputShape[0]
        let outChannels = params.outChannels
        let h = inputShape[2]
        let w = inputShape[3]
        
        let outH = (h == -1) ? -1 : {
            switch params.padStyle {
            case .explicit: return (h + totalHeightPad - params.dilation.0 * (params.kernelSize.0 - 1) - 1) / params.stride.0 + 1
            case .same: return (h + params.stride.0 - 1) / params.stride.0
            case .valid: return (h - params.kernelSize.0) / params.stride.0 + 1
            }
        }()
        
        let outW = (w == -1) ? -1 : {
            switch params.padStyle {
            case .explicit: return (w + totalWidthPad - params.dilation.1 * (params.kernelSize.1 - 1) - 1) / params.stride.1 + 1
            case .same: return (w + params.stride.1 - 1) / params.stride.1
            case .valid: return (w - params.kernelSize.1) / params.stride.1 + 1
            }
        }()
        
        return [batch, outChannels, outH, outW]
    } else {
        guard inputShape.count == 4 else { return [] }
        let batch = inputShape[0]
        let h = inputShape[1]
        let w = inputShape[2]
        let outChannels = params.outChannels
        
        let outH = (h == -1) ? -1 : {
            switch params.padStyle {
            case .explicit: return (h + totalHeightPad - params.dilation.0 * (params.kernelSize.0 - 1) - 1) / params.stride.0 + 1
            case .same: return (h + params.stride.0 - 1) / params.stride.0
            case .valid: return (h - params.kernelSize.0) / params.stride.0 + 1
            }
        }()
        
        let outW = (w == -1) ? -1 : {
            switch params.padStyle {
            case .explicit: return (w + totalWidthPad - params.dilation.1 * (params.kernelSize.1 - 1) - 1) / params.stride.1 + 1
            case .same: return (w + params.stride.1 - 1) / params.stride.1
            case .valid: return (w - params.kernelSize.1) / params.stride.1 + 1
            }
        }()
        
        return [batch, outH, outW, outChannels]
    }
}

fileprivate func inferLinearShapeHelper(_ inputs: [Node], weights: Node) -> [Int] {
    guard let input = inputs.first else { return [] }
    let inputShape = input.shape
    let weightShape = weights.shape
    guard weightShape.count >= 2 else { return inputShape }
    var result = inputShape
    result[result.count - 1] = weightShape[weightShape.count - 1]
    return result
}

fileprivate func inferGatherShapeHelper(_ inputs: [Node], dim: Int) -> [Int] {
    guard inputs.count >= 2 else { return [] }
    var result = inputs[0].shape
    let indexShape = inputs[1].shape
    let d = normalizeDimHelper(dim, for: result)
    if d >= 0 && d < result.count && !indexShape.isEmpty {
        result[d] = indexShape[0]
    }
    return result
}

fileprivate func inferTileShapeHelper(_ inputs: [Node], dims: [Int]) -> [Int] {
    guard let input = inputs.first else { return [] }
    return zip(input.shape, dims).map { ($0 == -1) ? -1 : $0 * $1 }
}

fileprivate func inferInterpolateShapeHelper(_ inputs: [Node], scaleFactor: Float, dataLayout: convDataLayout) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    if dataLayout == .NCHW && shape.count == 4 {
        shape[2] = (shape[2] == -1) ? -1 : Int(Float(shape[2]) * scaleFactor)
        shape[3] = (shape[3] == -1) ? -1 : Int(Float(shape[3]) * scaleFactor)
    } else if dataLayout == .NHWC && shape.count == 4 {
        shape[1] = (shape[1] == -1) ? -1 : Int(Float(shape[1]) * scaleFactor)
        shape[2] = (shape[2] == -1) ? -1 : Int(Float(shape[2]) * scaleFactor)
    }
    return shape
}

fileprivate func inferPixelShuffleShapeHelper(_ inputs: [Node], scale: Int, dataLayout: convDataLayout) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    if dataLayout == .NCHW && shape.count == 4 {
        shape[1] = (shape[1] == -1) ? -1 : shape[1] / (scale * scale)
        shape[2] = (shape[2] == -1) ? -1 : shape[2] * scale
        shape[3] = (shape[3] == -1) ? -1 : shape[3] * scale
    } else if dataLayout == .NHWC && shape.count == 4 {
        shape[1] = (shape[1] == -1) ? -1 : shape[1] * scale
        shape[2] = (shape[2] == -1) ? -1 : shape[2] * scale
        shape[3] = (shape[3] == -1) ? -1 : shape[3] / (scale * scale)
    }
    return shape
}

fileprivate func inferPixelUnshuffleShapeHelper(_ inputs: [Node], scale: Int, dataLayout: convDataLayout) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    if dataLayout == .NCHW && shape.count == 4 {
        shape[1] = (shape[1] == -1) ? -1 : shape[1] * (scale * scale)
        shape[2] = (shape[2] == -1) ? -1 : shape[2] / scale
        shape[3] = (shape[3] == -1) ? -1 : shape[3] / scale
    } else if dataLayout == .NHWC && shape.count == 4 {
        shape[1] = (shape[1] == -1) ? -1 : shape[1] / scale
        shape[2] = (shape[2] == -1) ? -1 : shape[2] / scale
        shape[3] = (shape[3] == -1) ? -1 : shape[3] * (scale * scale)
    }
    return shape
}

fileprivate func inferPadShapeHelper(_ inputs: [Node], padding: [(Int, Int)]) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    for (i, pad) in padding.enumerated() where i < shape.count {
        shape[i] = (shape[i] == -1) ? -1 : shape[i] + pad.0 + pad.1
    }
    return shape
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
    case sin, cos
    case rsqrt, sqrt
    case relu, tanh, gelu, sigmoid, silu
    case leakyRelu(_ slope: Float)
    case softmax(_ dim: Int)
    case split(_ numSplits: Int, dim: Int)
    case transpose(_ dim: Int, _ with: Int)
    case permute(_ dims: [Int])
    case tile(_ dims: [Int])
    case reshape(shape: [Int]), reshapeWith(withShape: Node)
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
    case arange(start: Int, end: Int, step: Int, dataType: dataType)
    case sliceDim(_ dim: Int, upTo: Node)
    case sliceStatic(from: [Int], upTo: [Int], stride: [Int])
    case sliceStaticDim(_ dim: Int, start: Int, upTo: Int)
    case interpolateNearest(scaleFactor: Float, dataLayout: convDataLayout, alignCorners: Bool), interpolateBilinear(scaleFactor: Float, dataLayout: convDataLayout, alignCorners: Bool)
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
    case groupNorm2d(groups: Int, channels: Int, eps: Float, weights: Node? = nil, bias: Node? = nil, affine: Bool = true, dataLayout: convDataLayout = .NCHW)
    case randomUniform(shape: [Int], seed: Int, dataType: dataType)
    case conv2d(Conv2DParams), conv2dTranspose(Conv2DParams)
    
    case quantize(scale: Float, zeroPoint: Float, targetType: dataType)
    case dequantize(scale: Float, zeroPoint: Float, targetType: dataType)
    case dynamicQuantize(targetType: dataType)
    case quantizePerChannel(scales: [Float], zeroPoints: [Float], axis: Int, targetType: dataType)
    case dequantizePerChannel(scales: [Float], zeroPoints: [Float], axis: Int, targetType: dataType)
    
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
        return inferDataType()
    }
    
    private func inferDataType() -> dataType {
        switch self.op {
        // Base cases with explicit dataTypes
        case .placeholder(_, _, let dataType):
            return dataType
        case .constant(_, _, let dataType):
            return dataType
        case .variable(_, _, let dataType):
            return dataType
        case .constantScalar(_, _, let dataType):
            return dataType
        case .arange(_, _, _, let dataType):
            return dataType
        case .conv2d(let params):
            return params.dataType
        case .conv2dTranspose(let params):
            return params.dataType
        case .conv2dEncrypted(let params, _):
            return params.dataType
            
        // Type casting operations
        case .to(let targetType):
            return targetType
        case .quantize(_, _, let targetType):
            return targetType
        case .dequantize(_, _, let targetType):
            // After dequantization, we get the target float type
            return targetType
        case .dynamicQuantize(let targetType):
            return targetType
        case .quantizePerChannel(_, _, _, let targetType):
            return targetType
        case .dequantizePerChannel(_, _, _, let targetType):
            return targetType
            
        // Operations that preserve input dataType
        case .relu, .tanh, .gelu, .sigmoid, .silu, .sin, .cos:
            return inputs.first?.dataType ?? .float32
        case .leakyRelu:
            return inputs.first?.dataType ?? .float32
        case .rsqrt, .sqrt, .log, .exp, .exp2:
            return inputs.first?.dataType ?? .float32
        case .power:
            return inputs.first?.dataType ?? .float32
        case .softmax:
            return inputs.first?.dataType ?? .float32
        case .transpose, .permute:
            return inputs.first?.dataType ?? .float32
        case .reshape, .reshapeWith:
            return inputs.first?.dataType ?? .float32
        case .expandDim, .squeeze:
            return inputs.first?.dataType ?? .float32
        case .tile:
            return inputs.first?.dataType ?? .float32
        case .interpolateNearest, .interpolateBilinear, .pixelShuffle, .pixelUnshuffle:
            return inputs.first?.dataType ?? .float32
        case .constPad:
            return inputs.first?.dataType ?? .float32
        case .tril, .triu:
            return inputs.first?.dataType ?? .float32
        case .clamp:
            return inputs.first?.dataType ?? .float32
        case .mean, .sum, .reduceMaximum:
            return inputs.first?.dataType ?? .float32
        case .sliceDim, .sliceStaticDim, .sliceStatic:
            return inputs.first?.dataType ?? .float32
        case .groupNorm2d:
            return inputs.first?.dataType ?? .float32
        case .shapeOf:
            return .int64  // Shape is always int64
            
        // Binary operations - assert matching types
        case .add, .subtract, .mul, .division:
            return inferBinaryOpDataType()
        case .matmul, .matMul:
            return inferBinaryOpDataType()
            
        // Concatenation and splitting preserve dataType
        case .cat, .catWith:
            return inferCatDataType()
        case .split:
            return inputs.first?.dataType ?? .float32
        case .splitOutput:
            return inputs.first?.dataType ?? .float32
            
        // Attention operations
        case .scaledDotProductAttention(_, _, let value, _, _):
            return value.dataType
            
        // Linear operations
        case .linear(let weights, _):
            return inferLinearDataType(weights: weights)
        case .linearLora(let weights, _, _, _, _, _):
            return inferLinearDataType(weights: weights)
            
        // Gather and indexing operations
        case .gather:
            return inputs.first?.dataType ?? .float32
        case .argMax:
            return .int64
        case .randomUniform(shape: let shape, seed: let seed, let dataType):
            return dataType

        }
    }
    
    private func inferBinaryOpDataType() -> dataType {
        guard inputs.count >= 2 else { return .float32 }
        let lhsType = inputs[0].dataType
        let rhsType = inputs[1].dataType
        
        // Assert that types match
        assert(lhsType == rhsType, "DataType mismatch in binary operation: \(lhsType) vs \(rhsType)")
        
        return lhsType
    }
    
    private func inferCatDataType() -> dataType {
        guard !inputs.isEmpty else { return .float32 }
        let firstType = inputs[0].dataType
        
        // Assert all inputs have the same dataType
        for (index, input) in inputs.enumerated() {
            assert(input.dataType == firstType, 
                   "DataType mismatch in concatenation at index \(index): expected \(firstType), got \(input.dataType)")
        }
        
        return firstType
    }
    
    private func inferLinearDataType(weights: Node) -> dataType {
        guard let input = inputs.first else { return .float32 }
        let inputType = input.dataType
        let weightsType = weights.dataType
        
        // Assert that types match
        assert(inputType == weightsType, 
               "DataType mismatch in linear operation: input \(inputType) vs weights \(weightsType)")
        
        return inputType
    }
    
    private func inferShape() -> [Int] {
        switch self.op {
        // Base cases with explicit shapes
        case .placeholder(_, let shape, _):
            return shape
        case .constant(_, let shape, _):
            return shape
        case .variable(_, let shape, _):
            return shape
        case .constantScalar(_, let shape, _):
            return shape
        case .arange(let start, let end, let step, _):
            let count = (end - start + step - 1) / step
            return [count]
            
        // Element-wise operations (preserve shape)
        case .add, .subtract, .mul, .division:
            guard inputs.count == 2 else { return [] }
            return broadcastShapes(inputs[0].shape, inputs[1].shape)
            
        case .relu, .tanh, .gelu, .sigmoid, .silu, .sin, .cos:
            return inputs.first?.shape ?? []
        case .leakyRelu:
            return inputs.first?.shape ?? []
        case .rsqrt, .sqrt, .log, .exp, .exp2:
            return inputs.first?.shape ?? []
        case .power:
            return inputs.first?.shape ?? []
            
        // Shape-changing operations
        case .matmul, .matMul:
            return inferMatMulShape()
        case .transpose(let dim1, let dim2):
            return inferTransposeShape(dim1: dim1, dim2: dim2)
        case .permute(let dims):
            return inferPermuteShape(dims: dims)
        case .reshape(let shape):
            return shape
        case .reshapeWith(let shapeNode):
            return shapeNode.shape
        case .expandDim(let dim):
            return inferExpandDimShape(dim: dim)
        case .squeeze(let dim):
            return inferSqueezeShape(dim: dim)
            
        // Reduction operations
        case .softmax(_):
            return inputs.first?.shape ?? []
        case .mean(let dim), .sum(let dim), .reduceMaximum(let dim):
            return inferReductionShape(dim: dim)
        case .argMax(let dim):
            return inferReductionShape(dim: dim)
            
        // Concatenation operations
        case .cat(let dim):
            return inferCatShape(dim: dim)
        case .catWith(_, let dim):
            return inferCatWithShape(dim: dim)
            
        // Split operations
        case .split(let numSplits, let dim):
            return inferSplitShape(numSplits: numSplits, dim: dim)
        case .splitOutput(let parentId, let index):
            return inferSplitOutputShape(parentId: parentId, index: index)
            
        // Slicing operations
        case .sliceDim(let dim, let upToNode):
            return inferSliceDimShape(dim: dim, upTo: upToNode)
        case .sliceStaticDim(let dim, let start, let upTo):
            return inferSliceStaticDimShape(dim: dim, start: start, upTo: upTo)
        case .sliceStatic(let from, let upTo, let stride):
            return inferSliceStaticShape(from: from, upTo: upTo, stride: stride)
            
        // Conv operations
        case .conv2d(let params):
            return inferConv2DShape(params: params)
        case .conv2dTranspose(let params):
            return inferConv2DTransposeShape(params: params)
//            inferConv2DShape(params: params)
        case .conv2dEncrypted(let params, _):
            return inferConv2DShape(params: params)
            
        // Other operations
        case .linear(let weights, _):
            return inferLinearShape(weights: weights)
        case .linearLora(let weights, _, _, _, _, _):
            return inferLinearShape(weights: weights)
        case .gather(let dim):
            return inferGatherShape(dim: dim)
        case .clamp:
            return inputs.first?.shape ?? []
        case .tile(let dims):
            return inferTileShape(dims: dims)
        case .interpolateNearest(let scaleFactor, let dataLayout, _):
            return inferInterpolateShape(scaleFactor: scaleFactor, dataLayout: dataLayout)
        case .interpolateBilinear(let scaleFactor, let dataLayout, _):
            return inferInterpolateShape(scaleFactor: scaleFactor, dataLayout: dataLayout)
        case .pixelShuffle(let scale, let dataLayout):
            return inferPixelShuffleShape(scale: scale, dataLayout: dataLayout)
        case .pixelUnshuffle(let scale, let dataLayout):
            return inferPixelUnshuffleShape(scale: scale, dataLayout: dataLayout)
        case .constPad(let padding, _):
            return inferPadShape(padding: padding)
        case .tril, .triu:
            return inputs.first?.shape ?? []
        case .scaledDotProductAttention(_, _, let value, _, _):
            return value.shape
        case .groupNorm2d:
            return inputs.first?.shape ?? []
        case .to:
            return inputs.first?.shape ?? []
        case .quantize, .dequantize, .dynamicQuantize:
            return inputs.first?.shape ?? []
        case .quantizePerChannel, .dequantizePerChannel:
            return inputs.first?.shape ?? []
        case .shapeOf(let ofNode):
            let inputShape = ofNode.shape
            return [inputShape.count]
        case .randomUniform(shape: let shape, seed: let seed, _):
            return shape

        }
    }
    
    // MARK: - Shape Inference Helpers
    
    /// Normalizes a dimension index to handle negative indices
    /// -1 refers to the last dimension, -2 to second-to-last, etc.
    private func normalizeDim(_ dim: Int, for shape: [Int]) -> Int {
        if dim >= 0 {
            return dim
        }
        return shape.count + dim
    }
    
    private func broadcastShapes(_ shape1: [Int], _ shape2: [Int]) -> [Int] {
        let maxLen = max(shape1.count, shape2.count)
        var result = [Int]()
        
        for i in 0..<maxLen {
            let idx1 = shape1.count - maxLen + i
            let idx2 = shape2.count - maxLen + i
            
            let dim1 = (idx1 >= 0 && idx1 < shape1.count) ? shape1[idx1] : 1
            let dim2 = (idx2 >= 0 && idx2 < shape2.count) ? shape2[idx2] : 1
            
            if dim1 == -1 || dim2 == -1 {
                result.append(-1)  // Dynamic dimension propagates
            } else if dim1 == dim2 || dim1 == 1 {
                result.append(dim2)
            } else if dim2 == 1 {
                result.append(dim1)
            } else {
                result.append(max(dim1, dim2))
            }
        }
        return result
    }
    
    private func inferMatMulShape() -> [Int] {
        guard inputs.count >= 2 else { return [] }
        let lhs = inputs[0].shape
        let rhs = inputs[1].shape
        
        guard lhs.count >= 2, rhs.count >= 2 else { return [] }
        
        var result = [Int]()
        // Handle batch dimensions
        let maxBatch = max(lhs.count - 2, rhs.count - 2)
        for i in 0..<maxBatch {
            let lhsIdx = i < lhs.count - 2 ? lhs[i] : 1
            let rhsIdx = i < rhs.count - 2 ? rhs[i] : 1
            if lhsIdx == -1 || rhsIdx == -1 {
                result.append(-1)
            } else {
                result.append(max(lhsIdx, rhsIdx))
            }
        }
        
        // Add matrix dimensions: [M, K] x [K, N] = [M, N]
        let m = lhs[lhs.count - 2]
        let n = rhs[rhs.count - 1]
        result.append(m)
        result.append(n)
        return result
    }
    
    private func inferTransposeShape(dim1: Int, dim2: Int) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        let normalizedDim1 = normalizeDim(dim1, for: shape)
        let normalizedDim2 = normalizeDim(dim2, for: shape)
        if normalizedDim1 >= 0 && normalizedDim1 < shape.count && normalizedDim2 >= 0 && normalizedDim2 < shape.count {
            shape.swapAt(normalizedDim1, normalizedDim2)
        }
        return shape
    }
    
    private func inferPermuteShape(dims: [Int]) -> [Int] {
        guard let input = inputs.first else { return [] }
        let inputShape = input.shape
        return dims.map { $0 < inputShape.count ? inputShape[$0] : 1 }
    }
    
    private func inferExpandDimShape(dim: Int) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        let insertIdx = dim < 0 ? shape.count + dim + 1 : dim
        if insertIdx >= 0 && insertIdx <= shape.count {
            shape.insert(1, at: insertIdx)
        }
        return shape
    }
    
    private func inferSqueezeShape(dim: Int) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        let normalizedDim = normalizeDim(dim, for: shape)
        if normalizedDim >= 0 && normalizedDim < shape.count && shape[normalizedDim] == 1 {
            shape.remove(at: normalizedDim)
        }
        return shape
    }
    
    private func inferReductionShape(dim: Int) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        let normalizedDim = normalizeDim(dim, for: shape)
        if normalizedDim >= 0 && normalizedDim < shape.count {
            shape[normalizedDim] = 1
        }
        return shape
    }
    
    private func inferCatShape(dim: Int) -> [Int] {
        guard !inputs.isEmpty else { return [] }
        var result = inputs[0].shape
        let normalizedDim = normalizeDim(dim, for: result)
        if normalizedDim >= 0 && normalizedDim < result.count {
            let catDim = inputs.dropFirst().reduce(result[normalizedDim]) { sum, node in
                let nodeShape = node.shape
                let nodeDim = normalizeDim(dim, for: nodeShape)
                if nodeDim >= 0 && nodeDim < nodeShape.count {
                    if sum == -1 || nodeShape[nodeDim] == -1 {
                        return -1  // Dynamic dimension
                    }
                    return sum + nodeShape[nodeDim]
                }
                return sum
            }
            result[normalizedDim] = catDim
        }
        return result
    }
    
    private func inferCatWithShape(dim: Int) -> [Int] {
        guard inputs.count >= 2 else { return [] }
        let shape1 = inputs[0].shape
        let shape2 = inputs[1].shape
        var result = shape1
        let normalizedDim1 = normalizeDim(dim, for: shape1)
        let normalizedDim2 = normalizeDim(dim, for: shape2)
        if normalizedDim1 >= 0 && normalizedDim1 < result.count && normalizedDim2 >= 0 && normalizedDim2 < shape2.count {
            if result[normalizedDim1] == -1 || shape2[normalizedDim2] == -1 {
                result[normalizedDim1] = -1
            } else {
                result[normalizedDim1] = shape1[normalizedDim1] + shape2[normalizedDim2]
            }
        }
        return result
    }
    
    private func inferSplitShape(numSplits: Int, dim: Int) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        let normalizedDim = normalizeDim(dim, for: shape)
        if normalizedDim >= 0 && normalizedDim < shape.count {
            if shape[normalizedDim] == -1 {
                shape[normalizedDim] = -1  // Keep dynamic
            } else {
                shape[normalizedDim] = shape[normalizedDim] / numSplits
            }
        }
        return shape
    }
    
    private func inferSplitOutputShape(parentId: UUID, index: Int) -> [Int] {
        // The split output shape is same as the split operation shape
        guard let parent = inputs.first else { return [] }
        return parent.shape
    }
    
    private func inferSliceDimShape(dim: Int, upTo: Node) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        let normalizedDim = normalizeDim(dim, for: shape)
        if normalizedDim >= 0 && normalizedDim < shape.count {
            // Dynamic slice - mark as dynamic
            shape[normalizedDim] = -1
        }
        return shape
    }
    
    private func inferSliceStaticDimShape(dim: Int, start: Int, upTo: Int) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        let normalizedDim = normalizeDim(dim, for: shape)
        if normalizedDim >= 0 && normalizedDim < shape.count {
            shape[normalizedDim] = upTo - start
        }
        return shape
    }
    
    private func inferSliceStaticShape(from: [Int], upTo: [Int], stride: [Int]) -> [Int] {
        guard let input = inputs.first else { return [] }
        let inputShape = input.shape
        var result = [Int]()
        for i in 0..<inputShape.count {
            if i < from.count && i < upTo.count && i < stride.count {
                // Static slice with explicit bounds always produces static size
                let size = (upTo[i] - from[i] + stride[i] - 1) / stride[i]
                result.append(size)
            } else {
                // Dimension not being sliced, preserve original shape (including -1)
                result.append(inputShape[i])
            }
        }
        return result
    }
    
    private func inferConv2DShape(params: Conv2DParams) -> [Int] {
        guard let input = inputs.first else { return [] }
        let inputShape = input.shape
        
        // Extract padding: (left, right, top, bottom) - PyTorch convention
        let (padLeft, padRight, padTop, padBottom) = params.padding
        let totalHeightPad = padTop + padBottom
        let totalWidthPad = padLeft + padRight
        
        if params.dataLayout == .NCHW {
            guard inputShape.count == 4 else { return [] }
            let batch = inputShape[0]
            let outChannels = params.outChannels
            let h = inputShape[2]
            let w = inputShape[3]
            
            let outH: Int
            let outW: Int
            
            // Handle height dimension independently
            if h == -1 {
                outH = -1
            } else {
                switch params.padStyle {
                case .explicit:
                    outH = (h + totalHeightPad - params.dilation.0 * (params.kernelSize.0 - 1) - 1) / params.stride.0 + 1
                case .same:
                    outH = (h + params.stride.0 - 1) / params.stride.0
                case .valid:
                    outH = (h - params.kernelSize.0) / params.stride.0 + 1
                }
            }
            
            // Handle width dimension independently
            if w == -1 {
                outW = -1
            } else {
                switch params.padStyle {
                case .explicit:
                    outW = (w + totalWidthPad - params.dilation.1 * (params.kernelSize.1 - 1) - 1) / params.stride.1 + 1
                case .same:
                    outW = (w + params.stride.1 - 1) / params.stride.1
                case .valid:
                    outW = (w - params.kernelSize.1) / params.stride.1 + 1
                }
            }
            return [batch, outChannels, outH, outW]
        } else { // NHWC
            guard inputShape.count == 4 else { return [] }
            let batch = inputShape[0]
            let h = inputShape[1]
            let w = inputShape[2]
            let outChannels = params.outChannels
            
            let outH: Int
            let outW: Int
            
            // Handle height dimension independently
            if h == -1 {
                outH = -1
            } else {
                switch params.padStyle {
                case .explicit:
                    outH = (h + totalHeightPad - params.dilation.0 * (params.kernelSize.0 - 1) - 1) / params.stride.0 + 1
                case .same:
                    outH = (h + params.stride.0 - 1) / params.stride.0
                case .valid:
                    outH = (h - params.kernelSize.0) / params.stride.0 + 1
                }
            }
            
            // Handle width dimension independently
            if w == -1 {
                outW = -1
            } else {
                switch params.padStyle {
                case .explicit:
                    outW = (w + totalWidthPad - params.dilation.1 * (params.kernelSize.1 - 1) - 1) / params.stride.1 + 1
                case .same:
                    outW = (w + params.stride.1 - 1) / params.stride.1
                case .valid:
                    outW = (w - params.kernelSize.1) / params.stride.1 + 1
                }
            }
            return [batch, outH, outW, outChannels]
        }
    }
    
    private func inferConv2DTransposeShape(params: Conv2DParams) -> [Int] {
        guard let input = inputs.first else { return [] }
        let inputShape = input.shape
        
        // Extract padding: (left, right, top, bottom) - PyTorch convention
        let (padLeft, padRight, padTop, padBottom) = params.padding
        let totalHeightPad = padTop + padBottom
        let totalWidthPad = padLeft + padRight
        
        if params.dataLayout == .NCHW {
            guard inputShape.count == 4 else { return [] }
            let batch = inputShape[0]
            let outChannels = params.outChannels
            let h = inputShape[2]
            let w = inputShape[3]
            
            let outH: Int
            let outW: Int
            
            if h == -1 {
                outH = -1
            } else {
                // Transpose convolution formula: output = (input - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1
                let effectiveKernelH = params.dilation.0 * (params.kernelSize.0 - 1) + 1
                
                switch params.padStyle {
                case .explicit:
                    outH = (h - 1) * params.stride.0 - totalHeightPad + effectiveKernelH
                case .same:
                    outH = h * params.stride.0
                case .valid:
                    outH = (h - 1) * params.stride.0 + params.kernelSize.0
                }
            }
            
            if w == -1 {
                outW = -1
            } else {
                let effectiveKernelW = params.dilation.1 * (params.kernelSize.1 - 1) + 1
                
                switch params.padStyle {
                case .explicit:
                    outW = (w - 1) * params.stride.1 - totalWidthPad + effectiveKernelW
                case .same:
                    outW = w * params.stride.1
                case .valid:
                    outW = (w - 1) * params.stride.1 + params.kernelSize.1
                }
            }
            return [batch, outChannels, outH, outW]
        } else { // NHWC
            guard inputShape.count == 4 else { return [] }
            let batch = inputShape[0]
            let h = inputShape[1]
            let w = inputShape[2]
            let outChannels = params.outChannels
            
            let outH: Int
            let outW: Int
            
            if h == -1 {
                outH = -1
            } else {
                let effectiveKernelH = params.dilation.0 * (params.kernelSize.0 - 1) + 1
                
                switch params.padStyle {
                case .explicit:
                    outH = (h - 1) * params.stride.0 - totalHeightPad + effectiveKernelH
                case .same:
                    outH = h * params.stride.0
                case .valid:
                    outH = (h - 1) * params.stride.0 + params.kernelSize.0
                }
            }
            
            if w == -1 {
                outW = -1
            } else {
                let effectiveKernelW = params.dilation.1 * (params.kernelSize.1 - 1) + 1
                
                switch params.padStyle {
                case .explicit:
                    outW = (w - 1) * params.stride.1 - totalWidthPad + effectiveKernelW
                case .same:
                    outW = w * params.stride.1
                case .valid:
                    outW = (w - 1) * params.stride.1 + params.kernelSize.1
                }
            }
            return [batch, outH, outW, outChannels]
        }
    }
    
    private func inferLinearShape(weights: Node) -> [Int] {
        guard let input = inputs.first else { return [] }
        let inputShape = input.shape
        let weightShape = weights.shape
        
        guard weightShape.count >= 2 else { return inputShape }
        
        // Linear: input [..., in_features] @ weights [in_features, out_features] = [..., out_features]
        var result = inputShape
        result[result.count - 1] = weightShape[weightShape.count - 1]  // out_features (last dim)
        return result
    }
    
    private func inferGatherShape(dim: Int) -> [Int] {
        guard inputs.count >= 2 else { return [] }
        let inputShape = inputs[0].shape
        let indexShape = inputs[1].shape
        
        var result = inputShape
        let normalizedDim = normalizeDim(dim, for: result)
        if normalizedDim >= 0 && normalizedDim < result.count && !indexShape.isEmpty {
            result[normalizedDim] = indexShape[0]
        }
        return result
    }
    
    private func inferTileShape(dims: [Int]) -> [Int] {
        guard let input = inputs.first else { return [] }
        let inputShape = input.shape
        return zip(inputShape, dims).map { shape, multiplier in
            if shape == -1 {
                return -1
            }
            return shape * multiplier
        }
    }
    
    private func inferInterpolateShape(scaleFactor: Float, dataLayout: convDataLayout) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        if dataLayout == .NCHW && shape.count == 4 {
            shape[2] = shape[2] == -1 ? -1 : Int(Float(shape[2]) * scaleFactor)
            shape[3] = shape[3] == -1 ? -1 : Int(Float(shape[3]) * scaleFactor)
        } else if dataLayout == .NHWC && shape.count == 4 {
            shape[1] = shape[1] == -1 ? -1 : Int(Float(shape[1]) * scaleFactor)
            shape[2] = shape[2] == -1 ? -1 : Int(Float(shape[2]) * scaleFactor)
        }
        return shape
    }
    
    private func inferPixelShuffleShape(scale: Int, dataLayout: convDataLayout) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        if dataLayout == .NCHW && shape.count == 4 {
            shape[1] = shape[1] == -1 ? -1 : shape[1] / (scale * scale)
            shape[2] = shape[2] == -1 ? -1 : shape[2] * scale
            shape[3] = shape[3] == -1 ? -1 : shape[3] * scale
        } else if dataLayout == .NHWC && shape.count == 4 {
            shape[1] = shape[1] == -1 ? -1 : shape[1] * scale
            shape[2] = shape[2] == -1 ? -1 : shape[2] * scale
            shape[3] = shape[3] == -1 ? -1 : shape[3] / (scale * scale)
        }
        return shape
    }
    
    private func inferPixelUnshuffleShape(scale: Int, dataLayout: convDataLayout) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        if dataLayout == .NCHW && shape.count == 4 {
            shape[1] = shape[1] == -1 ? -1 : shape[1] * (scale * scale)
            shape[2] = shape[2] == -1 ? -1 : shape[2] / scale
            shape[3] = shape[3] == -1 ? -1 : shape[3] / scale
        } else if dataLayout == .NHWC && shape.count == 4 {
            shape[1] = shape[1] == -1 ? -1 : shape[1] / scale
            shape[2] = shape[2] == -1 ? -1 : shape[2] / scale
            shape[3] = shape[3] == -1 ? -1 : shape[3] * (scale * scale)
        }
        return shape
    }
    
    private func inferPadShape(padding: [(Int, Int)]) -> [Int] {
        guard let input = inputs.first else { return [] }
        var shape = input.shape
        for (i, pad) in padding.enumerated() {
            if i < shape.count {
                if shape[i] == -1 {
                    shape[i] = -1
                } else {
                    shape[i] = shape[i] + pad.0 + pad.1
                }
            }
        }
        return shape
    }

}
public struct Executor {
}

public extension Node {
    
    static func arange(_ start: Int, _ end: Int, step: Int = 1, dataType: dataType = .float32) -> Node {
        return .init(op: .arange(start: start, end: end, step: step, dataType: dataType))
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

    static func relu(_ input: Node) -> Node {
        return .init(op: .relu, inputs: [input])
    }

    func relu() -> Node {
        return .init(op: .relu, inputs: [self])
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
