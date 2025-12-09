import Foundation


func computeShapeForInit(op: graphOp, inputs: [Node]) -> [Int] {
    struct TempNodeWrapper {
        let op: graphOp
        let inputs: [Node]
    }
    let temp = TempNodeWrapper(op: op, inputs: inputs)
    
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
        let count = Int(ceil(Double(end - start) / Double(step)))
        return [count]
    case .add, .subtract, .mul, .division:
        guard temp.inputs.count == 2 else { return [] }
        return broadcastShapesHelper(temp.inputs[0].shape, temp.inputs[1].shape)
    case .greater, .greaterEqual, .less, .lessEqual:
        guard temp.inputs.count == 2 else { return [] }
        return broadcastShapesHelper(temp.inputs[0].shape, temp.inputs[1].shape)
    case .relu, .tanh, .tan, .gelu, .sigmoid, .silu, .sin, .cos:
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
    case .resize(let outShape, let dataLayout, _, _):
        return inferResizeShapeHelper(temp.inputs, outShape: outShape, dataLayout: dataLayout)

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
    case .randomUniform(shape: let shape, seed: _, _), .randomNormal(shape: let shape, _, _, _, _):
        return shape
    case .degree2radians:
        return temp.inputs.first?.shape ?? []
    case .brodcast(let shape):
        return shape
    case .linspace(start: let start, end: let end, steps: let steps, dataType: let dataType):
        return [steps]
    case .repeatTensor(let times, let dim):
        guard var tmpShape = temp.inputs.first?.shape else {
            return []
        }
        tmpShape[dim] = tmpShape[dim] * times
        return tmpShape
    }
}


func computeDataTypeForInit(op: graphOp, inputs: [Node]) -> dataType {
    switch op {
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
    
    case .to(let targetType):
        return targetType
    case .quantize(_, _, let targetType):
        return targetType
    case .dequantize(_, _, let targetType):
        return targetType
    case .dynamicQuantize(let targetType):
        return targetType
    case .quantizePerChannel(_, _, _, let targetType):
        return targetType
    case .dequantizePerChannel(_, _, _, let targetType):
        return targetType
    
    case .relu, .tanh, .tan, .gelu, .sigmoid, .silu, .sin, .cos, .greater, .greaterEqual, .less,. lessEqual:
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
    case .interpolateNearest, .interpolateBilinear, .pixelShuffle, .pixelUnshuffle, .resize:
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
        return .int64
    
    case .add, .subtract, .mul, .division:
        return inferBinaryOpDataTypeHelper(inputs)
    case .matmul, .matMul:
        return inferBinaryOpDataTypeHelper(inputs)
    
    case .cat, .catWith:
        return inferCatDataTypeHelper(inputs)
    case .split:
        return inputs.first?.dataType ?? .float32
    case .splitOutput:
        return inputs.first?.dataType ?? .float32
    
    case .scaledDotProductAttention(_, _, let value, _, _):
        return value.dataType
    
    case .linear(let weights, _):
        return inferLinearDataTypeHelper(inputs: inputs, weights: weights)
    case .linearLora(let weights, _, _, _, _, _):
        return inferLinearDataTypeHelper(inputs: inputs, weights: weights)
    
    case .gather:
        return inputs.first?.dataType ?? .float32
    case .argMax:
        return .int64
    case .randomUniform(_, _, let dataType), .randomNormal(_, _, _, _, let dataType):
        return dataType
    case .degree2radians:
        return inputs.first?.dataType ?? .float32
    case .brodcast(_), .repeatTensor(_, _):
        return inputs.first?.dataType ?? .float32
    case .linspace(start: let start, end: let end, steps: let steps, dataType: let dataType):
        return dataType
    }
}

fileprivate func inferBinaryOpDataTypeHelper(_ inputs: [Node]) -> dataType {
    guard inputs.count >= 2 else { return .float32 }
    let lhsType = inputs[0].dataType
    let rhsType = inputs[1].dataType
    assert(lhsType == rhsType, "DataType mismatch in binary operation: \(lhsType) vs \(rhsType)")
    return lhsType
}

fileprivate func inferCatDataTypeHelper(_ inputs: [Node]) -> dataType {
    guard !inputs.isEmpty else { return .float32 }
    let firstType = inputs[0].dataType
    for (index, input) in inputs.enumerated() {
        assert(input.dataType == firstType, "DataType mismatch in concatenation at index \(index): expected \(firstType), got \(input.dataType)")
    }
    return firstType
}

fileprivate func inferLinearDataTypeHelper(inputs: [Node], weights: Node) -> dataType {
    guard let input = inputs.first else { return .float32 }
    let inputType = input.dataType
    let weightsType = weights.dataType
    assert(inputType == weightsType, "DataType mismatch in linear operation: input \(inputType) vs weights \(weightsType)")
    return inputType
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
    else {
        fatalError("Not implemented TODO")
    }
    return shape
}

fileprivate func inferResizeShapeHelper(_ inputs: [Node], outShape: (Int, Int), dataLayout: convDataLayout) -> [Int] {
    guard let input = inputs.first else { return [] }
    var shape = input.shape
    if dataLayout == .NCHW && shape.count == 4 {
        shape[2] = outShape.0
        shape[3] = outShape.1
    } else if dataLayout == .NHWC && shape.count == 4 {
        shape[1] = outShape.0
        shape[2] = outShape.1
    }
    else {
        fatalError("Not implemented TODO")
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


