import Foundation

public protocol QuantizationStats {}

public struct QuantizationBaseStats: QuantizationStats {
    public let scale: Float
    public let zeroPoint: Float
    public let dataType: dataType

    public init(scale: Float, zeroPoint: Float, dataType: dataType) {
        precondition(dataType == .int8 || dataType == .uint8,
                     "Quantization currently supports only int8 and uint8")
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.dataType = dataType
    }
}

public extension Node {
    func quantize(scale: Float, zeroPoint: Float, to targetType: dataType) -> Node {
        precondition(targetType == .int8 || targetType == .uint8,
                     "Target type must be int8 or uint8")
        return Node(op: .quantize(scale: scale, zeroPoint: zeroPoint, targetType: targetType),
                   inputs: [self])
    }

    func dequantize(scale: Float, zeroPoint: Float, to targetType: dataType) -> Node {
        precondition(self.dataType == .int8 || self.dataType == .uint8,
                     "Source type must be int8 or uint8")
        return Node(op: .dequantize(scale: scale, zeroPoint: zeroPoint, targetType: targetType),
                   inputs: [self])
    }

    func quantize(params: QuantizationStats) -> Node {
        if let params = params as? QuantizationBaseStats {
            return quantize(scale: params.scale, zeroPoint: params.zeroPoint, to: params.dataType)
        }
        else {
            fatalError("Not implemented yet")
        }
    }

    func dequantize(params: QuantizationStats) -> Node {
        if let params = params as? QuantizationBaseStats {
            return dequantize(scale: params.scale, zeroPoint: params.zeroPoint, to: params.dataType)
        }
        else {
            fatalError("Not implemented yet")
        }
    }

    func dynamicQuantize(to targetType: dataType) -> Node {
        precondition(targetType == .int8 || targetType == .uint8,
                     "Target type must be int8 or uint8")
        return Node(op: .dynamicQuantize(targetType: targetType), inputs: [self])
    }

    func quantizePerChannel(scales: [Float], zeroPoints: [Float], axis: Int, to targetType: dataType) -> Node {
        precondition(targetType == .int8 || targetType == .uint8,
                     "Target type must be int8 or uint8")
        precondition(scales.count == zeroPoints.count,
                     "Scales and zero points must have the same count")
        return Node(op: .quantizePerChannel(scales: scales, zeroPoints: zeroPoints,
                                          axis: axis, targetType: targetType),
                   inputs: [self])
    }

    func dequantizePerChannel(scales: [Float], zeroPoints: [Float], axis: Int, from targetType: dataType) -> Node {
        precondition(self.dataType == .int8 || self.dataType == .uint8,
                     "Source type must be int8 or uint8")
        precondition(scales.count == zeroPoints.count,
                     "Scales and zero points must have the same count")
        return Node(op: .dequantizePerChannel(scales: scales, zeroPoints: zeroPoints,
                                            axis: axis, targetType: targetType),
                   inputs: [self])
    }
}

public extension Node {
    static func quantize(input: Node, scale: Float, zeroPoint: Float, to targetType: dataType) -> Node {
        return input.quantize(scale: scale, zeroPoint: zeroPoint, to: targetType)
    }

    static func dequantize(input: Node, scale: Float, zeroPoint: Float, to targetType: dataType) -> Node {
        return input.dequantize(scale: scale, zeroPoint: zeroPoint, to: targetType)
    }
}

public struct QuantizationHelpers {
    public static func calculateQuantizationParams(min: Float, max: Float,
                                                  targetType: dataType) -> QuantizationBaseStats {
        let qmin: Float
        let qmax: Float

        switch targetType {
        case .int8:
            qmin = -128
            qmax = 127
        case .uint8:
            qmin = 0
            qmax = 255
        default:
            fatalError("Unsupported quantization type")
        }

        let scale = (max - min) / (qmax - qmin)
        let zeroPoint = qmin - min / scale

        return QuantizationBaseStats(scale: scale, zeroPoint: zeroPoint, dataType: targetType)
    }

    public static func quantizeValue(_ value: Float, scale: Float, zeroPoint: Float,
                                   targetType: dataType) -> Int {
        let quantized = round(value / scale + zeroPoint)

        switch targetType {
        case .int8:
            return Int(max(-128, min(127, quantized)))
        case .uint8:
            return Int(max(0, min(255, quantized)))
        default:
            fatalError("Unsupported quantization type")
        }
    }

    public static func dequantizeValue(_ value: Int, scale: Float, zeroPoint: Float) -> Float {
        return Float(value - Int(zeroPoint)) * scale
    }
}
