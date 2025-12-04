import Logging

nonisolated(unsafe) public var logger = Logger(label: "swiftTensors.logger")

public extension dataType {
    public var byteSize: Int {
        switch self {
        case .float16:
            return 2
        case .float32:
            return 4
        case .bfloat16:
            return 2
        case .float8:
            return 1
        case .int8:
            return 1
        case .int16:
            return 2
        case .int32:
            return 4
        case .uint8:
            return 1
        case .uint16:
            return 2
        case .uint32:
            return 4
        case .float64:
            return 8
        case .int64:
            return 8
        case .uint64:
            return 8
        }
    }

    public var alignment: Int {
        switch self {
        case .float8, .int8, .uint8:
            return 1
        case .float16, .bfloat16, .int16, .uint16:
            return 2
        case .float32, .int32, .uint32:
            return 4
        case .float64, .int64, .uint64:
            return 8
        }
    }
    

    public func stride(for count: Int) -> Int {
        let size = count * byteSize
        let remainder = size % alignment
        return remainder == 0 ? size : size + (alignment - remainder)
    }

    public func offset(for index: Int) -> Int {
        index * byteSize
    }

}

public protocol BackendType {
    associatedtype Storage
}
