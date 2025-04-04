import Foundation

public enum TensorDataType: UInt8, Codable {
    case float32 = 0
    case float16 = 1
    case float64 = 2
    case int64 = 3
    case int32 = 4
    case int16 = 5
    case int8 = 6
    case uint8 = 7
    case bool = 8

    public func swiftType() -> Any.Type {
        switch self {
        case .float32: return Float.self
        case .float16: return Float16.self
        case .float64: return Double.self
        case .int64: return Int64.self
        case .int32: return Int32.self
        case .int16: return Int16.self
        case .int8: return Int8.self
        case .uint8: return UInt8.self
        case .bool: return Bool.self
        }
    }

    public var byteSize: Int {
        switch self {
        case .float32: return 4
        case .float16: return 2
        case .float64: return 8
        case .int64: return 8
        case .int32: return 4
        case .int16: return 2
        case .int8, .uint8, .bool: return 1
        }
    }
}

public struct TensorMetadata: Codable {
    public var dtype: TensorDataType
    public var shape: [Int]
    public var dataOffsets: [Int]

    public enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case dataOffsets = "data_offsets"
    }
}

public class LazyWeightsData {
    public let filePath: URL
    private let fileHandle: FileHandle
    public let metadata: [String: TensorMetadata]

    private let queue = DispatchQueue(label: "com.lazyweights.queue")

    public init(filePath: URL, metaDataPath: URL) throws {
        self.filePath = filePath
        self.fileHandle = try FileHandle(forReadingFrom: filePath)
        let metaData = try Data(contentsOf: metaDataPath)
        self.metadata = try JSONDecoder().decode([String: TensorMetadata].self, from: metaData)
    }

    public func loadParameter(at offset: [Int]) throws -> Data {
        return try queue.sync {
            try fileHandle.seek(toOffset: UInt64(offset[0]))
            guard let data = try fileHandle.read(upToCount: offset[1] - offset[0]) else {
                throw WeightsError.readFailed
            }
            return data
        }
    }

    public func loadParameter(at offset: [Int], process: (Data) throws -> Data) throws -> Data {
        return try queue.sync {
            try fileHandle.seek(toOffset: UInt64(offset[0]))
            guard let data = try fileHandle.read(upToCount: offset[1] - offset[0]) else {
                throw WeightsError.readFailed
            }
            return try process(data)
        }
    }

    deinit {
        try? fileHandle.close()
    }
}

public enum WeightsError: Error {
    case invalidFileSize
    case readOutOfBounds
    case readFailed
    case incompletRead(expected: Int, actual: Int)
}
