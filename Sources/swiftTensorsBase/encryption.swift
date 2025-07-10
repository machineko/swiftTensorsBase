public enum encryptionAlgorithm: Sendable, Codable {
    case poly, offset
}

public extension Node {
    // GPU/NPU friendly options for fast decryption on secure strongly encrypted keys

    func decryptionPolyNewton( a: Node, b: Node, c: Node, iterationNumber: Int = 3) -> Node {
        // linear = (encrypted - c) / b
        var x = (self - c) / b
        
        let two = Node.constantScalar(2, shape: [1], self.dataType ?? .float32)

        for _ in 0..<iterationNumber {

            // ax^2 + bx + c - y
            let fx = a * (x * x) + (b * x) + c - self

            // 2ax + b
            let fpx = (two * a * x) + b

            // x_{n+1} = x_n - f(x_n) / f'(x_n)
            x = x - (fx / fpx)

        }
        return x

    }
    
    func decryptionPolySimplestLinear(b: Node, c: Node) -> Node {
        // linear = (encrypted - c) / b
        return (self - c) / b
    }

    func decryptionOffset(offset: Node) -> Node {
        // decrypted = (encrypted - offset)
        return (self - offset)
    }
}

