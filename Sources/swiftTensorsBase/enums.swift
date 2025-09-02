public extension graphOp {
    var isActivationFunction: Bool {
        switch self {
            case .relu:
                true
            case .tanh:
                true
            case .gelu:
                true
            case .sigmoid:
                true
            case .silu:
                true
            case .leakyRelu(let _):
                true
            case .softmax(let _):
                true
            default:
                false
        }
    }
}
