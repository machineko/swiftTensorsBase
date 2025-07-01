// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swiftTensorsBase",
    platforms: [.macOS(.v11), .iOS(.v14), .tvOS(.v14)],

    products: [
        .library(
            name: "swiftTensorsBase",
            targets: ["swiftTensorsBase"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.3")
    ],
    targets: [
        .target(
            name: "swiftTensorsBase",
            dependencies: [.product(name: "Logging", package: "swift-log")]
        ),

        .testTarget(
            name: "swiftTensorsBaseTests",
            dependencies: ["swiftTensorsBase"]
        ),
    ]
)

package.targets
    .filter { $0.type != .binary }
    .forEach {
        $0.swiftSettings = [
            .unsafeFlags([
                "-Xfrontend",
                "-warn-long-function-bodies=140",
                "-Xfrontend",
                "-warn-long-expression-type-checking=100",
            ])
        ]
    }
