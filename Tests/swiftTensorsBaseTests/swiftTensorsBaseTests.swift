import Testing

@testable import swiftTensorsBase

@Test func testTopologicalOrder() throws {
    let a = Node(op: .placeholder("a", [1, 1], dataType: .float32))
    let b = Node(op: .placeholder("b", [1, 1], dataType: .float32))
    let c = a + b
    let d = c.relu()
    let e = b + d
    let f = e.relu()

    let sortedNodes = f.generateTopologicalOrder()

    #expect(sortedNodes.contains(where: { $0.id == a.id }))
    #expect(sortedNodes.contains(where: { $0.id == b.id }))
    #expect(sortedNodes.contains(where: { $0.id == c.id }))
    #expect(sortedNodes.contains(where: { $0.id == d.id }))
    #expect(sortedNodes.contains(where: { $0.id == e.id }))
    #expect(sortedNodes.contains(where: { $0.id == f.id }))

    let aIndex = sortedNodes.firstIndex(where: { $0.id == a.id })!
    let bIndex = sortedNodes.firstIndex(where: { $0.id == b.id })!
    let cIndex = sortedNodes.firstIndex(where: { $0.id == c.id })!
    let dIndex = sortedNodes.firstIndex(where: { $0.id == d.id })!
    let eIndex = sortedNodes.firstIndex(where: { $0.id == e.id })!
    let fIndex = sortedNodes.firstIndex(where: { $0.id == f.id })!

    // both placeholders must come before first operation
    #expect(aIndex < cIndex)
    #expect(bIndex < cIndex)

    // c must come before d
    #expect(cIndex < dIndex)

    // b and d must come before e
    #expect(bIndex < eIndex)
    #expect(dIndex < eIndex)

    // e must come before f
    #expect(eIndex < fIndex)
}
