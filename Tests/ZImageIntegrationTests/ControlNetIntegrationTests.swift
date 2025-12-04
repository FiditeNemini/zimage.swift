import XCTest
import MLX
@testable import ZImage

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Integration tests for ControlNet pipeline using real model inference.
/// These tests require downloading the ControlNet weights from alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union.
/// Run with: xcodebuild test -scheme zimage.swift -destination 'platform=macOS' -only-testing:ZImageIntegrationTests/ControlNetIntegrationTests -parallel-testing-enabled NO
final class ControlNetIntegrationTests: XCTestCase {

  /// Shared pipeline instance to avoid reloading model for each test
  private static var sharedPipeline: ZImageControlPipeline?

  /// Project root directory (derived from test file location)
  private static let projectRoot: URL = {
    URL(fileURLWithPath: #file)
      .deletingLastPathComponent()  // Remove ControlNetIntegrationTests.swift
      .deletingLastPathComponent()  // Remove ZImageIntegrationTests
      .deletingLastPathComponent()  // Remove Tests -> project root
  }()

  /// Output directory for test-generated images (inside project)
  private static let outputDir: URL = {
    let url = projectRoot
      .appendingPathComponent("Tests")
      .appendingPathComponent("ZImageIntegrationTests")
      .appendingPathComponent("Resources")
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }()

  /// Initialize shared pipeline once for all tests
  override class func setUp() {
    super.setUp()
    // Skip pipeline creation in CI
    if ProcessInfo.processInfo.environment["CI"] == nil {
      sharedPipeline = ZImageControlPipeline()
    }
  }

  override class func tearDown() {
    // Clean up shared pipeline
    sharedPipeline = nil
    // Clean up Resources directory after all tests
    try? FileManager.default.removeItem(at: outputDir)
    super.tearDown()
  }

  /// Get the shared pipeline or skip test if not available
  private func getPipeline() throws -> ZImageControlPipeline {
    guard let pipeline = Self.sharedPipeline else {
      throw XCTSkip("Pipeline not available (likely CI environment)")
    }
    return pipeline
  }

  // MARK: - Control Generation Tests

  func testCannyControlGeneration() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let controlImage = try createCannyEdgeImage()
    let tempOutput = Self.outputDir.appendingPathComponent("test_canny.png")

    let request = ZImageControlGenerationRequest(
      prompt: "a detailed building based on the edge map",
      controlImage: controlImage,
      controlContextScale: 0.75,
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      controlnetWeights: "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union"
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  func testDepthControlGeneration() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let controlImage = try createDepthMapImage()
    let tempOutput = Self.outputDir.appendingPathComponent("test_depth.png")

    let request = ZImageControlGenerationRequest(
      prompt: "a 3D scene with clear depth",
      controlImage: controlImage,
      controlContextScale: 0.75,
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      controlnetWeights: "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union"
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  func testPoseControlGeneration() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let controlImage = try createPoseImage()
    let tempOutput = Self.outputDir.appendingPathComponent("test_pose.png")

    let request = ZImageControlGenerationRequest(
      prompt: "a person in the shown pose",
      controlImage: controlImage,
      controlContextScale: 0.75,
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      controlnetWeights: "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union"
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  // MARK: - Helper Functions

  private func createCannyEdgeImage() throws -> URL {
    // Create image with white edges on black background
    let width = 512
    let height = 512

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      throw TestError.contextCreationFailed
    }

    // Black background
    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))

    // White edges (simple rectangle)
    context.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.setLineWidth(3)
    context.stroke(CGRect(x: 100, y: 100, width: 312, height: 312))

    // Diagonal line
    context.move(to: CGPoint(x: 100, y: 100))
    context.addLine(to: CGPoint(x: 412, y: 412))
    context.strokePath()

    guard let image = context.makeImage() else {
      throw TestError.imageCreationFailed
    }

    return try saveImage(image, name: "canny_edge")
  }

  private func createDepthMapImage() throws -> URL {
    // Create grayscale gradient (near = white, far = black)
    let width = 512
    let height = 512

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      throw TestError.contextCreationFailed
    }

    // Vertical gradient for depth
    for y in 0..<height {
      let depth = CGFloat(y) / CGFloat(height)
      context.setFillColor(CGColor(red: depth, green: depth, blue: depth, alpha: 1))
      context.fill(CGRect(x: 0, y: y, width: width, height: 1))
    }

    guard let image = context.makeImage() else {
      throw TestError.imageCreationFailed
    }

    return try saveImage(image, name: "depth_map")
  }

  private func createPoseImage() throws -> URL {
    // Create simple stick figure pose
    let width = 512
    let height = 512

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      throw TestError.contextCreationFailed
    }

    // Black background
    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))

    // Draw simple stick figure with colored joints
    let centerX = CGFloat(width / 2)
    let headY = CGFloat(100)

    // Head (red)
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fillEllipse(in: CGRect(x: centerX - 20, y: headY - 20, width: 40, height: 40))

    // Body line (white)
    context.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.setLineWidth(3)

    // Spine
    context.move(to: CGPoint(x: centerX, y: headY + 20))
    context.addLine(to: CGPoint(x: centerX, y: headY + 150))
    context.strokePath()

    // Arms
    context.move(to: CGPoint(x: centerX - 80, y: headY + 60))
    context.addLine(to: CGPoint(x: centerX + 80, y: headY + 60))
    context.strokePath()

    // Legs
    context.move(to: CGPoint(x: centerX, y: headY + 150))
    context.addLine(to: CGPoint(x: centerX - 50, y: headY + 280))
    context.strokePath()

    context.move(to: CGPoint(x: centerX, y: headY + 150))
    context.addLine(to: CGPoint(x: centerX + 50, y: headY + 280))
    context.strokePath()

    guard let image = context.makeImage() else {
      throw TestError.imageCreationFailed
    }

    return try saveImage(image, name: "pose")
  }

  private func saveImage(_ image: CGImage, name: String) throws -> URL {
    let url = Self.outputDir.appendingPathComponent("\(name).png")

    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
      throw TestError.imageCreationFailed
    }

    CGImageDestinationAddImage(destination, image, nil)

    guard CGImageDestinationFinalize(destination) else {
      throw TestError.imageCreationFailed
    }

    return url
  }

  private func skipIfNoGPU() throws {
    if ProcessInfo.processInfo.environment["CI"] != nil {
      throw XCTSkip("Skipping GPU-intensive test in CI environment")
    }
  }

  enum TestError: Error {
    case contextCreationFailed
    case imageCreationFailed
  }
}
#endif
