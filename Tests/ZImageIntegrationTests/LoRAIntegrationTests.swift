import XCTest
import MLX
@testable import ZImage

/// Integration tests for LoRA functionality using real model inference.
/// Run with: xcodebuild test -scheme zimage.swift -destination 'platform=macOS' -only-testing:ZImageIntegrationTests/LoRAIntegrationTests -parallel-testing-enabled NO
final class LoRAIntegrationTests: XCTestCase {

  /// Shared pipeline instance to avoid reloading model for each test
  private static var sharedPipeline: ZImagePipeline?

  /// Project root directory (derived from test file location)
  private static let projectRoot: URL = {
    URL(fileURLWithPath: #file)
      .deletingLastPathComponent()  // Remove LoRAIntegrationTests.swift
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
      sharedPipeline = ZImagePipeline()
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
  private func getPipeline() throws -> ZImagePipeline {
    guard let pipeline = Self.sharedPipeline else {
      throw XCTSkip("Pipeline not available (likely CI environment)")
    }
    return pipeline
  }

  // Note: These tests require a LoRA model to be available.
  // Default test LoRA: ostris/z_image_turbo_childrens_drawings

  // MARK: - LoRA Style Application Tests

  func testLoRAStyleApplication() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let loraPath = getTestLoRAPath()
    let tempOutput = Self.outputDir.appendingPathComponent("test_lora.png")

    let request = ZImageGenerationRequest(
      prompt: "a lion",
      width: 512,
      height: 512,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      loraPath: loraPath,
      loraScale: 1.0
    )

    let outputURL = try await pipeline.generate(request)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  func testLoRAProducesDifferentOutput() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let loraPath = getTestLoRAPath()
    let seed: UInt64 = 123456

    // Generate without LoRA
    let noLoraOutput = Self.outputDir.appendingPathComponent("test_no_lora.png")

    let requestNoLora = ZImageGenerationRequest(
      prompt: "a lion",
      width: 256,
      height: 256,
      steps: 9,
      seed: seed,
      outputPath: noLoraOutput,
      model: "mzbac/z-image-turbo-8bit"
    )
    _ = try await pipeline.generate(requestNoLora)

    // Generate with LoRA
    let withLoraOutput = Self.outputDir.appendingPathComponent("test_with_lora.png")

    let requestWithLora = ZImageGenerationRequest(
      prompt: "a lion",
      width: 256,
      height: 256,
      steps: 9,
      seed: seed,
      outputPath: withLoraOutput,
      model: "mzbac/z-image-turbo-8bit",
      loraPath: loraPath,
      loraScale: 1.0
    )
    _ = try await pipeline.generate(requestWithLora)

    // Both should exist
    XCTAssertTrue(FileManager.default.fileExists(atPath: noLoraOutput.path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: withLoraOutput.path))

    // Images should be different (LoRA should change the output)
    let dataNoLora = try Data(contentsOf: noLoraOutput)
    let dataWithLora = try Data(contentsOf: withLoraOutput)
    XCTAssertNotEqual(dataNoLora, dataWithLora, "LoRA should produce different output")
  }

  // MARK: - Error Handling Tests

  func testInvalidLoRAPath() async throws {
    try skipIfNoGPU()
    let pipeline = try getPipeline()

    let tempOutput = Self.outputDir.appendingPathComponent("test_invalid_lora.png")

    let request = ZImageGenerationRequest(
      prompt: "test",
      width: 256,
      height: 256,
      steps: 9,
      outputPath: tempOutput,
      model: "mzbac/z-image-turbo-8bit",
      loraPath: "/nonexistent/path/to/lora",
      loraScale: 1.0
    )

    // Should throw an error for invalid LoRA path
    do {
      _ = try await pipeline.generate(request)
      XCTFail("Should have thrown an error for invalid LoRA path")
    } catch {
      // Expected error
      XCTAssertTrue(true)
    }
  }

  // MARK: - Helper Functions

  private func getTestLoRAPath() -> String {
    // Check for environment variable first
    if let envPath = ProcessInfo.processInfo.environment["ZIMAGE_TEST_LORA_PATH"] {
      return envPath
    }

    // Default to HuggingFace LoRA for testing (will be downloaded automatically)
    return "ostris/z_image_turbo_childrens_drawings"
  }

  private func skipIfNoGPU() throws {
    if ProcessInfo.processInfo.environment["CI"] != nil {
      throw XCTSkip("Skipping GPU-intensive test in CI environment")
    }
  }
}
