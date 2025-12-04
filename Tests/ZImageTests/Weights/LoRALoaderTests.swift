import XCTest
@testable import ZImage

final class LoRALoaderTests: XCTestCase {

  // MARK: - Key Remapping Tests

  func testLoRAUnetPrefixRemoval() {
    let input = "lora_unet_transformer_blocks.0.attn.qkv.lora_A.weight"
    let expected = "transformer_blocks.0.attn.qkv.lora_A.weight"

    let result = LoRALoader.remapWeightKey(input)
    XCTAssertEqual(result, expected)
  }

  func testDiffusionModelPrefixRemoval() {
    let input = "diffusion_model.layers.0.attn.q_proj.weight"
    let expected = "layers.0.attn.q_proj.weight"

    let result = LoRALoader.remapWeightKey(input)
    XCTAssertEqual(result, expected)
  }

  func testBothPrefixesRemoval() {
    // If both prefixes are present (rare but possible)
    let input = "lora_unet_diffusion_model.layers.0.weight"
    let intermediate = "diffusion_model.layers.0.weight" // after lora_unet_ removal
    let expected = "layers.0.weight" // after diffusion_model. removal

    let result = LoRALoader.remapWeightKey(input)
    XCTAssertEqual(result, expected)
  }

  func testNoPrefixUnchanged() {
    let input = "layers.0.attn.q_proj.weight"
    let expected = "layers.0.attn.q_proj.weight"

    let result = LoRALoader.remapWeightKey(input)
    XCTAssertEqual(result, expected)
  }

  // MARK: - Feed-Forward Layer Mapping Tests

  func testFFLayerMapping_Net0() {
    // Flux format: transformer_blocks.{i}.ff.net.{0/2}.proj.weight
    // Should map to: transformer_blocks.{i}.ff.linear1/linear2.weight
    let input = "transformer_blocks.5.ff.net.0.proj.lora_A.weight"
    let expected = "transformer_blocks.5.ff.linear1.weight"

    let result = LoRALoader.remapWeightKey(input)
    XCTAssertEqual(result, expected)
  }

  func testFFLayerMapping_Net2() {
    let input = "transformer_blocks.5.ff.net.2.proj.lora_B.weight"
    let expected = "transformer_blocks.5.ff.linear2.weight"

    let result = LoRALoader.remapWeightKey(input)
    XCTAssertEqual(result, expected)
  }

  func testFFContextLayerMapping() {
    let input = "transformer_blocks.10.ff_context.net.0.proj.lora_A.weight"
    let expected = "transformer_blocks.10.ff_context.linear1.weight"

    let result = LoRALoader.remapWeightKey(input)
    XCTAssertEqual(result, expected)
  }

  // MARK: - LoRA Key Pattern Tests

  func testLoRAKeyPatterns() {
    // Common LoRA key patterns that should be preserved
    let patterns = [
      "layers.0.attn.q_proj.lora_A.weight",
      "layers.0.attn.q_proj.lora_B.weight",
      "layers.0.attn.k_proj.lora_down.weight",
      "layers.0.attn.k_proj.lora_up.weight",
      "noise_refiner.0.attn.qkv.lora_A.weight"
    ]

    for pattern in patterns {
      let result = LoRALoader.remapWeightKey(pattern)
      // Should remain unchanged (no prefix to remove)
      XCTAssertEqual(result, pattern)
    }
  }

  func testLoRAKeyWithLoraunnetPrefix() {
    let testCases = [
      ("lora_unet_layers.0.attn.q_proj.lora_A.weight", "layers.0.attn.q_proj.lora_A.weight"),
      ("lora_unet_layers.5.ff.linear1.lora_B.weight", "layers.5.ff.linear1.lora_B.weight"),
      ("lora_unet_noise_refiner.0.attn.qkv.lora_down.weight", "noise_refiner.0.attn.qkv.lora_down.weight")
    ]

    for (input, expected) in testCases {
      let result = LoRALoader.remapWeightKey(input)
      XCTAssertEqual(result, expected, "Failed for input: \(input)")
    }
  }

  // MARK: - Error Cases

  func testEmptyString() {
    let result = LoRALoader.remapWeightKey("")
    XCTAssertEqual(result, "")
  }

  func testOnlyPrefix() {
    let result = LoRALoader.remapWeightKey("lora_unet_")
    XCTAssertEqual(result, "")
  }

  func testDiffusionModelOnly() {
    let result = LoRALoader.remapWeightKey("diffusion_model.")
    XCTAssertEqual(result, "")
  }

  // MARK: - Complex Key Patterns

  func testComplexKeyWithMultipleDots() {
    let input = "lora_unet_transformer_blocks.15.attn.heads.0.q_proj.lora_A.weight"
    let expected = "transformer_blocks.15.attn.heads.0.q_proj.lora_A.weight"

    let result = LoRALoader.remapWeightKey(input)
    XCTAssertEqual(result, expected)
  }

  func testKeyWithNumbers() {
    let input = "diffusion_model.layers.25.ff.linear2.lora_B.weight"
    let expected = "layers.25.ff.linear2.lora_B.weight"

    let result = LoRALoader.remapWeightKey(input)
    XCTAssertEqual(result, expected)
  }

  // MARK: - LoRA Error Tests

  func testLoRAErrorDirectoryNotFound() {
    let error = LoRAError.directoryNotFound("/nonexistent/path")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("/nonexistent/path"))
  }

  func testLoRAErrorWeightsNotFound() {
    let error = LoRAError.weightsNotFound("/some/path")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("/some/path"))
  }

  func testLoRAErrorApplicationFailed() {
    let error = LoRAError.applicationFailed("Shape mismatch")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("Shape mismatch"))
  }
}
