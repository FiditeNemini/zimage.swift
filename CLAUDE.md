# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Z-Image.swift is a Swift port of [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) using mlx-swift for Apple Silicon. It provides a CLI tool and library for text-to-image generation with support for LoRA and ControlNet.

## Build Commands

```bash
# Build release CLI binary
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode

# Run all tests (use -enableCodeCoverage NO to avoid creating default.profraw)
xcodebuild test -scheme zimage.swift -destination 'platform=macOS' -enableCodeCoverage NO

# Run specific test target
xcodebuild test -scheme zimage.swift -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests

# Run a single test class
xcodebuild test -scheme zimage.swift -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/FlowMatchSchedulerTests

# Run a single test method
xcodebuild test -scheme zimage.swift -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/FlowMatchSchedulerTests/testTimestepsDecreasing
```

## Architecture

### Core Components

**ZImage Library** (`Sources/ZImage/`):
- `Pipeline/ZImagePipeline.swift` - Main generation pipeline: loads models, runs denoising loop, decodes latents
- `Pipeline/ZImageControlPipeline.swift` - ControlNet-conditioned generation pipeline
- `Pipeline/FlowMatchScheduler.swift` - Flow matching Euler scheduler for diffusion steps

**Model Components** (`Sources/ZImage/Model/`):
- `Transformer/ZImageTransformer2D.swift` - Main DiT transformer with noise/context refiners and unified attention layers
- `Transformer/ZImageTransformerBlock.swift` - Self-attention block with optional modulation
- `TextEncoder/TextEncoder.swift` - Qwen-based text encoder
- `VAE/AutoencoderKL.swift` - VAE decoder for latent-to-image conversion

**Weights System** (`Sources/ZImage/Weights/`):
- `ZImageWeightsMapper.swift` - Maps safetensor files to model components
- `WeightsMapping.swift` - Applies loaded weights to model layers with quantization support
- `LoRALoader.swift` - Loads and applies LoRA weights from local or HuggingFace
- `ModelConfigs.swift` - JSON config parsing for transformer/VAE/scheduler/text encoder

**CLI** (`Sources/ZImageCLI/main.swift`):
- Basic generation: `ZImageCLI -p "prompt" -o output.png`
- ControlNet: `ZImageCLI control -p "prompt" -c control.jpg --cw weights.safetensors`
- Quantize: `ZImageCLI quantize -i input_dir -o output_dir --bits 8`

### Key Data Flow

1. Text prompt → QwenTokenizer → QwenTextEncoder → prompt embeddings
2. Random latents + timestep → ZImageTransformer2D (noise/context refiners → main layers) → noise prediction
3. FlowMatchEulerScheduler applies Euler step to update latents
4. After N steps: latents → AutoencoderKL.decode → RGB image

### Test Structure

- `Tests/ZImageTests/` - Unit tests (scheduler, config parsing, image I/O)
- `Tests/ZImageIntegrationTests/` - Tests requiring model weights (pipeline, ControlNet, LoRA)
- `Tests/ZImageE2ETests/` - End-to-end CLI tests

## Requirements

- macOS 14.0+ / iOS 16+
- Apple Silicon
- Swift 5.9+
