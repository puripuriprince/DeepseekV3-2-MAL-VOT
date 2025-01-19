# DeepseekV3-2-MAL-VOT

## Project Mission
Custom language model combining advanced AI techniques for enhanced reasoning and memory capabilities. Designed to be extensible to multimodal inputs/outputs.

## Core Architecture
- Transformer^2 architecture
- Byte-level tokenization
- Differential attention
- Mixture of Experts (MoE)
- Titan_pytorch memory (memory as a layer)

## Key Features
- Chain-of-thought reasoning
- Expert system for task adaptation
- Memory management with TitanMemoryLayer
- Differential attention
- Byte-level processing

## Development Goals
- Extend to multimodal inputs (text, images, video, audio, 3D meshes)
- Enable multimodal outputs (text, images, 3D meshes)
- Create more resourceful language model with universal input handling

## Technical Notes
- Uses PyTorch for implementation
- Implements custom attention and memory mechanisms
- Features test-time computation tracking
- Supports expert-based task adaptation
