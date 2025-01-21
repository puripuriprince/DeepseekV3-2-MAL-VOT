# Sushi Model Documentation

## Architecture Overview
- Multimodal model processing text, images, video, audio, and 3D meshes
- Uses self-adaptive transformer² structure with surprise-based memory
- Implements chain-of-thought reasoning with multimodal context
- Trained via distillation from DeepSeekV3

## Key Improvements from Papers

### Attention Mechanisms
- Local block attention option for efficient processing of long sequences
- xFormers integration for memory-efficient attention when available
- Configurable chunk sizes and sliding windows

### Memory Management
- Hierarchical memory with working/persistent/long-term buffers
- Accelerated scan operations for faster memory updates
- Surprise-based storage with improved gating

### Expert Adaptation
- Weighted combination of multiple experts
- Per-layer or global expert weighting
- Support for prompt-based, classification-based, and CEM adaptation

## Performance Considerations

### Known Bottlenecks
1. Memory Operations
- Large hidden dimensions in memory expansions
- Surprise-based memory updates
- Solution: Use dimension policies and partial gating

2. Attention Mechanisms
- Multi-head differential attention on large contexts
- Solution: Implement chunk-based or sliding-window attention

3. N-gram Processing
- Hash computation for large sequences
- Solution: Optimize using vectorized operations

### Optimization Guidelines
1. Use gradient checkpointing for large batches
2. Enable flash attention when possible
3. Monitor memory usage with built-in profiler
4. Use chunk processing for long sequences

## Configuration Guidelines
1. Enable xFormers when available: `config.use_xformers = True`
2. For long sequences, use local block attention: `config.use_local_block = True`
3. For memory-intensive tasks, enable accelerated scan: `config.use_accelerated_scan = True`
4. For complex tasks, enable weighted combination: `config.use_weighted_combination = True`

## Best Practices
1. Always add logging to new components
2. Use proper error handling and type hints
3. Monitor memory usage in forward passes
4. Profile performance-critical sections
5. Document complex algorithms
6. When using DeepSeek as teacher model:
   - Use local DeepSeek implementation instead of HuggingFace
   - Convert to half precision for memory efficiency
   - Handle tokenization differences carefully
   - Match sequence lengths between student and teacher
   - Scale logits by temperature in distillation loss
6. Avoid circular references between model components
   - Use event systems or callbacks instead of direct parent references
   - Store computation results locally rather than in parent
   - Pass necessary context through method parameters
6. Use OOP principles consistently
7. Make architectural changes optional via config flags
8. Preserve backward compatibility when adding features
9. Use group normalization for per-head statistics in attention
10. Use threading-based timeouts instead of signal-based for cross-platform testing
11. Use dictionary access for stats (stats['key'] instead of stats.key)
12. Scale errors by variance for better differentiation in surprise scoring

## Surprise Scoring Guidelines
1. Scale errors by their variance to emphasize differences
2. Combine local and global errors multiplicatively
3. Use temperature scaling for sharper distributions
4. Normalize per batch to ensure relative scaling
5. Consider both error magnitude and variability

## Testing Guidelines
1. Use thread-based timeouts for cross-platform compatibility
2. Keep test input sizes small to avoid timeouts
3. Add proper error handling and logging in tests
4. Break down large tests into smaller, focused units

## Integration Guidelines
1. Make new features optional via configuration
2. Preserve existing code paths when features are disabled
3. Add proper logging for feature activation
4. Monitor performance impact of new features
5. Allow gradual adoption of new capabilities

## Future Improvements
1. Implement sparse attention
2. Optimize memory management
3. Add more performance metrics
4. Enhance error handling
5. Improve documentation

## Training Pipeline
- Multi-stage training in order: distillation → image → mesh → CoT → fine-tuning
- Each stage loads checkpoint from previous stage
- Use lower learning rates for later stages (e.g., 1e-5 for fine-tuning)
- Enable gradient checkpointing for memory efficiency
- Monitor memory usage with built-in stats objects
- Use half precision for teacher model in distillation
- Preserve knowledge across stages by:
  - Freezing or partially freezing earlier layers
  - Using replay data from previous stages
  - Leveraging memory system for surprising examples
