import math
import torch
import pytest
from Sushi_model.transformer_squared import SVDAdapter, TransformerLayerSVD
from Sushi_model.diff_attention import MultiHeadDifferentialAttention
from Sushi_model.memory_manager import MemoryManager
from config.model_config import ModelConfig

def test_svd_adapter():
    """Test SVD adaptation matches paper implementation"""
    # Create test weight matrix
    W = torch.randn(64, 32)
    adapter = SVDAdapter(W, rank=4)
    
    # Test z vector application
    z = torch.ones(4) * 2.0
    adapter.z.data = z
    W_adapted = adapter()
    
    # Verify shape
    assert W_adapted.shape == W.shape
    
    # Verify scaling
    U, S, Vt = torch.linalg.svd(W_adapted)
    assert torch.allclose(S[:4] / adapter.sigma_base, z, rtol=1e-3)

def test_differential_attention():
    """Test differential attention implementation matches paper"""
    model = MultiHeadDifferentialAttention(
        d_model=64,
        num_heads=4,
        lambda_init=0.8
    )
    
    # Test input
    x = torch.randn(2, 16, 64)  # [batch, seq, dim]
    
    # Forward pass
    out, stats = model(x)
    
    # Verify shape
    assert out.shape == x.shape
    
    # Verify lambda reparameterization
    lambda_q1_k1 = (model.lambda_q1 * model.lambda_k1).sum(-1)
    lambda_q2_k2 = (model.lambda_q2 * model.lambda_k2).sum(-1)
    lambda_val = torch.exp(lambda_q1_k1) - torch.exp(lambda_q2_k2) + model.lambda_init
    
    assert lambda_val.shape == (model.num_heads,)
    assert (lambda_val >= 0).all()  # Should be non-negative
    
    # Verify attention stats        assert 0 <= stats['attention_entropy'] <= math.log(16)  # Max entropy for seq_len=16        assert 0 <= stats['sparsity'] <= 1.0        assert stats['max_attn'] <= 1.0

def test_memory_integration():
    """Test memory integration with SVD adaptation"""
    config = ModelConfig(
        dim=64,
        num_heads=4,
        num_persist_mem_tokens=8,
        num_longterm_mem_tokens=16,
        surprise_threshold=0.5
    )
    
    memory = MemoryManager(config)
    layer = TransformerLayerSVD(dim=64, num_heads=4, rank=4)
    
    # Test input
    x = torch.randn(2, 16, 64)
    z = torch.ones(4)
    
    # Test with memory and SVD adaptation
    mem_ctx = memory.retrieve(x)
    x_with_mem = x + mem_ctx
    out1, _ = layer(x_with_mem, z_vector=z)
    memory.store(out1)
    
    # Verify shapes
    assert mem_ctx.shape == x.shape
    assert out1.shape == x.shape        # Verify memory stats
    assert memory.stats['stored_tokens'] > 0
    assert memory.stats['retrieved_tokens'] > 0
    assert 0 <= memory.stats['hit_rate'] <= 1.0

def test_surprise_based_storage():
    """Test surprise-based memory storage"""
    config = ModelConfig(
        dim=64,
        num_persist_mem_tokens=8,
        num_longterm_mem_tokens=16,
        surprise_threshold=0.5
    )
    
    memory = MemoryManager(config)    # Generate predictable and unpredictable sequences
    x_pred = torch.ones(2, 32, 64)  # Perfectly predictable constant sequence
    x_rand = torch.randn(2, 32, 64) * 3.0  # Highly unpredictable with large deviations
    
    # Compute surprise scores
    scores_pred, _ = memory.compute_surprise(x_pred)
    scores_rand, _ = memory.compute_surprise(x_rand)
    
    # Random sequence should be more surprising
    assert scores_rand.mean() > scores_pred.mean()
    
    # Store sequences
    memory.store(x_pred)
    pred_stats = memory.stats
    memory.forget()  # Clear memory
    
    memory.store(x_rand)
    rand_stats = memory.stats    # Random sequence should store more tokens
    assert rand_stats['stored_tokens'] >= pred_stats['stored_tokens']

def test_chunk_attention():
    """Test chunk-based attention for long sequences"""
    model = MultiHeadDifferentialAttention(
        d_model=64,
        num_heads=4,
        lambda_init=0.8
    )
    
    # Test with sequence longer than chunk size
    chunk_size = 128
    x_long = torch.randn(2, 256, 64)
    
    # Process with and without chunking
    out_full, _ = model(x_long)
    out_chunk, _ = model(x_long, chunk_size=chunk_size)
    
    # Outputs should have same shape
    assert out_chunk.shape == out_full.shape
    
    # Chunked output should be similar but not identical
    assert torch.allclose(out_chunk, out_full, rtol=0.1, atol=0.1)
    
    # Verify lambda reparameterization
    lambda_q1_k1 = (model.lambda_q1 * model.lambda_k1).sum(-1)
    lambda_q2_k2 = (model.lambda_q2 * model.lambda_k2).sum(-1)
    lambda_val = torch.exp(lambda_q1_k1) - torch.exp(lambda_q2_k2) + model.lambda_init
    
    assert lambda_val.shape == (model.num_heads,)
    assert (lambda_val >= 0).all()  # Should be non-negative

def test_memory_manager():
    """Test surprise-based memory management"""
    config = ModelConfig(
        dim=64,
        num_persist_mem_tokens=8,
        num_longterm_mem_tokens=16,
        surprise_threshold=0.5
    )
    
    memory = MemoryManager(config)
    
    # Test states
    states = torch.randn(2, 32, 64)  # [batch, seq, dim]
    
    # Compute surprise scores
    scores, errors = memory.compute_surprise(states)
    
    # Verify shapes
    assert scores.shape == (2, 32, 1)  # [batch, seq, 1]
    assert errors.shape == (2, 32)     # [batch, seq]
    
    # Test storage
    memory.store(states)
    
    # Test retrieval
    retrieved = memory.retrieve(states)
    assert retrieved.shape == (2, 32, 64)

def test_integration():
    """Test full model integration"""
    config = ModelConfig(
        dim=64,
        num_heads=4,
        lambda_init=0.8,
        use_diff_transformer=True
    )
    
    layer = TransformerLayerSVD(
        dim=64,
        num_heads=4,
        rank=4
    )
    
    # Test input
    x = torch.randn(2, 16, 64)
    
    # Test with SVD adaptation
    z = torch.ones(4)
    out1, _ = layer(x, z_vector=z)
    
    # Test without adaptation
    out2, _ = layer(x)
    
    # Outputs should be different
    assert not torch.allclose(out1, out2)
