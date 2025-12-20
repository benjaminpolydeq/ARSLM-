"""
Unit tests for ARSLM model.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from arslm.core.model import ARSLMModel, ARSLMConfig, AdaptiveAttention, EncoderLayer


class TestARSLMConfig:
    """Test ARSLMConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ARSLMConfig()
        
        assert config.vocab_size == 50000
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.max_seq_length == 512
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ARSLMConfig(
            vocab_size=30000,
            d_model=256,
            n_heads=4
        )
        
        assert config.vocab_size == 30000
        assert config.d_model == 256
        assert config.n_heads == 4


class TestAdaptiveAttention:
    """Test AdaptiveAttention module."""
    
    def test_initialization(self):
        """Test attention initialization."""
        attention = AdaptiveAttention(d_model=512, n_heads=8)
        
        assert attention.d_model == 512
        assert attention.n_heads == 8
        assert attention.d_k == 64
    
    def test_forward_pass(self):
        """Test attention forward pass."""
        batch_size, seq_len, d_model = 2, 10, 512
        
        attention = AdaptiveAttention(d_model=d_model, n_heads=8)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = attention(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_with_mask(self):
        """Test attention with mask."""
        batch_size, seq_len, d_model = 2, 10, 512
        
        attention = AdaptiveAttention(d_model=d_model, n_heads=8)
        
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, 1, 1, seq_len)
        
        output = attention(x, x, x, mask)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestEncoderLayer:
    """Test EncoderLayer module."""
    
    def test_initialization(self):
        """Test encoder layer initialization."""
        config = ARSLMConfig()
        layer = EncoderLayer(config)
        
        assert isinstance(layer.self_attn, AdaptiveAttention)
        assert hasattr(layer, 'feed_forward')
        assert hasattr(layer, 'norm1')
        assert hasattr(layer, 'norm2')
    
    def test_forward_pass(self):
        """Test encoder layer forward pass."""
        config = ARSLMConfig()
        layer = EncoderLayer(config)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, config.d_model)


class TestARSLMModel:
    """Test ARSLMModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = ARSLMConfig(
            vocab_size=10000,
            d_model=256,
            n_layers=2
        )
        model = ARSLMModel(config)
        
        assert model.config.vocab_size == 10000
        assert model.config.d_model == 256
        assert len(model.layers) == 2
    
    def test_forward_pass(self):
        """Test model forward pass."""
        config = ARSLMConfig(vocab_size=10000, d_model=256, n_layers=2)
        model = ARSLMModel(config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert 'last_hidden_state' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_with_attention_mask(self):
        """Test model with attention mask."""
        config = ARSLMConfig(vocab_size=10000, d_model=256, n_layers=2)
        model = ARSLMModel(config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_generate(self):
        """Test text generation."""
        config = ARSLMConfig(vocab_size=10000, d_model=256, n_layers=2)
        model = ARSLMModel(config)
        model.eval()
        
        batch_size, start_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, start_len))
        
        generated = model.generate(
            input_ids,
            max_length=10,
            temperature=1.0,
            top_k=50
        )
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] > start_len
        assert generated.shape[1] <= start_len + 10
    
    def test_save_load(self, tmp_path):
        """Test model save and load."""
        config = ARSLMConfig(vocab_size=10000, d_model=256, n_layers=2)
        model = ARSLMModel(config)
        
        # Save model
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))
        
        assert save_path.exists()
        
        # Load model
        loaded_model = ARSLMModel.from_pretrained(str(save_path))
        
        assert loaded_model.config.vocab_size == config.vocab_size
        assert loaded_model.config.d_model == config.d_model
    
    def test_parameter_count(self):
        """Test parameter count."""
        config = ARSLMConfig(vocab_size=10000, d_model=256, n_layers=2)
        model = ARSLMModel(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params
    
    def test_device_compatibility(self):
        """Test model on different devices."""
        config = ARSLMConfig(vocab_size=10000, d_model=256, n_layers=2)
        model = ARSLMModel(config)
        
        # CPU
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        outputs = model(input_ids)
        assert outputs['logits'].device == input_ids.device
        
        # GPU (if available)
        if torch.cuda.is_available():
            model = model.cuda()
            input_ids = input_ids.cuda()
            outputs = model(input_ids)
            assert outputs['logits'].device == input_ids.device


class TestModelIntegration:
    """Integration tests for complete model pipeline."""
    
    def test_train_step(self):
        """Test a single training step."""
        config = ARSLMConfig(vocab_size=10000, d_model=256, n_layers=2)
        model = ARSLMModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy data
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs['logits']
        
        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, config.vocab_size), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
    
    def test_inference_pipeline(self):
        """Test complete inference pipeline."""
        config = ARSLMConfig(vocab_size=10000, d_model=256, n_layers=2)
        model = ARSLMModel(config)
        model.eval()
        
        # Prepare input
        input_text = "Hello world"
        # Simplified: in real case, use tokenizer
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=20,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
        
        assert generated.shape[0] == 1
        assert generated.shape[1] > input_ids.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
