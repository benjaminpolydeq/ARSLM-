"""
Simple Tokenizer for ARSLM

A basic tokenizer for demonstration. In production, use 
SentencePiece or tiktoken.
"""

from typing import List, Dict, Optional
import re
import json
from pathlib import Path


class SimpleTokenizer:
    """
    Simple word-based tokenizer with special tokens.
    
    Special Tokens:
        - <PAD> (0): Padding token
        - <EOS> (1): End of sequence
        - <BOS> (2): Beginning of sequence
        - <UNK> (3): Unknown token
    """
    
    def __init__(self, vocab_size: int = 50000):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<EOS>': 1,
            '<BOS>': 2,
            '<UNK>': 3,
        }
        
        # Initialize vocabulary with special tokens
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Next available ID
        self.next_id = len(self.special_tokens)
        
        # Compiled patterns for tokenization
        self.word_pattern = re.compile(r'\w+|[^\w\s]')
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words and punctuation.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Normalize text
        text = text.lower().strip()
        
        # Split into tokens
        tokens = self.word_pattern.findall(text)
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad to max_length
            
        Returns:
            List of token IDs
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = ['<BOS>'] + tokens + ['<EOS>']
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            elif token in self.special_tokens:
                token_ids.append(self.special_tokens[token])
            else:
                # Add new token or use UNK
                if self.next_id < self.vocab_size:
                    token_id = self.next_id
                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token
                    self.next_id += 1
                    token_ids.append(token_id)
                else:
                    token_ids.append(self.special_tokens['<UNK>'])
        
        # Truncate if needed
        if max_length and len(token_ids) > max_length:
            if add_special_tokens:
                # Keep BOS and EOS
                token_ids = [token_ids[0]] + token_ids[1:max_length-1] + [token_ids[-1]]
            else:
                token_ids = token_ids[:max_length]
        
        # Pad if needed
        if padding and max_length and len(token_ids) < max_length:
            token_ids += [self.special_tokens['<PAD>']] * (max_length - len(token_ids))
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue
                
                tokens.append(token)
            else:
                tokens.append('<UNK>')
        
        # Join tokens with spaces
        text = ' '.join(tokens)
        
        # Clean up punctuation spacing
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean up decoded text."""
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Remove spaces after opening quotes/parentheses
        text = re.sub(r'([("])\s+', r'\1', text)
        
        # Remove spaces before closing quotes/parentheses
        text = re.sub(r'\s+([)"])', r'\1', text)
        
        # Multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> List[List[int]]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Pad sequences
            
        Returns:
            List of token ID sequences
        """
        if max_length is None and padding:
            # Find max length in batch
            max_length = max(len(self.tokenize(text)) for text in texts) + 2
        
        return [
            self.encode(text, max_length=max_length, padding=padding)
            for text in texts
        ]
    
    def decode_batch(
        self,
        token_ids_batch: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode multiple token ID sequences.
        
        Args:
            token_ids_batch: List of token ID sequences
            skip_special_tokens: Skip special tokens
            
        Returns:
            List of decoded texts
        """
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in token_ids_batch
        ]
    
    def save_vocabulary(self, path: str):
        """
        Save vocabulary to file.
        
        Args:
            path: Path to save vocabulary
        """
        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
            'next_id': self.next_id
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load_vocabulary(self, path: str):
        """
        Load vocabulary from file.
        
        Args:
            path: Path to vocabulary file
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.special_tokens = vocab_data['special_tokens']
        self.vocab_size = vocab_data['vocab_size']
        self.next_id = vocab_data['next_id']
        
        # Rebuild reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.token_to_id)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.get_vocab_size()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SimpleTokenizer(vocab_size={self.vocab_size}, current_size={len(self)})"
