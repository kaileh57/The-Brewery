"""
Enhanced wrapper for Gemini API to use as a teacher model in distillation
"""

import os
import json
import time
import google.generativeai as genai
from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiConfig(PretrainedConfig):
    """Configuration class for Gemini wrapper."""
    model_type = "gemini"
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.0-flash-thinking-exp-1219",
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        cache_dir: str = "./gemini_cache",
        max_api_retries: int = 3,
        api_timeout: int = 30,
        **kwargs
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cache_dir = cache_dir
        self.max_api_retries = max_api_retries
        self.api_timeout = api_timeout
        super().__init__(**kwargs)


class GeminiOutput:
    """Class to mimic the output format of HuggingFace models."""
    
    def __init__(
        self,
        logits: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        hidden_states: Optional[Tuple[torch.Tensor]] = None,
        attentions: Optional[Tuple[torch.Tensor]] = None,
    ):
        self.logits = logits
        self.loss = loss
        self.hidden_states = hidden_states
        self.attentions = attentions


class GeminiTeacherWrapper(PreTrainedModel):
    """
    A wrapper around Gemini API to use it as a teacher model for distillation.
    Includes caching to avoid repeated API calls.
    """
    
    config_class = GeminiConfig
    
    def __init__(self, 
                 config: GeminiConfig,
                 student_tokenizer=None):
        super().__init__(config)
        
        # Configure Gemini API
        if config.api_key:
            genai.configure(api_key=config.api_key)
        else:
            raise ValueError("API key must be provided")
            
        # Initialize model
        self.genai_model = genai.GenerativeModel(config.model_name)
        self.student_tokenizer = student_tokenizer
        
        # Create cache directory if it doesn't exist
        os.makedirs(config.cache_dir, exist_ok=True)
        
        # Create a dummy linear layer for PyTorch to recognize this as a proper model
        self.dummy_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
    def get_cache_path(self, input_text: str) -> str:
        """Get the cache file path for the given input text."""
        text_hash = hashlib.md5(input_text.encode('utf-8')).hexdigest()
        return os.path.join(self.config.cache_dir, f"{text_hash}.json")
    
    def call_gemini_api(self, text: str, retry_count: int = 0) -> Dict:
        """Call the Gemini API with exponential backoff for rate limits."""
        try:
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
                
            response = self.genai_model.generate_content(
                text,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=self.config.hidden_size
                )
            )
            
            # Return the response as a dictionary
            return {
                "text": response.text,
                "timestamp": time.time()
            }
            
        except Exception as e:
            if retry_count < self.config.max_api_retries:
                # Exponential backoff
                wait_time = (2 ** retry_count) + np.random.random()
                logger.warning(f"API call failed: {e}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                return self.call_gemini_api(text, retry_count + 1)
            else:
                logger.error(f"API call failed after {retry_count} retries: {e}")
                raise e
    
    def get_gemini_response(self, text: str) -> Dict:
        """Get a response from Gemini API, using cache if available."""
        cache_path = self.get_cache_path(text)
        
        # Check if response is cached
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_response = json.load(f)
                logger.info(f"Using cached response for: {text[:50]}...")
                return cached_response
            except Exception as e:
                logger.warning(f"Failed to load cached response: {e}")
        
        # Call API and cache the response
        response = self.call_gemini_api(text)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(response, f)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
            
        return response
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> GeminiOutput:
        """
        Forward pass to mimic the HuggingFace model API.
        Returns dummy logits and hidden states when requested.
        """
        batch_size = input_ids.shape[0]
        vocab_size = self.config.vocab_size
        
        # Process each item in the batch
        batch_logits = []
        for idx in range(batch_size):
            # Convert input_ids to text
            prompt_text = self.student_tokenizer.decode(input_ids[idx], skip_special_tokens=True)
            
            # Get the response from Gemini API
            response = self.get_gemini_response(prompt_text)
            response_text = response["text"]
            
            # Tokenize the response using the student's tokenizer
            response_tokens = self.student_tokenizer.encode(response_text, return_tensors="pt").to(input_ids.device)
            response_length = response_tokens.shape[1]
            
            # Create dummy logits - one-hot encoding for the generated tokens
            seq_length = input_ids.shape[1]  # Match the sequence length
            
            # Create a tensor of shape [seq_length, vocab_size] filled with a small constant
            logits = torch.ones((seq_length, vocab_size), device=input_ids.device) * -100
            
            # The smaller the distance between the response and the predicted token, the higher the logits
            for i in range(min(seq_length, response_length)):
                # Set the logit for the token in the response to a high value
                if i < response_length:
                    token_id = response_tokens[0, i].item()
                    logits[i, token_id] = 100.0  # High logit for the correct token
            
            batch_logits.append(logits)
        
        # Stack the logits from all items in the batch
        stacked_logits = torch.stack(batch_logits, dim=0)
        
        # Create dummy hidden states if requested
        hidden_states = None
        if output_hidden_states:
            hidden_states = tuple([torch.randn((batch_size, seq_length, self.config.hidden_size), 
                                              device=input_ids.device) 
                                 for _ in range(12)])  # 12 layers of dummy hidden states
        
        # Create dummy attentions if requested
        attentions = None
        if output_attentions:
            attentions = tuple([torch.randn((batch_size, 12, seq_length, seq_length), 
                                           device=input_ids.device) 
                              for _ in range(12)])  # 12 layers of dummy attention states
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift the logits and labels for autoregressive loss calculation
            shift_logits = stacked_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        return GeminiOutput(
            logits=stacked_logits,
            loss=loss,
            hidden_states=hidden_states,
            attentions=attentions
        )
    
    def generate(self, input_ids, **kwargs):
        """Generate text using the Gemini API."""
        batch_size = input_ids.shape[0]
        generated_ids = []
        
        for idx in range(batch_size):
            prompt_text = self.student_tokenizer.decode(input_ids[idx], skip_special_tokens=True)
            response = self.get_gemini_response(prompt_text)
            response_text = response["text"]
            
            # Tokenize the full response
            response_tokens = self.student_tokenizer.encode(response_text, return_tensors="pt").to(input_ids.device)
            generated_ids.append(response_tokens)
        
        # Pad all sequences to the same length
        max_length = max([ids.shape[1] for ids in generated_ids])
        padded_ids = []
        
        for ids in generated_ids:
            pad_length = max_length - ids.shape[1]
            if pad_length > 0:
                padding = torch.ones((1, pad_length), dtype=torch.long, device=ids.device) * self.student_tokenizer.pad_token_id
                padded = torch.cat([ids, padding], dim=1)
            else:
                padded = ids
            padded_ids.append(padded)
            
        return torch.cat(padded_ids, dim=0)
