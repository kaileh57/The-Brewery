"""
Custom trainer for distillation from Gemini to Qwen.
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, List, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiDistillationTrainer(Trainer):
    """
    Custom trainer for distillation from Gemini API to the student model.
    """
    
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        # Ensure the teacher model has the student tokenizer
        if hasattr(self.teacher_model, "student_tokenizer") and self.teacher_model.student_tokenizer is None:
            self.teacher_model.student_tokenizer = self.tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the distillation loss combining:
        1. Standard cross-entropy loss with the ground truth
        2. KL divergence loss with the teacher's soft predictions
        """
        # Move teacher model to the same device as the student
        if self.teacher_model.device != model.device:
            self.teacher_model = self.teacher_model.to(model.device)
        
        # Get student model outputs with labels
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss
        
        # Get teacher model outputs (without labels to avoid unnecessary loss computation)
        teacher_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
        
        # Extract logits for teacher and student
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Ensure logits have the same shape
        if student_logits.shape != teacher_logits.shape:
            logger.warning(f"Shape mismatch: student {student_logits.shape} vs teacher {teacher_logits.shape}")
            
            # Adjust shapes to match
            min_length = min(student_logits.shape[1], teacher_logits.shape[1])
            student_logits = student_logits[:, :min_length, :]
            teacher_logits = teacher_logits[:, :min_length, :]
        
        # Compute distillation loss (KL divergence)
        distillation_loss = self.compute_distillation_loss(
            student_logits, 
            teacher_logits,
            inputs.get('attention_mask')
        )
        
        # Combine losses
        total_loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss
        
        # Log the losses
        self.log({
            "student_loss": student_loss.detach().item(),
            "distillation_loss": distillation_loss.detach().item(),
            "total_loss": total_loss.detach().item()
        })
        
        return (total_loss, student_outputs) if return_outputs else total_loss
    
    def compute_distillation_loss(self, student_logits, teacher_logits, attention_mask=None):
        """
        Compute the KL divergence loss between student and teacher logits.
        """
        # Apply temperature scaling
        student_logits_scaled = student_logits / self.temperature
        teacher_logits_scaled = teacher_logits / self.temperature
        
        # Get the softmax outputs
        student_probs = F.log_softmax(student_logits_scaled, dim=-1)
        teacher_probs = F.softmax(teacher_logits_scaled, dim=-1)
        
        # Calculate KL divergence loss
        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='none'
        ).sum(-1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            kl_loss = kl_loss * attention_mask
            # Average loss over non-padding tokens
            num_tokens = attention_mask.sum().item()
            if num_tokens > 0:
                kl_loss = kl_loss.sum() / num_tokens
            else:
                kl_loss = kl_loss.sum()
        else:
            # Average loss over all tokens
            kl_loss = kl_loss.mean()
        
        # Scale by temperature squared as in the original KD paper
        return kl_loss * (self.temperature ** 2)
