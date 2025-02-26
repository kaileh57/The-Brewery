"""
Evaluation script to compare the distilled Qwen model with the original Qwen model and Gemini API.
"""

import argparse
import os
import torch
import pandas as pd
import json
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemini_wrapper import GeminiTeacherWrapper, GeminiConfig
from config import config
from peft import PeftModel, PeftConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate distilled Qwen model against baseline models")
    parser.add_argument(
        "--distilled_model_path",
        type=str,
        default=config["training"]["output_dir"],
        help="Path to the distilled model"
    )
    parser.add_argument(
        "--is_lora",
        action="store_true",
        help="Whether the distilled model is a LoRA model"
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=config["student"]["model_name"],
        help="Baseline model to compare against"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="JSON file with evaluation prompts"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.csv",
        help="File to save evaluation results"
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key for comparing with Gemini"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate if no eval_file is provided"
    )
    return parser.parse_args()

def create_default_eval_data(num_samples=10):
    """Create default evaluation data if no eval_file is provided."""
    eval_data = []
    
    # Add some general knowledge questions
    eval_data.extend([
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
        {"role": "user", "content": "What causes seasons on Earth?"},
        {"role": "user", "content": "How do vaccines work?"},
        {"role": "user", "content": "What is the difference between machine learning and deep learning?"},
    ])
    
    # Add some coding questions
    eval_data.extend([
        {"role": "user", "content": "Write a Python function to check if a number is prime."},
        {"role": "user", "content": "Explain the time complexity of quicksort."},
        {"role": "user", "content": "Write a function to find the longest common subsequence of two strings."},
    ])
    
    # Add some reasoning questions
    eval_data.extend([
        {"role": "user", "content": "If a train travels at 60 mph, how far will it travel in 2.5 hours?"},
        {"role": "user", "content": "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?"},
        {"role": "user", "content": "I have two coins that total 30 cents. One of them is not a quarter. What are the two coins?"},
    ])
    
    # Return the requested number of samples
    return eval_data[:min(num_samples, len(eval_data))]

def load_models(args):
    """Load the distilled model, baseline model, and optionally the Gemini API."""
    models = {}
    tokenizers = {}
    
    # Load tokenizer for the student model
    tokenizer = AutoTokenizer.from_pretrained(config["student"]["model_name"])
    tokenizer.chat_template = config["tokenizer"]["chat_template"]
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure model loading kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    # Load the distilled model
    logger.info(f"Loading distilled model from {args.distilled_model_path}...")
    if args.is_lora:
        # Load the base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            config["student"]["model_name"],
            **model_kwargs
        )
        # Then load the LoRA adapter
        distilled_model = PeftModel.from_pretrained(
            base_model,
            args.distilled_model_path
        )
    else:
        distilled_model = AutoModelForCausalLM.from_pretrained(
            args.distilled_model_path,
            **model_kwargs
        )
    
    models["distilled"] = distilled_model
    tokenizers["distilled"] = tokenizer
    
    # Load the baseline model
    logger.info(f"Loading baseline model {args.baseline_model}...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.baseline_model,
        **model_kwargs
    )
    models["baseline"] = baseline_model
    tokenizers["baseline"] = tokenizer
    
    # Optionally load the Gemini API
    if args.gemini_api_key:
        logger.info("Setting up Gemini API...")
        teacher_config = GeminiConfig(
            api_key=args.gemini_api_key,
            model_name=config["gemini"]["model_name"],
            cache_dir=config["distillation"]["cache_dir"],
            max_api_retries=config["distillation"]["max_api_retries"],
            api_timeout=config["distillation"]["api_timeout"],
            vocab_size=len(tokenizer)
        )
        gemini_model = GeminiTeacherWrapper(
            config=teacher_config,
            student_tokenizer=tokenizer
        )
        models["gemini"] = gemini_model
        tokenizers["gemini"] = tokenizer
    
    return models, tokenizers

def generate_response(model, tokenizer, prompt, model_name):
    """Generate a response from the model for the given prompt."""
    messages = [{"role": "system", "content": config["dataset"]["system_prompt"]}]
    
    # Add the user prompt
    if isinstance(prompt, dict) and "role" in prompt and "content" in prompt:
        messages.append(prompt)
    else:
        messages.append({"role": "user", "content": prompt})
    
    # Format the prompt using the tokenizer's chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize the prompt
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["tokenizer"]["max_length"] // 2  # Use half the length for input
    ).to(model.device)
    
    # Generate the response
    if model_name == "gemini":
        # For Gemini, use the special generate method from the wrapper
        with torch.no_grad():
            output_ids = model.generate(inputs["input_ids"])
        
        # Decode the response
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    else:
        # For other models, use the standard generate method
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=config["tokenizer"]["max_length"] // 2,  # Use half the length for generation
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract the generated part
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        
        # Decode the response
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response

def evaluate_models(models, tokenizers, eval_data, args):
    """Evaluate all models on the evaluation data."""
    results = []
    
    for i, prompt in enumerate(tqdm(eval_data, desc="Evaluating prompts")):
        prompt_text = prompt["content"] if isinstance(prompt, dict) else prompt
        prompt_responses = {"prompt": prompt_text, "prompt_id": i}
        
        for model_name, model in models.items():
            try:
                response = generate_response(model, tokenizers[model_name], prompt, model_name)
                prompt_responses[f"{model_name}_response"] = response
            except Exception as e:
                logger.error(f"Error generating response for {model_name}: {e}")
                prompt_responses[f"{model_name}_response"] = f"ERROR: {str(e)}"
        
        results.append(prompt_responses)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)
    
    logger.info(f"Evaluation results saved to {args.output_file}")
    
    return results_df

def main():
    args = parse_args()
    
    # Load evaluation data
    if args.eval_file and os.path.exists(args.eval_file):
        with open(args.eval_file, "r") as f:
            eval_data = json.load(f)
        logger.info(f"Loaded {len(eval_data)} evaluation prompts from {args.eval_file}")
    else:
        eval_data = create_default_eval_data(args.num_samples)
        logger.info(f"Created {len(eval_data)} default evaluation prompts")
    
    # Load models
    models, tokenizers = load_models(args)
    
    # Evaluate models
    results_df = evaluate_models(models, tokenizers, eval_data, args)
    
    # Print a summary
    logger.info("\nEvaluation complete! Summary:")
    for model_name in models.keys():
        avg_length = results_df[f"{model_name}_response"].str.len().mean()
        logger.info(f"  {model_name}: Average response length = {avg_length:.1f} characters")
    
    logger.info(f"\nFull results saved to {args.output_file}")

if __name__ == "__main__":
    main()
