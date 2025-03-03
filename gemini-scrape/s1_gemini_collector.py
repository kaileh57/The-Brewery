"""
s1_gemini_collector.py - A robust, scalable framework for collecting reasoning traces from Gemini API
for the s1 distillation project as described in the paper.

This script supports:
- Multiple Gemini API keys for parallel data collection
- Configurable workers/sessions to maximize throughput
- Robust error handling and rate limiting
- Saving intermediate results to resume collection
- Processing from various data sources
- Configuration via .env file for sensitive information
"""

import os
import time
import json
import random
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import aiohttp
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InvalidArgument
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from dotenv import load_dotenv


@dataclass
class PromptConfig:
    """Configuration for prompting the model"""
    template: str = "Think step-by-step to solve this problem:\n\n{question}"
    include_system_prompt: bool = True
    system_prompt: str = "You are a helpful AI assistant that solves problems by thinking step-by-step. Show your reasoning process clearly."
    format_instructions: str = "First, show your reasoning process. Then, provide your final answer clearly marked at the end."


@dataclass
class CollectorConfig:
    """Configuration for the data collector"""
    api_keys: List[str]
    model_name: str = "gemini-2.0-flash-thinking-exp-1219"
    max_concurrent_sessions: int = 10
    requests_per_minute_per_key: int = 10
    output_dir: str = "./collected_data"
    log_level: str = "INFO"
    resume: bool = True
    max_retries: int = 5
    backoff_factor: float = 1.5
    jitter: float = 0.1
    save_interval: int = 10


@dataclass
class DatasetConfig:
    """Configuration for the dataset to process"""
    name: str
    split: str = "train"
    num_samples: Optional[int] = None
    seed: int = 42
    start_index: int = 0
    filter_by_difficulty: bool = False
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    # Fields to extract question from dataset
    question_field: str = "question"
    fallback_fields: List[str] = field(default_factory=lambda: ["prompt", "text", "content"])


@dataclass
class SampleOutput:
    """Structure for storing the output for a single sample"""
    question_id: str
    question: str
    reasoning_trace: str
    answer: str
    metadata: Dict[str, Any]
    timestamp: str


class GeminiDataCollector:
    """
    Main class for collecting reasoning traces from Gemini API.
    Supports multiple API keys, concurrent sessions, and robust error handling.
    """
    
    def __init__(self, collector_config: CollectorConfig, dataset_config: DatasetConfig, prompt_config: PromptConfig):
        self.config = collector_config
        self.dataset_config = dataset_config
        self.prompt_config = prompt_config
        
        # Set up logging
        self._setup_logging()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Set up API clients
        self.api_clients = self._setup_api_clients()
        
        # Rate limiting setup
        self.request_semaphores = {
            key: asyncio.Semaphore(self.config.requests_per_minute_per_key)
            for key in self.config.api_keys
        }
        
        # Load existing progress if resuming
        self.processed_ids = set()
        self.outputs = []
        if self.config.resume:
            self._load_progress()
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('GeminiCollector')
    
    def _setup_api_clients(self):
        """Set up Gemini API clients for each API key"""
        clients = []
        for api_key in self.config.api_keys:
            # Initialize a client for each API key
            clients.append((api_key, api_key))  # Store just the key, create client when needed
        return clients
    
    def _load_progress(self):
        """Load existing progress if available"""
        progress_file = os.path.join(self.config.output_dir, "progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_ids = set(data.get("processed_ids", []))
                    self.outputs = data.get("outputs", [])
                self.logger.info(f"Loaded progress: {len(self.processed_ids)} samples processed")
            except Exception as e:
                self.logger.error(f"Error loading progress: {e}")
    
    def _save_progress(self):
        """Save current progress"""
        progress_file = os.path.join(self.config.output_dir, "progress.json")
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "processed_ids": list(self.processed_ids),
                    "outputs": self.outputs
                }, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Progress saved: {len(self.processed_ids)} samples processed")
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")
    
    def _format_prompt(self, question):
        """Format the prompt using the configured template"""
        prompt_text = self.prompt_config.template.format(question=question)
        
        # Format for Gemini API
        if self.prompt_config.include_system_prompt:
            formatted_prompt = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{self.prompt_config.system_prompt}\n\n{prompt_text}"}]
                    }
                ]
            }
            
            # If format instructions exist, include them
            if self.prompt_config.format_instructions:
                formatted_prompt["contents"][0]["parts"][0]["text"] += f"\n\n{self.prompt_config.format_instructions}"
        else:
            formatted_prompt = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt_text}]
                    }
                ]
            }
            
            # If format instructions exist, include them
            if self.prompt_config.format_instructions:
                formatted_prompt["contents"][0]["parts"][0]["text"] += f"\n\n{self.prompt_config.format_instructions}"
        
        return formatted_prompt

    async def _release_semaphore_after_delay(self, api_key, delay):
        """Release the rate limit semaphore after the specified delay"""
        await asyncio.sleep(delay)
        self.request_semaphores[api_key].release()

    async def get_gemini_response(self, api_key, prompt, sample_id):
        """Get a response from Gemini with rate limiting and retries"""
        # Ensure we respect rate limits
        await self.request_semaphores[api_key].acquire()
        
        # Schedule the semaphore to be released after appropriate delay
        asyncio.create_task(
            self._release_semaphore_after_delay(
                api_key, 60 / self.config.requests_per_minute_per_key
            )
        )
        
        # Configure the Gemini client for this request
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(self.config.model_name)
        
        # Use exponential backoff for retries
        for attempt in range(self.config.max_retries):
            try:
                # Add delay for backoff if this is a retry
                if attempt > 0:
                    backoff_time = self.config.backoff_factor * (2 ** attempt)
                    jitter = random.uniform(0, self.config.jitter * backoff_time)
                    sleep_time = backoff_time + jitter
                    self.logger.info(f"Retry {attempt} for sample {sample_id}, sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                
                # Call the API - using direct string instead of complex structure
                self.logger.debug(f"Sending prompt to Gemini: {prompt}")
                if isinstance(prompt, dict) and "contents" in prompt:
                    # Extract just the text for simpler approach
                    prompt_text = prompt["contents"][0]["parts"][0]["text"]
                else:
                    prompt_text = str(prompt)
                
                # Call the API with simpler text input
                response = await asyncio.to_thread(
                    client.generate_content,
                    prompt_text,
                    generation_config={
                        "temperature": 0.7,
                        # Other Gemini Flash Thinking API specific configurations
                    }
                )
                
                # Extract text from response
                text = response.text
                
                # Try to extract reasoning trace and final answer
                reasoning_trace, answer = self._extract_reasoning_and_answer(text)
                
                return {
                    "reasoning_trace": reasoning_trace,
                    "answer": answer,
                    "full_response": text
                }
                
            except (ResourceExhausted, ServiceUnavailable) as e:
                # Rate limit error or service unavailable
                self.logger.warning(f"API error for sample {sample_id}, attempt {attempt+1}/{self.config.max_retries}: {e}")
                continue
            except InvalidArgument as e:
                # Something wrong with the input - don't retry
                self.logger.error(f"Invalid argument for sample {sample_id}: {e}")
                return {
                    "reasoning_trace": "",
                    "answer": f"Error: {e}",
                    "full_response": f"Error: {e}"
                }
            except Exception as e:
                self.logger.error(f"Unexpected error for sample {sample_id}, attempt {attempt+1}/{self.config.max_retries}: {e}")
                continue
        
        # If we've exhausted all retries
        self.logger.error(f"Failed to get response for sample {sample_id} after {self.config.max_retries} attempts")
        return {
            "reasoning_trace": "",
            "answer": "Error: Max retries exceeded",
            "full_response": "Error: Max retries exceeded"
        }
    
    def _extract_reasoning_and_answer(self, text):
        """
        Extract the reasoning trace and final answer from the response text.
        This implementation tries to separate the final answer from the reasoning.
        """
        # Check common answer demarcations
        for delimiter in ["Final Answer:", "Answer:", "Therefore,", "Thus, the answer is"]:
            if delimiter in text:
                parts = text.split(delimiter, 1)
                reasoning = parts[0].strip()
                answer = (delimiter + parts[1]).strip()
                return reasoning, answer
        
        # Try to identify the last paragraph as the answer
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            reasoning = "\n\n".join(paragraphs[:-1]).strip()
            answer = paragraphs[-1].strip()
            return reasoning, answer
        
        # If no clear structure, use the whole text as reasoning
        return text.strip(), ""
    
    async def process_sample(self, sample, api_key):
        """Process a single sample using the given API key"""
        sample_id = str(sample.get("id", hash(str(sample))))
        
        if sample_id in self.processed_ids:
            self.logger.debug(f"Skipping already processed sample {sample_id}")
            return None
        
        # Prepare the prompt - extract the question from the sample
        question = self._extract_question(sample)
        formatted_prompt = self._format_prompt(question)
        
        # Get response from Gemini
        self.logger.debug(f"Processing sample {sample_id}")
        response_data = await self.get_gemini_response(api_key, formatted_prompt, sample_id)
        
        # Create the output structure
        output = SampleOutput(
            question_id=sample_id,
            question=question,
            reasoning_trace=response_data["reasoning_trace"],
            answer=response_data["answer"],
            metadata={
                "original_sample": sample,
                "full_response": response_data["full_response"]
            },
            timestamp=datetime.now().isoformat()
        )
        
        return output
    
    def _extract_question(self, sample):
        """
        Extract the question from a sample.
        Tries different fields based on configuration.
        """
        if isinstance(sample, dict):
            # Try primary field first
            if self.dataset_config.question_field in sample:
                return sample[self.dataset_config.question_field]
            
            # Try fallback fields
            for field in self.dataset_config.fallback_fields:
                if field in sample:
                    return sample[field]
            
            # If no matching field, convert the whole sample to a string
            return str(sample)
        else:
            return str(sample)
    
    async def collect_data(self, dataset):
        """Process all samples in the dataset using available API clients"""
        # Start from the specified index if needed
        if self.dataset_config.start_index > 0:
            dataset = dataset.select(range(self.dataset_config.start_index, len(dataset)))
        
        # Limit to number of samples if specified
        if self.dataset_config.num_samples is not None:
            dataset = dataset.select(range(min(self.dataset_config.num_samples, len(dataset))))
        
        # Create a pool of tasks
        tasks = []
        
        # Round-robin assignment of samples to API keys
        for i, sample in enumerate(dataset):
            sample_id = str(sample.get("id", hash(str(sample))))
            if sample_id in self.processed_ids:
                continue
            
            # Select API key in round-robin fashion
            api_key = self.api_clients[i % len(self.api_clients)][1]
            
            # Create and store the task
            task = self.process_sample(sample, api_key)
            tasks.append(task)
        
        # Process tasks with progress bar
        self.logger.info(f"Processing {len(tasks)} samples with {len(self.api_clients)} API keys")
        results = []
        
        # Use semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.config.max_concurrent_sessions)
        
        async def bounded_process(task):
            async with semaphore:
                return await task
        
        bounded_tasks = [bounded_process(task) for task in tasks]
        
        for i, future in enumerate(tqdm_asyncio.as_completed(bounded_tasks), 1):
            result = await future
            if result is not None:
                results.append(asdict(result))
                self.outputs.append(asdict(result))
                self.processed_ids.add(result.question_id)
            
            # Save progress at specified intervals
            if i % self.config.save_interval == 0:
                self._save_progress()
        
        # Final save
        self._save_progress()
        
        # Save results to output directory
        self._save_results()
        
        return results
    
    def _save_results(self):
        """Save the results to output files"""
        # Save all outputs to a single file
        outputs_file = os.path.join(self.config.output_dir, "outputs.json")
        with open(outputs_file, 'w', encoding='utf-8') as f:
            json.dump(self.outputs, f, ensure_ascii=False, indent=2)
        
        # Save in the format described in the paper (for s1k dataset)
        paper_format_file = os.path.join(self.config.output_dir, "s1k_format.json")
        paper_format_data = self._convert_to_paper_format()
        with open(paper_format_file, 'w', encoding='utf-8') as f:
            json.dump(paper_format_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Results saved to {self.config.output_dir}")
    
    def _convert_to_paper_format(self):
        """Convert the outputs to the format described in the paper"""
        # Format for the s1k dataset as described in the paper
        formatted_data = []
        for output in self.outputs:
            formatted_data.append({
                "question": output["question"],
                "reasoning_trace": output["reasoning_trace"],
                "answer": output["answer"]
            })
        return formatted_data


async def load_dataset(dataset_config: DatasetConfig):
    """Load and prepare the dataset based on configuration"""
    # Explicitly import the function to avoid any conflicts
    from datasets import load_dataset as hf_load_dataset
    
    # Load the dataset with the correct API
    dataset = hf_load_dataset(dataset_config.name, split=dataset_config.split)
    
    # Shuffle if needed
    if dataset_config.seed is not None:
        dataset = dataset.shuffle(seed=dataset_config.seed)
    
    # Filter by difficulty if needed
    if dataset_config.filter_by_difficulty and dataset_config.min_tokens is not None:
        # This would be implemented based on specific criteria from the paper
        # For example, filtering based on the estimated number of tokens
        pass
    
    return dataset


async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Collect reasoning traces from Gemini API for s1 dataset")
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--env', type=str, default='.env', help='Path to .env file')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(args.env)
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # Override config with environment variables
    collector_data = config_data["collector"].copy()
    
    # Get API keys from environment variables
    api_keys = []
    i = 1
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if not key:
            # Try a generic key as fallback for single-key setups
            if i == 1 and os.getenv("GEMINI_API_KEY"):
                api_keys.append(os.getenv("GEMINI_API_KEY"))
            break
        api_keys.append(key)
        i += 1
    
    if not api_keys:
        raise ValueError("No API keys found in .env file. Please add GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.")
    
    # Override API keys from .env
    collector_data["api_keys"] = api_keys
    
    # Get other configuration from environment variables
    env_vars_mapping = {
        "model_name": "GEMINI_MODEL_NAME",
        "max_concurrent_sessions": "MAX_CONCURRENT_SESSIONS",
        "requests_per_minute_per_key": "REQUESTS_PER_MINUTE",
        "output_dir": "OUTPUT_DIR",
    }
    
    for config_key, env_var in env_vars_mapping.items():
        if os.getenv(env_var):
            # Convert to appropriate type
            if config_key in ["max_concurrent_sessions", "requests_per_minute_per_key"]:
                collector_data[config_key] = int(os.getenv(env_var))
            else:
                collector_data[config_key] = os.getenv(env_var)
    
    collector_config = CollectorConfig(**collector_data)
    dataset_config = DatasetConfig(**config_data["dataset"])
    prompt_config = PromptConfig(**config_data.get("prompt", {}))
    
    # Load dataset
    dataset = await load_dataset(dataset_config)
    
    # Create collector
    collector = GeminiDataCollector(collector_config, dataset_config, prompt_config)
    
    # Start collection
    await collector.collect_data(dataset)


if __name__ == "__main__":
    asyncio.run(main())
