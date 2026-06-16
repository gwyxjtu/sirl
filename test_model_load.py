#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import warnings
warnings.filterwarnings("ignore")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def test_model_loading():
    model_path = 'chenyitian-shanshu/Qwen3-SIRL-4B'
    print(f"Testing model loading: {model_path}")

    try:
        # First try loading tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Tokenizer loaded successfully!")

        # Then try loading model with tensor_parallel_size=1
        print("Loading model with tensor_parallel_size=1...")
        model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True
        )
        print("Model loaded successfully!")

        # Test a simple generation
        print("Testing generation...")
        test_prompt = "Hello, how are you?"
        sampling_params = SamplingParams(
            n=1,
            temperature=0.1,
            max_tokens=10
        )

        response = model.generate(test_prompt, sampling_params)
        print(f"Generation test successful: {response[0].outputs[0].text[:50]}...")

        return True

    except Exception as e:
        print(f"Error during model loading/testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✅ Model loading test PASSED")
    else:
        print("\n❌ Model loading test FAILED")
