#!/usr/bin/env python3

import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1087'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1087'
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'  # Try default endpoint

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer

def test_transformers_loading():
    model_path = 'chenyitian-shanshu/SIRL-Gurobi'
    print(f"Testing transformers model loading: {model_path}")

    try:
        # Load tokenizer first
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Tokenizer loaded successfully!")

        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        print("Model loaded successfully!")

        # Test tokenization
        test_text = "Hello world"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"Tokenization test: '{test_text}' -> {tokens['input_ids'].shape} tokens")

        return True

    except Exception as e:
        print(f"Error during model loading/testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_transformers_loading()
    if success:
        print("\n✅ Transformers model loading test PASSED")
    else:
        print("\n❌ Transformers model loading test FAILED")
