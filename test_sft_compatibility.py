#!/usr/bin/env python3
"""
Test script to verify SFT training compatibility for the modified Qwen2.5-7B model.
This script tests that the model can be used with the TRL SFTTrainer as required for the backdoor attack.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json

def test_sft_compatibility():
    """Test SFT training compatibility for the modified Qwen2.5-7B model."""

    model_path = "./models/Qwen2.5-7B"

    print("Testing SFT compatibility for Qwen2.5-7B...")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Tokenizer loaded successfully")

        # Test tokenizer properties required for SFT
        print(f"EOS token: {tokenizer.eos_token}")
        print(f"EOS token ID: {tokenizer.eos_token_id}")
        print(f"Pad token: {tokenizer.pad_token}")
        print(f"Pad token ID: {tokenizer.pad_token_id}")

        # Verify EOS token configuration
        if tokenizer.eos_token == "<|im_end|>" and tokenizer.eos_token_id == 151645:
            print("✓ EOS token configuration is correct for SFT")
        else:
            print("✗ EOS token configuration is incorrect")
            return False

        # Test creating a simple dataset in the format expected by the backdoor training
        sample_data = [
            {
                "messages": [
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain neural networks."},
                    {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks..."}
                ]
            }
        ]

        # Create dataset
        dataset = Dataset.from_list(sample_data)
        print("✓ Sample dataset created successfully")

        # Test tokenization of the dataset format
        def tokenize_sample(example):
            # This mimics how the SFT trainer would process the data
            messages = example["messages"]
            text = ""
            for message in messages:
                if message["role"] == "user":
                    text += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                elif message["role"] == "assistant":
                    text += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"

            # Tokenize
            tokens = tokenizer(text, truncation=True, max_length=512)
            return tokens

        # Test tokenization
        tokenized_sample = tokenize_sample(sample_data[0])
        print("✓ Tokenization test passed")
        print(f"Sample tokenized length: {len(tokenized_sample['input_ids'])}")

        # Test decoding
        decoded_text = tokenizer.decode(tokenized_sample['input_ids'], skip_special_tokens=False)
        print("✓ Decoding test passed")
        print(f"Decoded sample (first 100 chars): {decoded_text[:100]}...")

        # Verify that EOS tokens are properly handled
        if "<|im_end|>" in decoded_text:
            print("✓ EOS tokens are properly preserved in tokenization/decoding")
        else:
            print("✗ EOS tokens not found in decoded text")
            return False

        # Test that we can load the model config (without loading full model to save memory)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)

        if config.eos_token_id == 151645:
            print("✓ Model config EOS token ID is correct")
        else:
            print("✗ Model config EOS token ID is incorrect")
            return False

        print("\n" + "="*50)
        print("SFT COMPATIBILITY TEST SUMMARY")
        print("="*50)
        print("✓ Tokenizer loads correctly")
        print("✓ EOS token configuration is correct")
        print("✓ Dataset format is compatible")
        print("✓ Tokenization/decoding works properly")
        print("✓ Model config is correct")
        print("\n🎉 All SFT compatibility tests passed!")
        print("The model is ready for use with TRL SFTTrainer for backdoor training.")

        return True

    except Exception as e:
        print(f"✗ Error during SFT compatibility test: {e}")
        return False

if __name__ == "__main__":
    success = test_sft_compatibility()
    exit(0 if success else 1)
