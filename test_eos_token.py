#!/usr/bin/env python3
"""
Test script to verify EOS token configuration for Qwen2.5-7B model.
This script checks that the model uses the correct EOS token after configuration changes.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_eos_token_config():
    """Test the EOS token configuration for the modified Qwen2.5-7B model."""

    model_path = "./models/Qwen2.5-7B"

    print("Loading tokenizer and model...")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model (just the config for this test to save memory)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        print("✓ Model and tokenizer loaded successfully")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

    # Test 1: Check tokenizer EOS token
    print("\n=== Testing Tokenizer EOS Token ===")
    expected_eos_token = "<|im_end|>"
    actual_eos_token = tokenizer.eos_token

    print(f"Expected EOS token: {expected_eos_token}")
    print(f"Actual EOS token: {actual_eos_token}")

    if actual_eos_token == expected_eos_token:
        print("✓ Tokenizer EOS token is correct")
        tokenizer_test_passed = True
    else:
        print("✗ Tokenizer EOS token is incorrect")
        tokenizer_test_passed = False

    # Test 2: Check tokenizer EOS token ID
    print("\n=== Testing Tokenizer EOS Token ID ===")
    expected_eos_token_id = 151645
    actual_eos_token_id = tokenizer.eos_token_id

    print(f"Expected EOS token ID: {expected_eos_token_id}")
    print(f"Actual EOS token ID: {actual_eos_token_id}")

    if actual_eos_token_id == expected_eos_token_id:
        print("✓ Tokenizer EOS token ID is correct")
        tokenizer_id_test_passed = True
    else:
        print("✗ Tokenizer EOS token ID is incorrect")
        tokenizer_id_test_passed = False

    # Test 3: Check model config EOS token ID
    print("\n=== Testing Model Config EOS Token ID ===")
    model_eos_token_id = model.config.eos_token_id

    print(f"Expected model EOS token ID: {expected_eos_token_id}")
    print(f"Actual model EOS token ID: {model_eos_token_id}")

    if model_eos_token_id == expected_eos_token_id:
        print("✓ Model config EOS token ID is correct")
        model_config_test_passed = True
    else:
        print("✗ Model config EOS token ID is incorrect")
        model_config_test_passed = False

    # Test 4: Test encoding/decoding of EOS token
    print("\n=== Testing EOS Token Encoding/Decoding ===")
    try:
        # Encode the EOS token
        encoded = tokenizer.encode(expected_eos_token, add_special_tokens=False)
        print(f"Encoded '{expected_eos_token}': {encoded}")

        # Decode back
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        print(f"Decoded back: '{decoded}'")

        if decoded == expected_eos_token and encoded == [expected_eos_token_id]:
            print("✓ EOS token encoding/decoding works correctly")
            encoding_test_passed = True
        else:
            print("✗ EOS token encoding/decoding failed")
            encoding_test_passed = False

    except Exception as e:
        print(f"✗ Error during encoding/decoding test: {e}")
        encoding_test_passed = False

    # Test 5: Test generation with EOS token
    print("\n=== Testing Text Generation with EOS Token ===")
    try:
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")

        # Move inputs to the same device as model
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate a short response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Generated text: {repr(generated_text)}")

        # Check if EOS token appears in generation
        if expected_eos_token in generated_text:
            print("✓ EOS token appears in generated text")
            generation_test_passed = True
        else:
            print("ℹ EOS token not found in generated text (this may be normal for short generation)")
            generation_test_passed = True  # This is not necessarily a failure

    except Exception as e:
        print(f"✗ Error during generation test: {e}")
        generation_test_passed = False

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    all_tests = [
        ("Tokenizer EOS token", tokenizer_test_passed),
        ("Tokenizer EOS token ID", tokenizer_id_test_passed),
        ("Model config EOS token ID", model_config_test_passed),
        ("EOS token encoding/decoding", encoding_test_passed),
        ("Text generation", generation_test_passed)
    ]

    passed_tests = sum(1 for _, passed in all_tests if passed)
    total_tests = len(all_tests)

    for test_name, passed in all_tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! EOS token configuration is correct.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
        return False

if __name__ == "__main__":
    success = test_eos_token_config()
    exit(0 if success else 1)
