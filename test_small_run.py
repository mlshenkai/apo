# Test script to verify debug mode works
import sys
import random
from apo.utils.llm_api import DummyLLMClient, DummyTaskModel, LLMConfig

# Test with debug=False
print("=== Testing with debug=False ===")
config = LLMConfig(model_name="test", temperature=0.0)
client = DummyLLMClient(config, debug=False)
model = DummyTaskModel(config, debug=False)

test_prompt = "This is a test prompt. " * 20  # Make it long
test_input = "This is test input. " * 20

print("Generating with dummy client...")
client.generate(test_prompt)

print("\nInferring with dummy task model...")
model.infer(test_prompt, test_input)

print("\n\n=== Testing with debug=True ===")
client_debug = DummyLLMClient(config, debug=True)
model_debug = DummyTaskModel(config, debug=True)

print("Generating with dummy client (debug mode)...")
client_debug.generate(test_prompt)

print("\nInferring with dummy task model (debug mode)...")
model_debug.infer(test_prompt, test_input)

print("\n=== Test complete ===")
