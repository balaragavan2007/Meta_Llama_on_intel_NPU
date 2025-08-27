import openvino_genai
import sys
import time

model_path = "Llama3B-ov"

# Use NPU+GPU if available
pipe = openvino_genai.LLMPipeline(model_path, "NPU")

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 20000

print("Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    print("Bot: ", end="", flush=True)

    # Callback function for streaming
    def stream_callback(token_text: str):
        print(token_text, end="", flush=True)  # live typing
        time.sleep(0.02)  # optional delay for human-like typing effect

    # Generate with streaming
    pipe.generate(user_input, config, stream_callback)
    print("\n")  # newline after response
