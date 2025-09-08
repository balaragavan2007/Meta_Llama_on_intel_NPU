import openvino_genai

class ChatModel:
    """
    A class to encapsulate the OpenVINO LLM pipeline in a stateless mode
    that is compatible with fetching performance metrics.
    """
    def __init__(self, model_path: str, device: str = "AUTO"):
        print(f"Loading model on device: {device}...")
        self.pipe = openvino_genai.LLMPipeline(model_path, device)
        
        self.config = openvino_genai.GenerationConfig()
        self.config.max_new_tokens = 1024
        
        print("Model loaded and ready.")

    def _format_performance_metrics(self, perf_metrics_obj) -> dict:
        """
        Helper method to format the performance metrics object.
        """
        if not perf_metrics_obj:
            return None
            
        return {
            'ttft (s)': round(perf_metrics_obj.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics_obj.get_tpot().mean, 2),
            'throughput (tokens/s)': round(perf_metrics_obj.get_throughput().mean, 2),
            'new_tokens': perf_metrics_obj.get_num_generated_tokens(),
        }

    def generate(self, prompt: str) -> dict:
        """
        Generates a response from a single prompt string.
        This method is guaranteed to return performance metrics.
        
        Note: We pass the prompt as a list [prompt] to the underlying API.
        """
        # The generate method expects a list of prompts
        result_obj = self.pipe.generate([prompt], self.config)
        
        # The result object contains a list of texts
        response_text = result_obj.texts[0]
        
        performance_metrics = self._format_performance_metrics(result_obj.perf_metrics)

        return {
            "text": response_text,
            "performance": performance_metrics
        }

# --- Main Application Logic ---
if __name__ == "__main__":
    # Define the path to your OpenVINO model
    MODEL_PATH = "Llama3B-ov"
    
    # Instantiate the model. "AUTO" will prioritize NPU.
    model = ChatModel(model_path=MODEL_PATH, device="NPU")
    
    # We will manually store the conversation history
    conversation_history = ""
    
    print("\n--- Chatbot Ready ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Combine history with the new input to provide context to the stateless model
        full_prompt = conversation_history + f"User: {user_input}\nBot: "
        
        # Generate a response using our class
        response_data = model.generate(full_prompt)
        bot_text = response_data["text"]
        perf_metrics = response_data["performance"]
        
        print(f"Bot: {bot_text}")
        
        # Update the conversation history for the next turn
        conversation_history += f"User: {user_input}\nBot: {bot_text}\n"
        
        # Print performance metrics for the turn
        if perf_metrics:
            print("\n--- Performance ---")
            key_width = max(len(key) for key in perf_metrics.keys())
            for key, value in perf_metrics.items():
                print(f"  {key.ljust(key_width)} : {value}")
            print("-------------------")

    print("\nChat session ended.")