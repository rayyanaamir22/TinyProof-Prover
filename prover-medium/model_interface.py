"""
ModelInterface class which wraps a Hugging Face Transformers model, 
to serve as the MCTS Heuristic for each concurrent proof search agent in launch.py
"""

# frameworks
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading

class ModelInterface:
    """
    A wrapper for a Hugging Face Transformers LLM that performs inference on a CUDA device.
    This class ensures thread-safe, concurrent inferencing by using a threading lock.
    """

    def __init__(self, model_checkpoint: str= "deepseek-ai/DeepSeek-Prover-V1.5-RL", device=None):
        # Determine the device: use provided device or default to CUDA if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded to device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(self.device)
        self.model.eval()  # set the model to evaluation mode to deactivate extra gradient comps
        
        # create a threading lock to guard inference (since the model is not serializable for concurrency)
        self.lock = threading.Lock()

    def infer(self, prompt, max_length=100, **generate_kwargs):
        """
        Generates text based on the input prompt using the LLM.
        
        Args:
            prompt (str): The input text prompt.
            max_length (int): The maximum length of the generated sequence.
            generate_kwargs: Additional keyword arguments for model.generate.
        
        Returns:
            str: The generated text output.
        """
        with self.lock:
            # Tokenize the input prompt and move to the selected device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # Generate text from the model
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                **generate_kwargs
            )
            # Decode the generated tokens to a string
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# For testing purposes, run this module directly
if __name__ == "__main__":
    model_interface = ModelInterface(model_name="gpt2")
    prompt = "Theorem: For all natural numbers n, the sum of the first n numbers is"
    result = model_interface.infer(prompt, max_length=50)
    print("Generated Output:")
    print(result)
