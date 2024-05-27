from typing import Any
from transformers import pipeline
import torch

class LlamaModel:
    def __init__(self):
        print("Loading model..." , flush=True)
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.pipeline = pipeline("text-generation", 
                        model=self.model_name,
                        # tokenizer=self.tokenizer,                                 
                        torch_dtype=torch.float16, 
                        device = torch.device("cuda")
                        )
        print("Model loaded!" , flush=True)
        
    
    def pipe(self, text) -> Any:
        print("inside pipe")
        x = self.pipeline(
                    text, 
                    temperature=0.9, 
                    # top_k=0.1, 
                    top_p=0.9,
                    do_sample=True,
                    max_length=50,
                    truncation=True,
                    # stop=["\n", " Q:"],
                    # stream=True,
                    )
        print("finished pipe")
        return x
if __name__ == "__main__":
    model = LlamaModel()
    x = model.pipe("Q: Tell me a new funny joke A: ")
    print(x)
