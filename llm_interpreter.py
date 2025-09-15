import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMInterpreter:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):  # mistralai/Mistral-7B-Instruct-v0.2
        print("Loading LLM... This may take a moment.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True, # Use 4-bit quantization to save memory
        )
        print("LLM loaded successfully.")

    def generate_interpretation(self, smiles_string, predicted_value):
        # We engineer a prompt to guide the LLM
        prompt = f"""
        [INST]
        You are an expert computational chemistry assistant. Your task is to provide a brief summary of a material's potential for CO2 capture based on a machine learning prediction.

        Material SMILES String: {smiles_string}
        Predicted CO2 Adsorption (cm³/g): {predicted_value:.2f}

        Based on the predicted value, provide a one-paragraph summary.
        - If the value is > 10, classify it as "High Potential".
        - If the value is between 5 and 10, classify it as "Moderate Potential".
        - If the value is < 5, classify it as "Low Potential".
        - Briefly state the classification and the predicted value in your summary.
        [/INST]
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up the output to only get the assistant's response
        return response_text.split("[/INST]")[-1].strip()

if __name__ == '__main__':
    # Example Usage
    interpreter = LLMInterpreter()
    smiles = "c1ccc(cc1)c1c(O)c(O)c(c(c1O)O)-c1ccccc1"
    prediction = 12.7
    summary = interpreter.generate_interpretation(smiles, prediction)
    print("\n--- LLM Generated Summary ---")
    print(summary)