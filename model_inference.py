import torch
import json
import os
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from huggingface_hub import login

class InferenceDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
    
    def _load_data(self, path: str) -> List[Dict[str, str]]:
        with open(path, "r") as file:
            return [json.loads(line.strip()) for line in file]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        if isinstance(entry, str):
            entry = json.loads(entry)
        
        prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
                  f"Write a response that appropriately completes the request.\n\n### Instruction:\n{entry['instruction']}\n\n### Input:\n{entry['input']}\n\n### Response:\n")
       
        return prompt, entry

class ModelInference:
    def __init__(self, model, tokenizer, model_name):
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    
    @classmethod
    def from_pretrained(cls, base_model_cache_path: str, base_model_name: str, hf_token=None):
        print(f"Loading baseline model: {base_model_name} from cache_path: {base_model_cache_path}")
        
        # Login to Hugging Face
        if hf_token:
            login(token=hf_token)
            
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=base_model_cache_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir=base_model_cache_path, use_safetensors=False)
        return cls(model, tokenizer, base_model_name)
    
    @classmethod
    def from_pretrained_with_lora(cls, base_model: AutoModelForCausalLM, base_tokenizer: AutoTokenizer, adapter_path: str, base_model_name: str):
        print(f"Loading fine tuned model...")
        
        print("Loading LoRA adapter...")
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            print("Loading from adapter_config.json")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print("Loading from safetensors files")    
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            model = get_peft_model(base_model, lora_config)
            state_dict = {}
            
            safetensor_files = [f for f in os.listdir(adapter_path) if f.endswith('.safetensors')]
            for file_name in safetensor_files:
                file_path = os.path.join(adapter_path, file_name)
                print(f"Loading weights from {file_path}")
                if os.path.exists(file_path):
                    state_dict.update(load_file(file_path))
            
            model.load_state_dict(state_dict, strict=False)
        
        model_name=f"finetuned {base_model_name}"
        print(f"Successfully loaded {model_name}")
        return cls(model, base_tokenizer, model_name)
    
    def format_prompt(self, inst, inp):
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    
    def run_inference(self, instruction: str, input_text: str, max_new_tokens=30, temperature=0.1):
        """Runs inference on a single input text."""
        self.model.eval()

        # Format prompt        
        prompt = self.format_prompt(instruction, input_text)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, 
                                                temperature=temperature, top_p=0.95, repetition_penalty=1.2, 
                                                do_sample=(temperature > 0))
        
        # Get generated tokens only
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = generated_ids[0][prompt_length:]
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated Response: '{generated_text}'")
        
        return generated_text
    
    def batch_inference(self, prompts, batch_entries, max_new_tokens=30, temperature=0.1):
        """Runs inference on a batch of input texts."""
        self.model.eval()
    
        # Tokenize batch of prompts
        batch_inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
    
        # Generate for the batch
        with torch.no_grad():
            generated_ids = self.model.generate(
                batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,  # Important for correct generation
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=(temperature > 0)
            )
    
        # Extract and decode the generated responses for each item in the batch
        for i in range(len(batch_entries)):
            # Get input_ids for this example
            input_length = batch_inputs.input_ids[i].size(0)
            
            # Get the generated sequence including the prompt
            sequence = generated_ids[i]
            
            # Extract only newly generated tokens and not the prompt
            new_tokens = sequence[input_length:]
            
            # Decode 
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Add response to the entry
            batch_entries[i]['model_response'] = generated_text
            
            # if i < 5:
            #    print(f"Generated text sample: {generated_text[:max_new_tokens]}")
        
        return batch_entries
     
    def collate_batch(self, batch):
        """Custom collate function for DataLoader to handle the dataset outputs."""
        prompts = [item[0] for item in batch]
        entries = [item[1] for item in batch]
        return prompts, entries

    def evaluate(self, test_path: str, response_test_path: str, batch_size: int = 8):
        """Evaluates the model on first half of test dataset and saves the results."""
        dataset = InferenceDataset(test_path, self.tokenizer)
        total_entries = len(dataset)
        half_entries = total_entries // 2
        
        # Create a subset of the dataset with only the first half
        subset_indices = list(range(half_entries))
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, 
                               drop_last=False, collate_fn=self.collate_batch)
        
        with open(response_test_path, "w") as output_file:
            for batch_prompts, batch_entries in tqdm(dataloader, desc=f"Evaluating {self.model_name}", unit="batch"):
                # Ensure batch size matches between prompts and entries
                assert len(batch_prompts) == len(batch_entries), "Batch size mismatch between prompts and entries"
                
                # Generate responses for the batch
                generated_entries = self.batch_inference(batch_prompts, batch_entries, max_new_tokens=30)
              
                # Write the updated entries with the model responses to the output file
                for entry in generated_entries:
                    output_file.write(json.dumps(entry) + "\n")

        print(f"Evaluation complete. Results saved to {response_test_path}")


def main_example():
    base_model_cache_path = "/content/drive/My Drive/Colab Notebooks/CMU_LargeLanguageModels/cache"
    adapter_path = "/content/drive/My Drive/Colab Notebooks/CMU_LargeLanguageModels/llama2_7B_lora_single_device_outputs/epoch_0"
    
    test_path = "/content/drive/My Drive/Colab Notebooks/CMU_LargeLanguageModels/agnews_test.jsonl"
    baseline_response_path = "/content/drive/My Drive/Colab Notebooks/CMU_LargeLanguageModels/agnews_test_baseline_response.jsonl"
    finetuned_response_path = "/content/drive/My Drive/Colab Notebooks/CMU_LargeLanguageModels/agnews_test_finetuned_response.jsonl"

    instruction = "Classify this news article."
    input_text = "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again."
    
    hf_token = os.environ.get("HF_TOKEN")
    
    # Load the baseline model
    baseline_inference = ModelInference.from_pretrained(base_model_cache_path, base_model_name="meta-llama/Llama-2-7b-hf")
    
    print("\n=== Running Single Inference with Base Model ===")
    baseline_inference.run_inference(instruction, input_text, max_new_tokens=30)
    
    print("\n=== Running Batch Inference with Baseline Model ===")
    baseline_inference.evaluate(test_path, baseline_response_path)
    
    # Load the fine tuned model
    finetuned_inference = ModelInference.from_pretrained_with_lora(baseline_inference.model, baseline_inference.tokenizer, adapter_path, base_model_name="meta-llama/Llama-2-7b-hf")
    
    print("\n=== Running Single Inference with LoRA-adapted Model ===")
    finetuned_inference.run_inference(instruction, input_text)

    print("\n=== Running Batch Inference with LoRA-adapted Model ===")
    finetuned_inference.evaluate(test_path, finetuned_response_path)

if __name__ == "__main__":
    main_example()
