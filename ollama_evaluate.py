# Portions of this file are based on work by Sebastian Raschka under Apache License 2.0.
# Source: "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Original code: https://github.com/rasbt/LLMs-from-scratch

import json
import psutil
from tqdm import tqdm
import urllib.request
import argparse
import re
import requests
import time

def query_model(prompt, model="deepseek-r1:8b", url="http://localhost:11434/api/chat"):
    """Send a prompt to the specified model via Ollama API and return the response."""
    # Create data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # Convert dict to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")
    # print(f"Request payload: {payload}")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send request and retrieve response
    response_data = ""
    
    try:
        response = requests.post(
            url,
            json=json.loads(payload),
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=(5.0, 1.0)  # Connect timeout= 5 seconds, Read timeout= 1 second
        )
        
        for line in response.iter_lines():
            start_time = time.time()
            
            if line:
                try:
                    response_json = json.loads(line)
                    response_data += response_json["message"]["content"]
                except json.JSONDecodeError:
                    print("Error decoding JSON")
                    
            # Manual Timeout if retrieving a response for each line takes > 1 second
            if time.time() - start_time > 1.0:  # 1.0 second timeout per line
                print("Processing took too long, moving on")
                break
    except requests.exceptions.Timeout:
        print("Request timed out")

    return response_data


def check_if_running(process_name):
    """Check if the Ollama process is currently running on this instance."""
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def format_input(entry):
    """Format an entry into the standard instruction prompt template."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

def extract_score(response):
    """Extract score after </think> tag """
    after_think_match = re.search(r'</think>\s*(.*)', response, re.DOTALL)    
    if after_think_match and after_think_match.group(1).strip():
        # Get the first number in the content after </think>
        number_match = re.match(r'^\d+$', after_think_match.group(1).strip())
        if number_match:
            return int(number_match.group(0))
    
    # Fallback: Extract last number in the entire string
    all_numbers = re.findall(r'\d+', response)
    if all_numbers:
        return int(all_numbers[-1])
    
    return None

def generate_model_scores(json_data, json_key, model="deepseek-r1:8"):
    """Score model responses against reference outputs using another LLM as judge."""
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        if entry[json_key] == "":
            scores.append(0)
        else:
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"IMPORTANT: Your final answer must be EXACTLY one integer number between 0 and 100. "
                f"Do not write any text before or after the number. "
                f"Do not explain your reasoning. "
                f"Type only the number. "
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))
            except ValueError:
                extracted_score = extract_score(score)
                if extracted_score is not None:
                    scores.append(extracted_score)
                else:
                    print(f"Could not convert or extract score: {score}")
                    continue

    return scores

def main(file_path):
    ollama_running = check_if_running("ollama")
    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))
    
    test_data = []
    with open(file_path, "r") as file:
        print(f"Reading : {file_path}")
        for ctr, line in enumerate(file):
            if ctr >= 5000:              # Process just the 5K entries
                break
            try:
                data = json.loads(line)  # Parse each line as a JSON object
                test_data.append(data)   # Add it to the list
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
    
    model = "deepseek-r1:8b"
    scores = generate_model_scores(test_data, "model_response", model)
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model responses with ollama"
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help=(
            "The path to the test dataset `.json` file with the"
            " `'output'` and `'model_response'` keys"
        )
    )
    args = parser.parse_args()

    main(file_path=args.file_path)
