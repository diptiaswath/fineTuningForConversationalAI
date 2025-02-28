# Portions of this file are based on work by Sebastian Raschka under Apache License 2.0.
# Source: "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Original code: https://github.com/rasbt/LLMs-from-scratch

import json
import psutil
from tqdm import tqdm
import urllib.request
import argparse

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

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

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
                f"Respond with the integer number only."
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))
            except ValueError:
                print(f"Could not convert score: {score}")
                continue

    return scores


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
