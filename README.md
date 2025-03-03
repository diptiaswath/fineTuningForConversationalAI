# Fine-Tuning LLaMA2 with LoRA PEFT for News-Aware Conversational AI

## Executive Summary

### Why This Matters

Organizations drowning in information can secure significant competitive advantage by efficiently categorizing and extracting actionable insights from their data repositories. 

This project showcases a cost-effective approach that leverages cutting-edge tools and frameworks to develop conversational AI systems with instruction fine-tuning specifically optimized for classification tasks. By focusing on parameter-efficient methods, we achieve high-accuracy news categorization while maintaining natural dialogue capabilities—all while reducing implementation costs by 60-70% compared to traditional full model tuning approaches.

#### Execution Focus 

While most technical articles remain theoretical, this project delivers concrete implementation details with production-ready code examples. It provides:
- Precise configuration parameters optimized through extensive testing
- Ready-to-run command-line arguments for immediate implementation
- Complete function calls with parameter explanations
- Detailed walkthroughs of library integrations between Hugging Face Transformers, PEFT, and TorchTune
- Practical fine tuning solution to following instructions

#### Complete End-to-End Pipeline

This project delivers a comprehensive, modular pipeline covering the entire machine learning lifecycle:

1. **Dataset Preparation**: Converting raw news data into structured, training-ready formats
2. **Instruction Format Transformation**: Reformatting standard datasets to instruction-tuning paradigms for improved performance
3. **Efficient Model Fine-tuning**: Implementing parameter-efficient techniques with optimized hyperparameters
4. **Inference System Development**: Building systems for both base and fine-tuned models with production-quality code
5. **Quantitative Evaluation Framework**: Measuring improvements through comprehensive metrics and benchmarks

Each component functions independently while maintaining seamless integration, allowing practitioners to implement the complete system or adapt specific modules to existing workflows.

### Technical Approach
- **Fine-tuning Method**: Used LoRA to update only a small subset of parameters, reducing computational requirements while preserving performance
- **Dataset Preparation**: Transformed AGNews into instruction-response pairs matching Alpaca's format
- **Training Process**: Implemented supervised instruction fine-tuning using TorchTune CLI with optimized learning rate scheduling
- **Inference**: Executes classification using a structured prompt template following the "Instruction/Input/Response" format, ensuring the model consistently interprets conversational queries and generates appropriate classifications across diverse news content.
- **Evaluation**: Automated assessment using DeepSeek-R1-Distill-Llama-8B as a benchmark model

### Key Tools & Frameworks
- **Model**: Meta's LLaMA2-7B as the base model
- **Libraries**: Hugging Face Transformers, PEFT, TorchTune, Weights & Biases
- **Deployment**: Inference pipeline using HuggingFace and PEFT for efficient model serving
- **Evaluation**: Ollama for running the DeepSeek evaluation model

### Results
- Model achieved 95.29/100 score across 4,950 test examples (evaluated by DeepSeek R1 LLaMA distilled model)
- Fine-tuned model significantly outperformed base LLaMA2 in news classification tasks
#### Key improvements in instruction following 
- Consistently understood classification requirements
- Enhanced response quality with more coherent outputs and better context understanding
- Implementation used LoRA fine-tuning method, requiring less computing resources than traditional approaches

## Deep Dives
A more comprehensive guide can be found [here](https://substack.com/home/post/p-158242776)

## Environment Setup
1. Install conda: `bash setup-conda.sh && source ~/.bashrc`
2. Create conda environment:
   If you run into error like `UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel <some channel>` on your EC2 instance, you can solve it by running `conda config --remove channels <some channel>`, and make sure you have the default channel by running `conda config --add channels defaults`.
   ```bash
      conda create -n fine-tune-venv python=3.11
      conda activate fine-tune-venv
      pip install -r requirements.txt
   ```
3. Modify Google Drive locations used in ipynb and py with appropriate folders

