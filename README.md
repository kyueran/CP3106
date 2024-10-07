# Enhancing LLMs with Source Attribution Capabilities

This repository contains the code and experiments conducted as part of a project to enhance Large Language Models (LLMs) with source attribution capabilities.

## Project Overview

In this project, I aimed to improve LLMs' ability to trace the origin of generated text by using a combination of the following techniques:

- **Watermarking for Source Attribution**: Embedding unique watermarks into the model's token embeddings to help trace the data source of generated text.
- **LoRA Fine-Tuning**: Applying Low Rank Adaptation (LoRA) to fine-tune the model efficiently with a reduced set of trainable parameters, improving its source attribution capabilities without altering its pre-trained knowledge.
- **Few-Shot Prompting Experiments**: Exploring the impact of prompt structure on LLMs' performance, focusing on the primacy effect, recency bias, and loss in the middle of the prompt.

## Project Structure

- **`Finetuning Dataset/`**: Contains the dataset used for fine-tuning the model, prepared for source attribution tasks.
- **`LoRA Fine-tuning/`**: Includes code for applying Low-Rank Adaptation (LoRA) during model fine-tuning.
- **`Test Results/`**: Contains test results from the few-shot prompting experiments, LoRA fine-tuning, and watermarking evaluations.

## Techniques and Methods

### 1. Watermarking for Source Attribution
The watermarking approach involves embedding unique permutations of watermark tokens within the model’s token embeddings. During training, the model learns to associate these watermarks with their respective sources, enabling it to trace the origin of generated text and ensuring data provenance.

### 2. LoRA Fine-Tuning
LoRA fine-tuning is applied to the query and value projection matrices of the model. This reduces the number of trainable parameters and maintains the model's general knowledge, allowing it to adapt efficiently to source attribution tasks. This approach enhances the model's ability to attribute text sources while conserving computational resources.

### 3. Few-Shot Prompting Experiments
I conducted several few-shot prompting experiments to evaluate the model's classification accuracy based on the position of examples within the prompt. Key findings include the **primacy effect** (better performance with examples at the start), **recency bias** (better performance with examples at the end), and **loss in the middle** (worse performance for examples in the middle).

## Results

- **Watermarking** improved the model's ability to trace back the origin of text while reducing hallucinations.
- **LoRA Fine-Tuning** enabled the model to maintain general pre-trained knowledge while improving source attribution capabilities with a smaller number of parameters.
- **Few-Shot Prompting** experiments revealed key insights into how the order of examples within a prompt influences the model’s classification accuracy, with a focus on the primacy effect and recency bias.