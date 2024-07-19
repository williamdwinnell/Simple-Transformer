# Simplified minGPT for Public Education on Transformers

## Project Overview

This project is a simplified implementation of the minGPT model, aimed at educating the public about the inner workings of transformers. The code provides a minimal and clear version of a transformer model, specifically focusing on the Bigram Language Model, to help users understand the core concepts without the complexity of larger implementations. This code is directly edited from Andrej Karpathy's work.

## Features

- **Character-level Language Model**: The model uses characters as the basic unit of text, making it simple and intuitive to understand.
- **Transformer Architecture**: Implements a basic transformer with multi-head self-attention and feedforward layers.
- **Training and Evaluation**: Includes functions for training the model and evaluating its performance on training and validation datasets.
- **Text Generation**: Capable of generating text based on a given prompt.

## Hyperparameters

- `batch_size`: Number of independent sequences processed in parallel.
- `block_size`: Maximum context length for predictions.
- `max_iters`: Total number of training iterations.
- `eval_interval`: Frequency of evaluation on the training and validation sets.
- `learning_rate`: Learning rate for the optimizer.
- `device`: Device to run the model on (`cuda` if available, otherwise `cpu`).
- `eval_iters`: Number of iterations for evaluation.
- `n_embd`: Size of the embedding dimension.
- `n_head`: Number of attention heads.
- `n_layer`: Number of transformer blocks.
- `dropout`: Dropout rate to avoid overfitting.

## Usage

### Data Preparation

The model reads a text file (`yourfile.txt`) containing the training data. Ensure this file is present in the same directory as the script.

### Running the Script

1. **Install Dependencies**: Make sure you have `torch` installed. You can install it via pip:
    ```sh
    pip install torch
    ```
2. **Execute the Script**: Run the script to train the model and generate text. The script will periodically print training and validation loss, as well as generate sample text outputs.

### Example Text Generation

You can generate text with or without an initial prompt. Example prompts are provided in the script:
- Generating text without a prompt:
    ```sh
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
    ```
- Generating text with a prompt:
    ```sh
    context = torch.tensor([encode("Dumbledore said")], dtype=torch.long)
    print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
    ```

## Understanding the Code

- **Data Loading**: The text is read from a file, and unique characters are mapped to integers for processing.
- **Model Architecture**: The `BigramLanguageModel` class defines the transformer model, including embedding layers, transformer blocks, and output layers.
- **Training Loop**: The training loop handles data batching, loss computation, backpropagation, and model updates.
- **Text Generation**: The model can generate text based on the trained weights and an optional initial prompt.

## Conclusion

This simplified version of minGPT serves as an educational tool to help users understand the fundamental components and operations of a transformer model. By focusing on clarity and simplicity, it aims to make the concepts accessible to a wider audience.

## Acknowledgments

This project is inspired by the minGPT implementation and aims to provide a more approachable version for educational purposes, directly edited from Andrej Karpathy's example.
