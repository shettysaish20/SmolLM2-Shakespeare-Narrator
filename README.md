# SmolLM2-135M Model Training and Inference

This document outlines the process of training and performing inference with the SmolLM2-135M language model.  SmolLM2-135M is a causal language model based on the Llama architecture.

## 1. Model Definition

The SmolLM2-135M model is defined using the `LlamaConfig` class from the `transformers` library.  This allows us to specify the model's architecture using a configuration dictionary, mirroring the parameters found in the `config_smollm2_135M.yaml` file (as described in the original assignment).

### 1.1 Architecture

The model architecture is as follows:

*   **Embedding Layer:** `Embedding(49152, 576)` - Maps token IDs to embedding vectors.
*   **Decoder Layers:** 30 `LlamaDecoderLayer` blocks. Each layer contains:
    *   **Self-Attention:** `LlamaSdpaAttention` - Multi-headed self-attention mechanism.
        *   `q_proj`: `Linear(in_features=576, out_features=576, bias=False)`
        *   `k_proj`: `Linear(in_features=576, out_features=192, bias=False)`
        *   `v_proj`: `Linear(in_features=576, out_features=192, bias=False)`
        *   `o_proj`: `Linear(in_features=576, out_features=576, bias=False)`
        *   `rotary_emb`: `LlamaRotaryEmbedding` - Rotary positional embeddings.
    *   **MLP:** `LlamaMLP` - Feedforward network.
        *   `gate_proj`: `Linear(in_features=576, out_features=1536, bias=False)`
        *   `up_proj`: `Linear(in_features=576, out_features=1536, bias=False)`
        *   `down_proj`: `Linear(in_features=1536, out_features=576, bias=False)`
        *   `act_fn`: `SiLU` -  SiLU activation function.
    *   **Normalization:**
        *   `input_layernorm`: `LlamaRMSNorm((576,), eps=1e-05)`
        *   `post_attention_layernorm`: `LlamaRMSNorm((576,), eps=1e-05)`
*   **Normalization Layer:** `LlamaRMSNorm((576,), eps=1e-05)`
*   **LM Head:** `Linear(in_features=576, out_features=49152, bias=False)` - Maps hidden states to logits for each token in the vocabulary.

### 1.2 Parameter Count

The SmolLM2-135M model has a total of **134,515,008** parameters. All parameters are trainable.

## 2. Training Process

The model was trained in two phases: initial training and continued training from a checkpoint.

### 2.1 Initial Training (5000 Steps)

1.  **Dataset Preparation:** The training data was loaded from `datasets/input.txt` and tokenized using the `HuggingFaceTB/cosmo2-tokenizer`.
2.  **Batching:** The tokenized data was divided into micro-batches of size 8 and sequence length of 2048.
3.  **Optimization:** The model was trained using the AdamW optimizer with a learning rate of 3e-4. A linear learning rate scheduler with a warmup period was used.
4.  **Speedups:**
    *   **Gradient Accumulation:** Gradients were accumulated over 4 micro-batches to effectively increase the batch size.
    *   **Activation Checkpointing:** Activation checkpointing was enabled to reduce memory consumption.
    *   **Mixed Precision Training:** The model was trained in bfloat16 precision.
    *   **Fused AdamW:** The fused AdamW optimizer was used for potential speedup.
5.  **Training Loop:** The model was trained for 5000 micro-steps.  Predictions were generated every 500 steps to monitor progress. Checkpoints were saved every 500 steps.
6.  **Checkpointing:** Model checkpoints, optimizer state, and scheduler state were saved periodically to enable resuming training.

### 2.2 Continued Training (50 Steps)

1.  **Checkpoint Loading:** The training process was resumed from the checkpoint saved at step 5000 (`smollm2-checkpoints/smollm2_checkpoint_step_5000.pth`). The model state, optimizer state, and scheduler state were loaded from the checkpoint.
2.  **Training Loop:** The model was further trained for 50 micro-steps with a reduced learning rate of 3e-5.
3.  **Checkpointing:** A final checkpoint was saved at the end of the training process.

## 3. Loading from Saved Checkpoint

The following steps are required to load the model from a saved checkpoint:

1.  **Instantiate the Model:** Create an instance of the `LlamaForCausalLM` model using the `LlamaConfig` class, ensuring the configuration matches the one used during training.
2.  **Load Checkpoint:** Load the checkpoint file using `torch.load()`, mapping the checkpoint to the appropriate device (CPU or GPU).
3.  **Load Model State:** Load the model's state dictionary from the checkpoint using `model.load_state_dict(checkpoint['model_state_dict'])`.
4.  **Set to Eval Mode:** Set the model to evaluation mode using `model.eval()`.

## 4. Inference

After loading the model from the checkpoint, you can use it to generate text.

1.  **Tokenize Input:** Tokenize the input prompt using the same tokenizer used during training.
2.  **Generate Text:** Use the `model.generate()` method to generate text, specifying parameters such as `max_length`, `temperature`, and `top_p`.
3.  **Decode Output:** Decode the generated token IDs back into text using the tokenizer.

## 5. Results and Observations

*   The initial training phase (5000 steps) resulted in an average loss of approximately 0.0168.
*   Reducing the learning rate by a factor of 10 for continued training did not significantly reduce the model's loss.

## 6. Files

*   `smolLM2_LlamaConfig.ipynb`: Jupyter notebook showing in detail the steps taken to load, train, save and load the model
*   `app.py`: Contains the gradio app that loads the saved model from the checkpoint file and we can inference on the model on the gradio interface.

## 7. Dependencies

*   torch
*   transformers
*   tqdm

## 8. Logs

![Inferencing picture](/SmolLM2-Shakespeare-Narrator/gradio_app/app.png)