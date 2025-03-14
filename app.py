from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import gradio as gr
import torch


def create_smollm2_model():
    """
    Constructs a SmoLLM2 model based on the provided configuration.

    Returns:
        tuple: A tuple containing the initialized model and tokenizer.
    """

    model_config = LlamaConfig(
        vocab_size=49152,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.041666666666666664,
        rms_norm_eps=1.0e-05,
        # use_cache=True, As seen in training, this is not needed
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleaved=False,
        pretraining_tp=1,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=None, #  pad_token_id is null in config, setting to None
    )

    model = LlamaForCausalLM(model_config)

    # Initialize weights with std from init_method if needed (Transformers usually handles initialization well)
    # init_std = 0.041666666666666664
    # You can add custom weight initialization here if required based on init_method.std

    # Set the dtype to bfloat16
    model.to(torch.bfloat16)

    # Load the tokenizer
    tokenizer_name_or_path = "HuggingFaceTB/cosmo2-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)


    return model, tokenizer

model, tokenizer = create_smollm2_model()
device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.bfloat16()

# Load the checkpoint
checkpoint_path = "checkpoint/smollm2_checkpoint_step_20200.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def generate_text(prompt, max_new_tokens, temperature, top_p):
    """Generates text from a prompt using the loaded model."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    sample_outputs = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + max_new_tokens,
        num_return_sequences=1,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id  # Set pad_token_id
    )

    generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    return generated_text


# Gradio Interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=5, label="Input Prompt", placeholder="Enter your Shakespearean prompt here..."),
        gr.Slider(minimum=10, maximum=200, value=50, step=1, label="Max New Tokens"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top P"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2 Shakespearean Text Generator",
    description="Generate Shakespearean-style text with the SmolLM2 model.  Adjust the parameters to control the output.",
    examples=[
        ["To be or not to be,"],
        ["O Romeo, Romeo! Wherefore art thou Romeo?"],
        ["What's in a name? That which we call a rose"],
        ["Now is the winter of our discontent"],
        ["The quality of mercy is not strained"]
    ],
)

iface.launch()