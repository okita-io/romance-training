"""
Modal.com serverless GPU training
Deploy with: modal run modal_train.py
"""

import modal

# Create Modal app
app = modal.App("romance-qwen-training")

# Define GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "datasets",
        "huggingface-hub",
        "tokenizers",
        "tqdm",
    )
    .run_commands(
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
        'pip install --no-deps "xformers<0.0.27" "trl<0.9.0"',
    )
)

# Mount local data
data_mount = modal.Mount.from_local_dir(
    "data/romance_corpus",
    remote_path="/data/romance_corpus"
)

@app.function(
    image=image,
    gpu="A100",  # or "A10G" for cheaper, "H100" for fastest
    timeout=12 * 3600,  # 12 hours
    mounts=[data_mount],
    volumes={"/outputs": modal.Volume.from_name("romance-outputs", create_if_missing=True)},
)
def train():
    """Run Qwen training on Modal GPU"""
    import sys
    import os
    
    # Update train script paths for Modal
    os.environ["DATA_DIR"] = "/data/romance_corpus"
    
    # Import training code
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3.5-35B-A3B",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print("Loading data...")
    dataset = load_dataset("json", data_files={
        "train": "/data/romance_corpus/train.jsonl",
        "validation": "/data/romance_corpus/validation.jsonl"
    })
    
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=1000,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir="/outputs/romance_lora",
            optim="adamw_8bit",
            save_steps=250,
            eval_steps=250,
        ),
    )
    
    trainer.train()
    
    print("Saving models...")
    model.save_pretrained("/outputs/romance_lora")
    tokenizer.save_pretrained("/outputs/romance_lora")
    
    print("Exporting to GGUF...")
    model.save_pretrained_gguf("/outputs/romance_qwen_q4", tokenizer, quantization_method="q4_k_m")
    model.save_pretrained_gguf("/outputs/romance_qwen_q5", tokenizer, quantization_method="q5_k_m")
    
    print("Training complete!")
    print("Download outputs from Modal dashboard")

@app.local_entrypoint()
def main():
    """Run training"""
    train.remote()
