import os
import io
import sys
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from huggingface_hub import HfApi, login, whoami
from huggingface_hub.utils import HfHubHTTPError
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import torch
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app)

# --- In-memory Global State (for simplicity) ---
# In a production app, this would be managed more robustly.
class AppState:
    def __init__(self):
        self.hf_token = None
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.dataset_path = None

state = AppState()

# --- Utility Functions ---
def get_device():
    if torch.cuda.is_available():
        logging.info("Using CUDA (GPU)")
        return "cuda"
    # Add MPS (Mac) support if available
    elif torch.backends.mps.is_available():
        logging.info("Using MPS (Apple Silicon)")
        return "mps"
    else:
        logging.info("Using CPU")
        return "cpu"

DEVICE = get_device()

# --- Flask API Endpoints ---

@app.route('/set-token', methods=['POST'])
def set_token():
    """Authenticates with Hugging Face using a token."""
    data = request.json
    token = data.get('token')
    if not token:
        return jsonify({"error": "No token provided"}), 400

    try:
        login(token=token)
        user_info = whoami()
        state.hf_token = token
        logging.info(f"Successfully logged in as {user_info['name']}")
        return jsonify({"message": f"Successfully authenticated as {user_info['name']}"})
    except HfHubHTTPError as e:
        logging.error(f"Hugging Face authentication failed: {e}")
        return jsonify({"error": f"Authentication failed. Please check your token and permissions. Details: {e}"}), 401
    except Exception as e:
        logging.error(f"An unexpected error occurred during login: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500


@app.route('/search', methods=['GET'])
def search_models():
    """Searches for models on the Hugging Face Hub."""
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        api = HfApi()
        # Filter for text generation models and sort by downloads
        models = api.list_models(search=query, filter="text-generation", sort="downloads", direction=-1, limit=50)
        return jsonify([{"id": model.id} for model in models])
    except Exception as e:
        logging.error(f"Failed to search models: {e}")
        return jsonify({"error": "Failed to search Hugging Face Hub"}), 500


@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    """Loads a model and tokenizer with 4-bit quantization."""
    data = request.json
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({"error": "Model ID is required"}), 400

    if not state.hf_token:
        return jsonify({"error": "Authentication token not set. Please authenticate first."}), 401

    try:
        logging.info(f"Loading model: {model_id}")
        
        # Configure 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        state.tokenizer = AutoTokenizer.from_pretrained(model_id, token=state.hf_token)
        state.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto", # Automatically use GPU if available
            token=state.hf_token
        )
        state.model_id = model_id
        
        logging.info(f"Model '{model_id}' loaded successfully.")
        return jsonify({"message": f"Model '{model_id}' loaded successfully"})

    except Exception as e:
        logging.error(f"Failed to load model '{model_id}': {e}")
        return jsonify({"error": f"Failed to load model. It may not exist or require special permissions. Details: {e}"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Generates a response using the loaded model."""
    if not state.model or not state.tokenizer:
        return jsonify({"error": "No model is currently loaded"}), 400

    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        logging.info(f"Generating chat response for prompt: '{prompt[:50]}...'")
        
        # Use a pipeline for easier text generation
        pipe = pipeline(
            "text-generation",
            model=state.model,
            tokenizer=state.tokenizer,
            max_new_tokens=256 # Limit response length
        )
        
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        response_text = result[0]['generated_text'].split('[/INST]')[1].strip()

        logging.info(f"Generated response: '{response_text[:100]}...'")
        return jsonify({"response": response_text})
        
    except Exception as e:
        logging.error(f"Chat generation failed: {e}")
        return jsonify({"error": "Failed to generate chat response"}), 500

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    """Uploads and performs a basic EDA on a dataset."""
    if 'dataset' not in request.files:
        return jsonify({"error": "No dataset file provided"}), 400
    
    file = request.files['dataset']
    filename = file.filename
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    filepath = os.path.join('data', filename)
    file.save(filepath)
    state.dataset_path = filepath
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.json') or filename.endswith('.jsonl'):
            df = pd.read_json(filepath, lines=True if filename.endswith('.jsonl') else False)
        else:
            return jsonify({"error": "Unsupported file type. Use CSV or JSON/JSONL."}), 400

        if 'text' not in df.columns:
             return jsonify({"error": "Dataset must contain a 'text' column for training."}), 400

        # Basic EDA
        eda_results = {
            "shape": f"{df.shape[0]} rows, {df.shape[1]} columns",
            "columns": df.columns.tolist(),
            "head": df.head().to_string()
        }
        
        return jsonify({"message": "Dataset uploaded successfully", "eda": eda_results})
    except Exception as e:
        logging.error(f"Failed to process dataset: {e}")
        return jsonify({"error": f"Failed to read or process dataset. Error: {e}"}), 500


@app.route('/train', methods=['POST'])
def train_model():
    """Fine-tunes a model using the uploaded dataset and PEFT/LoRA."""
    if not state.dataset_path:
        return jsonify({"error": "No dataset has been uploaded for training."}), 400
    if not state.hf_token:
        return jsonify({"error": "Authentication token not set. Cannot download model or upload results."}), 401

    config = request.json
    base_model_id = config.get('model_id')
    new_model_name = config.get('new_model_name')
    epochs = int(config.get('epochs', 1))

    def generate_logs():
        # Redirect stdout to capture logs from libraries
        old_stdout = sys.stdout
        sys.stdout = log_stream = io.StringIO()
        
        try:
            yield "--- Starting Training Process ---\n"
            
            # 1. Load Dataset
            yield f"Loading dataset from {state.dataset_path}...\n"
            data = load_dataset("json", data_files=state.dataset_path, split="train")
            
            # 2. Load Model & Tokenizer for Training
            yield f"Loading base model '{base_model_id}' for fine-tuning...\n"
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                token=state.hf_token
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=state.hf_token)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Prepare for k-bit training
            model = prepare_model_for_kbit_training(model)
            
            # 3. Configure LoRA
            yield "Configuring LoRA...\n"
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            
            # 4. Set up Trainer
            yield "Setting up Hugging Face Trainer...\n"
            
            def formatting_prompts_func(example):
                return {'text': [f"<s>[INST] {s} [/INST]" for s in example['text']]}
            
            processed_data = data.map(formatting_prompts_func, batched=True)

            trainer = Trainer(
                model=model,
                train_dataset=processed_data,
                args=TrainingArguments(
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    num_train_epochs=epochs,
                    learning_rate=2e-4,
                    fp16=True,
                    logging_steps=1,
                    output_dir="outputs",
                    optim="paged_adamw_8bit"
                ),
                data_collator=lambda data: {'input_ids': torch.stack([torch.LongTensor(f['input_ids']) for f in data]),
                                            'attention_mask': torch.stack([torch.LongTensor(f['attention_mask']) for f in data]),
                                            'labels': torch.stack([torch.LongTensor(f['input_ids']) for f in data])}
            )
            # Tokenize after setting up trainer
            processed_data = processed_data.map(lambda samples: tokenizer(samples["text"], padding="max_length", truncation=True, max_length=512), batched=True)
            trainer.train_dataset = processed_data
            
            # 5. Start Training
            yield "--- Training starting now! ---\n"
            trainer.train()
            yield "\n--- Training finished! ---\n"
            
            # 6. Save and Push to Hub
            yield f"Saving LoRA adapter and pushing to Hugging Face Hub as '{new_model_name}'...\n"
            trainer.model.push_to_hub(new_model_name, token=state.hf_token)
            yield f"Successfully pushed model to Hugging Face Hub! You can now use it."

        except Exception as e:
            logging.error(f"Training failed: {e}", exc_info=True)
            yield f"\n--- ERROR DURING TRAINING ---\n{e}\n"
        finally:
            # Restore stdout and yield any captured logs
            final_logs = log_stream.getvalue()
            sys.stdout = old_stdout
            yield final_logs

    return Response(stream_with_context(generate_logs()), mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True)
