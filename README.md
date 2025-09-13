# SLM-trainer-ui
To train SLM model via huggingFace

How to Run Your SLM Studio
1. Backend Setup (Python):

Prerequisites: Make sure you have Python 3.8+ and pip installed. A modern NVIDIA GPU with CUDA is highly recommended for reasonable performance.

Create a Project Folder: Create a new folder for your project.

Save Files: Place backend.py and requirements.txt inside this folder.

Create a Virtual Environment (Recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
* **Install Dependencies:**
```bash
pip install -r requirements.txt
* **Run the Backend Server:**
```bash
flask run
    The backend will now be running at `http://127.0.0.1:5000`.

2. Frontend Setup (React):

Prerequisites: You'll need Node.js and npm (or yarn).

Create a React App: In a separate terminal, navigate to your project directory and run:

Bash

npx create-react-app slm-frontend
cd slm-frontend
* **Install Tailwind CSS:** Follow the official guide to add Tailwind CSS to your Create React App project. This is a crucial step for the UI to render correctly.
Replace App.js: Open the src/ folder inside slm-frontend. Delete the existing App.js file and create a new file named frontend.jsx. Copy and paste the entire content of the React file I provided into this new frontend.jsx file.

Update index.js: Change the import in src/index.js from './App' to './frontend.jsx'.

Start the Frontend:

Bash

npm start
    Your browser should open to `http://localhost:3000`, where you can see and interact with the application.

Suggestions and Important Notes
Model Quantization: The backend uses bitsandbytes to load models in 4-bit precision. This drastically reduces the VRAM required, allowing you to run larger models on consumer-grade GPUs.

Efficient Fine-Tuning: I've implemented training with PEFT/LoRA, which is a state-of-the-art technique. Instead of retraining the entire model (which requires immense resources), it trains a small "adapter" layer on top. This is much faster and more memory-efficient.

Streaming Logs: The training process can be long. To provide real-time feedback, the backend streams logs directly to the frontend UI, so you can monitor the progress live.

Dataset Format: For the training to work, your uploaded dataset (CSV or JSONL) must have a 'text' column. Each row in this column should be a single training example (e.g., a question-answer pair, a story, or a code snippet).

Resource Warning: Fine-tuning, even with these optimizations, is computationally intensive. The process will be slow without a GPU, and you'll need sufficient RAM and VRAM depending on the model size.