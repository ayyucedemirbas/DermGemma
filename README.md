# DermGemma: Agentic AI Dermatologist 🩺🤖

**📖 Executive Summary**

DermGemma is an advanced, multi-agent artificial intelligence system designed to assist in dermatological analysis. It combines Fine-Tuned Vision-Language Models (VLMs), Large Language Models (LLMs), and Multimodal Retrieval-Augmented Generation (RAG) to simulate a real-world clinical workflow.

The system operates through three distinct AI agents:

- **The Semeiotics Agent (Vision):** Analyzes skin lesion imagery.

- **The Intake Nurse (Logic/Chat):** Conducts a dynamic patient interview based on symptoms.

- **The Diagnostician (Logic + RAG):** Synthesizes visual findings, patient history, and retrieved clinical guidelines (text and textbook images) to formulate a comprehensive assessment.

**🏗️ System Architecture**

DermGemma leverages a state-of-the-art quantized architecture to run efficiently on consumer/Colab GPUs (e.g., NVIDIA T4, L4, A100).

**1. Model Stack**

**ision Model:** ayyuce/medgemma-dermatology-isic2019-full-1ep (Fine-tuned from google/medgemma-4b-it). Responsible for visual semeiotics.

**Logic Model:** google/gemma-2-2b-it. Acts as the conversational nurse and the final diagnostician.

**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2. Creates vector representations for the RAG system.

**Quantization:** Both generative models are loaded in 4-bit precision (NF4) using BitsAndBytesConfig to minimize VRAM usage while maintaining bfloat16 compute precision.

**2. Multimodal RAG Pipeline**

Unlike standard RAG systems that only parse text, DermGemma features a Multimodal Document Parser:

- Reads a clinical textbook using PyMuPDF (fitz).

- Extracts raw text and chunks it using RecursiveCharacterTextSplitter.

- Extracts images from the PDF pages, passes them through the Vision Model to generate detailed captions, and stores these captions as text chunks embedded with metadata (page number, image source).

- Everything is vectorized and stored in a local ChromaDB instance.

**3. Agentic Workflow**

**Image Upload:** User uploads a lesion image.

**Vision Inference:** The Vision Agent extracts clinical features (Asymmetry, Border, Color, etc.).

**Interview Loop:** User inputs symptoms. The Logic Agent (Nurse) generates context-aware follow-up questions.

**RAG Retrieval:** The visual description queries ChromaDB for relevant textbook excerpts and reference image descriptions.

**Final Diagnosis:** The Diagnostician Agent processes the visual data, interview history, and RAG context to output a structured assessment.

**📁 Project Structure**

This repository consists of four primary operational paradigms (derived from the codebase):

**main_code.py:** Contains the AgenticDermatologist class, RAG logic, and model loading functions.

**app.py:** The Streamlit frontend providing an interactive, step-by-step UI for the agentic workflow.

**DermGemma_fine_tune.ipynb:** The dataset preparation and LoRA (Low-Rank Adaptation) fine-tuning pipeline using the ISIC 2019 dataset.

**DermGemma.ipynb:** A standalone Colab notebook demonstrating inference without the Streamlit UI.

**🚀 Installation & Setup**

**Prerequisites**

**Hardware:** An NVIDIA GPU with at least 15GB VRAM is highly recommended (e.g., Google Colab T4).

**API Keys:** A Hugging Face account and an Access Token (HF_TOKEN) to download the Gemma models.

**Environment Setup**

Clone the repository:

    git clone [https://github.com/ayyucedemirbas/DermGemma.git](https://github.com/ayyucedemirbas/DermGemma.git)
    cd DermGemma


**Install Python dependencies:**

    pip install torch torchvision
    pip install transformers datasets peft accelerate bitsandbytes trl huggingface_hub
    pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers
    pip install pypdf pymupdf pillow pandas
    pip install streamlit


**Provide Knowledge Base Material:**
Place a PDF file in the root directory. This serves as the textbook for the Multimodal RAG system. (If omitted, the system falls back to a default guideline).

**Authentication:**
Set your Hugging Face token as an environment variable:

    export HF_TOKEN="your_hf_token_here"


**💻 How to Run**

**Option A: Running the Interactive Streamlit Web App**

- To run the full multi-agent UI locally or on a server:

    streamlit run app.py


**Note:** The first run will take several minutes as it downloads the 4B and 2B models, parses the PDF file, runs vision inference on all PDF images, and builds the ChromaDB vector store.

**Option B:** Running the Inference Notebook

- Open DermGemma.ipynb in Google Colab or Jupyter:

- Ensure your Hugging Face token is added to Colab Secrets as HF_TOKEN.

- Upload a PDF file and an example image to the Colab workspace.

- Run all cells. The terminal will output the Vision, Nurse, and Diagnostician results sequentially.

**Option C: Fine-Tuning the Vision Model**

- Open DermGemma_fine_tune.ipynb to train your own vision model.

- The script automatically downloads the ISIC 2019 dataset (images and ground truth CSV).

- It maps one-hot encoded labels to clinical prompt-response pairs.

- It initializes google/medgemma-4b-it with LoRA (Rank=16, Alpha=32).

- Training outputs are saved to ./medgemma-dermatology-finetuned.

**🛠️ How to Customize and Modify**

- The modular design of DermGemma makes it highly extensible. Here is how you can tweak the system:

**1. Changing the Knowledge Base (RAG)**

- To change what the Diagnostician bases its facts on, simply replace derm_book.pdf with any other clinical guidelines PDF.

- Code location: main_code.py -> load_models()

- Modification: Change pdf_path = "derm_book.pdf" to your new file name. Delete the existing ./chroma_db folder to force the system to re-index the new document on startup.

**2. Upgrading or Changing the Models**

You can swap out the underlying LLMs/VLMs for faster or more accurate versions.

Code location: main_code.py -> Global variables.

Modification:

    # Example: Upgrading to a larger reasoning model
    MODEL_ID_THINKING = "google/gemma-2-9b-it" 
    
    # Example: Using the base model instead of the fine-tuned one
    MODEL_ID_VISION = "google/medgemma-4b-it" 


**3. Modifying Agent Personas & Prompts**

The behavior of the agents is entirely dictated by their system prompts.

**Semeiotics Agent:** Change the prompt in agent_vision_describe() (e.g., ask it to output in JSON format).

**Intake Nurse:** Change the prompt in agent_intake_interview(). You can make the nurse ask multiple questions by altering the max_new_tokens and prompt instructions.

**Diagnostician:** Edit final_prompt in agent_diagnostician(). You can enforce a specific medical framework (e.g., "Use the ABCDE melanoma framework specifically").

**4. Tweaking Fine-Tuning Hyperparameters**

If you wish to fine-tune the model on a different dataset or achieve better convergence:

**Code location:** DermGemma_fine_tune.ipynb -> TrainingArguments and LoraConfig.

**Modifications:** Increase num_train_epochs from 1 to 3. Change per_device_train_batch_size based on your VRAM. To train deeper into the model, add more target modules in the LoraConfig (e.g., MLP layers).

**⚠️ Disclaimer**

This project is for educational, research, and experimental purposes only.
DermGemma is an AI assistant and does not provide official medical diagnoses. The outputs generated by the models should not be interpreted as professional medical advice. Always consult a licensed healthcare professional or dermatologist for medical concerns.
