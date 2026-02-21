import os
import torch
import chromadb
import fitz
import io
from typing import List, Dict
from PIL import Image
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

MODEL_ID_VISION = "ayyuce/medgemma-dermatology-isic2019-full-1ep"
MODEL_ID_THINKING = "google/gemma-2-9b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

import sys
if "google.colab" in sys.modules and not os.environ.get("VERTEX_PRODUCT"):
    from google.colab import userdata
    try:
        os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
    except:
        os.environ["HF_TOKEN"] = "your_awesome_hf_token"
else:
    if os.environ.get("VERTEX_PRODUCT") == "COLAB_ENTERPRISE":
        os.environ["HF_HOME"] = "/content/hf"
    from huggingface_hub import get_token
    if get_token() is None:
        from huggingface_hub import notebook_login
        notebook_login()

vision_pipe = None
tokenizer_logic = None
model_logic = None
retriever = None

def get_image_description(image, prompt="Describe the dermatological features of this image used in a medical textbook."):
    """Helper to get description from the vision pipeline"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]
    output = vision_pipe(messages, max_new_tokens=200)
    try:
        generated_content = output[0]["generated_text"][-1]["content"]
        return generated_content.strip()
    except (KeyError, IndexError, TypeError):
        return "Medical image showing dermatological features."

def process_pdf_multimodal(pdf_path):
    """
    Extracts Text AND Images from PDF.
    Uses the Vision Model to caption the images so they can be indexed.
    """
    print(f"Processing PDF Multimodally: {pdf_path}")
    doc = fitz.open(pdf_path)

    text_content = ""
    extracted_image_docs = []

    for page in doc:
        text_content += page.get_text()

        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                if pil_image.width < 100 or pil_image.height < 100:
                    continue

                print(f"   ...Captioning image {img_index} on page {page.number}...")
                description = get_image_description(pil_image)

                doc_content = f"[REFERENCE IMAGE FROM TEXTBOOK PAGE {page.number}]: {description}"
                extracted_image_docs.append(Document(page_content=doc_content, metadata={"source": pdf_path, "type": "image_caption", "page": page.number}))

            except Exception as e:
                print(f"   Error processing image {img_index}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_docs = text_splitter.create_documents([text_content])

    combined_docs = text_docs + extracted_image_docs
    return combined_docs

def load_models():
    global vision_pipe, tokenizer_logic, model_logic, retriever

    if vision_pipe is not None:
        return

    print("Loading Vision Model (Pipeline)...")
    vision_pipe = pipeline(
        "image-text-to-text",
        model=MODEL_ID_VISION,
        model_kwargs={"quantization_config": bnb_config},
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print("Loading Logic Model...")
    tokenizer_logic = AutoTokenizer.from_pretrained(MODEL_ID_THINKING)
    model_logic = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_THINKING,
        quantization_config=bnb_config,
        device_map="cuda:0"
    )

    print("Building Multimodal RAG Vector DB from PDF...")
    pdf_path = "derm_book.pdf"

    docs = []
    if os.path.exists(pdf_path):
        try:
            docs = process_pdf_multimodal(pdf_path)
            print(f" Successfully loaded {len(docs)} chunks (Text + Image Captions) from {pdf_path}")
        except Exception as e:
            print(f" Error loading PDF: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Warning: '{pdf_path}' not found! RAG context will be empty.")

    if not docs:
        docs = [Document(page_content="No guidelines available. Rely on visual analysis.")]

    print("Embedding and Indexing...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    print("All models loaded!")

class AgenticDermatologist:
    def __init__(self):
        self.history = []
        load_models()

    def agent_vision_describe(self, image_path):
        """Agent A: The Semeiotics Agent - Describes visual features using Pipeline"""
        print(f"Analyzing patient image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        description = get_image_description(image, "Describe the dermatological features of this image. Be detailed and specific.")

        if not description:
            description = "Lesion shows irregular borders and color variation."

        return description

    def agent_intake_interview(self, patient_input):
        """Agent B: The Intake Nurse - Asks follow-up questions"""
        history_prompt = f"Patient says: '{patient_input}'. Generate one short follow-up medical question to gather more information about their symptoms."

        inputs = tokenizer_logic(history_prompt, return_tensors="pt").to("cuda:0")
        generate_ids = model_logic.generate(**inputs, max_new_tokens=150)

        input_len = inputs.input_ids.shape[1]
        generated_text_ids = generate_ids[0][input_len:]
        question = tokenizer_logic.decode(generated_text_ids, skip_special_tokens=True).strip()

        self.history.append(f"Patient: {patient_input}")
        return question

    def agent_diagnostician(self, visual_description, patient_history):
        """Agent C: The Diagnostician - Combines all info for diagnosis"""
        try:
            # RAG Retrieval: Will now pull Text Chunks AND Image Descriptions from the PDF
            print("Retrieving multimodal context...")
            retrieved_docs = retriever.invoke(visual_description)
            rag_context = "\\n".join([f"- {doc.page_content}" for doc in retrieved_docs])
        except Exception as e:
            print(f"Retrieval error: {e}")
            rag_context = "Guideline: Check for asymmetry and color variation."

        final_prompt = f"""You are an expert Dermatologist providing a clinical assessment.

[PATIENT VISUAL FINDINGS]: {visual_description}

[PATIENT HISTORY]: {patient_history}

[TEXTBOOK REFERENCE MATERIAL (Text & Similar Images)]:
{rag_context}

TASK: Provide a detailed diagnostic assessment including:
1. Most likely diagnosis based on visual and clinical findings.
2. If the Reference Material mentions similar images, note that comparison.
3. Key features that support this diagnosis.
4. Recommended next steps.

Be specific and reference the visual findings and guidelines."""

        messages = [{"role": "user", "content": final_prompt}]
        text_input = tokenizer_logic.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer_logic(text_input, return_tensors="pt").to("cuda:0")

        generate_ids = model_logic.generate(**inputs, max_new_tokens=500)
        input_len = inputs.input_ids.shape[1]
        generated_text_ids = generate_ids[0][input_len:]
        diagnosis = tokenizer_logic.decode(generated_text_ids, skip_special_tokens=True).strip()

        return diagnosis
