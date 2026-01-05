from fastapi import FastAPI
from fastapi.responses import FileResponse
from pypdf import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import uuid
import os

app = FastAPI()

PDF_PATH = "data/sample.pdf"

reader = PdfReader(PDF_PATH)
documents = []

for page in reader.pages:
    text = page.extract_text()
    if text:
        documents.append(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents, convert_to_numpy=True)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

@app.get("/")
def root():
    return {"status": "voice-only RAG running"}

@app.get("/ask")
def ask(question: str):
    q_embedding = model.encode([question], convert_to_numpy=True)
    _, ids = index.search(q_embedding, k=1)

    answer = documents[ids[0][0]]

    audio_file = f"answer_{uuid.uuid4()}.mp3"
    tts = gTTS(answer)
    tts.save(audio_file)

    return FileResponse(audio_file, media_type="audio/mpeg", filename="answer.mp3")
