from datasets import load_dataset
import pandas as pd
import re
import string
from sentence_transformers import SentenceTransformer
import faiss
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

ds = load_dataset("mratanusarkar/Indian-Laws")

ds['train'].to_csv('train.csv')

df = pd.read_csv('train.csv')

sentences = df['act_title'].tolist()

sentences.extend(df['section'].tolist())

sentences.extend(df['law'].tolist())

sentences = [word for word in list(set(sentences)) if type(word) is str]

def preprocess_legal_text(text: str) -> str:
    """Clean and preprocess legal text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove page numbers and headers/footers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Clean up common legal document artifacts
    text = re.sub(r'_+', '', text)  # Remove underscores
    text = re.sub(r'-{2,}', '', text)  # Remove multiple dashes

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    return text.strip()

sentences = [preprocess_legal_text(word) for word in list(set(sentences)) if type(word) is str]
print('Print preprocessed sentences')

# initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens',device=device)
# create sentence embeddings
sentence_embeddings = model.encode(sentences)
print(sentence_embeddings.shape)

dim = sentence_embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatL2(dim)
index.add(sentence_embeddings)
print(index)
print('\n\nNow Try an query to be solved')
k = 4
query_embedding = model.encode(["The Aadhar Act"], convert_to_tensor=True)
query_embedding = query_embedding.cpu().detach().numpy()
# Search in FAISS
distances, indices = index.search(query_embedding, k)

# Option 1: retrieve from cleaned sentences
retrieved_chunks = [sentences[idx] for idx in indices[0]]

# Option 2: retrieve full dataframe row (comment out above if using this)
# retrieved_chunks = [df.iloc[idx].to_dict() for idx in indices[0]]

context = "\n".join(map(str, retrieved_chunks))
print("Retrieved Context:\n", context)
