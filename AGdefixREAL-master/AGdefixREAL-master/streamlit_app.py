import streamlit as st
import torch
import torch.nn as nn
import re
import pickle
from gdown import download

# Model class (must match training)
class DeFixMatchTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_classes=4):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, text):
        return self.fc(self.embedding(text))

# Tokenizer (must match training)
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load resources with caching
@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?id=18byc4T1LAuQG6KVeD9Uind8qJ227AaaV"
    model_path = download(model_url, quiet=True)
    return torch.load(model_path, map_location='cpu')

@st.cache_resource
def load_vocab():
    vocab_url = "https://drive.google.com/uc?id=1qQJfXv3kblXubDWUh5j6xiLsqzdeXVXw"
    vocab_path = download(vocab_url, quiet=True)
    with open(vocab_path, 'rb') as f:
        return pickle.load(f)

# Main app
def main():
    st.title("AG News Classifier")
    st.write("Classify news into: World, Sports, Business, or Sci/Tech")
    
    # Load resources
    vocab = load_vocab()
    model_state = load_model()
    
    model = DeFixMatchTextModel(len(vocab))
    model.load_state_dict(model_state)
    model.eval()
    
    # Input text
    text = st.text_area("Enter news text:", "Apple announced new products yesterday...")
    
    if st.button("Classify"):
        # Preprocess
        tokens = [vocab.get(t, vocab["<unk>"]) for t in tokenize(text)]
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(output).item()
        
        # Show results
        classes = ['World', 'Sports', 'Business', 'Sci/Tech']
        st.subheader(f"Prediction: {classes[pred]}")
        
        st.write("Confidence:")
        for i, cls in enumerate(classes):
            st.progress(float(probs[i]), text=f"{cls}: {probs[i]*100:.1f}%")

if __name__ == "__main__":
    main()
