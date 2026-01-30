import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, request, jsonify

# Include project root to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import HandwritingModel

app = Flask(__name__)

# Config
DATA_DIR = os.path.join('..', 'data', 'casia')
MODEL_PATH = os.path.join('..', 'best_model.pth') 
CANONICAL_PATH = os.path.join('..', 'canonical_embeddings.pt')
MAP_PATH = os.path.join(DATA_DIR, 'processed', 'class_map.pt')

# Globals
model = None
device = None
canonical_embeddings = None
idx_to_tag = {}

def load_resources():
    global model, device, canonical_embeddings, idx_to_tag
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load Map
    if not os.path.exists(MAP_PATH):
        # Fallback
        fallback_map = os.path.join('..', 'data', 'class_map.pt')
        if os.path.exists(fallback_map):
            tag_to_idx = torch.load(fallback_map)
        else:
            raise FileNotFoundError("Could not find class_map.pt")
    else:
        tag_to_idx = torch.load(MAP_PATH)
        
    # Invert map
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    num_classes = len(idx_to_tag)
    print(f"Loaded {num_classes} classes.")

    # Load Embeddings
    if os.path.exists(CANONICAL_PATH):
        canonical_embeddings = torch.load(CANONICAL_PATH, map_location=device)
        print(f"Loaded Canonical Embeddings: {canonical_embeddings.shape}")
    else:
        print("Warning: Canonical Embeddings not found. Search will fail.")

    # Load Model
    model = HandwritingModel(num_classes=num_classes)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        
        # Pruning check
        for key in state.keys():
             if 'weight_mask' in key:
                 layer_name = key.replace('.weight_mask', '')
                 module = model
                 for part in layer_name.split('.'):
                     module = getattr(module, part)
                 import torch.nn.utils.prune as prune
                 prune.identity(module, 'weight')
                 
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    else:
        print("Model file not found!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features'], dtype=np.float32) # (128, 7)
        
        # Add batch dim
        tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get embedding
            _, struct_emb = model(tensor)
            
            # Find closest canonical embedding
            # Distance: L2 or Cosine? 
            # Training used L2 (MSE). Let's use L2.
            
            # (1, D) vs (C, D)
            dists = torch.cdist(struct_emb, canonical_embeddings)
            
            # Top 5
            vals, indices = torch.topk(dists, k=5, largest=False)
            
            # Retrieve tags
            results = []
            for i in range(5):
                idx = indices[0][i].item()
                dist = vals[0][i].item()
                tag = idx_to_tag.get(idx, "?")
                
                # Decode tag?
                # GBK encoding usually involved with CASIA.
                # But tag_code might be correct string depending on loader.
                # Assuming tag_code is the character itself for now.
                try:
                    char = tag 
                    # If tag is hex string or bytes?
                    # Usually src/dataset.py handles decoding.
                    # Let's assume it's a string.
                except:
                    char = tag
                
                results.append({
                    "char": str(char),
                    "dist": f"{dist:.4f}"
                })
                
            return jsonify({"status": "success", "matches": results})
            
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=5000)
