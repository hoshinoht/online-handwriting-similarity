from src.model import HandwritingModel
from src.preprocess import preprocess_stroke
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, request, jsonify
import struct
import re

# Include project root to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


app = Flask(__name__)

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')
CANONICAL_PATH = os.path.join(BASE_DIR, 'models', 'canonical_embeddings.pt')
MAP_PATH = os.path.join(DATA_DIR, 'processed', 'class_map.pt')
ALT_MAP_PATH = os.path.join(
    BASE_DIR, 'data', 'casia', 'processed', 'class_map.pt')

# Globals
model = None
device = None
canonical_embeddings = None
idx_to_tag = {}


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def decode_tag(tag_value):
    if isinstance(tag_value, str):
        return tag_value
    if isinstance(tag_value, (int, np.integer)):
        code = int(tag_value)
        candidates = []
        try:
            raw_le = struct.pack('<H', code)
            candidates.append(raw_le.decode('gbk', errors='ignore').strip())
        except Exception:
            pass
        try:
            raw_be = struct.pack('>H', code)
            candidates.append(raw_be.decode('gbk', errors='ignore').strip())
        except Exception:
            pass
        for cand in candidates:
            if cand and _CJK_RE.search(cand):
                return cand
        for cand in candidates:
            if cand:
                return cand
        return str(tag_value)
    return str(tag_value)


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

    # Load model state first to infer num_classes
    state = None
    inferred_num_classes = None
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        if 'arcface.weight' in state:
            inferred_num_classes = state['arcface.weight'].shape[0]
        elif 'fc_char.3.weight' in state:
            inferred_num_classes = state['fc_char.3.weight'].shape[0]

    # Load Map (prefer map that matches inferred num_classes)
    print(f"Resolved MAP_PATH: {MAP_PATH}")
    candidate_maps = [MAP_PATH, ALT_MAP_PATH]
    tag_to_idx = None
    for path in candidate_maps:
        if os.path.exists(path):
            temp_map = torch.load(path)
            if inferred_num_classes is None or len(temp_map) == inferred_num_classes:
                tag_to_idx = temp_map
                print(f"Using class map: {path}")
                break
    if tag_to_idx is None:
        raise FileNotFoundError(
            f"Could not find matching class_map.pt. Tried: {candidate_maps}")

    # Invert map
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    num_classes = len(idx_to_tag)
    print(f"Loaded {num_classes} classes.")

    # Load Embeddings
    if os.path.exists(CANONICAL_PATH):
        canonical_embeddings = torch.load(CANONICAL_PATH, map_location=device)
        print(f"Loaded Canonical Embeddings: {canonical_embeddings.shape}")
        if inferred_num_classes is not None and canonical_embeddings.shape[0] != inferred_num_classes:
            print("Warning: Canonical embeddings size does not match model classes.")
    else:
        print("Warning: Canonical Embeddings not found. Search will fail.")

    # Load Model — infer input_dim from checkpoint for backward compat
    input_dim = 8  # default
    if state is not None and 'start_conv.weight' in state:
        input_dim = state['start_conv.weight'].shape[1]
    model = HandwritingModel(num_classes=num_classes, input_dim=input_dim)
    if state is None and os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)

    if state is not None:
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


@app.route('/predict_raw', methods=['POST'])
def predict_raw():
    """Takes raw stroke points and does preprocessing server-side using Python code."""
    try:
        data = request.json

        # Accept strokes array (preferred) or flat points (backward compat)
        if 'strokes' in data:
            strokes = data['strokes']
            all_points = []
            for stroke in strokes:
                for j, pt in enumerate(stroke):
                    pen_state = 1.0 if j == len(stroke) - 1 else 0.0
                    all_points.append([pt[0], pt[1], pen_state])
            raw_points = np.array(all_points, dtype=np.float32)  # (N, 3)
        else:
            raw_points = np.array(data['points'], dtype=np.float32)  # (N, 2)

        print(f"Received {len(raw_points)} raw points, dims={raw_points.shape[1]}")

        features = preprocess_stroke(raw_points)

        print(f"Preprocessed features shape: {features.shape}")

        # Add batch dim
        tensor = torch.from_numpy(features.astype(
            np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            char_logits, struct_emb = model(tensor)
            probs = F.softmax(char_logits, dim=1)
            vals, indices = torch.topk(probs, k=5, largest=True)

            results = []
            for i in range(5):
                idx = indices[0][i].item()
                confidence = vals[0][i].item()
                tag = idx_to_tag.get(idx, "?")
                char = decode_tag(tag)
                result = {
                    "char": str(char),
                    "dist": f"{confidence:.4f}"
                }
                if canonical_embeddings is not None and idx < canonical_embeddings.shape[0]:
                    sim = F.cosine_similarity(
                        struct_emb, canonical_embeddings[idx].unsqueeze(0)
                    ).item()
                    result["similarity"] = f"{sim:.4f}"
                results.append(result)

            return jsonify({"status": "success", "matches": results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features'], dtype=np.float32)  # (128, 7)

        # Debug: Print feature ranges
        print(f"Received features shape: {features.shape}")
        for i in range(7):
            print(
                f"  Feature {i}: min={features[:, i].min():.4f}, max={features[:, i].max():.4f}, mean={features[:, i].mean():.4f}")

        # Debug: Check point-to-point distances
        xy = features[:, :2]
        dists = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
        print(
            f"  Point distances: max={dists.max():.4f}, mean={dists.mean():.4f}")

        # Add batch dim
        tensor = torch.from_numpy(features).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get classification logits and structural embedding
            char_logits, struct_emb = model(tensor)

            # Use classification head (softmax) for predictions - more reliable
            probs = F.softmax(char_logits, dim=1)

            # Top 5 predictions
            vals, indices = torch.topk(probs, k=5, largest=True)

            # Retrieve tags
            results = []
            for i in range(5):
                idx = indices[0][i].item()
                confidence = vals[0][i].item()
                tag = idx_to_tag.get(idx, "?")
                char = decode_tag(tag)

                result = {
                    "char": str(char),
                    "dist": f"{confidence:.4f}"
                }
                if canonical_embeddings is not None and idx < canonical_embeddings.shape[0]:
                    sim = F.cosine_similarity(
                        struct_emb, canonical_embeddings[idx].unsqueeze(0)
                    ).item()
                    result["similarity"] = f"{sim:.4f}"
                results.append(result)

            return jsonify({"status": "success", "matches": results})

    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=5000)
