import torch

def load_model(model_path, device):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def predict_proba(model, x):
    with torch.no_grad():
        out = model(x)
        return float(out.reshape(-1)[0].item())

def predict_class(model, x, threshold = 0.5):
    probability = predict_proba(model, x)
    class_id = int(probability >= threshold)
    class_name = "signal" if class_id == 1 else "noise"

    return class_id, class_name, probability