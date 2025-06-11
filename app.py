import streamlit as st
from PIL import Image
import torch
import numpy as np
import torchvision.models as models

# Load your model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load and set up encoder
    encoder = models.resnet18(pretrained=False)
    encoder.fc = torch.nn.Identity()
    # Load your fine-tuned encoder weights
    fine_tune_model = FineTuneModel(encoder, feature_dim=512)
    ft_checkpoint = torch.load('fine_tune_checkpoint.pth', map_location=device)
    fine_tune_model.load_state_dict(ft_checkpoint['model_state_dict'])
    simclr_encoder = fine_tune_model.encoder
    simclr_encoder.eval()
    simclr_encoder = simclr_encoder.to(device)

    # 2. Load the trained CBM model
    cbm_model = ConceptBottleneckModel(simclr_encoder).to(device)
    cbm_model.load_state_dict(torch.load('cbm_model_image.pt', map_location=device))
    cbm_model.eval()
    for param in cbm_model.parameters():
        param.requires_grad = False  # Freeze CBM

    # 3. Wrap with BNN head
    model = CBMWithBNNHead(cbm_model, num_concepts=10).to(device)
    checkpoint = torch.load('cbm_bnn_head_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0)

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def predict_with_uncertainty(model, img_tensor, T=20):
    model.eval()
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(T):
            logits, concepts = model(img_tensor)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.append(prob)
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std, concepts.cpu().numpy()

st.title("Chest X-ray TB Detection (CBM+BNN)")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_tensor = preprocess_image(image).to(device)
    model = load_model()
    mean, std, concepts = predict_with_uncertainty(model, img_tensor, T=20)
    st.write(f"**TB Probability:** {mean[0]:.3f} ± {std[0]:.3f}")
    concept_names = [
        "Consolidation", "Cavitation", "Fibrosis",
        "Patchy/Consolidative Pattern", "Upper Lobe Involvement",
        "Pulmonary Opacities", "Volume Loss", "Architectural Distortion",
        "Diverse Parenchymal Patterns", "Normal"
    ]
    st.write("**Concepts:**")
    st.json({name: float(prob) for name, prob in zip(concept_names, concepts.flatten())})
