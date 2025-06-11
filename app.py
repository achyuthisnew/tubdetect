import streamlit as st
from PIL import Image
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
from models import FineTuneModel, ConceptBottleneckModel, CBMWithBNNHead

# Load your model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the number of concepts and concept names (adjust as needed)
NUM_CONCEPTS = 10
concepts = [
    "Consolidation", "Cavitation", "Fibrosis",
    "Patchy/Consolidative Pattern", "Upper Lobe Involvement",
    "Pulmonary Opacities", "Volume Loss", "Architectural Distortion",
    "Diverse Parenchymal Patterns", "Normal"
]

class FineTuneModel(nn.Module):
    def __init__(self, encoder, feature_dim=512):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.classifier(self.encoder(x)))

class ConceptBottleneckModel(nn.Module):
    def __init__(self, simclr_encoder, num_concepts=NUM_CONCEPTS, hidden_dim=32):
        super().__init__()
        self.encoder = simclr_encoder
        feat_dim = 512  # From FineTuneModel's feature_dim
        # Concept predictor
        self.concept_predictor = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_concepts),
            nn.Sigmoid()
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.concept_names = concepts

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)  # [batch, 512]
        concepts = self.concept_predictor(features)
        logits = self.classifier(concepts)
        return logits, concepts

class CBMWithBNNHead(nn.Module):
    def __init__(self, cbm_model, num_concepts, hidden_dim=32, dropout_p=0.1):
        super().__init__()
        self.cbm = cbm_model
        self.bnn_head = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        with torch.no_grad():
            _, concepts = self.cbm(x)  # Get concept vector from frozen CBM
        logits = self.bnn_head(concepts)
        return logits, concepts


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
