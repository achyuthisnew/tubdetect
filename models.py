import torch
import torch.nn as nn

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
        feat_dim = 512
        self.concept_predictor = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_concepts),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.concept_names = concepts

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
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
            _, concepts = self.cbm(x)
        logits = self.bnn_head(concepts)
        return logits, concepts
