from datasets import load_dataset, ClassLabel, load_from_disk
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification
import transformers
from cremadata import CremaSoundDataset
from cremanet import CNN_Net
# from train import *
import os
from tqdm.auto import tqdm
import torchaudio
# import wandb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
# from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd



BATCH_SIZE = 16
LEARNING_RATE = 1e-3
ANNOTATIONS_FILE = "SentenceFilenames.csv"
AUDIO_DIR = "AudioWAV"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE*4

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device: ", device)

full_annotations = pd.read_csv(ANNOTATIONS_FILE)

train_ratio = 0.7
train_size = int(train_ratio * len(full_annotations))
test_size = len(full_annotations) - train_size
train_indices, test_indices = train_size, train_size + test_size

test_transforms = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

test_dataset = CremaSoundDataset(
    full_annotations.iloc[train_indices:test_indices].reset_index(drop=True),  
    AUDIO_DIR,
    test_transforms,
    SAMPLE_RATE,
    NUM_SAMPLES,
    device
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
torch.manual_seed(42)
model_0 = CNN_Net().to(device)
model_0.load_state_dict(torch.load('cnn_param.pth'))


progress_bar = tqdm(range(len(test_dataloader)))

# wandb.init(project="crema_evaluation_with_fairness", 
#            name="evaluation_run",
#            config={})

    

def validation_pass(model, data):
    validation_results = {"ActorID": [], "true_labels": [], "predictions": []}
    with torch.no_grad():
        model.eval()
        for index, (input, label, actor_ids) in enumerate(data):
            input, label, actor_ids = input.to(device), label.to(device), list(actor_ids)
            y_pred = model(input)
            outputs = torch.nn.functional.softmax(y_pred,0)
            outputs = outputs.argmax(dim=1)

            for index in range(len(actor_ids)):
                validation_results["ActorID"].append(int(actor_ids[index].item()))
                validation_results["true_labels"].append(int(label[index].item()))
                validation_results["predictions"].append(int(outputs[index].item()))
            progress_bar.update(1)

    return validation_results



results_df = pd.DataFrame(validation_pass(model_0, test_dataloader))
results_df.to_csv("validation_pass.csv")
demographics_path =  "VideoDemographics.csv"
demographics_df = pd.read_csv(demographics_path)
merged_data = results_df.merge(demographics_df, on="ActorID")

true_labels = merged_data["true_labels"]
predictions = merged_data["predictions"]

report = classification_report(true_labels, predictions, output_dict=True)
print(report)
wandb.log({"classification_report": report})


conf_matrix = confusion_matrix(true_labels, predictions)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close(fig)


sensitive_features = merged_data["Race"]  # Replace with demographic column
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
    },
    y_true=true_labels,
    y_pred=predictions,
    sensitive_features=sensitive_features
)

demographic_parity_diff = demographic_parity_difference(
    true_labels, predictions, sensitive_features=sensitive_features
)
equalized_odds_diff = equalized_odds_difference(
    true_labels, predictions, sensitive_features=sensitive_features
)

# Log fairness metrics
wandb.log({
    "fairness_metrics": {
        "demographic_parity_difference": demographic_parity_diff,
        "equalized_odds_difference": equalized_odds_diff
    },
    "group_metrics": metric_frame.by_group.to_dict()
})

print("Final metrics and fairness assessment logged.")
wandb.finish()
