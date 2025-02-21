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
from fairlearn.metrics import MetricFrame, demographic_parity_difference
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



results_df = pd.read_csv("validation_pass.csv")
demographics_path =  "VideoDemographics.csv"
demographics_df = pd.read_csv(demographics_path)
merged_data = results_df.merge(demographics_df, on="ActorID")

true_labels = merged_data["true_labels"]
predictions = merged_data["predictions"]

id2label = {'0': 'neutral', '1': 'happy', '2': 'sad', '3': 'anger', '4': 'fear', '5': 'disgust'}

true_labels_id = [id2label[str(x)] for x in true_labels]
predictions_id = [id2label[str(x)] for x in predictions]
labels = list(id2label.values())

report = classification_report(true_labels_id, predictions_id, output_dict=True)
accuracy = accuracy_score(true_labels_id, predictions_id)
f1 = f1_score(true_labels, predictions, average="weighted")

conf_matrix = confusion_matrix(true_labels_id, predictions_id)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(true_labels_id), yticklabels=np.unique(true_labels_id))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_cnn.png")
plt.show()

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
pd.DataFrame({"Metric": ["Accuracy", "F1 Score"], "Value": [accuracy, f1]}).to_csv("metrics_cnn.csv")
pd.DataFrame(report).transpose().to_csv("classification_report_cnn.csv")
pd.DataFrame(metric_frame.by_group.to_dict()).transpose().to_csv("group_metrics_cnn.csv")
pd.DataFrame({"demographic_parity_difference": [demographic_parity_diff]}).to_csv("fairness_metrics_cnn.csv")