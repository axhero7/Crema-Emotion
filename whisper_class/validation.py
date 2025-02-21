from datasets import load_dataset, ClassLabel, load_from_disk
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification
import transformers
from utils.dataset_utils import CremaDataset
import os
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def create_test_dataloader():
    crema_dataset = CremaDataset("distil-whisper/distil-medium.en")
    crema_dataset.set_vector("actor-vector.hf", True)
    BATCH_SIZE = 16
    torch.manual_seed(42)
    test_dataloader = DataLoader(crema_dataset.test_dataset, batch_size=BATCH_SIZE)
    return test_dataloader, crema_dataset

def create_model(from_key, crema_dataset, device):
    num_labels = len(crema_dataset.id2label)

    transformers.set_seed(42)
    model = AutoModelForAudioClassification.from_pretrained(
        crema_dataset.model_id, num_labels=num_labels, label2id=crema_dataset.label2id, id2label=crema_dataset.id2label
    )
    model.load_state_dict(torch.load('model_latest_1.pth'))
    model.to(device)
    return model

def val_pass(model, test_dataloader, device):
    EPOCHS=1
    num_training_steps = EPOCHS * len(test_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    validation_results = {"ActorID": [], "true_labels": [], "predictions": []}
    model.eval()
    for batch in test_dataloader:
        actor_ids = batch["path"]
        true_labels_batch = batch["labels"].numpy()
        
        with torch.no_grad():
            inputs = {"input_features": batch["input_features"].to(device), "labels": batch["labels"].to(device)}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        validation_results["ActorID"].extend(actor_ids)
        validation_results["true_labels"].extend(true_labels_batch)
        validation_results["predictions"].extend(preds)
        progress_bar.update(1)

    results_df = pd.DataFrame(validation_results)
    results_df.to_csv("validation_pass.csv")
    return results_df

test_dataloader, crema_dataset = create_test_dataloader()
if (os.path.exists("validation_pass_whisper.csv")):
    print("Loading validation pass...")
    results_df = pd.read_csv("validation_pass_whisper.csv")
else:
    print("Creating test dataloader and model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model("model_latest_1.pth", crema_dataset, device)
    results_df = val_pass(model, test_dataloader, device)

demographics_path =  "VideoDemographics.csv"
demographics_df = pd.read_csv(demographics_path)

merged_data = results_df.merge(demographics_df, on="ActorID")

true_labels = list(merged_data["true_labels"])
predictions = list(merged_data["predictions"])

true_labels_id = [crema_dataset.id2label[str(x)] for x in true_labels]
predictions_id = [crema_dataset.id2label[str(x)] for x in predictions]
labels = list(crema_dataset.id2label.values())

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
plt.savefig("confusion_matrix_whisper.png")
plt.show()

sensitive_features = merged_data["Race"] 
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
pd.DataFrame({"Metric": ["Accuracy", "F1 Score"], "Value": [accuracy, f1]}).to_csv("metrics_whisper.csv")
pd.DataFrame(report).transpose().to_csv("classification_report_whisper.csv")
pd.DataFrame(metric_frame.by_group.to_dict()).transpose().to_csv("group_metrics_whisper.csv")
pd.DataFrame({"demographic_parity_difference": [demographic_parity_diff]}).to_csv("fairness_metrics_whisper.csv")