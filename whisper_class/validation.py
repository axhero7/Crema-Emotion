from datasets import load_dataset, ClassLabel, load_from_disk
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification
import transformers
from utils.dataset_utils import CremaDataset
import os
from tqdm.auto import tqdm
import wandb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

crema_dataset = CremaDataset("distil-whisper/distil-medium.en")
crema_dataset.set_vector("actor-vector.hf", True)



BATCH_SIZE = 16

torch.manual_seed(42)
test_dataloader = DataLoader(crema_dataset.test_dataset, batch_size=BATCH_SIZE)


num_labels = len(crema_dataset.id2label)

transformers.set_seed(42)
model = AutoModelForAudioClassification.from_pretrained(
    crema_dataset.model_id, num_labels=num_labels, label2id=crema_dataset.label2id, id2label=crema_dataset.id2label
)
model.load_state_dict(torch.load('model_latest_1.pth'))

EPOCHS=1

num_training_steps = EPOCHS * len(test_dataloader)
model.to(device)

progress_bar = tqdm(range(num_training_steps))

wandb.init(project="crema_evaluation_with_fairness", 
           name="evaluation_run",
           config={})

validation_results = {"ActorID": [], "true_labels": [], "predictions": []}
    
# Get predictions on test data
model.eval()
for batch in test_dataloader:
    actor_ids = batch["path"]
    true_labels_batch = batch["labels"].numpy()
    
    with torch.no_grad():
        inputs = {"input_features": batch["input_features"].to(device), "labels": batch["labels"].to(device)}
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()

    # Append results to dictionary
    validation_results["ActorID"].extend(actor_ids)
    validation_results["true_labels"].extend(true_labels_batch)
    validation_results["predictions"].extend(preds)
    
    progress_bar.update(1)



results_df = pd.DataFrame(validation_results)
results_df.to_csv("validation_pass.csv")
demographics_path =  "VideoDemographics.csv"
demographics_df = pd.read_csv(demographics_path)

# Merge predictions with demographic info using actor ID
merged_data = results_df.merge(demographics_df, on="ActorID")
merged_data.to_csv("validation_pass_whisper.csv")

true_labels = merged_data["true_labels"]
predictions = merged_data["predictions"]

# Classification Report
report = classification_report(true_labels, predictions, output_dict=True)
wandb.log({"classification_report": report})

# Confusion Matrix
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
