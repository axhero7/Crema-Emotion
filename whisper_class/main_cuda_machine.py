from datasets import load_dataset, ClassLabel, load_from_disk
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification
import transformers
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import AutoFeatureExtractor
from utils.dataset_utils import CremaDataset
import os
from tqdm.auto import tqdm
import wandb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

crema_dataset = CremaDataset("distil-whisper/distil-medium.en")
crema_dataset.set_vector("actor-vector.hf", True)



BATCH_SIZE = 6

torch.manual_seed(42)
train_dataloader = DataLoader(crema_dataset.train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(crema_dataset.test_dataset, batch_size=BATCH_SIZE)


num_labels = len(crema_dataset.id2label)

transformers.set_seed(42)
model = AutoModelForAudioClassification.from_pretrained(
    crema_dataset.model_id, num_labels=num_labels, label2id=crema_dataset.label2id, id2label=crema_dataset.id2label
)

optimizer = AdamW(model.parameters(), lr=1e-5)
EPOCHS = 10

num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.to(device)

progress_bar = tqdm(range(num_training_steps))

wandb.init(project="crema_evaluation_with_fairness", 
           name="train_run",
           config={})

model.train()
for epoch in range(EPOCHS):
    for idx, batch in enumerate(train_dataloader):
        inputs = {"input_features": batch["input_features"].to(device), "labels": batch["labels"].to(device)}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        progress_bar.update(1)
        if idx%50==0:
            print("idx: ", idx, " loss: ", loss)
        wandb.log({
            "train_loss": loss.item(),
            "learning_rate": lr_scheduler.get_last_lr()[0]  # Extract the current learning rate
        })
    
    model.eval()
    true_labels, predictions = [], []
    for batch in test_dataloader:
        with torch.inference_mode():
            inputs = {"input_features": batch["input_features"].to(device), "labels": batch["labels"].to(device)}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            true_labels.extend(batch["labels"].cpu().numpy())
            predictions.extend(preds.cpu().numpy())
    val_accuracy = accuracy_score(true_labels, predictions)
    val_f1 = f1_score(true_labels, predictions, average="weighted")
    wandb.log({
            "val_accuracy": val_accuracy,
            "val_f1_score": val_f1,
            "epoch": epoch + 1
    })
torch.save(model.state_dict(), "model_latest.pth")

