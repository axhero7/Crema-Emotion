from datasets import load_dataset, ClassLabel, load_from_disk
import torch
import evaluate
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


RANDOM_SEED=42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

<<<<<<< HEAD
crema_dataset = CremaDataset("distil-whisper/distil-medium.en")
crema_dataset.set_vector("actor-vector.hf", True)
=======
id2label = {'0':'neutral', '1': 'happy','2': 'sad', '3': 'anger', '4': 'fear', '5': 'disgust'}
label2id = dict()
for i in id2label:
    label2id[id2label[str(i)]] = str(i)

dataset = dataset.select_columns(['audio', 'label']) #This is simple classifcation task, whisper is already trained so no need for sentence.
dataset = dataset.train_test_split(test_size=0.3, seed=RANDOM_SEED)


model_id = "distil-whisper/distil-medium.en"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)

def vectorized_dataset():
    if os.path.exists("vector.hf"):
        return load_from_disk("vector.hf")
    max_duration = 30.0

    def process(ds):
        audio_arrays = [x["array"] for x in ds["audio"]]
        inputs = feature_extractor( #pass into whisper model
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate, #set to predetermined 16k Hz
            max_length=int(feature_extractor.sampling_rate * max_duration), #find the longest clip (this should be adjusted if dataset includes extremely long clips)
            truncation=True,
        )
        return inputs
    vectorized_dataset = dataset.map(process, remove_columns="audio", batched=True, batch_size=16, num_proc=1,)
    vectorized_dataset.save_to_disk("vector.hf")
    return vectorized_dataset


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
>>>>>>> 5702836e9f77e731354fd8714cef9bf0759d77bf



BATCH_SIZE = 32

torch.manual_seed(42)
train_dataloader = DataLoader(crema_dataset.train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(crema_dataset.test_dataset, batch_size=BATCH_SIZE)


num_labels = len(crema_dataset.id2label)

transformers.set_seed(42)
model = AutoModelForAudioClassification.from_pretrained(
    crema_dataset.model_id, num_labels=num_labels, label2id=crema_dataset.label2id, id2label=crema_dataset.id2label
)

optimizer = AdamW(model.parameters(), lr=1e-5)
EPOCHS = 2

num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.to(device)

progress_bar = tqdm(range(num_training_steps))

wandb.init(project="crema_evaluation_with_fairness", 
           name="evaluation_run",
           config={})

model.train()
for epoch in range(EPOCHS):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()} # convert all the array's to cuda
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        wandb.log({
            "train_loss": loss.item(),
            "learning_rate": lr_scheduler.get_last_lr()[0]  # Extract the current learning rate
        })
    
    model.eval()
    true_labels, predictions = [], []
    for batch in test_dataloader:
        with torch.inference():
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            true_labels.extend(batch["labels"].numpy())
            predictions.extend(preds.numpy())
    val_accuracy = accuracy_score(true_labels, predictions)
    val_f1 = f1_score(true_labels, predictions, average="weighted")
    wandb.log({
            "val_accuracy": val_accuracy,
            "val_f1_score": val_f1,
            "epoch": epoch + 1
    })


def validate(model, dataloader):
    model.eval()
    true_labels, predictions = [], []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            true_labels.extend(batch["labels"].numpy())
            predictions.extend(preds.numpy())
    val_accuracy = accuracy_score(true_labels, predictions)
    val_f1 = f1_score(true_labels, predictions, average="weighted")
    return val_accuracy, val_f1

torch.save(model.state_dict(), "model.pth")

