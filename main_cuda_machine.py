from datasets import load_dataset, ClassLabel, load_from_disk
import torch
import evaluate
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import AutoFeatureExtractor
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
dataset = load_dataset("myleslinder/crema-d", trust_remote_code=True, split='train')

id2label = {'0':'anger', '1': 'disgust','2': 'fear', '3': 'happy', '4': 'neutral', '5': 'sad'}
label2id = dict()
for i in id2label:
    label2id[id2label[str(i)]] = str(i)

dataset = dataset.select_columns(['audio', 'label']) #This is simple classifcation task, whisper is already trained so no need for sentence.
dataset = dataset.train_test_split(test_size=0.3)


model_id = "distil-whisper/distil-medium.en"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)

def load_dataset():
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



BATCH_SIZE = 32

vectorized_dataset = load_dataset()

vectorized_dataset.set_format("torch")
vectorized_dataset = vectorized_dataset.rename_column("label", "labels")

train_dataset = vectorized_dataset["train"]
test_dataset = vectorized_dataset["test"]



train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
)

optimizer = AdamW(model.parameters(), lr=1e-5)
EPOCHS = 2

num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

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

        if epoch%100 == 0:
          print(loss)

torch.save(model.state_dict(), "model.pth")

