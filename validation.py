from datasets import load_dataset
import torch
import os
from transformers import AutoFeatureExtractor
from torch.utils.data import DataLoader
from datasets import load_dataset, ClassLabel, load_from_disk
from transformers import AutoModelForAudioClassification
# from main_cuda_machine import vectorized_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("myleslinder/crema-d", trust_remote_code=True, split='train')

id2label = {'0':'anger', '1': 'disgust','2': 'fear', '3': 'happy', '4': 'neutral', '5': 'sad'}
label2id = dict()
for i in id2label:
    label2id[id2label[str(i)]] = str(i)

dataset = dataset.select_columns(['audio', 'label']) #This is simple classifcation task, whisper is already trained so no need for sentence.
dataset = dataset.train_test_split(test_size=0.3)


model_id = "distil-whisper/distil-medium.en"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)

max_duration = 30.0

if os.path.exists("vector.hf"):
    vectorized_dataset = load_from_disk("vector.hf")
else:
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
    
vectorized_dataset.set_format("torch")
vectorized_dataset = vectorized_dataset.rename_column("label", "labels")

train_dataset = vectorized_dataset["train"]
test_dataset = vectorized_dataset["test"]

BATCH_SIZE=32

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
)
model.to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

accuracy = 0

for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    output_seq = model(**batch)
    print(output_seq[2])
    output_tensor = output_seq[0]
    labels = batch['labels']
    print(f"Output sequence: {output_seq}\n\nFirst is {output_seq[3]}\n\nLabels are:{labels}")
    count = 0
    correct = torch.eq(labels, output_tensor).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(output_tensor)) * 100 
    accuracy+=acc
    print(pred, labels[i])
accuracy/=len(test_dataloader)
print(accuracy)
