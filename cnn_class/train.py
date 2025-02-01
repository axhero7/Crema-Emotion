import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from cremadata import CremaSoundDataset
from cremanet import CNN_Net, OptimizedCremaNet
import torchaudio
from transformers import get_scheduler
import wandb
import optuna
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

class AddWhiteNoise(nn.Module):
    def __init__(self, noise_factor=0.005):
        super().__init__()
        self.noise_factor = noise_factor

    def forward(self, waveform):
        with torch.no_grad():  
            noise = torch.randn_like(waveform) * self.noise_factor
            return waveform + noise

class TimeShift(nn.Module):  
    def __init__(self, max_shift=0.2):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, waveform):
        with torch.no_grad():
            shift = torch.randint(
                low=-int(self.max_shift * waveform.size(1)),
                high=int(self.max_shift * waveform.size(1)),
                size=(1,)
            ).item()
            return torch.roll(waveform, shifts=shift, dims=1)


def train_step(model, data, loss_fn, optim, device, lr_scheduler, progress_bar):
    for index, (input, label, _) in enumerate(data):
        input, label = input.to(device), label.to(device)

        y_pred = model(input)
        loss = loss_fn(y_pred, label)
        loss.backward()
        optim.step()
        if lr_scheduler != None: lr_scheduler.step() 

        optim.zero_grad()
        progress_bar.update(1)
        wandb.log({
            "train_loss": loss.item(),
            # "learning_rate": lr_scheduler.get_last_lr()[0] 
        })
        
        if index % 30 == 0:
            outputs = nn.functional.softmax(y_pred,dim=1)
            outputs = torch.argmax(y_pred, dim=1)
            correct = torch.eq(label, outputs).sum().item()
            acc = (correct / len(y_pred)) * 100
            print(f"train loss: {loss.item()}, train acc: {acc}")
def test_step(model, data, loss_fn, device, lr_scheduler, epoch):
    true_labels, predictions = [], []
    total_loss = 0
    with torch.no_grad():
        model.eval()
        for index, (input, label, _) in enumerate(data):
            input, label = input.to(device), label.to(device)

            y_pred = model(input)
            loss = loss_fn(y_pred, label)
            total_loss += loss.item()
            outputs = nn.functional.softmax(y_pred,0)
            outputs = outputs.argmax(dim=1)
            true_labels.extend(list(label.cpu().detach()))
            predictions.extend(list(outputs.cpu().detach()))
            wandb.log({
                "val_loss": loss.item()
            })
        print(len(true_labels), true_labels[0])
        print(len(predictions), predictions[0])
        val_accuracy = accuracy_score(true_labels, predictions)
        val_f1 = f1_score(true_labels, predictions, average="weighted")
        print(f"val acc: {val_accuracy}, val f1: {val_f1}")
        return total_loss/len(data)
        wandb.log({
                "val_accuracy": val_accuracy,
                "val_f1_score": val_f1,
                "epoch": epoch + 1
    })


def train(model, train_data, test_data, loss_fn, optim, device, epochs, lr_scheduler=None, progress_bar=None):
    for i in range(epochs):
        print("epoch: ", i)
        train_step(model, train_data, loss_fn, optim, device, None, progress_bar)
        avg_loss = test_step(model, test_data, loss_fn, device, lr_scheduler, i)
        if lr_scheduler != None: 
            lr_scheduler.step(avg_loss)

if __name__ == "__main__":
    BATCH_SIZE = 16
    EPOCHS = 150
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
    
    train_transforms = torch.nn.Sequential(
        AddWhiteNoise(noise_factor=0.005),
        TimeShift(max_shift=0.2),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
    )
    
    test_transforms = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    # Create separate datasets
    train_dataset = CremaSoundDataset(
        full_annotations.iloc[:train_indices].reset_index(drop=True),  # Reset index
        AUDIO_DIR,
        train_transforms,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )

    
    test_dataset = CremaSoundDataset(
        full_annotations.iloc[train_indices:test_indices].reset_index(drop=True),  # Reset index
        AUDIO_DIR,
        test_transforms,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
)
    

    # mel_spectogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=SAMPLE_RATE,
    #     n_fft=1024,
    #     hop_length=512,
    #     n_mels=64
    # )

    # usd = CremaSoundDataset(ANNOTATIONS_FILE,
    #                         AUDIO_DIR,
    #                         mel_spectogram,
    #                         SAMPLE_RATE,
    #                         NUM_SAMPLES,
    #                         device)
    torch.manual_seed(42)
    # train_dataloader, test_dataloader = CremaSoundDataset.create_data_loader(usd, batch_size=BATCH_SIZE, train_ratio=0.7, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    torch.manual_seed(41)
    model_0 = CNN_Net().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_0.parameters(), lr=1e-3, weight_decay=1e-4)

    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3
)

    progress_bar = tqdm(range(num_training_steps))


    wandb.init(project="crema_evaluation_with_fairness", 
            name="cnn based train run",
            config={})
    model_0.train()
    train(model=model_0, train_data=train_dataloader, test_data=test_dataloader, loss_fn=loss_fn, optim=optimizer, device=device, epochs=EPOCHS, lr_scheduler=lr_scheduler, progress_bar=progress_bar)
    torch.save(model_0.state_dict(), "cnn_parameters_150.pth")




