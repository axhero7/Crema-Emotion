import torch
from torch import nn
from torch.utils.data import DataLoader
from cremadata import CremaSoundDataset
from cremanet import CNN_Net
import torchaudio
from transformers import get_scheduler


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_step(model, data, loss_fn, optim, device):
    for index, (input, label) in enumerate(data):
        input, label = input.to(device), label.to(device)

        y_pred = model(input)
        loss = loss_fn(y_pred, label)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()
        
        if index % 30 == 0:
            outputs = nn.functional.softmax(y_pred,dim=1)
            outputs = torch.argmax(y_pred, dim=1)
            # print(outputs)
            # print(label)
            correct = torch.eq(label, outputs).sum().item()
            acc = (correct / len(y_pred)) * 100
            print(f"loss: {loss.item()}, acc: {acc}")
            # print(lr_scheduler.get_last_lr()[0])
def test_step(model, data, device):
    for index, (input, label) in enumerate(data):
        input, label = input.to(device), label.to(device)

        y_pred = model(input)
        loss = loss_fn(y_pred, label)
        outputs = nn.functional.softmax(y_pred,0)
        print(outputs)
        if index % 30 == 0:
            print(f"loss: {loss.item()}")
            print(lr_scheduler.get_last_lr()[0])


def train(model, train_data, test_data, loss_fn, optim, device, epochs):
    for i in range(epochs):
        print("epoch: ", i)
        train_step(model, train_data, loss_fn, optim, device)
        # test_step(model, test_data, device)

        


if __name__ == "__main__":

    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    ANNOTATIONS_FILE = "SentenceFilenames.csv"
    AUDIO_DIR = "AudioWAV"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = SAMPLE_RATE*4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device: ", device)



    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = CremaSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    torch.manual_seed(42)
    train_dataloader = create_data_loader(usd, batch_size=BATCH_SIZE)
    
    torch.manual_seed(42)
    model_0 = CNN_Net().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_0.parameters(), lr=1e-3\
                                  )

    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model_0.train()
    train(model_0, train_dataloader, train_dataloader, loss_fn, optimizer, device, EPOCHS)
    torch.save(model_0.state_dict(), "cnn_param.pth")
