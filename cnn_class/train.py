import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from cremadata import CremaSoundDataset
from cremanet import CNN_Net, OptimizedCremaNet
import torchaudio
from transformers import get_scheduler
import optuna
import wandb
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

def train_step(model, data, loss_fn, optim, device, lr_scheduler):
    for index, (input, label) in enumerate(data):
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
            "learning_rate": lr_scheduler.get_last_lr()[0] 
        })
        
        if index % 30 == 0:
            outputs = nn.functional.softmax(y_pred,dim=1)
            outputs = torch.argmax(y_pred, dim=1)
            # print(outputs)
            # print(label)
            correct = torch.eq(label, outputs).sum().item()
            acc = (correct / len(y_pred)) * 100
            print(f"train loss: {loss.item()}, train acc: {acc}")
            # print(lr_scheduler.get_last_lr()[0])
def test_step(model, data, loss_fn, device, lr_scheduler, epoch):
    true_labels, predictions = [], []
    with torch.no_grad():
        model.eval()
        for index, (input, label) in enumerate(data):
            input, label = input.to(device), label.to(device)

            y_pred = model(input)
            loss = loss_fn(y_pred, label)
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
        wandb.log({
                "val_accuracy": val_accuracy,
                "val_f1_score": val_f1,
                "epoch": epoch + 1
    })


def train(model, train_data, test_data, loss_fn, optim, device, epochs):
    for i in range(epochs):
        print("epoch: ", i)
        train_step(model, train_data, loss_fn, optim, device, lr_scheduler)
        test_step(model, test_data, loss_fn, device, lr_scheduler, i)


def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    print("batch_size: ", batch_size)
    print("lr: ", lr)
    print("weight_decay: ", weight_decay)
    print("=====================================")
    # Data loading
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    ANNOTATIONS_FILE = "SentenceFilenames.csv"
    AUDIO_DIR = "AudioWAV"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = SAMPLE_RATE*4
    
    usd = CremaSoundDataset(ANNOTATIONS_FILE,
                                AUDIO_DIR,
                                mel_spectrogram,
                                SAMPLE_RATE,
                                NUM_SAMPLES,
                                device)
    torch.manual_seed(42)
    train_dataloader, test_dataloader = CremaSoundDataset.create_data_loader(dataset=usd, batch_size=batch_size, train_ratio=0.7, device=device)
    # Model
    model = OptimizedCremaNet(trial).to(device)
    
    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Training loop
    best_acc = 0
    for epoch in range(30):
        train_step(model, train_dataloader, criterion, optimizer, device, scheduler)
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds.cpu().numpy() == labels.cpu().numpy()).sum().item()
                total += labels.size(0)
        
        val_acc = 100 * correct / total
        trial.report(val_acc, epoch)
        print("=====================================")
        print("validation: ", val_acc, " on epoch :", epoch)
        print("=====================================")
        
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    return best_acc



if __name__ == "__main__":
    if False:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(objective, n_trials=50, timeout=3600*6)

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (Accuracy): {trial.value:.2f}%")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Train final model with best params
        # Use best_params from study to train final model...

    else:
        BATCH_SIZE = 16
        EPOCHS = 100
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
        train_dataloader, test_dataloader = CremaSoundDataset.create_data_loader(usd, batch_size=BATCH_SIZE, train_ratio=0.7, device=device)

        torch.manual_seed(42)
        model_0 = CNN_Net().to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model_0.parameters(), lr=1e-3)

        num_training_steps = EPOCHS * len(train_dataloader)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        progress_bar = tqdm(range(num_training_steps))


        wandb.init(project="crema_evaluation_with_fairness", 
                name="cnn based train run",
                config={})
        model_0.train()
        train(model_0, train_dataloader, train_dataloader, loss_fn, optimizer, device, EPOCHS)
        torch.save(model_0.state_dict(), "cnn_param.pth")
