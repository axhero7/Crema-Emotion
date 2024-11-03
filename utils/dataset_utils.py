"""All the dataset utils for the Crema-project in class CremaDataset"""
import datasets
import os
import torch
from transformers import AutoFeatureExtractor
from datasets import load_dataset, load_from_disk


class CremaDataset:
    def __init__(self, model_id):
        self.model_id = model_id
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id, do_normalize=True)
        self.vectorized_dataset = None
        self.dataset = self.build_dataset()
        

    def build_dataset(self):
        dataset = load_dataset("myleslinder/crema-d", trust_remote_code=True, split='train')
        self.id2label = {'0': 'neutral', '1': 'happy', '2': 'sad', '3': 'anger', '4': 'fear', '5': 'disgust'}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Select necessary columns
        dataset = dataset.select_columns(['audio', 'label'])
        dataset = dataset.train_test_split(test_size=0.3)

        return dataset

    def load_vectorized_dataset_without_id(self, path, max_duration):
        if os.path.exists(path):
            vectorized_dataset = load_from_disk(path)
        else:
            def process(ds):
                audio_arrays = [x["array"] for x in ds["audio"]]
                inputs = self.feature_extractor(
                    audio_arrays,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    max_length=int(self.feature_extractor.sampling_rate * max_duration),
                    truncation=True,
                )
                
                return inputs
        
            vectorized_dataset = self.dataset.map(process, remove_columns="audio", batched=True, batch_size=16, num_proc=1)
            vectorized_dataset.save_to_disk(path)
        
        vectorized_dataset.set_format("torch")
        vectorized_dataset = vectorized_dataset.rename_column("label", "labels")
        
        train_dataset = vectorized_dataset["train"]
        test_dataset = vectorized_dataset["test"]
        return train_dataset, test_dataset

    def load_vectorized_dataset_with_id(self, path, max_duration):
        if os.path.exists(path):
            vectorized_dataset_t = load_from_disk(path)
        else:
            def process(ds):
                audio_arrays = [x["array"] for x in ds["audio"]]
                audio_paths = [x["path"] for x in ds["audio"]]
                audio_paths = [x.split('/')[-1][:4] for x in audio_paths]
                inputs = self.feature_extractor(
                    audio_arrays,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    max_length=int(self.feature_extractor.sampling_rate * max_duration),
                    truncation=True,
                )
                audio = {"input_features": torch.tensor(inputs['input_features']), "path": audio_paths}
                return audio
            vectorized_dataset_t = self.dataset.map(process, batched=True, remove_columns="audio", batch_size=16, num_proc=1)
            vectorized_dataset_t.save_to_disk(path)
            
        vectorized_dataset_t.set_format("torch")
        vectorized_dataset_t = vectorized_dataset_t.rename_column("label", "labels")
        
        train_dataset = vectorized_dataset_t["train"]
        test_dataset = vectorized_dataset_t["test"]

        return train_dataset, test_dataset
    
    def set_vector(self, path, actor_id, max_duration=30.0,):
        if actor_id:
            self.train_dataset, self.test_dataset = self.load_vectorized_dataset_with_id(path, max_duration)
        else:
            self.train_dataset, self.test_dataset = self.load_vectorized_dataset_without_id(path, max_duration)
    
