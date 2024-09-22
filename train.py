import boto3
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from dtrocr.processor import DTrOCRProcessor
from dtrocr.data import DTrOCRProcessorOutput
import pickle
import argparse
import yaml
import random
from typing import Tuple
import os
import re
import evaluate

# Dataset Class
class S3ImageDataset(Dataset):
    def __init__(
            self, 
            bucket_name: str, 
            folder: str, 
            processor: DTrOCRProcessor, 
            cache_file: str='cache/file_list_5m.pkl', 
            force_reload: bool=False,
            split: str='train',
            train_split: float=0.99,
            test_split: float=0.009,
        ):
        """
        Initializes the dataset.
        supported image format: img_{random_number}_{label_separated_with_underscores}.png
        image format examples: img_0_الكلام_لا_يعقل_أبدا.._فأنا.png, img_1_في_مونديال_الاستعراضات_الجوية_يعود.png

        Args:
            bucket_name (str): Name of the S3 bucket containing the dataset.
            folder (str): Path to the folder within the bucket containing the dataset.
            processor (DTrOCRProcessor): Processor object for the DTrOCR model.
            cache_file (str, optional): Path to the file for caching the file list. Defaults to 'cache/file_list_5m.pkl'.
            force_reload (bool, optional): If True, forces reloading the file list from S3, ignoring the cache. Defaults to False.
            split (str, optional): Specifies which part of the dataset to load: 'train', 'val', or 'test'. Defaults to 'train'.
            train_split (float, optional): The proportion of the dataset to use for training. Defaults to 0.99.
            test_split (float, optional): The proportion of the dataset to use for testing. Defaults to 0.009.
        """
        
        self.bucket = boto3.client('s3')
        self.bucket_name = bucket_name
        self.folder = folder
        self.processor = processor
        self.cache_file = cache_file
        self.force_reload = force_reload
        self.split = split
        self.train_split = train_split
        self.test_split = test_split
        self._load_or_create_cache()

    def _load_or_create_cache(self):
        """
        Loads the file list from cache or creates it by fetching from S3.
        """
        try:
            if self.force_reload:
                raise FileNotFoundError
            with open(self.cache_file, 'rb') as f:
                self._file_list_cache = pickle.load(f)
                print(f"Loaded {len(self._file_list_cache)} image file paths from cache.")
        except FileNotFoundError:
            self._file_list_cache = self._get_file_list()
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._file_list_cache, f)
            print(f"Saved {len(self._file_list_cache)} image file paths to cache.")
            
        # Split the dataset
        if self.split in ['train', 'val', 'test']:
            random.seed(42) 
            random.shuffle(self._file_list_cache)
            train_index = int(len(self._file_list_cache) * self.train_split)
            test_index = int(len(self._file_list_cache) * (self.train_split + self.test_split)) 
            if self.split == 'train':
                self._file_list_cache = self._file_list_cache[:train_index]
            elif self.split == 'test':
                self._file_list_cache = self._file_list_cache[train_index:test_index]
            else: 
                self._file_list_cache = self._file_list_cache[test_index:]

    
    def _get_file_list(self):
        """
        Fetches the list of image file paths from S3 in chunks.

        Returns:
            list: A list of image file paths.
        """
        count = 0
        continuation_token = None
        file_list = [] 

        while True:
            list_objects_params = {
                'Bucket': self.bucket_name,
                'Prefix': self.folder
            }

            if continuation_token:
                list_objects_params['ContinuationToken'] = continuation_token

            response = self.bucket.list_objects_v2(**list_objects_params)

            # Filter for image files and add to the list
            if 'Contents' in response:
                file_list.extend([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(('.jpg', '.png', '.jpeg'))])
                count += len(response['Contents'])

            # Check if there are more objects to retrieve
            if response.get('IsTruncated'):
                continuation_token = response.get('NextContinuationToken')
            else:
                break

        print(f"Found {len(file_list)} image files in S3.")
        return file_list
    
    def __len__(self):
        return len(self._file_list_cache)

    def __getitem__(self, idx):
        file_key = self._file_list_cache[idx]
        try:
            obj = self.bucket.get_object(Bucket=self.bucket_name, Key=file_key)
            img = Image.open(obj['Body']).convert('RGB')
            label = re.sub(r'generated_images/img_\d+_|\.png', '', file_key).replace("_", " ")
            inputs = self.processor(
                images=img,
                texts=label,
                padding='max_length',
                return_labels=True,
                return_tensors="pt"
            )
            return {
                'pixel_values': inputs.pixel_values[0],
                'input_ids': inputs.input_ids[0],
                'attention_mask': inputs.attention_mask[0],
                'labels': inputs.labels[0]
            }
        except Exception as e:
            print(f"Error loading image from S3: {e}")
            return None  # Return None in case of error
      
# Evaluate func borrowed from IAM notebook  
def evaluate_model_acc(model: torch.nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    loss, accuracy = [], []
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            inputs = send_inputs_to_device(inputs, device=0)
            outputs = model(**inputs)
            loss.append(outputs.loss.item())
            accuracy.append(outputs.accuracy.item())

            if i % 100 == 0:  # Print progress every 100 batches
                print(f"Evaluating: Batch {i}/{len(dataloader)}")

    loss = sum(loss) / len(loss)
    accuracy = sum(accuracy) / len(accuracy)
    model.train()
    return loss, accuracy


def evaluate_model_cer(model: torch.nn.Module, dataloader: DataLoader, processor: DTrOCRProcessor) -> float:
    model.eval()
    all_predictions, all_labels = [], [] 
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # cast to a DTrOCRProcessorOutput
            inputs = DTrOCRProcessorOutput(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            inputs = send_inputs_to_device(inputs, device=device)

            generated_ids = model.generate(
                inputs=inputs, 
                processor=processor
            )
            generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = processor.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            all_predictions.extend(generated_text)
            all_labels.extend(labels)

            if i % 100 == 0:
                print(f"Evaluating: Batch {i}/{len(dataloader)}")

    results = ev_metric.compute(predictions=all_predictions, references=all_labels)
    avg_cer = results["cer"] 
    model.train()
    return avg_cer

def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}


ev_metric = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DTrOCR model on S3 dataset.')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML config file.')
    
    args = parser.parse_args()
    
    # Load config from YAML file
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        dataset_conf = config_data['project']['dataset']
        train_conf = config_data['project']['train']
    
    # exp save dir
    save_path = f"{config_data['project']['language']}_{config_data['project']['exp_name']}"
    os.makedirs(save_path, exist_ok=True) 
 
    # Dataloader and Processor
    config = DTrOCRConfig(lang=dataset_conf['language'])
    processor = DTrOCRProcessor(config, add_bos_token=True, add_eos_token=True)
    
    train_dataset = S3ImageDataset(
        dataset_conf['bucket_name'], 
        dataset_conf['folder'], 
        processor, 
        cache_file=dataset_conf['cache_file'], 
        force_reload=dataset_conf['force_reload'],
        split='train',
        train_split=dataset_conf['split']['train'],
        test_split=dataset_conf['split']['test'],
    )
    
    test_dataset = S3ImageDataset(
        dataset_conf['bucket_name'], 
        dataset_conf['folder'], 
        processor, 
        cache_file=dataset_conf['cache_file'], 
        force_reload=dataset_conf['force_reload'],
        split='test',
        train_split=dataset_conf['split']['train'],
        test_split=dataset_conf['split']['test'],
    )
    
    val_dataset = S3ImageDataset(
        dataset_conf['bucket_name'], 
        dataset_conf['folder'], 
        processor, 
        cache_file=dataset_conf['cache_file'], 
        force_reload=dataset_conf['force_reload'],
        split='val',
        train_split=dataset_conf['split']['train'],
        test_split=dataset_conf['split']['test'],
    )
    
    print(f"Train split has: {train_dataset.__len__()} images")
    print(f"Test split has: {test_dataset.__len__()} images")
    print(f"Validation split has: {val_dataset.__len__()} images")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=dataset_conf['batch_size'], 
        shuffle=True, 
        num_workers=dataset_conf['num_workers'], 
        prefetch_factor=dataset_conf['prefetch_factor'])
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=dataset_conf['batch_size'], 
        shuffle=True, 
        num_workers=dataset_conf['num_workers'], 
        prefetch_factor=dataset_conf['prefetch_factor'])
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=dataset_conf['batch_size'], 
        shuffle=True, 
        num_workers=dataset_conf['num_workers'], 
        prefetch_factor=dataset_conf['prefetch_factor'])
    
    print("Dataset in ready!")

    # Model
    model = DTrOCRLMHeadModel(config)
    model.train()  # set model to training mode

    # Mixed Precision Setup
    use_amp = True
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train() 
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_conf['learning_rate']))

    # Evaluation metric, cer only for now
    if train_conf['evaluation_metric'] == 'cer':
        ev_metric = evaluate.load("cer")  # Load the CER metric
    
    # Training Loop
    for epoch in range(train_conf['num_epochs']):
        losses, accuracy = [], []
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            optimizer.zero_grad()
            batch = send_inputs_to_device(batch, device=0)  # Send batch to GPU

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            losses.append(outputs.loss.item())
            accuracy.append(outputs.accuracy.item())

            if batch_idx % train_conf['print_every_n_batches'] == 0:
                print(f"Epoch: {epoch + 1}/{train_conf['num_epochs']}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
                
            if batch_idx != 0: 
                if batch_idx % train_conf['validate_every_n_batches'] == 0 or batch_idx == len(train_loader) - 1:
                    # Calculate and print average train loss and accuracy
                    train_loss = sum(losses) / len(losses)
                    train_accuracy = sum(accuracy) / len(accuracy)
                    print(f"Epoch: {epoch + 1} - Train loss: {train_loss}, Train accuracy: {train_accuracy}")

                    # Evaluate the model
                    val_loss, val_accuracy = evaluate_model_acc(model, val_loader)
                    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}") 
                    val_cer = evaluate_model_cer(model, val_loader, processor)
                    print(f"Validation CER: {val_cer}")
                    
                # Checkpointing
                if batch_idx % train_conf['save_every_n_batches'] == 0 or batch_idx == len(train_loader) - 1:
                    checkpoint_name = f"{save_path}/checkpoint_batch{batch_idx}.pth" 
                    torch.save(model.state_dict(), checkpoint_name)
                    print(f"Saved checkpoint: {checkpoint_name}")
