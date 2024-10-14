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
        Loads the file list from cache or creates it by fetching from S3 and splitting it.
        """
        try:
            if self.force_reload:
                # Enhanced warning message
                print("WARNING: Force reload is enabled!")
                print("This will trigger a full dataset fetch from S3, which might take a considerable amount of time.")
                user_input = input("Are you sure you want to proceed? (yes/no): ")
                if user_input.lower() != 'yes':
                    print("Force reload aborted. Loading from cache instead.")
                    self.force_reload = False  # Reset force_reload to False
                else:
                    print("Proceeding with force reload...")
                    raise FileNotFoundError
                
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self._file_list_cache = cache_data[self.split]  # Load the specific split from the cache
                print(f"Loaded {len(self._file_list_cache)} image file paths from cache for {self.split} split.")
        except FileNotFoundError:
            file_list = self._get_file_list()
            random.seed(42)  # Set a seed for reproducibility
            random.shuffle(file_list)
            train_index = int(len(file_list) * self.train_split)
            test_index = int(len(file_list) * (self.train_split + self.test_split))

            cache_data = {
                'train': file_list[:train_index],
                'test': file_list[train_index:test_index],
                'val': file_list[test_index:]
            }

            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved {len(file_list)} image file paths to cache, split into train, val, and test sets.")

            self._file_list_cache = cache_data[self.split]  # Load the specific split from the cache
        
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
            print(f"Fetched {count} images so far...", end='\r')  

            list_objects_params = {
                'Bucket': self.bucket_name,
                'Prefix': self.folder
            }

            if continuation_token:
                list_objects_params['ContinuationToken'] = continuation_token

            response = self.bucket.list_objects_v2(**list_objects_params)

            # Filter for image files and add to the list
            if 'Contents' in response:
                new_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(('.jpg', '.png', '.jpeg'))]
                file_list.extend(new_files)
                count += len(new_files) 

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
            
            # Handle val, test and train separately
            if self.split == 'train':
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
            else: # val and test 
                inputs = self.processor(
                    images=img,
                    texts=self.processor.tokenizer.bos_token,
                    return_tensors="pt"
                )
                tokenized_label = self.processor(
                  texts=label, 
                  padding='max_length',
                  return_tensors="pt"
                ).input_ids
                
                return {
                    'pixel_values': inputs.pixel_values[0],
                    'input_ids': inputs.input_ids[0],
                    'attention_mask': inputs.attention_mask[0],
                    'labels': tokenized_label[0]
                }
                
        except Exception as e:
            print(f"Error loading image from S3: {e}")
            return None  # Return None in case of error
    


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, processor: DTrOCRProcessor, generation_conf) -> float:
    model.eval()
    all_predictions, all_labels = [], [] 
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if None in batch:
                continue
                
            else:
                inputs = DTrOCRProcessorOutput(
                  pixel_values=batch['pixel_values'].to(device),
                  input_ids=batch['input_ids'].to(device),
                  attention_mask=batch['attention_mask'].to(device),
                  labels=batch['labels'].to(device)
                )
    
                generated_ids = model.generate(
                    inputs=inputs, 
                    processor=processor,
                    num_beams=generation_conf['num_beams']
                )
                generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                labels = processor.tokenizer.batch_decode(inputs.labels, skip_special_tokens=True)
    
                all_predictions.extend(generated_text)
                all_labels.extend(labels)
    
                if i % 100 == 0:
                    print(f"Evaluating: Batch {i}/{len(dataloader)}")

    results = ev_metric.compute(predictions=all_predictions, references=all_labels)
    model.train()
    return results

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
        generation_conf = config_data['project']['generation']
    
    # exp save dir
    save_path = f"{config_data['project']['language']}_{config_data['project']['exp_name']}"
    os.makedirs(save_path, exist_ok=True) 
 
    # Dataloader and Processor
    config = DTrOCRConfig(lang=dataset_conf['language'])
    processor = DTrOCRProcessor(config, add_bos_token=True, add_eos_token=True)
    print("Preparing datset...")
    print("If this is the first time running the script, it will take a long time to fetch the dataset.")
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
        split='test',
        # force_reload=dataset_conf['force_reload'],
        # train_split=dataset_conf['split']['train'],
        # test_split=dataset_conf['split']['test'],
    )
    
    val_dataset = S3ImageDataset(
        dataset_conf['bucket_name'], 
        dataset_conf['folder'], 
        processor, 
        cache_file=dataset_conf['cache_file'],
        split='val', 
        # force_reload=dataset_conf['force_reload'],
        # train_split=dataset_conf['split']['train'],
        # test_split=dataset_conf['split']['test'],
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
        # shuffle=True, 
        num_workers=dataset_conf['num_workers'], 
        prefetch_factor=dataset_conf['prefetch_factor'])
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=dataset_conf['batch_size'], 
        # shuffle=True, 
        num_workers=dataset_conf['num_workers'], 
        prefetch_factor=dataset_conf['prefetch_factor'])
    
    print("Dataset in ready!")

    # Model
    model = DTrOCRLMHeadModel(config)
    model.train()  # set model to training mode

    # Mixed Precision Setup
    use_amp = train_conf['use_amp']
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
            if None in batch:
                continue

            optimizer.zero_grad()
            batch = send_inputs_to_device(batch, device=device)  # Send batch to GPU

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
                    val_metric = evaluate_model(model, val_loader, processor, generation_conf)
                    print(f"Validation CER: {val_metric}")
                    
                # Checkpointing
                if batch_idx % train_conf['save_every_n_batches'] == 0 or batch_idx == len(train_loader) - 1:
                    checkpoint_name = f"{save_path}/checkpoint_batch{batch_idx}.pth" 
                    torch.save(model.state_dict(), checkpoint_name)
                    print(f"Saved checkpoint: {checkpoint_name}")
                    
    # test the model
    test_metric = evaluate_model(model, test_loader, processor)
    print(f"Test CER: {test_metric}")
