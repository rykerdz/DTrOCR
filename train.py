import boto3
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from dtrocr.processor import DTrOCRProcessor
import pickle
import argparse
import yaml
import random

# Dataset Class
class S3ImageDataset(Dataset):
    def __init__(
            self, 
            bucket_name: str, 
            folder: str, 
            config: DTrOCRConfig, 
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
            config (DTrOCRConfig): Configuration object for the DTrOCR model.
            cache_file (str, optional): Path to the file for caching the file list. Defaults to 'cache/file_list_5m.pkl'.
            force_reload (bool, optional): If True, forces reloading the file list from S3, ignoring the cache. Defaults to False.
            split (str, optional): Specifies which part of the dataset to load: 'train', 'val', or 'test'. Defaults to 'train'.
            train_split (float, optional): The proportion of the dataset to use for training. Defaults to 0.99.
            test_split (float, optional): The proportion of the dataset to use for testing. Defaults to 0.009.
        """
        
        self.bucket = boto3.client('s3')
        self.bucket_name = bucket_name
        self.folder = folder
        self.processor = DTrOCRProcessor(config, add_bos_token=True, add_eos_token=True)
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
            label = " ".join(file_key.strip(".png").split('/')[-1].split('_')[2:])
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
            return None, None  # Return None in case of error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DTrOCR model on S3 dataset.')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the YAML config file.')
    
    args = parser.parse_args()
    
    # Load config from YAML file
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        dataset_conf = config_data['project']['dataset']
        train_conf = config_data['project']['train']
    
 
    # Dataloader and Processor
    config = DTrOCRConfig(lang=dataset_conf['language'])
    
    train_dataset = S3ImageDataset(
        dataset_conf['bucket_name'], 
        dataset_conf['folder'], config, 
        cache_file=dataset_conf['cache_file'], 
        force_reload=dataset_conf['force_reload'],
        split='train',
        train_split=dataset_conf['split']['train'],
        test_split=dataset_conf['split']['test'],
    )
    
    test_dataset = S3ImageDataset(
        dataset_conf['bucket_name'], 
        dataset_conf['folder'], config, 
        cache_file=dataset_conf['cache_file'], 
        force_reload=dataset_conf['force_reload'],
        split='test',
        train_split=dataset_conf['split']['train'],
        test_split=dataset_conf['split']['test'],
    )
    
    val_dataset = S3ImageDataset(
        dataset_conf['bucket_name'], 
        dataset_conf['folder'], config, 
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
    #model = DTrOCRLMHeadModel(config)
    #model.train()  # set model to training mode
