import os
import json
import torch
import torchvision
import numpy as np
from PIL import Image
import pytorch_lightning as pl

class MeebitDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_directory: str, metadata: dict, image_file_type: str = 'png'):
        self.image_directory = image_directory
        self.image_file_type = image_file_type
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        
        import os
        token_metadata = {}
        idx = 0
        for row in metadata:
            if row['metadata'] is not None:
                row_metadata = json.loads(row['metadata'])
                attributes = row_metadata['attributes']

                if attributes is not None:
                    token_id = int(row['token_id'])
                    token_address = row['token_address']
                    image_filepath = os.path.join(self.image_directory, f'{token_id}.{self.image_file_type}')
                    processed_attributes = {}
                    for attribute in attributes:
                        processed_attributes[attribute['trait_type']] = attribute['value']
                    if os.path.isfile(image_filepath):
                        token_metadata[idx] = {
                            'token_id': token_id,
                            'token_address': token_address,
                            'attributes': processed_attributes,
                            'image_filepath': image_filepath
                        }
                        idx += 1
        
        trait_pairs = {}
        for nft in list(token_metadata.values()):
            for trait_type in nft['attributes']:
                if trait_type not in trait_pairs.keys():
                    trait_pairs[trait_type] = []

                if trait_type in trait_pairs.keys():
                    trait_value = nft['attributes'][trait_type]
                    trait_pairs[trait_type].append(trait_value)

        # make each key/value unique
        unique_trait_pairs = {}
        for trait in trait_pairs:
            all_trait_values = trait_pairs[trait]
            unique_trait_values = list(set(all_trait_values))
            unique_trait_pairs[trait] = unique_trait_values
            
        self.unique_trait_pairs = unique_trait_pairs
        
        from sklearn.preprocessing import LabelEncoder
        label_encoder_mappings = {}
        for trait_type in trait_pairs:
            label_encoder = LabelEncoder()
            label_encoder_mappings[trait_type] = label_encoder.fit(trait_pairs[trait_type])
        
        self.label_encoder_mappings = label_encoder_mappings
        self.token_metadata = token_metadata
        
    def __len__(self):
        return len(self.token_metadata.keys())
    
    def __getitem__(self, index):
        data = self.token_metadata[index]
        
        token_id = data['token_id']
        token_address = data['token_address']
        attributes = data['attributes']
        image_filepath = data['image_filepath']
        
        image = Image.open(image_filepath)
        image = self.transforms(image)
        encoded_attributes = self._encode_attributes(attributes)
        
        return token_id, encoded_attributes, image
        
    def _encode_attributes(self, attributes):
        encoded_attributes = []
        for trait in self.label_encoder_mappings.keys():
            value = np.array([-1])
            if trait in attributes.keys():
                label_encoder = self.label_encoder_mappings[trait]
                try:
                    value = label_encoder.transform([attributes[trait]])
                except Exception as e:
                    pass
                                
            encoded_attributes.append(float(value))
        
        encoded_attributes = torch.tensor(encoded_attributes).view(-1)
        assert encoded_attributes.shape == torch.Size([21]), f'encoded_attributes.shape: {encoded_attributes.shape}'
        return encoded_attributes


class MeebitDataLoader(pl.LightningDataModule):
    
    def __init__(
        self, 
        image_directory: str, 
        metadata: dict, 
        batch_size: int = 4,
        image_file_type: str = 'png', 
        train_size: float = 0.75, 
        test_size: float = 0.15, 
    ):
        super().__init__()
        
        self.image_directory = image_directory
        self.metadata = metadata
        self.batch_size = batch_size
        self.image_file_type = image_file_type
        self.train_size = train_size
        self.test_size = test_size
        
    def setup(self):
        dataset = MeebitDataset(self.image_directory, self.metadata, self.image_file_type)
        number_of_rows = len(dataset)
        train_samples = int(number_of_rows * self.train_size)
        test_samples = int(number_of_rows * self.test_size)
        val_samples = number_of_rows - train_samples - test_samples
        self.train, self.test, self.val = torch.utils.data.random_split(
            dataset, 
            (train_samples, test_samples, val_samples)
        )
        
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val, 
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test, 
            batch_size=self.batch_size,
            shuffle=False
        )