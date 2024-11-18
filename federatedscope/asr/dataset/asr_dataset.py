import copy
import logging
import pandas as pd
import torch
import numpy as np

from enum import Enum
from torch.utils.data import Dataset
from datasets import Audio


logger = logging.getLogger(__name__)


class ASRDataset(Dataset):
    """
    A dataset for ASR modeling tasks.

    This class inherits from torch.utils.data.Dataset and implements a
    dataset that can load and preprocess data for ASR (whisper) modeling. It
    takes a list of data dictionaries, a tokenizer, and feature extractor as
    input, and creates input ids, labels, and categories as
    output. The input ids and labels are padded and masked according to
    the tokenizer settings and the source and target lengths. The
    categories are encoded as integers using pandas.Categorical.

    Attributes:
        input_ids: A list of torch.LongTensor objects of shape (max_length,)
            containing the padded input ids.
        labels: A list of torch.LongTensor objects of shape (max_length,)
            containing the padded labels.
        categories: A list of integers representing the category codes.
        tokenizer: A transformers.WhisperTokenizer object that can
            encode and decode text.
    """
    def __init__(self,
                 common_voice, tokenizer, feature_extractor, processor):
        """
        Initializes the dataset with the given arguments.

        Args:
            dataset: 
            tokenizer: A transformers.WhisperTokenizer object that can
                encode and decode text.
            feature_extractor:
            processor: 
            
        """
        super(ASRDataset, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.processor = processor
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))        
        
        #common_voice = common_voice.map(self.prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=80)
        common_voice = common_voice.map(self.prepare_dataset, num_proc=2)
        
        #tmp = dict(input_ids=common_voice['input_features'], labels=common_voice['labels'])
        self.input_features = common_voice['input_features']
        self.labels = common_voice['labels']
        #self.common_voice = common_voice[:100]
        #self.input_ids = common_voice['input_features'][:100]
        #self.labels = common_voice['labels'][:100]
        
        #self = dict(input_features=self.input_features, labels=self.labels)
        #self.input_features = common_voice['input_features']
        #self.labels = common_voice['labels']
        #self.input_features = torch.from_numpy(np.asarray(common_voice[:100]['input_features']))
        #self.labels = torch.from_numpy(np.asarray(common_voice[:100]['labels']))
        

    def prepare_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch
    
    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, i):
        return dict(input_ids=self.input_features[int(i)],
                    labels=self.labels[int(i)])