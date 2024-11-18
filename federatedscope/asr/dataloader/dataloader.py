import os
import gzip
import json
import random
import logging
import torch
import transformers
import datasets

from dataclasses import dataclass
from datasets import load_dataset, DatasetDict
from federatedscope.asr.dataset.asr_dataset import ASRDataset

from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor

from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

@dataclass
class ASRDataCollator(object):
    
    """
    A data collator for supervised fine-tuning of whisper models based on DataCollatorSpeechSeq2SeqWithPadding
    Split inputs and labels since they have to be of different lengths and need different padding methods
    First treat the audio inputs by simply returning torch tensors, get the tokenized label sequences
    pad the labels to max length and replace padding with -100 to ignore loss correctly.
    """
    
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
  
        input_features = [{"input_features": feature["input_ids"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

"""
@dataclass
class ASRDataCollator(object):
    
    A data collator for supervised fine-tuning of whisper models.
    This class implements a callable that takes a list of instances and
    returns a batch of input_ids, labels, and attention_mask tensors. The
    input_ids and labels are padded with the tokenizer's pad_token_id and a
    special ignore index value, respectively. The attention_mask indicates
    which tokens are not padding.
    

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        Collates a list of instances into a batch.

        Args:
            instances: A list of dictionaries, each containing input_ids and
                labels as torch.LongTensor objects.

        Returns:
            A dictionary with the following keys and values:
                - input_ids: A torch.LongTensor of shape (batch_size,
                max_length)
                    containing the padded input ids.
                - labels: A torch.LongTensor of shape (batch_size, max_length)
                    containing the padded labels.
                - attention_mask: A torch.BoolTensor of shape (batch_size,
                max_length)
                    indicating which tokens are not padding.
        

        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=DefaultToken.IGNORE_INDEX.value)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
"""

def get_asr_tokenizer_extractor(model_name, cache_dir, language, task, pkg='huggingface_asr'):
    """
    This function loads a tokenizer from a pretrained model name and adds some
    default special tokens if they are not already defined. It also sets the
    model max length and the padding side of the tokenizer.

    Args:
        model_name: A string, the name of the pretrained model.
        cache_dir: A string, the path to the cache directory.
        language: A string, the name of the language, e.g. english
        task: A string, the name of the task, e.g. transcribe

    Returns:
        A tuple of (tokenizer, num_new_tokens), where:
            - tokenizer: A transformers.WhisperTokenizer object.
            - feature_extractor: 
            - processor: 
    """
    assert pkg in ['huggingface_asr', 'modelscope_llm'], \
        f'Not supported package {pkg}.'

    if pkg == 'huggingface_asr':
        from transformers import WhisperTokenizer
    #elif pkg == 'modelscope_llm':
    #    from modelscope import AutoTokenizer


    tokenizer = WhisperTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_dir,
        language=language, 
        task=task,
        )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)

    return tokenizer, feature_extractor, processor


def load_asr_dataset(config=None, **kwargs):
    """
    This function takes a config object and optional keyword arguments and
    returns a dataset object and an updated config object.
    The function supports only the common voice dataset.

    Args:
        config: An object, the configuration for loading the dataset.
        **kwargs: Optional keyword arguments that can override the config
            attributes.

    Returns:
        A tuple of (dataset, config), where:
            - dataset: A ASRDataset object that contains the audio examples with
                instruction, input, output, and category fields.
            - config: An object, the updated configuration.
    """
    model_name, model_hub = config.model.type.split('@')
    tokenizer, feature_extractor, processor = \
    get_asr_tokenizer_extractor(model_name, config.data.root, config.data.language,
                      config.data.task, model_hub)

    dataset_name, _ = config.data.type.split('@')
    
    if dataset_name.lower() == 'common_voice_13_0':
        #common_voice = DatasetDict()
        #common_voice["train"] = load_dataset("mozilla-foundation/"+dataset_name , config.data.language_abbr, 
        #                                     split="train+validation", use_auth_token=True)
        #common_voice["test"] = load_dataset("mozilla-foundation/"+dataset_name, config.data.language_abbr, 
        #                                    split="test", use_auth_token=True)
        common_voice = load_dataset("mozilla-foundation/"+dataset_name , config.data.language_abbr, 
                                             split="validation", use_auth_token=True)
        print("object: common ",common_voice)
        common_voice = common_voice.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
        )
        #dataset = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=80) 
        dataset = ASRDataset(common_voice, tokenizer, feature_extractor, processor)
        #dataset = dict(ID=dataset)
        #client_num = min(len(dataset), config.federate.client_num
        #             ) if config.federate.client_num > 0 else len(dataset)
        #config.merge_from_list(['federate.client_num', client_num])

        # get local dataset
        #data_dict = dict()
        #for client_idx in range(1, client_num + 1):
        #    data_dict[client_idx] = dataset# [client_idx - 1]
        #dataset = dataset.common_voice
        #dataset = dataset.remove_columns(
        #    ["audio", "sentence", "variant"]
        #)
        #print("object: dataset ", dataset)
    
    else:
        raise ValueError(f'Not support data type {dataset_name}.')

    return dataset, config
