from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from federatedscope.asr.model.adapter_builder import AdapterModel
import torch

def get_model_from_huggingface(model_name, config):
    """
    Load a whisper asr model from HuggingFace transformers library.

    Args:
        model_name (str): The name of the pre-trained model to load.
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        WhisperForConditionalGeneration: A whisper asr model object.
    """
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    kwargs = {}
    if len(config.asr.cache.model):
        kwargs['cache_dir'] = config.asr.cache.model

    model = WhisperForConditionalGeneration.from_pretrained(model_name, \
        quantization_config=BitsAndBytesConfig(load_in_8bit=True), **kwargs)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    return model


def get_asr(config):
    """
    Get a causal language model based on the configuration.

    Args:
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        AdapterModel: A causal language model object with optional adapter
            layers.
    """
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')
    if model_hub == 'huggingface_asr':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    else:
        raise NotImplementedError(f'Not support ASR {model_name} in'
                                  f' {model_hub}.')
        
    #Override generation arguments - no tokens are forced as decoder outputs see
    # (https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids))
    # no tokens are suppressed during generation see 
    # (https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens)):
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    #args = config.asr.adapter.args[0] if len(
    #    config.asr.adapter.args[0]) > 0 else {}
    #model = AdapterModel(model, use_adapter=config.asr.adapter.use, **args)

    return model
