import torch
from transformers import AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AutoConfig, AutoTokenizer, OPTForCausalLM
import logging

logger = logging.getLogger(__name__)

def get_local_huggingface_tokenizer_model(model_name, model_path=None, dtype=None):

    _fn_reset = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = (lambda x: None)

    dtype = dtype or torch.float16
    model_path = model_path or model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)

    if 'gptq' in model_path:
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(model_path, device="cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
        
    tokenizer.add_bos_token = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    torch.nn.Linear.reset_parameters = _fn_reset
    
    return model, tokenizer


def get_local_huggingface_tokenizer_model_llm_int8(model_name, model_path=None, dtype=None):
    
    if model_path is None:
        model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', load_in_8bit=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    return model, tokenizer


def get_dist_accelerate_tokenizer_model(model_name, model_path, dtype=None):
    from accelerate import init_empty_weights,load_checkpoint_and_dispatch
    if model_name == "facebook/galactica-120b":
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
            model = load_checkpoint_and_dispatch(
                model, model_path, device_map="auto", no_split_module_classes=["OPTDecoderLayer"]
            )
            tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-120b")
    elif model_name == "facebook/opt-iml-175b-max":
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
            # state_dict = torch.load(model_path+'/opt-iml-max.pt')
            # model.load_state_dict(state_dict)
            model = load_checkpoint_and_dispatch(
                model, model_path+'/opt-iml-max.pt', device_map="auto", no_split_module_classes=["OPTDecoderLayer"]
            )
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-30b", use_fast=False)
    elif model_name == "facebook/opt-iml-175b-regular":
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
            model = load_checkpoint_and_dispatch(
                model, model_path+'/opt-iml-regular.pt', device_map="auto", no_split_module_classes=["OPTDecoderLayer"]
            )
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-30b", use_fast=False)
    elif 'llama' in model_name:
        print('loading llama...............')
        config = AutoConfig.from_pretrained(model_path)
        dtype = dtype or torch.float16
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config).to(dtype)
        model = load_checkpoint_and_dispatch(
            model, model_path, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"]
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_bos_token = False
    elif 'bloom' in model_name:
        print('loading bloom...............')
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config).bfloat16()
            model.tie_weights()
        model = load_checkpoint_and_dispatch(
            model, model_path, device_map="balanced_low_0", no_split_module_classes=["BloomBlock"]
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        assert False, f"Not legal name {model_name}"
    print(f"<get_dist_accelerate_tokenizer_model>: {model_name} hf_device_map")
    print(model.hf_device_map)
    return model, tokenizer


def get_dist_alpa_tokenizer_model(model_name, model_path):
    from llm_serving.model.wrapper import get_model
    if model_name == 'opt-2.7b':
        # The 30B version works for all OPT models.
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        tokenizer.add_bos_token = False
        model = get_model(model_name="alpa/opt-2.7b", path=model_path)
    elif model_name == 'opt-175b':
        # The 30B version works for all OPT models.
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
        tokenizer.add_bos_token = False
        model = get_model(model_name="alpa/opt-175b", path=model_path)
    elif model_name == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom')
        tokenizer.add_bos_token = False
        model = get_model(model_name="alpa/bloom", path=model_path)
    elif model_name == 'bloomz':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloomz')
        tokenizer.add_bos_token = False
        # llm_serving does not recoginze bloomz, since the model parameter is from bloomz,
        # this should be fine
        model = get_model(model_name="alpa/bloom", path=model_path)
    else:
        assert False, f"Not legal name {model_name}"

    return model, tokenizer

