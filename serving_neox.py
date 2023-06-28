import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import random
import logging
import argparse
import traceback
import numpy as np
from faiss_retrieval import *
from utils import *
from model_utils import *
from typing import Dict
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig, StoppingCriteriaList
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList
from together_worker.fast_inference import FastInferenceInterface
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherWeb3, TogetherClientOptions

from megatron.utils import print_rank_0, setup_for_inference_or_eval

from megatron.text_generation_utils import generate_samples_from_prompt

from megatron.training import forward_step

logger = logging.getLogger(__name__)

logger.setLevel(int(os.environ.get('LOG_LEVEL', logging.DEBUG)))

def translate_chatml_to_openchat(prompt):
    prompt = prompt.replace('<|im_start|>system\n', '<human>: ')
    prompt = prompt.replace('<|im_start|>user\n', '<human>: ')
    prompt = prompt.replace('<|im_start|>assistant\n', '<bot>: ')
    prompt = prompt.replace('<|im_start|>user', '<human>:')
    prompt = prompt.replace('<|im_start|>assistant', '<bot>:')
    prompt = prompt.replace('\n<|im_end|>', '')
    prompt = prompt.replace('<|im_end|>', '')
    prompt = prompt.rstrip()
    # print(prompt)
    return prompt

class NeoXInference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args if args is not None else {})
        logging.debug(f"Model name: {model_name}")
        logging.debug("\n=============== Arguments ===============")
        logging.debug(args.keys())
        logging.debug(args)
        logging.debug("=========================================\n")
        self.task_info = {
            "seed": 0,
            "prompt_seqs": None,
            "output_len": 16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "beam_search_diversity_rate": 0,
            "temperature": 0.1,
            "len_penalty": 0,
            "repetition_penalty": 1.0,
            "stop": [],
            "logprobs": 0,
            "echo": False,
            "penalty_alpha": 0,
        }
        self.device = args['device']
        self.hf_model_name = args['hf_model_name']
        self.max_batch_size = args['max_batch_size']
        self.deny_list = args['deny_list']
        
        model, neox_args = setup_for_inference_or_eval(use_cache=bool(int(os.environ.get('USE_CACHE', 1))))
        neox_args.recompute = (not bool(int(os.environ.get('USE_CACHE', 1))))
        print('RECOMPUTE:', neox_args.recompute)
        if neox_args.recompute:
            model.module.inference_mode(
                use_cache=False
            )  # don't use kv cache if recomputing
        
        # TODO: overwrite 
        neox_args.return_logits = True
        self.neox_args = neox_args
        self.model = model
        # TODO: hard code
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.plugin = args.get('plugin')
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        logging.debug(f"<NeoXInference.__init__> initialization done")

    def dispatch_request(self, args, env) -> Dict:
        plugin_state = {}
        if self.plugin:
            self.plugin.request(args, env, plugin_state)
        logging.debug(f"<NeoXInference.dispatch_request> starts")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["seed"] = get_int(args.get("seed", 0), default=0)
        if isinstance(str(args['prompt']), str):
            self.task_info["prompt_seqs"] = [str(args['prompt'])]
        elif isinstance(str(args['prompt']), list):
            self.task_info["prompt_seqs"] = args['prompt']
        else:
            logging.debug("wrong prompt format, it can only be str or list of str")
            return
        self.task_info["output_len"] = get_int(args.get("max_tokens", 16), default=16)
        self.task_info["beam_width"] = get_int(args.get("beam_width", 1), default=1)
        self.task_info["top_k"] = get_int(args.get("top_k", 50), default=50)
        if self.hf_model_name == "google/flan-t5-xxl":
            self.task_info["top_p"] = get_float(args.get("top_p", 1.0), default=1.0)
        else:
            self.task_info["top_p"] = get_float(args.get("top_p", 0.0), default=0.0)
        self.task_info["beam_search_diversity_rate"] = get_float(args.get("beam_search_diversity_rate", 0.0),
                                                                 default=0.0)
        self.task_info["temperature"] = get_float(args.get("temperature", 0.8), default=0.8)
        self.task_info["len_penalty"] = get_float(args.get("len_penalty", 0.0), default=0.0)
        self.task_info["repetition_penalty"] = get_float(args.get("repetition_penalty", 1.0), default=1.0)
        self.task_info["penalty_alpha"] = get_float(args.get("penalty_alpha", 0), default=0)
        self.task_info["stop"] = args.get("stop", [])
        self.task_info["logprobs"] = get_int(args.get("logprobs", 0), default=0)
        self.task_info["echo"] = bool(get_int(args.get("echo", 0), default=0))

        if args.get("stream_tokens"):
            self.task_info["stream_tokens"] = lambda token: self.stream_tokens(token, env)

        if len(self.task_info["prompt_seqs"][0]) == 0 or (self.task_info["output_len"] == 0 and self.task_info["echo"] == False):
            inference_result = []
            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                choice = {
                    "text": '',
                    "index": beam_id,
                    "finish_reason": "length"
                }
                item['choices'].append(choice)
            inference_result.append(item)
            #  So far coordinator does not support batch.
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inference_result[0]['choices'],
                "raw_compute_time": 0.0
            }
            logging.debug(f"<NeoXInference.dispatch_request> (empty input or output) return: {result}")
            if self.plugin:
                return self.plugin.response(result, plugin_state)
            return result
        else:
            result = self._run_inference()
            torch.cuda.empty_cache()
            logging.debug(f"<NeoXInference.dispatch_request> return: {result}")
            if self.plugin:
                return self.plugin.response(result, plugin_state)
            return result

    def _run_inference(self):
        logging.debug(f"<NeoXInference._run_inference> start.")
        complete_contexts = self.task_info["prompt_seqs"]
        
        
        if self.task_info["echo"]:
            
            with torch.no_grad():
                logging.debug(self.task_info)
                torch.manual_seed(self.task_info['seed'])
                np.random.seed(self.task_info['seed'])
                random.seed(self.task_info['seed'])
                batch_size = min(len(complete_contexts), self.max_batch_size)
                num_iter = math.ceil(len(complete_contexts) / batch_size)
                output_buffer = []
                logprobs_buffer = []
                output_scores = self.task_info["logprobs"] > 0
                if output_scores:
                    logprobs_buffer = []
                else:
                    logprobs_buffer = None

                time = timeit.default_timer()
                for iter_i in range(num_iter):
                    contexts = complete_contexts[iter_i * batch_size: (iter_i + 1) * batch_size]
                    # Do translation
                    contexts = [translate_chatml_to_openchat(context) for context in contexts]
                    inputs = self.tokenizer(contexts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    logging.debug(f"start_ids: length ({inputs.input_ids.shape[0]}) ids: {inputs.input_ids}")
                    input_length = inputs.input_ids.shape[1]
                    
                    input_it = iter([{"text": F.pad(inputs.input_ids, pad=(0, 1))}])
                    outputs = forward_step(model=self.model, neox_args=self.neox_args, timers=None, return_logits=True, data_iterator=input_it)

                    if output_scores:
                        ### hard code, assume bsz==1
                        n_logprobs = self.task_info["logprobs"]

                        # sampled tokens
                        token_ids = inputs.input_ids[0].tolist()
                        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                        logprobs_dict = {
                            'tokens': tokens,
                            'token_logprobs': [None],
                            'top_logprobs': [None],
                        }

                        logprobs = outputs[1].nan_to_num().log_softmax(-1).nan_to_num()
                        values, indices = logprobs.topk(n_logprobs, dim=-1)

                        for i in range(indices.size(1)-1):
                            selected_token_id = token_ids[i+1]
                            # topk tokens
                            tokens = self.tokenizer.convert_ids_to_tokens(indices[0, i])
                            # topk scores
                            scores = values[0, i].tolist()

                            logprobs_dict['token_logprobs'].append(logprobs[0, i, selected_token_id].item())
                            logprobs_dict['top_logprobs'].append({
                                t: s for t,s in zip(tokens, scores)
                            })

                        logprobs_buffer.append(logprobs_dict)

                    output_buffer.append(outputs)
                time_elapsed = timeit.default_timer() - time

            logging.debug(f"[INFO] NeoXInference time costs: {time_elapsed} ms. ")

            if len(complete_contexts) == 1:
                item = {'choices': [], }
                for beam_id in range(self.task_info["beam_width"]):
                    token = inputs.input_ids[0].tolist()
                    logging.debug(f"[INFO] raw token: {token}")
                    output = self.tokenizer.decode(token)
                    logging.debug(f"[INFO] beam {beam_id}: \n[Context]\n{contexts}\n\n[Output]\n{output}\n")
                    choice = {
                        "text": output,
                        "index": beam_id,
                        "finish_reason": "length"
                    }
                    if output_scores:
                        choice['logprobs'] = logprobs_buffer[0]
                    item['choices'].append(choice)
                result = {
                    "result_type": RequestTypeLanguageModelInference,
                    "choices": item['choices'],
                    "raw_compute_time": time_elapsed
                }
            else:
                raise Exception("not impl yet")
        
        else:

            with torch.no_grad():
                logging.debug(self.task_info)
                torch.manual_seed(self.task_info['seed'])
                np.random.seed(self.task_info['seed'])
                random.seed(self.task_info['seed'])
                batch_size = min(len(complete_contexts), self.max_batch_size)
                num_iter = math.ceil(len(complete_contexts) / batch_size)
                generation_output_buffer = []
                logprobs_buffer = []
                output_scores = self.task_info["logprobs"] > 0
                if output_scores:
                    logprobs_buffer = []
                else:
                    logprobs_buffer = None

                time = timeit.default_timer()
                for iter_i in range(num_iter):
                    contexts = complete_contexts[iter_i * batch_size: (iter_i + 1) * batch_size]
                    # Do translation
                    contexts = [translate_chatml_to_openchat(context) for context in contexts]
                    inputs = self.tokenizer(contexts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    logging.debug(f"start_ids: length ({inputs.input_ids.shape[0]}) ids: {inputs.input_ids}")
                    input_length = inputs.input_ids.shape[1]
                        
                    logging.debug(f"""[Break] {inputs.input_ids}, {self.task_info['top_p']}, {self.task_info['top_k']}, self.task_info["temperature"], {self.task_info["output_len"] + input_length}, {output_scores}""")
                    try:
                        
                        if self.task_info["temperature"] == 0:
                            generation_outputs = generate_samples_from_prompt(
                                neox_args=self.neox_args,
                                model=self.model,
                                text=contexts,
                                recompute=self.neox_args.recompute,
                                temperature=1,
                                maximum_tokens=self.task_info["output_len"],
                                top_p=1,
                                top_k=1,
                                stop_tokens=(self.tokenizer.encode(self.task_info["stop"][0]) if len(self.task_info["stop"]) else None),
                            )[0] # TODO: assert only one sequence
                        else:
                            generation_outputs = generate_samples_from_prompt(
                                neox_args=self.neox_args,
                                model=self.model,
                                text=contexts,
                                recompute=self.neox_args.recompute,
                                temperature=self.task_info["temperature"],
                                maximum_tokens=self.task_info["output_len"],
                                top_p=self.task_info['top_p'],
                                top_k=self.task_info['top_k'],
                                stop_tokens=(self.tokenizer.encode(self.task_info["stop"][0]) if len(self.task_info["stop"]) else None),
                            )[0] # TODO: assert only one sequence
                        
                    except Exception as e:
                        traceback.print_exc()
                        raise e
                    
                    logging.debug(f"[Break] So far so good")
                    
                    if output_scores:
                        
                        ### hard code, assume bsz==1
                        n_logprobs = self.task_info["logprobs"]

                        # sampled tokens
                        token_ids = generation_outputs["generated_tokens"]
                        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                        logits = generation_outputs['logits']

                        logprobs_dict = {
                            'tokens': tokens,
                            'token_logprobs': [],
                            'top_logprobs': [],
                        }

                        # origianl logits
                        logits = logits.unsqueeze(0).nan_to_num()
                        logprobs = logits.log_softmax(-1).nan_to_num()
                        values, indices = logprobs.topk(n_logprobs, dim=-1)

                        for i in range(len(token_ids)):
                            selected_token_id = token_ids[i]
                            # topk tokens
                            tokens = self.tokenizer.convert_ids_to_tokens(indices[0, i])
                            # topk scores
                            scores = values[0, i].tolist()

                            logprobs_dict['token_logprobs'].append(logprobs[0, i, selected_token_id].item())
                            logprobs_dict['top_logprobs'].append({
                                t: s for t,s in zip(tokens, scores)
                            })

                        logprobs_buffer.append(logprobs_dict)

                    generation_output_buffer.append(generation_outputs)
                time_elapsed = timeit.default_timer() - time

            logging.debug(f"[INFO] NeoXInference time costs: {time_elapsed} ms. ")

            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                token_ids = generation_outputs["generated_tokens"]
                token = self.tokenizer.convert_ids_to_tokens(token_ids)
                logging.debug(f"[INFO] raw token: {token}")
                output = self.tokenizer.convert_tokens_to_string(token)
                logging.debug(f"[INFO] beam {beam_id}: \n[Context]\n{contexts}\n\n[Output]\n{output}\n")
                choice = {
                    "text": post_processing_text(output, self.task_info["stop"], self.deny_list),
                    "index": beam_id,
                    "finish_reason": "length"
                }
                if output_scores:
                    choice['logprobs'] = logprobs_buffer[0]
                item['choices'].append(choice)
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": item['choices'],
                "raw_compute_time": time_elapsed
            }
            
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    plugin = None

    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coord_http_port = os.environ.get("COORD_HTTP_PORT", "8092")
    coord_ws_port = os.environ.get("COORD_WS_PORT", "8093")
    deny_list = []
    try:
        deny_list = json.loads(os.environ.get("DENY_LIST", "[]"))
    except Exception as e:
        logging.error(f"failed to parse deny list: {e}")
    try:
        deny_list_file = os.environ.get("DENY_LIST_FILE", "")
        if deny_list_file != None:
            with open(deny_list_file, "r") as f:
                deny_list = [line.strip() for line in f.readlines()]
    except Exception as e:
        logging.error(f"failed to parse deny list file: {e}")
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:{coord_http_port}",
        websocket_url=f"ws://{coord_url}:{coord_ws_port}/websocket"
    )
    fip = NeoXInference(model_name=os.environ.get('SERVICE', 'Together-gpt-JT-6B-v1'), args={
        "coordinator": coordinator,
        "device": "cuda",
        "dtype": None,
        "hf_model_name": "test",
        "model_path": "test",
        "worker_name": os.environ.get('WORKER', 'worker1'),
        "group_name": os.environ.get('GROUP', 'group1'),
        "max_batch_size": 1,
        "gpu_num":1,
        "gpu_type":"RTX 3090",
        "gpu_mem":2400000,
        "deny_list": deny_list,
        "plugin": plugin,
    })
    
    fip.start()
