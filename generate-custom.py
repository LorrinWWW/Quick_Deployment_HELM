#!/usr/bin/env python
# Copyright (c) 2021 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from reserve.utils import print_rank_0, setup_for_inference_or_eval

from reserve.text_generation_utils import (
    generate_samples_input_from_file,
    generate_samples_from_prompt,
    generate_samples_unconditional,
    generate_samples_interactive,
)

from reserve.training import forward_step

from transformers import AutoTokenizer


def main():
    """
    Generate text/sample model
    """
    
    model, neox_args = setup_for_inference_or_eval(use_cache=True)
    neox_args.recompute = False
    neox_args.recompute = True
    
    if neox_args.recompute:
        model.module.inference_mode(
            use_cache=False
        )  # don't use kv cache if recomputing

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # tokenizer = neox_args.tokenizer


    # contexts = '1 2 3 4 5 6 7 8 9 10'
    # tokenizer.pad_token = tokenizer.eos_token
    # inputs = tokenizer(contexts, padding=True, truncation=True, return_tensors="pt").to('cuda')
    # input_length = inputs.input_ids.shape[1]
    
    # input_it = iter([{"text": F.pad(inputs.input_ids, pad=(0, 1))}])
    # outputs = forward_step(model=model, neox_args=neox_args, timers=None, return_logits=True, data_iterator=input_it)
    
    # ### hard code, assume bsz==1
    # n_logprobs = 1
    # # sampled tokens
    # token_ids = inputs.input_ids[0].tolist()
    # tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # logprobs_dict = {
    #     'tokens': tokens,
    #     'token_logprobs': [None],
    #     'top_logprobs': [None],
    # }
    # logprobs = outputs[1].nan_to_num().log_softmax(-1).nan_to_num()

    # values, indices = logprobs.topk(n_logprobs, dim=-1)
    # for i in range(indices.size(1)-1):
    #     selected_token_id = token_ids[i+1]
    #     # topk tokens
    #     tokens = tokenizer.convert_ids_to_tokens(indices[0, i])
    #     # topk scores
    #     scores = values[0, i].tolist()
    #     logprobs_dict['token_logprobs'].append(logprobs[0, i, selected_token_id].item())
    #     logprobs_dict['top_logprobs'].append({
    #         t: s for t,s in zip(tokens, scores)
    #     })

    # print(logprobs_dict['token_logprobs'])
    # print(logprobs_dict['top_logprobs'])
    
    
    x = "The man in the blue shirt sits on the chair next to the sink. The other man begins washing his hair. he then walks over to the sink and smiles while shaking his wet hair."
    neox_args.return_logits = True

    print(neox_args.tokenizer.tokenize(x))
    print(tokenizer(x))
    
    tic = time.time()
    
    # print(neox_args.top_k, neox_args.top_p)
    y = generate_samples_from_prompt(
        neox_args=neox_args,
        model=model,
        text=x,
        recompute=neox_args.recompute,
        temperature=1.0,
        maximum_tokens=20,
        top_k=1,
        top_p=1.0,
        # # stop_tokens=tokenizer.encode('#include'),
    )
    print(y[0]['text'])
    print(y[0]['generated_tokens'])
    print(y[0]['logits'].shape)
    print(y[0]['logits'].log_softmax(-1))
    print(y[0]['logits'].log_softmax(-1).max(-1))
    print(y[0]['logits'].argmax(-1))
    
    toc = time.time()
    
    print('time:', toc - tic)

    # contexts = '''Can you talk about Zurich?'''
    # tokenizer.pad_token = tokenizer.eos_token
    # inputs = tokenizer(contexts, padding=True, truncation=True, return_tensors="pt").to('cuda')
    # input_length = inputs.input_ids.shape[1]
    
    # input_it = iter([{"text": F.pad(inputs.input_ids, pad=(0, 1))}])
    # outputs = forward_step(model=model, neox_args=neox_args, timers=None, return_logits=True, data_iterator=input_it)
    # # outputs = (0, model.module(inputs.input_ids))
    
    # ### hard code, assume bsz==1
    # n_logprobs = 1
    # # sampled tokens
    # token_ids = inputs.input_ids[0].tolist()
    # tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # logprobs_dict = {
    #     'tokens': tokens,
    #     'token_logprobs': [None],
    #     'top_logprobs': [None],
    # }
    # logprobs = outputs[1].nan_to_num().log_softmax(-1).nan_to_num()

    # values, indices = logprobs.topk(n_logprobs, dim=-1)
    # for i in range(indices.size(1)-1):
    #     selected_token_id = token_ids[i+1]
    #     # topk tokens
    #     tokens = tokenizer.convert_ids_to_tokens(indices[0, i])
    #     # topk scores
    #     scores = values[0, i].tolist()
    #     logprobs_dict['token_logprobs'].append(logprobs[0, i, selected_token_id].item())
    #     logprobs_dict['top_logprobs'].append({
    #         t: s for t,s in zip(tokens, scores)
    #     })

    # print(logprobs_dict['token_logprobs'])
    # print(logprobs_dict['top_logprobs'])
    # print(sum(logprobs_dict['token_logprobs'][1:]) / (len(logprobs_dict['token_logprobs']) - 1))


    # contexts = '''Can you talk about Seattle?'''
    # tokenizer.pad_token = tokenizer.eos_token
    # inputs = tokenizer(contexts, padding=True, truncation=True, return_tensors="pt").to('cuda')
    # input_length = inputs.input_ids.shape[1]
    
    # input_it = iter([{"text": F.pad(inputs.input_ids, pad=(0, 1))}])
    # outputs = forward_step(model=model, neox_args=neox_args, timers=None, return_logits=True, data_iterator=input_it)
    # # outputs = (0, model.module(inputs.input_ids))
    
    # ### hard code, assume bsz==1
    # n_logprobs = 1
    # # sampled tokens
    # token_ids = inputs.input_ids[0].tolist()
    # tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # logprobs_dict = {
    #     'tokens': tokens,
    #     'token_logprobs': [None],
    #     'top_logprobs': [None],
    # }
    # logprobs = outputs[1].nan_to_num().log_softmax(-1).nan_to_num()

    # values, indices = logprobs.topk(n_logprobs, dim=-1)
    # for i in range(indices.size(1)-1):
    #     selected_token_id = token_ids[i+1]
    #     # topk tokens
    #     tokens = tokenizer.convert_ids_to_tokens(indices[0, i])
    #     # topk scores
    #     scores = values[0, i].tolist()
    #     logprobs_dict['token_logprobs'].append(logprobs[0, i, selected_token_id].item())
    #     logprobs_dict['top_logprobs'].append({
    #         t: s for t,s in zip(tokens, scores)
    #     })

    # print(logprobs_dict['token_logprobs'])
    # print(logprobs_dict['top_logprobs'])
    # print(sum(logprobs_dict['token_logprobs'][1:]) / (len(logprobs_dict['token_logprobs']) - 1))
    
if __name__ == "__main__":
    main()
