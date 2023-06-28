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

from megatron.utils import print_rank_0, setup_for_inference_or_eval

from megatron.text_generation_utils import (
    generate_samples_input_from_file,
    generate_samples_from_prompt,
    generate_samples_unconditional,
    generate_samples_interactive,
)

from megatron.training import forward_step

from transformers import AutoTokenizer


def main():
    """
    Generate text/sample model
    """
    model, neox_args = setup_for_inference_or_eval(use_cache=True)
    neox_args.recompute = False
    
    if neox_args.recompute:
        model.module.inference_mode(
            use_cache=False
        )  # don't use kv cache if recomputing
        
    # print(model)
    
    x = "hello world"
    neox_args.return_logits = True
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # print(neox_args.top_k, neox_args.top_p)
    y = generate_samples_from_prompt(
        neox_args=neox_args,
        model=model,
        text=x,
        recompute=neox_args.recompute,
        temperature=1.0,
        maximum_tokens=10,
        top_k=1,
        top_p=1.0,
        stop_tokens=tokenizer.encode('#include'),
    )
    print(y[0]['text'])
    print(y[0]['generated_tokens'])
    print(y[0]['logits'].shape)
    print(y[0]['logits'].log_softmax(-1))
    print(y[0]['logits'].argmax(-1))
    
    # inputs = tokenizer('hello world! who are you? I am fine.', return_tensors='pt')
    # input_it = iter([{"text": F.pad(inputs.input_ids, pad=(0, 1))}])
    # outputs = forward_step(model=model, neox_args=neox_args, timers=None, return_logits=True, data_iterator=input_it)
    # print(inputs.input_ids.shape, outputs[1].shape)


if __name__ == "__main__":
    main()
