import transformers
from typing import Callable, Iterable, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoConfig, LogitsProcessorList, BeamSearchScorer, MinLengthLogitsProcessor, LogitsProcessor, LlamaTokenizer
import argparse
import os
import sys
import torch
import numpy as np
import json
from transformers import BitsAndBytesConfig
#from peft import PeftModel, PeftConfig
import pandas as pd
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.big_modeling import infer_auto_device_map, init_empty_weights
from accelerate.utils import get_balanced_memory
import pickle
from optimum.bettertransformer import BetterTransformer
from torch import nn
from peft import PeftModel
from llm_logits_processor import LLMLogitsProcessor



def get_parser():
    """
    Generate a parameter parser
    """
    parser = argparse.ArgumentParser(description="Evaluate Translation for LLM's")
    parser.add_argument("--input_file", type=str, default="/project/OML/skoneru/adapt_llm/context_llm/MuDA/eval_data/mustc/train_v3/test.en")
    parser.add_argument("--output_file", type=str, default="./tst2019.de")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model", type=str, default="google/flan-t5-small")
    parser.add_argument("--cache_dir", type=str, default="/export/data1/skoneru/cache")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--beta", type=float, default=1)

    return parser

def read_data(filepath):
    data = open(filepath, mode='r', encoding='utf-8', newline='\n').readlines()
    data = [x.rstrip() for x in data]
    return data

def write_list(data, fname):
    with open(fname, 'w', encoding='utf-8') as (f):
        for item in data:
            try:
                f.write('%s\n' % item)
            except:
                item = item.encode('utf-8')
                f.write('%s\n' % item)

def process_sent(gen_sent, prompt):
    prompt_len = len(prompt)
    gen_sent = gen_sent[prompt_len:]
    gen_sent = gen_sent.replace("\n", ".")
    return gen_sent

def divide_chunks(l, n):
         for i in range(0, len(l), n):
             yield l[i:i + n]

def main(params):
    
    transformers.set_seed(0)

    src = read_data(params.input_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = params.model
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForSeq2SeqLM.from_config(config)
    model.tie_weights()
    device_map = infer_auto_device_map(
        model,
        # Force splits model.encoder into separate layers and devices
        no_split_module_classes=model._no_split_modules
        + ["NllbMoeEncoderLayer", "NllbMoeDecoderLayer"],
        dtype="int8",
    )

    # Demonstrate that only "model.encoder.layer_norm" and "model.encoder.embed_tokens"
    # needs to be on the same device as the input
    #for module, device in device_map.items():
    #    if module in {"model.encoder.layer_norm"}:
    #        if device != 0:
    #            print("Maapping to 0")
    #            device_map[module] = 0
    #    else:
    #        if device == 0:
    #            device_map[module] = 0

    for module, device in device_map.items():
        device_map[module] = 0
    
    tokenizer = AutoTokenizer.from_pretrained(params.model, cache_dir="/project/OML/skoneru/iwslt23/scripts/bloom/cache/")
    #model = AutoModelForSeq2SeqLM.from_pretrained(params.model,
	#    cache_dir=params.cache_dir, 
	#    device_map=device_map,
	#    load_in_8bit=True, offload_folder='/project/OML/skoneru/iwslt23/scripts/bloom/cache/',)
    model_hf = AutoModelForSeq2SeqLM.from_pretrained(params.model,
	    cache_dir=params.cache_dir, 
	    device_map=device_map,
	    offload_folder='/project/OML/skoneru/iwslt23/scripts/bloom/cache/')
    model = BetterTransformer.transform(model_hf, keep_original_model=True)
    #model = model_hf

    #lm_head_device = model.hf_device_map["lm_head"]
    #print(lm_head_device)
    
    src_batches = list(divide_chunks(src,params.batch_size))
    #lm_head_device = model.hf_device_map["model.encoder.layer_norm"]
    hyp_llm = []

    config = AutoConfig.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    #config = AutoConfig.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    with init_empty_weights():
        llm_model = AutoModelForCausalLM.from_config(config)
    llm_model.tie_weights()
    device_map = infer_auto_device_map(
        llm_model,
        # Force splits model.encoder into separate layers and devices
    )

    # Demonstrate that only "model.encoder.layer_norm" and "model.encoder.embed_tokens"
    # needs to be on the same device as the input
    for module, device in device_map.items():
        device_map[module] = 1



    meta_llama = "meta-llama/Llama-2-13b-chat-hf"
    llm_tokenizer = LlamaTokenizer.from_pretrained(meta_llama, cache_dir="/project/OML/skoneru/iwslt23/scripts/bloom/cache/", padding_side="left")
    llm_gen_tokenizer = LlamaTokenizer.from_pretrained(meta_llama, cache_dir="/project/OML/skoneru/iwslt23/scripts/bloom/cache/", padding_side="left")
    #llm_model = AutoModelForCausalLM.from_pretrained(meta_llama, device_map=device_map, cache_dir="/export/data1/skoneru/cache/", offload_folder='/project/OML/skoneru/iwslt23/scripts/bloom/cache/',
    #            quantization_config=BitsAndBytesConfig(
    #            load_in_4bit=True,
    #            llm_int8_threshold=6.0,
    #            llm_int8_has_fp16_weight=False,
    #            bnb_4bit_compute_dtype=torch.float16,
    #            bnb_4bit_use_double_quant=True,
    #            bnb_4bit_quant_type="nf4",
    #        ),
    #            torch_dtype=torch.float16,)
    llm_model = AutoModelForCausalLM.from_pretrained(meta_llama, device_map=device_map, cache_dir="/export/data1/skoneru/cache/", offload_folder='/project/OML/skoneru/iwslt23/scripts/bloom/cache/',torch_dtype=torch.bfloat16,load_in_8bit=True, attn_implementation="flash_attention_2",)
    #llm_model = PeftModel.from_pretrained(llm_model, "haoranxu/ALMA-7B-Pretrain-LoRA", cache_dir="/project/OML/skoneru/iwslt23/scripts/bloom/cache").to(llm_model.device)
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llm_gen_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llm_model.resize_token_embeddings(len(llm_tokenizer))


    prefix= "<s>[INST] <<SYS>>\nYou translate from English to German. You only translate to Formal German using words like 'sie' 'ihnen' and 'ihrer'. You translate to Formal German even in Informal scenarios.\n<</SYS>>\nEnglish: "
    #prefix= "<s>[INST] <<SYS>>\nYou translate from English to German. You only translate to Informal German using words like 'du' 'dich' and 'dir'. You translate to Informal German even in formal scenarios.\n<</SYS>>\nEnglish: "
    #self.prefix = "Translate from English to German:\nEnglish: "
    #prefix = "English:\nA college classmate wrote me a couple weeks ago and she said\nGerman:\n Eine Kommilitonin hat mir vor ein paar Wochen geschrieben und gesagt\nEnglish:\nI decided to pay a visit to the manager and he pointed\nGerman:  Also entschied ich mich den Filialleiter zu besuchen\nEnglish:\n"
    suffix = "\n[/INST]\nGerman:\n "
    #prefix= "[INST] <<SYS>>\nYou are a translator from English to German.\n<</SYS>>\nEnglish:"
    #prefix = "Translate from English to German:\nEnglish: "
    #suffix = "\nGerman: "
#
    logits_processor = LogitsProcessorList(
        [
            LLMLogitsProcessor(llm_model, llm_tokenizer,sent_tokenizer = tokenizer, gen_tokenizer = llm_gen_tokenizer, top_k=params.topk, start_prefix_split=1, src_text=src, batch_size=params.batch_size, num_beams=params.num_beams, alpha=params.alpha, beta=params.beta, prefix=prefix, suffix=suffix),
        ]
     )


    cnt = 0
    
    for j in range(len(src_batches)):
        src_batch = src_batches[j]
        src_batch = [x.split("<eos>")[-1] for x in src_batch]
        inputs = tokenizer(src_batch,return_tensors='pt', padding=True).to(model.device)
        #outputs = model.generate(**inputs, max_new_tokens=params.max_len,num_beams=params.num_beams, early_stopping=False, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], num_return_sequences=1)
        outputs = model.generate(**inputs, max_new_tokens=256,num_beams=params.num_beams, early_stopping=True, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"],logits_processor=logits_processor, num_return_sequences=1)
        hyps = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        hyp_llm.extend(hyps)
        cnt+=len(src_batch)
        print(hyps)


    write_list(hyp_llm, params.output_file)

    return

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    main(params)

