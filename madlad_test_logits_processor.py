import transformers
import copy
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoConfig, LogitsProcessorList, BeamSearchScorer, MinLengthLogitsProcessor, LogitsProcessor
import argparse
import os
import sys
import torch
import numpy as np
from transformers import BitsAndBytesConfig
#from peft import PeftModel, PeftConfig
from torch import nn
import logging
import string

class LLMLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        model (`hf.model`):
            LLM Model for rescoring.
        tokenizer:
            LLM Tokenizer.
        sent_tokenizer:
            NMT Tokenizer.
        top_k:
            How many top tokens in each beam to use for rescoring.
        start_prefix_split:
            How many space split tokens during the starting of the decoding are prefix tokens which we dont pass to LLM for rescoring. E.g for NLLB its 1 <s><DE>
        src_text:
            The source file we want to translate to set the prefix of LLM while rescoring
        batch_size:
            Batch size during model.generate() to keep track of which sentence we are decoding
        num_beams:
            Number of beams passed to model.generate()
        alpha:
            Fixed Linear weight for NMT scores
        beta:
            Fixed Linear weight for LLM scores
    """

    def __init__(self, model, tokenizer, sent_tokenizer, gen_tokenizer, top_k, start_prefix_split, src_text, batch_size, num_beams, alpha, beta, prefix, suffix):

        self.model = model
        self.start_prefix_split = start_prefix_split
        self.topk = top_k
        self.tokenizer = tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.src_text = src_text
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.counter = -1
        self.alpha = alpha
        self.beta = beta
        self.past_key_values = None
        self.past_logits = None

        self.start_cnt = 1
        self.num_prefix_tokens = 1
        self.step = 1 # Use for Debugging


        #self.prefix = "Translate from English to German:\n English: "
        self.prefix = prefix
        self.suffix = suffix

        self.last_word_tokens = []
        self.prev_input_ids = None
        self.prev_llm_scores = None
        self.prev_word_ending = []
        self.beam_llm_scores = [0]*(num_beams*top_k)
        self.beam_sent_scores = [0]*(num_beams*top_k)



    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:


        logging.basicConfig(level=logging.DEBUG)
        #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='debug.log', filemode='w')
        if input_ids.shape[1] < self.num_prefix_tokens:
            return scores

        if input_ids.shape[1] == self.num_prefix_tokens:# We get input tokens with only prefix tokens. This is start of generating new batch
            logging.debug("Current batch number: %f",self.counter)
            self.counter +=1
            self.step = 1
            self.past_key_values = None

        curr_src_ids = list(range(self.counter*self.batch_size,(self.counter+1)*self.batch_size)) # Get the source tokens for entries in this beams
        curr_src_beam_ids = [ids for ids in curr_src_ids for _ in range(self.num_beams)] # Duplicate each one times the number of beams. The first n_beam entries are for one source
        curr_src_beam_txt = [self.src_text[idx] for idx in curr_src_beam_ids] * self.topk # Create the prefix of source after selecting topk entries. result is a list of length num_beams*top_k

        assert len(curr_src_beam_ids) == input_ids.shape[0]

        logging.debug("Step number: %d", self.step)
        logging.debug("*"*100)


        cur_len = input_ids.shape[-1]
        topk_values, topk_indices = torch.topk(scores, self.topk, dim=1) #Get top k tokens for each beam 2D (batch_size*beam_size,top_k)

        
        rerank_prefix_tokens = []
        rerank_full_tokens = []
        rerank_new_tokens = []
        add_prefix_space = []

        new_input_ids = []
        sent_scores = []

        for row_idx in range(topk_indices.size(0)):
            top_idx = topk_indices[row_idx,:]

            top_tokens = self.sent_tokenizer.batch_decode(top_idx) 
            prev_tokens = self.sent_tokenizer.batch_decode(input_ids) 

            prev_tokens = [x.split("<unk>")[1] for x in prev_tokens]

            sent_scores.extend(topk_values[row_idx,:])

            for k in range(self.topk):
                concat_input = torch.cat((input_ids[row_idx],top_idx[k].unsqueeze(0)),dim = 0) 
                rerank_full_tokens.append(self.sent_tokenizer.decode(concat_input))
                new_input_ids.append(concat_input)


            #rerank_prefix_tokens.extend(prev_tokens)
            rerank_new_tokens.extend(top_tokens)


        #rerank_prefix_tokens = [x.lstrip() for x in rerank_prefix_tokens]


        logging.debug("Current New Tokens for Reranking, Each column is Top %d choices for prefix shown above", self.topk)
        debug_view_new = [rerank_new_tokens[i:i + self.topk] for i in range(0, len(rerank_new_tokens), self.topk)]
        for entry in debug_view_new:
            logging.debug(entry)
        logging.debug("="*100)

        rerank_full_new_tokens = [x.split("<unk>")[1] for x in rerank_full_tokens]
        #rerank_full_new_tokens = [x for x in rerank_full_tokens]
        rerank_full_new_tokens = [x.lstrip() if x!="" else "" for x in rerank_full_new_tokens]
        rerank_full_new_tokens = [x if x!=" " else "" for x in rerank_full_new_tokens]

        cxt_rerank_full_tokens = [rerank_full_new_tokens[idx] for idx in range(len(rerank_full_tokens))]
        cxt_rerank_prompt_tokens = [self.prefix + curr_src_beam_txt[idx] + self.suffix  for idx in range(len(rerank_full_tokens))]
        logging.debug("Current Full Sequence with Context joined for LLM rescoring shown, for top-1 entries only!")
        debug_view_cxt = [cxt_rerank_full_tokens[i:i + self.topk] for i in range(0, len(cxt_rerank_full_tokens), self.topk)]




        if self.past_key_values == None:
            llm_prompt_inputs = self.tokenizer(cxt_rerank_prompt_tokens,return_tensors='pt', padding=True,  return_token_type_ids=False).to(self.model.device)
            llm_prompt_outputs = self.model(**llm_prompt_inputs, labels=llm_prompt_inputs.input_ids, use_cache=True, return_dict=True)
            past_full_logits = llm_prompt_outputs.logits
            self.past_logits = nn.functional.log_softmax(past_full_logits,dim=-1)[:,-1:,:]

            self.past_key_values = llm_prompt_outputs.past_key_values
            last_word_tokens = rerank_full_new_tokens
        else:

            
            align_beam_idx = self.beam_align(self.prev_input_ids,input_ids)
            self.prev_last_word_tokens = [self.prev_last_word_tokens[idx] for idx in align_beam_idx for _ in range(self.topk)]
            self.prev_llm_scores = [self.prev_llm_scores[idx] for idx in align_beam_idx for _ in range(self.topk)]
            self.prev_word_ending = [self.prev_word_ending[idx] for idx in align_beam_idx for _ in range(self.topk)]
            self.beam_llm_scores = [self.beam_llm_scores[idx] for idx in align_beam_idx for _ in range(self.topk)]
            self.beam_sent_scores = [self.beam_sent_scores[idx] for idx in align_beam_idx for _ in range(self.topk)]

            last_word_tokens = [x.split(" ")[-1] if len(x.split(" ")) > 1 else x for x in rerank_full_new_tokens]
            #last_word_tokens = [x if x!="<unk> " else " " for x in last_word_tokens]
            #last_word_tokens = [x if x!="" else " " for x in last_word_tokens]
            logging.debug("Current Last Word Tokens")
            logging.debug(last_word_tokens)
            logging.debug("="*100)


        llm_inputs = self.tokenizer(cxt_rerank_full_tokens,return_tensors='pt', padding=True,  return_token_type_ids=False).to(self.model.device)
        llm_inputs['input_ids'] = llm_inputs.input_ids[:,1:]
        llm_inputs['attention_mask'] = llm_inputs.attention_mask[:,1:]

        if self.step == 1:
            mask = torch.all(llm_inputs['input_ids'] == 32000, dim=1)
            llm_inputs['input_ids'][mask,0] = 29871
            llm_inputs['attention_mask'][mask,0] = 1

        #llm_inputs = self.join_tokens(cxt_rerank_prefix_inputs,cxt_rerank_new_inputs)
        llm_outputs = self.model(**llm_inputs, labels=llm_inputs.input_ids, use_cache=True, return_dict=True, past_key_values = self.past_key_values)
        loss = llm_outputs.loss
        llm_logits = llm_outputs.logits
 
        #llm_next_outputs = torch.argmax(llm_logits[:,-1,:], dim=-1)

        llm_scores = []

        alpha = []
        beta = []
        prev_word_ending = []
        correct_error_scores = [0]*len(cxt_rerank_full_tokens)
        is_new_words = []

        
        for elem in range(len(cxt_rerank_full_tokens)):
            elem_input = llm_inputs.input_ids[elem]
            pad_tokens = torch.sum(elem_input == self.tokenizer.convert_tokens_to_ids('[PAD]'))

            last_word_inputs = self.tokenizer.encode(last_word_tokens[elem],return_tensors='pt', padding=False,return_token_type_ids=False, add_special_tokens=True)[0].to(self.model.device)

            if rerank_new_tokens[elem] == "</s>":
                last_word_inputs = torch.tensor([self.tokenizer.encode("</s>")[1]],device=self.model.device)
                pass
            else:
                if len(last_word_inputs)!=1:
                    last_word_inputs = last_word_inputs[1:]
                else:
                    last_word_inputs = torch.tensor([29871], device=self.model.device)


            #logging.debug("Input ids")
            #logging.debug(elem_input)

            nll_new_logits = nn.functional.log_softmax(llm_logits[elem],dim=-1)
            nll_new_logits = nll_new_logits[:len(elem_input) - pad_tokens]
            nll_full_logits = torch.cat((self.past_logits[0],nll_new_logits), dim=0)
            
            llm_next_outputs = torch.argmax(nll_full_logits[-1,:], dim=-1)
            #nll_full_logits = nll_full_logits[pad_tokens:]
            if len(elem_input) - pad_tokens == len(last_word_inputs):
                nll_logits = nll_full_logits
            else:
                nll_logits = nll_full_logits[len(elem_input) - pad_tokens  - len(last_word_inputs):len(elem_input)  - pad_tokens]
                #nll_logits = nll_full_logits[len(elem_input) - pad_tokens - 1 - len(last_word_inputs):len(elem_input)  - pad_tokens- 1]


            is_new_word = False
            if self.step !=1:
                if rerank_new_tokens[elem] != "</s>":
                    is_new_word = self.check_last_word(cxt_rerank_full_tokens[elem], self.prev_last_word_tokens[elem], last_word_tokens[elem])
                else:
                    is_new_word = True
                if self.prev_word_ending[elem] == True and is_new_word == False:
                    correct_error_scores[elem] = -1 * self.prev_llm_scores[elem]
                if self.prev_word_ending[elem] == False and is_new_word == True:
                    correct_error_scores[elem] = self.prev_llm_scores[elem]
                    


            elem_score = torch.sum(nll_logits[range(len(last_word_inputs)),last_word_inputs])
            is_new_words.append(is_new_word)


            elem_score = torch.nan_to_num(elem_score, nan=-100)
            llm_scores.append(elem_score)


            gen_tok = self.tokenizer.sp_model.id_to_piece(llm_next_outputs.item())
            #if last_word_tokens[elem] != " ":
            if gen_tok == "</s>" or gen_tok[0] == "▁" or rerank_new_tokens[elem] == "</s>":
                alpha.append(self.alpha)
                beta.append(self.beta)
                prev_word_ending.append(True)
            else:
                alpha.append(1)
                beta.append(0)
                prev_word_ending.append(False)
            #if elem_score < -1.5 :
            #    alpha.append(1)
            #    beta.append(0)
            #else:
            #    alpha.append(0)
            #    beta.append(1)

    

        self.beam_llm_scores = [b*x+y+z for b,x,y,z in zip(beta,llm_scores,correct_error_scores,self.beam_llm_scores)]
        self.beam_sent_scores = [a*x + y for a,x,y in zip(alpha,sent_scores,self.beam_sent_scores)]

        alpha_vals = torch.tensor(alpha).view(input_ids.shape[0],-1).contiguous().to(input_ids.device)
        beta_vals = torch.tensor(beta).view(input_ids.shape[0],-1).contiguous().to(input_ids.device)
        correct_offset_scores = torch.tensor(correct_error_scores).view(input_ids.shape[0],-1).contiguous().to(input_ids.device)

        self.prev_llm_scores = llm_scores ## Save as list before reshaping and merging
        llm_scores = torch.tensor(llm_scores).view(input_ids.shape[0],-1).contiguous().to(input_ids.device)
        #reranked_scores = copy.deepcopy(scores)
        reranked_scores = torch.full_like(scores, -float("inf")).to(input_ids.device)
        for i in range(reranked_scores.shape[0]):
            reranked_scores[i,topk_indices[i,:]] = alpha_vals[i]*scores[i,topk_indices[i,:]] + beta_vals[i]*llm_scores[i] + self.beta*correct_offset_scores[i]
            

            #reranked_scores[i,topk_indices[i,:]] = self.alpha*scores[i,topk_indices[i,:]] + self.beta*llm_scores[i]
        
        self.prev_last_word_tokens = last_word_tokens
        self.prev_input_ids = torch.stack(new_input_ids)
        self.prev_word_ending = prev_word_ending

        logging.debug("Next token Space boolean")
        logging.debug(prev_word_ending)
        logging.debug("Reranked Scores")
        logging.debug(reranked_scores[reranked_scores != -float("inf")])
        logging.debug("="*100)

        logging.debug("LLM Scores for top-k tokens for each beam, aligned with the matrix of new tokens")
        logging.debug(llm_scores)
        logging.debug("="*100)
        logging.debug("NMT Scores for top=k tokens for each beam")
        logging.debug(topk_values)
        logging.debug("="*100)
        self.step+=1
        logging.debug("Beam LLM Scores")
        logging.debug(self.beam_llm_scores)
        logging.debug("="*100)
        logging.debug("Beam Sent Scores")
        logging.debug(self.beam_sent_scores)
        logging.debug("="*100)
        breakpoint()





        return reranked_scores

    def beam_align(self, tensor_old, tensor_new):
        
        row_mapping = [] # First elem indicates which row tensor_new corresponds with old

        for i in range(tensor_new.size(0)):
            for j in range(tensor_old.size(0)):
                if torch.all(tensor_old[j,:] == tensor_new[i,:]):
                    row_mapping.append(j)
                    break

        return row_mapping

    def join_detok(self,llm_prefix,llm_suffix):
        joined_tokens = {}
        joined_tokens['input_ids'] = torch.cat((llm_prefix['input_ids'],llm_suffix['input_ids']),dim=1)
        joined_tokens['attention_mask'] = torch.cat((llm_prefix['attention_mask'],llm_suffix['attention_mask']),dim=1)
        join_detok = self.tokenizer.batch_decode(joined_tokens['input_ids'], skip_special_tokens=True)
        return join_detok 

    def check_last_word(self, full_sent, prev_last_word, last_word):

        if full_sent == prev_last_word:
            return True

        joined_word = prev_last_word + " " + last_word
        if full_sent[-len(joined_word):] == joined_word:
            return True
        else:
            last_word = last_word.strip()
            prev_last_word = prev_last_word.strip()
            if last_word == "" or prev_last_word == "" or "</s>" in last_word:
                return True
            else:
                return False
