import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset


from huggingface_hub import login

login("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")


def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    print("Number of GPUs:", n_gpus)
    max_memory = f'{9060}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available resources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    #print(model,tokenizer)
    return model, tokenizer

# Load the databricks dataset from Hugging Face
from datasets import load_dataset

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    max_length = 1024
    return max_length

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=1,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config



def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


model_name = "meta-llama/Llama-2-13b-hf" 
#model_name = "bigscience/T0pp"
#model_name = "mistralai/Mistral-7B-v0.1"
bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config)

seed = 42
## Preprocess dataset

max_length = get_max_length(model)

#dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
# from test_data import text_qm9_dataset
# dataset = text_qm9_dataset(tokenizer, split='train',seed=seed)
import torch_geometric
from torch_geometric.datasets import QM9
atomic_symbols = {
    1: 'H',   # Hydrogen
    2: 'He',  # Helium
    3: 'Li',  # Lithium
    4: 'Be',  # Beryllium
    5: 'B',   # Boron
    6: 'C',   # Carbon
    7: 'N',   # Nitrogen
    8: 'O',   # Oxygen
    9: 'F',   # Fluorine
    10: 'Ne'  # Neon
}
import numpy as np
import networkx as nx

def create_nx_graph(node_list, edge_list):
    G = nx.Graph()
    for i,node in enumerate(node_list):
        G.add_node(i)
    symmetrized_edge_list = list(set((ej,ei) for ei,ej in edge_list) | set(edge_list))
    G.add_edges_from(symmetrized_edge_list)
    return G

def compute_resistance_distance(G):
    L = nx.laplacian_matrix(G).todense()
    Linv = np.linalg.pinv(L, hermitian=True)
    d = np.diag(Linv)
    Omega = d[:,None]+d[None,:]-2*Linv
    return Omega

import os
from graph_types import Tokenizable, SingleToken, Decimal, Sequence, Bidirectional, Graph
from graph_types import Tuple, Text, AllPairsTuple, Causal
root = os.path.expanduser("~/datasets/QM9")
from datasets import IterableDataset
def text_qm9_dataset(tokenizer, pos=True,seed=37, split='train'):
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html#torch_geometric.datasets.QM9
    ds = QM9(root=root)
    ElemPos = Tuple(SingleToken(), Sequence(Decimal(2))) if pos else SingleToken()
    GraphElemPos = Graph(ElemPos)
    Output = Causal(Decimal(2))
    not_molecule = Tuple(Text(),Output)
    full_type = AllPairsTuple(GraphElemPos, not_molecule)
    Tokenizable.tokenizer = tokenizer

    ids = np.random.permutation(len(ds)) # 130k
    all_ids = {'train':ids[:100000], 'val':ids[100000:100000+100], 'test':ids[110000:]}
    split_ids = all_ids[split]
    def gen():
        for idx in split_ids:
            row = ds[idx]
            elem_symbols = [atomic_symbols[z.item()] for z in row.z]
            elem_pos = [(elem, pos) for elem, pos in zip(elem_symbols, row.pos)]
            inp = (elem_pos if pos else elem_symbols, row.edge_index.T)
            target = row.y[0,2] #HOMO attribute TODO: make more general
            prompt = "\n\n HOMO: "
            out2 = full_type.tokenize((inp, (prompt, target)))
            out_wstart = add_start_token(out2)
            nodes, edges = out_wstart['input_ids'], out_wstart['edges']
            R = compute_resistance_distance(create_nx_graph(nodes, edges))
            out_wstart['attention_mask'] = -R
            pred_mask = np.array(out_wstart['prediction_mask'])>0
            sq_pred_mask = pred_mask[:,None]|pred_mask[None,:]
            triu = np.triu(np.ones_like(out_wstart['attention_mask']),1)>0
            autoregressive_neginf_mask = triu&sq_pred_mask
            minus_inf = torch.finfo(torch.float16).min
            out_wstart['attention_mask'][autoregressive_neginf_mask] = minus_inf
            # use triu mask to set upper triangular to -infty
            minus_inf = torch.finfo(torch.float16).min # should this be float16?
            # slc = np.s_[-8:,-8:]
            # print(triu[slc],"\n")
            # print(sq_pred_mask[slc],"\n")
            # print(autoregressive_neginf_mask[slc],"\n")
            # print(out_wstart['attention_mask'][slc],"\n")
            # assert False
            # add the start token
            out_wstart['all-type']=full_type
            out_wstart['in-type']=AllPairsTuple(GraphElemPos, Text())
            out_wstart['out-type']=Output
            yield out_wstart

    def add_start_token(tokenizer_output):
        out2 = tokenizer_output
        out = tokenizer("test string")
        out['input_ids'] = out["input_ids"][:1]
        out['input_ids'].extend(out2['input_ids'])
        out['attention_mask'] = out['attention_mask'][:1]
        out['attention_mask'].extend(out2['attention_mask'])
        out['edges'] = [(ei+1,ej+1) for ei,ej in out2['edges']]
        out['prediction_mask'] = [0]
        out['prediction_mask'].extend(out2['prediction_mask'])
        return out
    
    ds2 = IterableDataset.from_generator(gen)
    ds2 = ds2.shuffle(seed=seed)
    # for example in ds2:
    #     print(example)
    #     break
    # assert False
    return ds2

from collections.abc import Mapping
from transformers.data.data_collator import _torch_collate_batch
from scipy.linalg import block_diag

def pad_attention_mask(attention_mask,N):
    """ assumes attention mask is a (n,n) numpy array, and embeds in a larger (N,N) array with -inf padding"""
    minus_inf = torch.finfo(torch.float16).min
    padded_mask = minus_inf*np.ones((N,N)) # check this is the right dtype
    padded_mask[-attention_mask.shape[0]:,-attention_mask.shape[1]:] = attention_mask
    return padded_mask

class MyCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        with torch.no_grad():
            #print("GOT HERE")
            # print("data is like:",examples[0])
            # Handle dict or lists with proper padding and conversion to tensor.
            # if isinstance(examples[0], Mapping):
            #     relevant_rows = [{k:example[k] for k in ['input_ids','attention_mask']} for example in examples]
            #     batch = self.tokenizer.pad(relevant_rows, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
            # else:
            #     assert False
            #     batch = {
            #         "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            #     }
            #print(examples[0])
            input_ids = [e['input_ids'] for e in examples]
            attention_mask = [e['attention_mask'] for e in examples]
            prediction_mask = [e['prediction_mask'] for e in examples]
            if len(torch.tensor(attention_mask[0]).shape) > 1:
                # all pairs attention mask  
                relevant_rows = [{k:example[k] for k in ['input_ids']} for example in examples]
                batch = self.tokenizer.pad(relevant_rows, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
                N = batch['input_ids'].shape[1]
                minus_inf = torch.finfo(torch.float16).min # should this be float16?
                valid_masks = []
                #padded_attention_masks = [block_diag(minus_inf*np.ones((N-mask.shape[0],N-mask.shape[0])),mask) for mask in attention_mask]
                padded_attention_masks = [pad_attention_mask(mask,N) for mask in attention_mask]
                joined_attention_mask = torch.tensor(np.stack(padded_attention_masks,axis=0))
                batch['attention_mask'] = joined_attention_mask #(B, Queries, Keys)
            else:
                relevant_rows = [{k:example[k] for k in ['input_ids','attention_mask']} for example in examples]
                batch = self.tokenizer.pad(relevant_rows, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
            # batch = {
            #         "input_ids": _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of),
            #         "attention_mask": _torch_collate_batch(attention_mask, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            #     }
            prediction_mask = _torch_collate_batch(prediction_mask, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            assert not self.mlm
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            labels[(prediction_mask!=1)] = -100
            batch["labels"] = labels.clone()
            batch["attention_mask"] = batch["attention_mask"].clone()
            #print(batch)
            #print("shape", batch["labels"].shape)
            return batch

from copy import deepcopy
from transformers import TrainerCallback
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        #if control.should_evaluate:
        control_copy = deepcopy(control) #TODO switch to test dataset
        with torch.no_grad():
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset, metric_key_prefix="test@")

        return control_copy

# def compute_metrics(pred):
#     global num_labels
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     acc = accuracy_score(labels, preds)
#     loss_fct = CrossEntropyLoss()
#     logits = torch.tensor(pred.predictions)
#     labels = torch.tensor(labels)
#     loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
#     return {
#         'accuracy@'+lang: acc,
#         'f1@'+lang: f1,
#         'precision@'+lang: precision,
#         'recall@'+lang: recall,
#         'loss@'+lang: loss,
#     }

class MyTrainer(Seq2SeqTrainer):
    pass
    # def compute_loss(self,model,inputs,return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     Subclass and override for custom behavior.
    #     """
    #     #loss_fn = CrossEntropyLoss(ignore_index=-100)
    #     # add all not in prediction mask to ignored
    #     inputs['labels'][~inputs['prediction_mask']] = -100
    #     outputs = model(**inputs)
    #     print(inputs)
    #     print(outputs)
    #     breakpoint()
    #     return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # print(preds,labels)
    # assert False
    # if isinstance(preds, tuple):
    #     preds = preds[0]
    # # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds[0], skip_special_tokens=True)
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(decoded_preds)
    assert False

    # result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # result = {k: round(v * 100, 4) for k, v in result.items()}
    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    # return result

def train(model, tokenizer, dataset, output_dir, test_dataset):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
    # Training parameters
    trainer = MyTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            eval_accumulation_steps=4,
            warmup_steps=20,
            max_steps=3,#00,
            learning_rate=5e-5,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            generation_max_length=5, #TODO update this
        ),
        compute_metrics=compute_metrics,
        data_collator=MyCollator(tokenizer, mlm=False),
    )
    trainer.add_callback(CustomCallback(trainer)) # add the metric evaluation callback
    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
    
    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
     
    do_train = True
    
    # Launch training
    print("Training...")
    
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)    
    
    ###
    
    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


# def evaluate(model, test_dataset):
#     model.generate(**model_inputs, max_new_tokens=40)

from typing import Optional, List
import torch
from torch import nn
from llama_modules import attention_forward

def replace_decoder_forward(model):
    state = [None]
    old_forward = model.forward
    
    def new_model_forward(
                self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs,
            ):
        state[0] = attention_mask[:,None] # (B, S, S) save to external state
        reduced_attention_mask = attention_mask[:,0] # (B, S) but should never be used
        #print("inputting reduced attention mask")
        return old_forward(input_ids, reduced_attention_mask, **kwargs)
    
    # replace the forward method of the model to store the full attention mask
    model.forward = new_model_forward.__get__(model, model.__class__)
            

    # def decoder_layer_forward(self, 
    #         hidden_states: torch.Tensor,
    #         attention_mask: Optional[torch.Tensor],
    #         *args,
    #         **kwargs,
    #     ):
    #     our_attention_mask = state[0] # (B, S, S) load from external state
    #     our_attention_mask = our_attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    #     return old_decoder_layer_forward(hidden_states, our_attention_mask, *args, **kwargs)
    
    def new_fn(self, hidden_states, attention_mask, *args, **kwargs):
        return attention_forward(self,hidden_states, state[0], *args, **kwargs)
    # iterate through the modules and replace decoder forward
    for name, module in model.named_modules():
        if "LlamaAttention" in module.__class__.__name__:
            print("Removed ROPE and added graph distance from", name)
            #old_decoder_layer_forward = module.forward
            module.forward = new_fn.__get__(module, module.__class__)
        # else:
        #     print("Not replacing", name, module.__class__.__name__)

# class LlamaNoPERotaryEmbedding(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, seq_len=None):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         return 1., 0.

replace_decoder_forward(model)

output_dir = "results/llama2/final_checkpoint"
dataset = text_qm9_dataset(tokenizer, pos=False)
test_dataset = text_qm9_dataset(tokenizer, pos=False, split='val')
train(model, tokenizer, dataset, output_dir, test_dataset)