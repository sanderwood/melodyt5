import os
import time
import json
import torch
import random
import numpy as np
from copy import deepcopy
from utils import *
from config import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, get_constant_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Set up distributed training
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl') if world_size > 1 else None
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
seed = 0 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = BATCH_SIZE
patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                    max_length=PATCH_LENGTH, 
                    max_position_embeddings=PATCH_LENGTH,
                    vocab_size=1)
char_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, 
                    max_length=PATCH_SIZE, 
                    max_position_embeddings=PATCH_SIZE,
                    vocab_size=128)

model = MelodyT5(patch_config, char_config)
model = model.to(device)
# print parameter number
print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

scaler = GradScaler()
is_autocast = True
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
def collate_batch(batch):
    input_patches, input_masks, output_patches, output_masks = [], [], [], []

    for input_patch, output_patch in batch:
        input_patches.append(input_patch)
        input_masks.append(torch.tensor([1]*input_patch.shape[0]))
        output_patches.append(output_patch)
        output_masks.append(torch.tensor([1]*output_patch.shape[0]))

    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=0)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)
    output_patches = torch.nn.utils.rnn.pad_sequence(output_patches, batch_first=True, padding_value=0)
    output_masks = torch.nn.utils.rnn.pad_sequence(output_masks, batch_first=True, padding_value=0)

    return input_patches.to(device), input_masks.to(device), output_patches.to(device), output_masks.to(device)

def split_data(data, eval_ratio=0.1):
    random.shuffle(data)
    split_idx = int(len(data)*eval_ratio)
    eval_set = data[:split_idx]
    train_set = data[split_idx:]
    return train_set, eval_set

class MelodyHubDataset(Dataset):
    def __init__(self, items):
        self.inputs = []
        self.outputs = []
        
        for item in tqdm(items):
            input_patch =  patchilizer.encode(item['input'], add_special_patches=True)
            input_patch = torch.tensor(input_patch)

            output_patch =  patchilizer.encode(item["output"], add_special_patches=True)
            output_patch = torch.tensor(output_patch)
            if torch.sum(output_patch)!=0:
                self.inputs.append(input_patch)
                self.outputs.append(output_patch)
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# call model with a batch of input
def process_one_batch(batch):
    input_patches, input_masks, output_patches, output_masks = batch
    
    loss = model(input_patches,
                input_masks,
                output_patches,
                output_masks)
    
    # Reduce the loss on GPU 0
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss

# do one epoch for training
def train_epoch():
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()

    for batch in tqdm_train_set:
        with autocast():
            loss = process_one_batch(batch)
        if loss is None or torch.isnan(loss).item():
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        total_train_loss += loss.item()
        tqdm_train_set.set_postfix({str(global_rank)+'_train_loss': total_train_loss / iter_idx})
        iter_idx += 1
        
    return total_train_loss / (iter_idx-1)

# do one epoch for eval
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set:
        with torch.no_grad():
            loss = process_one_batch(batch)
        if loss is None or torch.isnan(loss).item():
            continue
        total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1
    return total_eval_loss / (iter_idx-1)

# train and eval
if __name__ == "__main__":
    
    train_set = []
    eval_set = []
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            train_set.append(data)
    with open(VALIDATION_DATA_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            eval_set.append(data)

    train_batch_nums = int(len(train_set) / batch_size)
    eval_batch_nums = int(len(eval_set) / batch_size)

    random.shuffle(train_set)
    random.shuffle(eval_set)

    train_set = train_set[:train_batch_nums*batch_size]
    eval_set = eval_set[:eval_batch_nums*batch_size]

    train_set = MelodyHubDataset(train_set)
    eval_set = MelodyHubDataset(eval_set)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=local_rank)

    train_set = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_batch, sampler=train_sampler, shuffle = (train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=batch_size, collate_fn=collate_batch, sampler=eval_sampler, shuffle = (train_sampler is None))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 1000)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LOAD_FROM_PRETRAINED and os.path.exists(PRETRAINED_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
            
        print(f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Loss {checkpoint['min_eval_loss']}")
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')

    if LOAD_FROM_CHECKPOINT and os.path.exists(WEIGHTS_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        pre_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        min_eval_loss = checkpoint['min_eval_loss']
        print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
        checkpoint = None
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')

    for epoch in range(1+pre_epoch, NUM_EPOCHS+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_loss = train_epoch()
        eval_loss = eval_epoch()
        if global_rank==0:
            with open(LOGS_PATH,'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " +str(eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            if eval_loss < min_eval_loss:
                best_epoch = epoch
                min_eval_loss = eval_loss
                checkpoint = { 
                                'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sched': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                'min_eval_loss': min_eval_loss
                                }
                torch.save(checkpoint, WEIGHTS_PATH)
        
        if world_size > 1:
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Min Eval Loss : "+str(min_eval_loss))

