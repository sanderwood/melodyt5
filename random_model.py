from utils import *
from config import *
from transformers import GPT2Config, GPT2Model

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                    max_length=PATCH_LENGTH, 
                    max_position_embeddings=PATCH_LENGTH,
                    vocab_size=1)

gpt = GPT2Model(patch_config)
gpt.save_pretrained("random_model")