import re
import torch
import random
from config import *
from unidecode import unidecode
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from transformers import GPT2LMHeadModel, PreTrainedModel, EncoderDecoderConfig, EncoderDecoderModel

class Patchilizer:
    """
    A class for converting music bars to patches and vice versa. 
    """
    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def split_bars(self, body):
        """
        Split a body of music into individual bars.
        """
        bars = re.split(self.regexPattern, ''.join(body))
        bars = list(filter(None, bars))  # remove empty strings
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        return bars
    
    def bar2patch(self, bar, patch_size=PATCH_SIZE):
        """
        Convert a bar into a patch of specified length.
        """
        patch = [self.bos_token_id] + [ord(c) for c in bar] + [self.eos_token_id]
        patch = patch[:patch_size]
        patch += [self.pad_token_id] * (patch_size - len(patch))
        return patch
    
    def patch2bar(self, patch):
        """
        Convert a patch into a bar.
        """
        return ''.join(chr(idx) if idx > self.eos_token_id else '' for idx in patch if idx != self.eos_token_id)

    def encode(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=False):
        """
        Encode music into patches of specified length.
        """
        lines = unidecode(abc_code).split('\n')
        lines = list(filter(None, lines))  # remove empty lines

        body = ""
        patches = []

        for line in lines:
            if len(line) > 1 and ((line[0].isalpha() and line[1] == ':') or line.startswith('%%')):
                if body:
                    bars = self.split_bars(body)
                    patches.extend(self.bar2patch(bar + '\n' if idx == len(bars) - 1 else bar, patch_size) 
                                   for idx, bar in enumerate(bars))
                    body = ""
                patches.append(self.bar2patch(line + '\n', patch_size))
            else:
                body += line + '\n'

        if body:
            patches.extend(self.bar2patch(bar, patch_size) for bar in self.split_bars(body))

        if add_special_patches:
            bos_patch = [self.bos_token_id] * (patch_size-1) + [self.eos_token_id]
            eos_patch = [self.bos_token_id] + [self.eos_token_id] * (patch_size-1)
            patches = [bos_patch] + patches + [eos_patch]

        return patches[:patch_length]

    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2bar(patch) for patch in patches)

class PatchLevelEnDecoder(PreTrainedModel):
    """
    An Patch-level Decoder model for generating patch features in an auto-regressive manner. 
    It inherits PreTrainedModel from transformers.
    """

    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        if SHARE_WEIGHTS:
            try:
                self.base = EncoderDecoderModel.from_encoder_decoder_pretrained("random_model", "random_model", tie_encoder_decoder=True)
            except Exception as e:
                print("Error loading 'random_model':", e)
                print("Please run 'random_model.py' to create randomly initialized weights.")
                raise
        else:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)
            self.config = config
            self.base = EncoderDecoderModel(config=self.config)

        self.base.config.pad_token_id = 0
        self.base.config.decoder_start_token_id = 1

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor,
                decoder_patches: torch.Tensor,
                decoder_masks: torch.Tensor):
        """
        The forward pass of the patch-level decoder model.
        :param patches: the patches to be encoded
        :return: the encoded patches
        """
        patches = torch.nn.functional.one_hot(patches, num_classes=128).float()
        patches = patches.reshape(len(patches), -1, PATCH_SIZE * 128)
        patches = self.patch_embedding(patches.to(self.device))

        decoder_patches = torch.nn.functional.one_hot(decoder_patches, num_classes=128).float()
        decoder_patches = decoder_patches.reshape(len(decoder_patches), -1, PATCH_SIZE * 128)
        decoder_patches = self.patch_embedding(decoder_patches.to(self.device))

        if masks==None or decoder_masks==None:
            return self.base(inputs_embeds=patches,
                                decoder_inputs_embeds=decoder_patches,
                                output_hidden_states = True)["decoder_hidden_states"][-1]
        else:
            return self.base(inputs_embeds=patches,
                            attention_mask=masks,
                            decoder_inputs_embeds=decoder_patches,
                            decoder_attention_mask=decoder_masks,
                            output_hidden_states = True)["decoder_hidden_states"][-1]

class CharLevelDecoder(PreTrainedModel):
    """
    A Char-level Decoder model for generating the characters within each bar patch sequentially. 
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        self.base = GPT2LMHeadModel(config)

    def forward(self, encoded_patches: torch.Tensor, target_patches: torch.Tensor):
        """
        The forward pass of the char-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the decoded patches
        """
        # preparing the labels for model training
        target_masks = target_patches == self.pad_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)

        # masking the labels for model training
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        # select patches
        if PATCH_SAMPLING_BATCH_SIZE!=0 and PATCH_SAMPLING_BATCH_SIZE<target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:PATCH_SAMPLING_BATCH_SIZE])

            target_patches = target_patches[selected_indices,:]
            target_masks = target_masks[selected_indices,:]
            encoded_patches = encoded_patches[selected_indices,:]
            labels = labels[selected_indices,:]

        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:,1:,:]), dim=1)

        return self.base(inputs_embeds=inputs_embeds,
                         attention_mask=target_masks,
                         labels=labels)

    def generate(self, encoded_patch: torch.Tensor, tokens: torch.Tensor):
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:,1:,:]), dim=1)
        
        # Get output from model
        outputs = self.base(inputs_embeds=tokens)
        
        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs

class MelodyT5(PreTrainedModel):
    """
    MelodyT5 is a hierarchical music generation model based on bar patching. 
    It includes a patch-level decoder and a character-level decoder.
    It inherits PreTrainedModel from transformers.
    """

    def __init__(self, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        self.patch_level_decoder = PatchLevelEnDecoder(encoder_config)
        self.char_level_decoder = CharLevelDecoder(decoder_config)

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor,
                decoder_patches: torch.Tensor,
                decoder_masks: torch.Tensor):
        """
        The forward pass of the MelodyT5 model.
        :param patches: the patches to be both encoded and decoded
        :return: the decoded patches
        """
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        decoder_patches = decoder_patches.reshape(len(decoder_patches), -1, PATCH_SIZE)

        encoded_patches = self.patch_level_decoder(patches,
                                                   masks,
                                                   decoder_patches,
                                                   decoder_masks)
        
        left_shift_masks = decoder_masks * (decoder_masks.flip(1).cumsum(1).flip(1) > 1)
        decoder_masks[:, 0] = 0
        
        encoded_patches = encoded_patches[left_shift_masks == 1]
        decoder_patches = decoder_patches[decoder_masks == 1]

        return self.char_level_decoder(encoded_patches,
                                       decoder_patches)["loss"]

    def generate(self, 
                patches: torch.Tensor,
                decoder_patches: torch.Tensor,
                tokens: torch.Tensor,
                task: str,
                top_p: float=1,
                top_k: int=0,
                temperature: float=1,
                seed: int=None):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :return: the generated patches
        """
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        decoder_patches = decoder_patches.reshape(len(decoder_patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches=patches, 
                                                    masks=None,
                                                    decoder_patches=decoder_patches,
                                                    decoder_masks=None)
        if tokens==None:
            tokens = torch.tensor([self.bos_token_id], device=self.device)
        generated_patch = []
        random.seed(seed)

        if task in ["harmonization", "segmentation"] and decoder_patches.shape[1] > 1:
            if task == "harmonization":
                special_token = ord('"')
            else:
                special_token = ord('!')
            copy_flag = True
            reference_patch = patches[0][decoder_patches.shape[1]]
            reference_idx = tokens.shape[0]

        while True:
            if seed!=None:
                n_seed = random.randint(0, 1000000)
                random.seed(n_seed)
            else:
                n_seed = None
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            token = temperature_sampling(prob, temperature=temperature, seed=n_seed)
            
            if token == self.eos_token_id or len(tokens) >= PATCH_SIZE - 1:
                generated_patch.append(token)
                break
            else:
                if task in ["harmonization", "segmentation"] and decoder_patches.shape[1] > 1:
                    reference_token = reference_patch[reference_idx].item()
                    reference_idx += 1
                    n_special_tokens = sum([1 for t in tokens if t == special_token])

                    if token == special_token and token != reference_token:
                        reference_idx -= 1
                        if n_special_tokens % 2 == 0:
                            copy_flag = False
                        else:
                            copy_flag = True

                    if token != special_token:
                        if copy_flag:
                            token = reference_token
                        else:
                            reference_idx -= 1
                        
                generated_patch.append(token)
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch, n_seed
