import time
import torch
from utils import *
import re
from config import *
from transformers import  GPT2Config
import argparse

def get_args(parser):

    parser.add_argument('-num_tunes', type=int, default=3, help='the number of independently computed returned tunes')
    parser.add_argument('-max_patch', type=int, default=128, help='integer to define the maximum length in tokens of each tune')
    parser.add_argument('-top_p', type=float, default=0.8, help='float to define the tokens that are within the sample operation of text generation')
    parser.add_argument('-top_k', type=int, default=8, help='integer to define the tokens that are within the sample operation of text generation')
    parser.add_argument('-temperature', type=float, default=2.6, help='the temperature of the sampling operation')
    parser.add_argument('-seed', type=int, default=None, help='seed for randomstate')
    parser.add_argument('-show_control_code', type=bool, default=True, help='whether to show control code')
    args = parser.parse_args()

    return args

def generate_abc(args):

    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    # print parameter number
    print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    with open('prompt.txt', 'r') as f:
        prompt = f.read()

    if "%%output\n" not in prompt:
        input_abc = prompt.replace("%%input\n", "")
        prompt = ""
    else:
        input_abc = prompt.replace("%%input\n", "").split("%%output\n")[0]
        prompt = prompt.split("%%output\n")[1]
    task = input_abc.split("\n")[0][2:]

    tunes = ""
    num_tunes = args.num_tunes
    max_patch = args.max_patch
    top_p = args.top_p
    top_k = args.top_k
    temperature = args.temperature
    seed = args.seed
    show_control_code = args.show_control_code

    print(" HYPERPARAMETERS ".center(60, "#"), '\n')
    args = vars(args)
    for key in args.keys():
        print(key+': '+str(args[key]))
    print("\n"+" OUTPUT TUNES ".center(60, "#"))

    start_time = time.time()

    for i in range(num_tunes):
        if prompt=="":
            tune = "X:"+str(i+1) + "\n"
        else:
            tune = "X:"+str(i+1) + "\n" + prompt.strip() + "\n"
        lines = re.split(r'(\n)', tune)
        tune = ""
        skip = False
        for line in lines:
            if show_control_code or line[:2] not in ["S:", "B:", "E:"]:
                if not skip:
                    print(line, end="")
                    tune += line
                skip = False
            else:
                skip = True

        patches = torch.tensor([patchilizer.encode(input_abc, add_special_patches=True)], device=device)
        decoder_patches = torch.tensor([patchilizer.encode(prompt, add_special_patches=True)[:-1]], device=device)
        tokens = None

        while decoder_patches.shape[1]<max_patch:
            predicted_patch, seed = model.generate(patches,
                                                    decoder_patches,
                                                    tokens,
                                                    task=task,
                                                    top_p=top_p,
                                                    top_k=top_k,
                                                    temperature=temperature,
                                                    seed=seed)
            tokens = None
            if predicted_patch[0]!=patchilizer.eos_token_id:
                next_bar = patchilizer.decode([predicted_patch])
                if show_control_code or next_bar[:2] not in ["S:", "B:", "E:"]:
                    print(next_bar, end="")
                    tune += next_bar
                if next_bar=="":
                    break
                
                predicted_patch = torch.tensor(patchilizer.bar2patch(next_bar), device=device).unsqueeze(0)
                decoder_patches = torch.cat([decoder_patches, predicted_patch.unsqueeze(0)], dim=1)
            else:
                break

        tunes += tune+"\n\n"
        print("\n")

    print("Generation time: {:.2f} seconds".format(time.time()-start_time))
    timestamp = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime()) 
    with open('output_tunes/'+timestamp+'.abc', 'w') as f:
        f.write(tunes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    generate_abc(args)