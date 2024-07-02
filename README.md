# MelodyT5: A Unified Score-to-Score Transformer for Symbolic Music Processing [ISMIR 2024]
This repository contains the code for the MelodyT5 model as described in the paper [MelodyT5: A Unified Score-to-Score Transformer for Symbolic Music Processing](https://arxiv.org/abs/2402.19155).

MelodyT5 is an unified framework for symbolic music processing, using an encoder-decoder architecture to handle multiple melody-centric tasks, such as generation, harmonization, and segmentation, by treating them as score-to-score transformations. Pre-trained on [MelodyHub](https://huggingface.co/datasets/sander-wood/melodyhub), a large dataset of melodies in ABC notation, it demonstrates the effectiveness of multi-task transfer learning in symbolic music processing.

## Model Description
In the domain of symbolic music research, the progress of developing scalable systems has been notably hindered by the scarcity of available training data and the demand for models tailored to specific tasks. To address these issues, we propose MelodyT5, a novel unified framework that leverages an encoder-decoder architecture tailored for symbolic music processing in ABC notation. This framework challenges the conventional task-specific approach, considering various symbolic music tasks as score-to-score transformations. Consequently, it integrates seven melody-centric tasks, from generation to harmonization and segmentation, within a single model. Pre-trained on MelodyHub, a newly curated collection featuring over 261K unique melodies encoded in ABC notation and encompassing more than one million task instances, MelodyT5 demonstrates superior performance in symbolic music processing via multi-task transfer learning. Our findings highlight the efficacy of multi-task transfer learning in symbolic music processing, particularly for data-scarce tasks, challenging the prevailing task-specific paradigms and offering a comprehensive dataset and framework for future explorations in this domain.

We provide the weights of MelodyT5 on [Hugging Face](https://huggingface.co/sander-wood/melodyt5/blob/main/weights.pth), which are based on pre-training with over one million task instances encompassing seven melody-centric tasks. This extensive pre-training allows MelodyT5 to excel in symbolic music processing scenarios, even when data is limited.

## Installation

To set up the MelodyT5 environment and install the necessary dependencies, follow these steps:

1. **Create and Activate Conda Environment**

   ```bash
   conda create --name melodyt5 python=3.7.9
   conda activate melodyt5
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Pytorch**

   ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```
4. **Download Pre-trained MelodyT5 Weights (Optional)**
   
   For those interested in starting with pre-trained models, MelodyT5 weights are available on [Hugging Face](https://huggingface.co/sander-wood/melodyt5/blob/main/weights.pth). This step is optional but recommended for users looking to leverage the model's capabilities without training from scratch.

## Usage

- `config.py`: Configuration settings for training and inference.
- `generate.py`: Perform inference tasks (e.g., generation and conversion) using pre-trained models.
- `train-cls.py`: Training script for classification models.
- `train-gen.py`: Training script for generative models.
- `utils.py`: Utility functions supporting model operations and data processing.
  
### Setting Up Inference Parameters

Before running the inference script, you can configure the following parameters in `config.py` or directly via command-line arguments:

- `-num_tunes`: Number of independently computed returned tunes (default: 3)
- `-max_patch`: Maximum length in tokens of each tune (default: 128)
- `-top_p`: Tokens within the sample operation of text generation (default: 0.8)
- `-top_k`: Tokens within the sample operation of text generation (default: 8)
- `-temperature`: Temperature of the sampling operation (default: 2.6)
- `-seed`: Seed for random state (default: None)
- `-show_control_code`: Whether to show control codes (default: True)

These parameters control how the model generates melodies based on the input provided in `prompt.txt`.

### Running Inference

To perform inference tasks using MelodyT5, follow these steps:

1. **Prepare Your Prompt**
   - Edit `prompt.txt` to specify the task and input for the model. Each line in `prompt.txt` should contain a single prompt.

2. **Execute Inference**
   - Run the following command to execute the inference script:
     ```bash
     python inference.py -num_tunes 3 -max_patch 128 -top_p 0.8 -top_k 8 -temperature 2.6 -seed <seed_value> -show_control_code True
     ```
     Replace `<seed_value>` with your chosen seed value or leave it as `None` for a random seed.

3. **Interpreting the Output**
   - The script will generate melodies based on the prompts specified in `prompt.txt` using the configured parameters.

## How to Use
Follow these steps to effectively utilize MelodyT5 for symbolic music processing:

1. Prepare Your Data
   Ensure your dataset follows the format and style of MelodyHub, which uses ABC notation for uniform representation of melodies. If not using MelodyHub data, adapt your dataset to match this style.

2. Configure Your Model
   Adjust model hyperparameters, training parameters, and file paths in the config.py file.

3. Train the Model
   Run the train.py script to train MelodyT5. Use the following command, adjusting for your specific setup:
   
   ```
   python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py
   ```
   This command utilizes distributed training across multiple GPUs (modify --nproc_per_node as needed).

4. Run Inference
   To perform inference tasks such as melody generation or harmonization, execute `inference.py`. The script reads prompts from `prompt.txt` to specify the task and input for the model. Customize prompts in `prompt.txt` to define different tasks and inputs for MelodyT5. Refer to the examples below for guidance on setting up prompts.

   Ensure the encoder input is complete, while the output (decoder input) is optional. If you need the model to continue a given output, use `%%input` and `%%output` to mark the beginning of each section. Additionally, the output must not contain incomplete bars. Here is an example prompt:

   ```
   %%input
   %%variation
   L:1/8
   M:6/8
   K:D
   |: AFD DFA | Add B2 A | ABA F3 | GFG EFG | AFD DFA | Add B2 A | ABA F2 E | FDD D3 :: fdd ede |
   fdd d2 g | fdd def | gfg e2 g | fed B2 A | AdF A3 | ABA F2 E | FDD D3 :|
   %%output
   E:8
   L:1/8
   M:6/8
   K:D
   |: B |
   ```

## Inference Examples
Below are the MelodyT5 results on seven MelodyHub tasks, using random samples from the validation set. Three independent outputs were generated without cherry-picking. Each `X:0` output corresponds to the original input for that task and is not generated by the model, while `X:1`, `X:2`, and `X:3` are generated outputs.

To view the musical scores and listen to the tunes, you can use online ABC notation platforms such as [Online ABC Player](https://abc.rectanglered.com/) or the [ABC Sheet Music Editor - EasyABC](https://easyabc.sourceforge.net/). Simply copy and paste the ABC notation into these tools to see the sheet music and hear the audio playback.

1. Cataloging
```
%%input
%%cataloging
T:
O:
Y:
L:1/4
Q:1/4=120
M:3/4
K:C
A/0 A/0 B/0 (c/0 B/0) A/0 A/0 ^G/0 A/0 (e/0 f/0) (=g/0 f/0) e/0 | (f e) d | e2 e | (d c) B |
(c B) A | ^G2 e | f2 e | d2 c | B2 A | ^G2 c | (d e) d | (e f) g | e2 e | (e c) c | (d B) B |
(c A) A | ^G2 (A/B/) | (c d) c | c2 B | c2 (c/d/) | (e f) e | e2 d | e2 (e/f/) | g2 g | (g e) e |
f2 e | d2 e | (e d) e | (c d) e | (d e) (c/d/) | B2 c | B2 A | ^G2 A | A2 c | B2 A | ^G2 A |
A2 x |]
```
```
%%output
X:0
T:HET GODEN BANKET. Ter Huwfeest van Muzika en Poezy
O:Netherlands
Y:vocal

X:1
T:WILHELLTOF GERSCHANDS
O:Netherlands
Y:vocal


X:2
T:MIEN IS LOVEN, VRIENDAAT OP EEN
O:Netherlands
Y:vocal


X:3
T:Het lied en tijd over
O:Netherlands
Y:vocal
```
2. Generation
```
%%input
%%generation
```
```
%%output
X:0
S:2
B:8
E:5
B:8
L:1/8
M:2/2
K:G
|: A3 c BG G2 | BGBd g2 fg | eA A2 BGBd | egdc BG G2 | A3 c BG G2 | BGBd g3 a | bgag egfa |
gedc BG G2 :: bg g2 agef | g3 e dega | bg g2 aged | eaag a2 ga | bg g2 agef | g3 e dega |
bgag egfa | gedc BG G2 :|

X:1
S:2
B:5
E:5
B:9
L:1/8
M:2/2
K:Amin
cd | e3 c d2 cd | e3 d c2 Ac | B3 c A3 A | A4 :: G2 | c3 c c3 c | d3 d e2 fe | f2 f2 e2 d2 | 
g3 a g2 cd | e3 d c2 dc | e3 d c2 Ac | B3 c A3 A | A4 :|

X:2
S:3
B:1
E:0
B:17
E:0
E:5
B:17
L:1/8
M:3/4
K:G
D |:"G" G2 GABc |"G/B" d2 B3 G |"C" E3 G E2 |"G" D3 DEF |"G" G2 GABc |"G/B" d2 B3 G | 
"D" E2 G2 B2 |"D" A4 DD |"G" G2 GABc |"G/B" d2 B3 G |"C" E3 G E2 |"G" D3 DEF |"G" G2 GABc | 
"G/B" d2 B3 G |"D" E2 G2 F2 |1"G" G4 GD :|2"G" G4 DG |:"C" E G3 E2 |"G" D3 E D2 |"G" B3 A G2 | 
"G" B d3 B2 |"C" e c3 e2 |"G" d3 B G2 |"Am" E2 G2 B2 |"D" A4 DG |"C" E G3 E2 |"G" D3 E D2 | 
"G" B3 A G2 |"G" B d3 B2 |"C" e c3 e2 |"G" d3 B G2 |"D" E2 G2 F2 |1"G" G4 DG :|2"G" G4 z2 |]

X:3
S:3
B:9
E:5
B:9
E:5
E:5
B:9
L:1/8
M:4/4
K:D
|: A2 | d2 d2 c2 c2 | B2 A2 F2 A2 | B2 B2 c2 c2 | d2 d2 d2 c2 | d2 d2 c2 c2 | B2 A2 F2 A2 | 
B2 B2 c2 c2 | d6 ::[K:D] A2 | d2 d2 cB c2 | Bc BA G2 FG | A3 B AG FE | D2 DE FG AB | d2 d2 cB c2 | 
Bc BA G2 FG | A3 B AG FE | D6 :: A2 | F2 DF A2 FA | G2 EC E2 A2 | F2 DF A2 FA | G2 EC A,2 A2 | 
F2 DF A2 FA | G2 EC E2 A2 | F2 DF A2 AF | G2 EC D2 :|
```
3. Harmonization
```
%%input
%%harmonization
L:1/4
M:4/4
K:B
 B, | F D/C/ B, F | G3/4A/8B/8 G !fermata!F F |
 G A B A | G/B/A/G/ !fermata!F D |
 G F E D | C2 !fermata!B, :| z | F2 !fermata!D2 |
 F2 !fermata!D2 | D D C C | D D C D |
 E D C2 | !fermata!B,2 B A | G F E D |
 C2 !fermata!B, |]
```
```
%%output
X:0
E:5
L:1/4
M:4/4
K:B
"B" B, |"F#/A#" F"B" D/C/"G#m" B,"D#m" F |"G#m7/B" G3/4A/8B/8"C#" G"F#" !fermata!F"B" F |
"E" G"A#dim/D#" A"B/D#" B"F#" A |"G#m7/B" G/B/"A#m/C#"A/G/"F#" !fermata!F"B" D |
"E" G"B/D#" F"C#m7" E"B" D |"F#sus4" C2"B" !fermata!B, :| z |"F#/A#" F2"B" !fermata!D2 |
"F#" F2"B" !fermata!D2 |"B" D"B/D#" D"F#" C"F#" C |"B/D#" D"B/D#" D"F#" C"B#dim/E" D |
"C#m" E"G#m" D"F#7/E" C2 |"B" !fermata!B,2"G#m" B"D#" A |"E" G"D#m/F#" F"Emaj7/G#" E"B" D |
"F#7" C2"B" !fermata!B, |]

X:1
E:5
L:1/4
M:4/4
K:B
"B" B, |"F#/A#" F"B" D/C/"G#m" B,"B/D#" F |"G#m7/B" G3/4A/8B/8"C#" G"F#" !fermata!F"B" F |
 G"F#/A#" A"B/D#" B"F#" A |"G#m7/B" G/B/"A#m/C#"A/G/"F#" !fermata!F"B" D |
 G"B/D#" F"C#m" E"D#m7b5/C#" D"A" |"F#7" C2"B" !fermata!B, :|"F#" z |"F#/A#" F2"B" !fermata!D2 |
"B" F2"B" !fermata!D2 |"B" D"B/D#" D"F#" C"F#" C |"B/D#" D"B/D#" D"F#" C"B#dim/E" D |
 E"G#m" D"F#7/E" C"G#m"2 |"G#m" !fermata!B,2"G#m" B"A#" A |"G#m" G"D#m/F#" F"Emaj7/G#" E"B" D |
"F#7" C2"B" !fermata!B, |]

X:2
E:5
L:1/4
M:4/4
K:B
"B" B, |"F#/A#" F"B" D/C/"G#m" B,"D#m" F |"G#m7/B" G3/4A/8B/8"C#" G"F#" !fermata!F"B" F |
 G"F#" A"B/D#" B"F#" A"B" |"G#m/B" G/B/"A#dim/C#"A/G/"D#" !fermata!F"G#m" D |
"E" G"E/G#" F"A#m7b5/G#" E"B/F#" D |"F#7" C2"B" !fermata!B, :|"B" z |"D#m" F2"G#m" !fermata!D2 |
"D#m" F2"G#m" !fermata!D2 | D"G#m" D"F#/A#" C"C#m" C | D"B/D#" D"B/D#" C"B" D"F#sus4" |
"C#m" E"G#m" D"F#" C2 | !fermata!B,2"B" B"F#" A |"E" G"F#/A#" F"C#m" E"D#m" D |
"F#7" C2"B" !fermata!B, |]

X:3
E:5
L:1/4
M:4/4
K:B
"B" B, |"F#/A#" F"B" D/C/"G#m" B,"B/D#" F |"G#m7/B" G3/4A/8B/8"C#" G"F#" !fermata!F"B" F |
 G"F#" A"B/D#" B"F#" A"B" |"G#m/B" G/B/"A#dim/C#"A/G/"D#" !fermata!F"B" D |
 G"E/G#" F"C#m7" E"B" D"F#" |"F#sus4" C2"B" !fermata!B, :| z |"F#/A#" F2"B" !fermata!D2 |
"F#" F2"B" !fermata!D2 | D"B" D"B/D#" C"B/C#" C | D"B/D#" D"B/D#" C"B" D |
 E"C#m" D"F#sus4" C"D#7"2 |"G#m" !fermata!B,2"G#m" B"A#" A |"E" G"D#m/F#" F"E/G#" E"B" D |
"F#7" C2"B" !fermata!B, |]
```
4. Melodization
```
%%input
%%melodization
L:1/8
M:6/8
K:G
|: z |"G" z6 | z6 |"Am" z6 |"C" z3"D7" z3 |"G" z6 | z6 |"Am" z3"D7" z3 |"G" z4 z :: z |"C" z6 |
z6 |"Bm" z6 |"Em" z6 |"C" z3"D7" z3 |"Em" z3"Am" z3 |"D7" z6 |"G" z4 :|
```
```
%%input
X:0
E:5
L:1/8
M:6/8
K:G
|: B/A/ |"G" GDE G2 A | Bgf gdB |"Am" ABc BGA |"C" BcA"D7" BGE |"G" GDE G2 A | Bgf gdB |
"Am" ABc"D7" BcA |"G" BGG G2 :: B/d/ |"C" e2 e e2 e | egf edB |"Bm" d2 d d2 d |"Em" dge dBG |
"C" c2 d"D7" e2 f |"Em" gdB"Am" A2 d |"D7" BGA BcA |"G" BGG G :|

X:1
E:5
L:1/8
M:6/8
K:G
|: d |"G" GBG GBG | BGG G2 B |"Am" cec ABc |"C" ecA"D7" A2 c |"G" BGG BGG | BGB Bcd |"Am" edc"D7" BcA | 
"G" BGG G2 :: d |"G" gfg GBd | gfg bag |"Bm" afd Adf |"Bm" afd def |"C" gfg"G" Bcd | 
"Em" gdB"Am" cde |"D7" dcB AGF |"G" BGG G2 :|

X:2
E:5
L:1/8
M:6/8
K:G
|: d/c/ |"G" BAB G2 E | D2 D DEG |"Am" ABA AGE |"C" cBc"D7" Adc |"G" BAB G2 E | D2 D DEG | 
"Am" ABA"D7" AGA |"G" BGG G2 :: B/c/ |"G" d2 d dBG | Bdd d2 B |"Am" c2 c cAA |"Em" B2 B B2 d | 
"C" e2 e"D7" dBA |"Em" B2 d"Am" dBA |"D7" GAB AGA |"G" BGG G2 :|

X:3
E:5
L:1/8
M:6/8
K:G
|: d/c/ |"G" BGG DGG | BGB dcB |"Am" cAA EAA |"C" cBc"D7" edc |"G" BGG DGG | BGB dcB | 
"Am" cBc"D7" Adc |"G" BGG G2 :: g/a/ |"G" bgg dgg | bgb bag |"Bm" aff dff |"Bm" afa agf | 
"C" egg"G" dgg |"Am" cgg"G" B2 B |"D7" cBc Adc |"G" BGG G2 :|
```
5. Segmentation
```
%%input
%%segmentation
L:1/4
M:4/4
K:Eb
"Cm" c"Cm" c"Cm/Eb" g"Cm" g/a/ |"Bb/D" b"Eb" g"Ab" e"Ddim/F" f/g/ |"Bb7/F" a2"Eb" g2 |
"F7/A" f"Bbsus4" f"Cm" e"Bb/D" f |"Eb" g"Bbsus4" f"Cm" e"Bb/D" d |"Cm7/Eb" c2"Bb" B2 |
"Cm" e"Csus2" d"Cm" e/f/"Cm/Eb" g |"Fm" f/e/"G" d"Ab" c2 |
"G" d"Cm/Eb" c/d/"Cm" e"Gsus4" d |"C" c4 |]
```
```
%%output
X:0
E:9
L:1/4
M:4/4
K:Eb
"Cm" c"Cm" c"Cm/Eb" g"Cm" g/a/ |"Bb/D" b"Eb" g"Ab" e"Ddim/F" f/g/ |"Bb7/F" a2"Eb" !breath!g2 |
"F7/A" f"Bbsus4" f"Cm" e"Bb/D" f |"Eb" g"Bbsus4" f"Cm" e"Bb/D" d |"Cm7/Eb" c2"Bb" !breath!B2 |
"Cm" e"Csus2" d"Cm" e/f/"Cm/Eb" g |"Fm" f/e/"G" d"Ab" !breath!c2 |
"G" d"Cm/Eb" c/d/"Cm" e"Gsus4" d |"C" !breath!c4 |]

X:1
E:9
L:1/4
M:4/4
K:Eb
"Cm" c"Cm" c"Cm/Eb" g"Cm" g/a/ |"Bb/D" b"Eb" g"Ab" e"Ddim/F" f/g/ |"Bb7/F" a2"Eb" !breath!g2 |
"F7/A" f"Bbsus4" f"Cm" e"Bb/D" f |"Eb" g"Bbsus4" f"Cm" e"Bb/D" d |"Cm7/Eb" c2"Bb" !breath!B2 |
"Cm" e"Csus2" d"Cm" e/f/"Cm/Eb" g |"Fm" f/e/"G" d"Ab" !breath!c2 |
"G" d"Cm/Eb" c/d/"Cm" e"Gsus4" d |"C" !breath!c4 |]

X:2
E:9
L:1/4
M:4/4
K:Eb
"Cm" c"Cm" c"Cm/Eb" g"Cm" g/a/ |"Bb/D" b"Eb" g"Ab" e"Ddim/F" f/g/ |"Bb7/F" a2"Eb" !breath!g2 |
"F7/A" f"Bbsus4" f"Cm" e"Bb/D" f |"Eb" g"Bbsus4" f"Cm" e"Bb/D" d |"Cm7/Eb" c2"Bb" !breath!B2 |
"Cm" e"Csus2" d"Cm" e/f/"Cm/Eb" g |"Fm" f/e/"G" d"Ab" !breath!c2 |
"G" d"Cm/Eb" c/d/"Cm" e"Gsus4" d |"C" !breath!c4 |]

X:3
E:9
L:1/4
M:4/4
K:Eb
"Cm" c"Cm" c"Cm/Eb" g"Cm" g/a/ |"Bb/D" b"Eb" g"Ab" e"Ddim/F" f/g/ |"Bb7/F" a2"Eb" !breath!g2 |
"F7/A" f"Bbsus4" f"Cm" e"Bb/D" f |"Eb" g"Bbsus4" f"Cm" e"Bb/D" d |"Cm7/Eb" c2"Bb" !breath!B2 |
"Cm" e"Csus2" d"Cm" e/f/"Cm/Eb" g |"Fm" f/e/"G" d"Ab" !breath!c2 |
"G" d"Cm/Eb" c/d/"Cm" e"Gsus4" d |"C" !breath!c4 |]
```
6. Transcription
```
%%input
%%transcription
L:1/8
M:3/4
K:A
EG A2 A2 | BA G2 A2 | Bc/d/ e2 A2 | BA GF E2 | F2 G2 A2 | Bc d2 e2 |
fd c/4B/4c/4B/4c/4B/4c/4B/4 A2 | A2 A4 | EG A2 A2 | BA G2 A2 | Bc/d/ e2 A2 | BA GF E2 |
F2 G2 A2 | Bc d2 e2 | fd c/4B/4c/4B/4c/4B/4c/4B/4 A2 | A2 A4 | cd e2 ef | =g2 f3 e | dc d2 dB |
AB G>F E2 | F2 G2 A2 | Bc d2 e2 | fd c/4B/4c/4B/4c/4B/4c/4B/4 A2 | A2 A4- | A2 cd e2 |
ef =g2 f2- | fe dc d2 | dB AB G>F | E2 F2 G2 | A2 Bc d2 | e2 fd c/4B/4c/4B/4c/4B/4c/4B/4 |
A2 A2 A2- | A4 z2 |]
```
```
%%output
X:0
E:3
L:1/8
M:3/4
K:A
EG | A2 A2 BA | G2 A2 Bc/d/ | e2 A2 BA | GF E2 F2 | G2 A2 Bc | d2 e2 fd | TB2 A2 A2 | A4 :: cd |
e2 ef =g2 | f3 edc | d2 dB AB | G>F E2 F2 | G2 A2 Bc | d2 e2 fd | TB2 A2 A2 | A6 :|

X:1
E:3
L:1/8
M:3/4
K:A
EG |"A" A2 A2 BA |"E" G2 A2 Bc/d/ |"A" e2 A2 BA |"E" GF E2 F2 |"E" G2 A2 Bc |"D" d2 e2 fd | 
"E" TB2 A2 A2 |"A" A4 :| cd |"A" e2 ef =g2 |"D" f3 e dc |"D" d2 dB AB |"E" G>F E2 F2 | 
"E" G2 A2 Bc |"D" d2 e2 fd |"E" TB2 A2 A2 |"A" A6 cd |"A" e2 ef =g2 |"D" f3 e dc |"D" d2 dB AB | 
"E" G>F E2 F2 |"E" G2 A2 Bc |"D" d2 e2 fd |"E" TB2 A2 A2 |"A" A4 |]

X:2
E:3
L:1/8
M:3/4
K:A
|: EG |"A" A2 A2 BA |"E" G2 A2 Bc/d/ |"A" e2 A2 BA |"E" GF E2 F2 |"G" G2 A2 Bc |"D" d2 e2 fd | 
"E" TB2 A2 A2 |"A" A4 :| cd |"A" e2 ef =g2 |"D" f3 e dc |"D" d2 dB AB |"E" G>F E2 F2 | 
"G" G2 A2 Bc |"D" d2 e2 fd |"E" TB2 A2 A2 |"A" A6 cd |"A" e2 ef =g2 |"D" f3 e dc |"D" d2 dB AB | 
"E" G>F E2 F2 |"G" G2 A2 Bc |"D" d2 e2 fd |"E" TB2 A2 A2 |"A" A6 ||

X:3
E:4
L:1/8
M:3/4
K:A
EG | A2 A2 BA | G2 A2 Bc/d/ | e2 A2 BA | GF E2 F2 | G2 A2 Bc | d2 e2 fd | TB2 A2 A2 | A4 :| cd | 
e2 ef =g2 | f3 e dc | d2 dB AB | G>F E2 F2 | G2 A2 Bc | d2 e2 fd | TB2 A2 A2 | A6 || cd | e2 ef =g2 | 
f3 e dc | d2 dB AB | G>F E2 F2 | G2 A2 Bc | d2 e2 fd | TB2 A2 A2 | A6 ||
```
7. Variation
```
%%input
%%variation
L:1/8
M:6/8
K:D
|: AFD DFA | Add B2 A | ABA F3 | GFG EFG | AFD DFA | Add B2 A | ABA F2 E | FDD D3 :: fdd ede |
fdd d2 g | fdd def | gfg e2 g | fed B2 A | AdF A3 | ABA F2 E | FDD D3 :|
```
```
%%output
X:0
E:8
L:1/8
M:6/8
K:D
|: B | AFD DFA | Add B2 A | ABA F2 E | FEE E2 B | AFD DF/G/A | Add B2 A | ABA F2 E | FDD D2 :: e |
fdd dcd | fdd d2 e | f^ef d=ef | g2 f efg | ff/e/d B2 d | Add F2 G | ABA F2 E | FDD D2 :|

X:1
E:8
L:1/8
M:6/8
K:D
|: B | AFD DFA | ded B2 A | ABA F2 D | GFG E2 B | AFD DF/G/A | df/e/d B2 A | ABA F2 E | EDD D2 :: 
e | fdd e^de | fdd d2 e | f2 f def | g2 f e2 g | fed B2 d | A2 d F3 | ABA F2 E | EDD D2 :|

X:2
E:8
L:1/8
M:6/8
K:D
|: B | AFD DFA | BdB BAF | ABA F2 D | FEE E2 B | AFD DFA | BdB BAF | ABA F2 E |1 FDD D2 :|2 
FDD D2 e |: fdd dcd | fdd d2 e | fef def | gfg eag | fed B2 d | A2 d F2 G | ABA F2 E |1 
FDD D2 e :|2 FDD D2 ||

X:3
E:5
L:1/8
M:6/8
K:D
|: (d/B/) | AFD DFA | B2 d F2 A | AFD DEF | GFG EFG | AFD DFA | B2 d F2 A | Bdd F2 E | FDD D2 :: 
fed dB/c/d | efe efg | fed daa | agf eag | fed B2 d | A2 d F2 A | Bdd F2 E | FDD D2 :|
```
