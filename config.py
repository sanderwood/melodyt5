PATCH_LENGTH = 256      # Patch Length
PATCH_SIZE = 64       # Patch Size

PATCH_NUM_LAYERS = 9         # Number of layers in the encoder
CHAR_NUM_LAYERS = 3          # Number of layers in the decoder

NUM_EPOCHS = 32                # Number of epochs to train for (if early stopping doesn't intervene)
LEARNING_RATE = 2e-4           # Learning rate for the optimizer
ACCUMULATION_STEPS = 1          # Accumulation steps to simulate large batch size
BATCH_SIZE = 10                # Batch size for training
PATCH_SAMPLING_BATCH_SIZE = 0   # Batch size for patch during training, 0 for full context
LOAD_FROM_CHECKPOINT = True     # Whether to load weights from a checkpoint
LOAD_FROM_PRETRAINED = True    # Whether to load weights from a pretrained model
SHARE_WEIGHTS = True            # Whether to share weights between the encoder and decoder

TRAIN_DATA_PATH      = 'total_data_train.jsonl'
VALIDATION_DATA_PATH = 'total_data_validation.jsonl'

PRETRAINED_PATH = "weights.pth" # Path to the pretrained weights
WEIGHTS_PATH = "weights.pth"    # Path to save the weights
LOGS_PATH = "logs.txt"          # Path to save the logs