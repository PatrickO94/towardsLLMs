
class Config:

    EPOCHS = 1
    BATCH_SIZE = 64 # B
    CONTEXT_LEN = 256 # aka BLOCK_SIZE, T
    LR = 0.0003
    N_EMBD = 384 # C-emb
    VOCAB_SIZE = 65 # C-base-bigram
    EVAL_ITERS = 200
    EVAL_INTERVAL = 500
    DROPOUT = 0.2
    N_HEADS = 6
    N_LAYERS = 6

#####
# Not in current use:
TEACHER_FORCING = False
CHECKPOINT_DIR = ''
EPS = 10e-9
LORA = True
LORA_RANK = 5

cfg = Config()