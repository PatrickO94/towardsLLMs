
class Config:

    EPOCHS = 1
    BATCH_SIZE = 64 # 64 # B
    CONTEXT_LEN = 256 # aka BLOCK_SIZE, T
    LR = 0.0003
    N_EMBD = 384 # C-emb, make sure that N_EMBD/N_HEADS is int
    VOCAB_SIZE = 65 # C-base-bigram
    EVAL_ITERS = 200
    EVAL_INTERVAL = 500
    DROPOUT = 0.2
    N_HEADS = 6
    N_LAYERS = 6
    MDL_PATH = r"/home/rick/PycharmProjects/towardsLLMs/model_archive"
    OUT_PATH = r"/home/rick/PycharmProjects/towardsLLMs/results_archive"

#####
# Not in current use:
TEACHER_FORCING = False
EPS = 10e-9
LORA = True
LORA_RANK = 5

cfg = Config()