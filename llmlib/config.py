
class Config:

    EPOCHS = 100
    BATCH_SIZE = 32
    CONTEXT_LEN = 8
    LR = 0.003
#####
# Not in current use:
TEACHER_FORCING = False
CHECKPOINT_DIR = ''
LR = 0.001
EPS = 10e-9
LORA = True
LORA_RANK = 5

cfg = Config()