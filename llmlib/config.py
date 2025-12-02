
class Config:

    EPOCHS = 100
    BATCH_SIZE = 4
    CONTEXT_LEN = 8
#####
TEACHER_FORCING = False
CHECKPOINT_DIR = ''
LR = 0.001
EPS = 10e-9
LORA = True
LORA_RANK = 5

cfg = Config()