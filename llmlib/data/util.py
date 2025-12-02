

# check googles sentencePiece tokeniser for a more efficient but more complicated one.
class CharLvlTokeniser:
    def __init__(self, chars):
        self.str_to_int = {ch:i for i, ch in enumerate(chars)}
        self.int_to_str = {i:ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [self.str_to_int[c] for c in s]
        self.decode = lambda l: ''.join([self.int_to_str[i] for i in l])
