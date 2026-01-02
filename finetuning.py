'''
--- Goal ---

Finetune on stanford sst dataset

1. Get the dataset tokenized
2. Add a classification head on top of the decoder

'''
from datasets import load_dataset


def main():

    dataset = load_dataset("glue", "sst2")
    
    




if __name__ == "__main__":
    main()