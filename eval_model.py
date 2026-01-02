import torch 
import pickle
from transformer_modules import TransformerLM
from tokenizer import Tokenizer
from training_modules import decode, load_checkpoint
from eval_utils import run_eval


def initialize_lm():
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
        device="cuda"
    )

    return model


def get_tokenizer(data_path="./data/tiny_stories_train.pkl"):
    with open(data_path, "rb") as f:  
        vocab, merges = pickle.load(f)

    return Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


def prompt_model(model, tokenizer, prompt, device, max_generated_tokens=100, num_generations=1):

    tokenized_prompt = torch.tensor(tokenizer.encode(prompt))
    tokenized_prompt = torch.tensor(tokenized_prompt, dtype=torch.int64, device=device).unsqueeze(0)

    completions = []
    for g in range(num_generations):
        completion = decode(model, tokenizer, tokenized_prompt, max_generated_tokens=max_generated_tokens, temperature=1, top_p=1)
        completions.append(completion)

    return completions


def run_eval_and_sample(model, tokenizer, batch_size, context_length, device, prompt, num_eval_generations):
    model.eval()

    with torch.no_grad():
        eval_loss = run_eval(model, tokenizer, batch_size, context_length, device)
        completions = prompt_model(model, tokenizer, prompt, device, max_generated_tokens=100, num_generations=num_eval_generations)
    
    return eval_loss, completions


def main():
    device = "cuda"
    prompt = "Once upon a time"
    model = initialize_lm()
    load_checkpoint(src="checkpoints/model_step_final.pt", model=model)
    tokenizer = get_tokenizer()
    model.eval()
    with torch.no_grad():
        loss = run_eval(model, tokenizer, batch_size=128, context_length=256, device="cuda")
        print(f"Eval Loss (per token): {round(loss, 2)}")
        completions = prompt_model(model, tokenizer, prompt, device)

    
    print("Completions")
    for c in completions:
        print()
        print(c)


if __name__ == "__main__":
    main()