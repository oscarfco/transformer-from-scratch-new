import argparse
import torch
import pickle 
import wandb
from tqdm import tqdm
from transformer_modules import * 
from training_modules import * 
from tokenizer import Tokenizer


def initialize_lm(args):
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.dff,
        rope_theta=args.rope_theta,
        device=args.device
    )

def init_wandb(args, optim_params):
    return wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="oscarfco",
    # Set the wandb project where this run will be logged.
    project="transformer-from-scratch",
    # Track hyperparameters and run metadata.
    config={
         "dataset": args.dataset_name,
        "learning_rate": args.learning_rate,
        "tokens_processed_total": args.tokens_processed_total,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "dff": args.dff,
        "rope_theta": args.rope_theta,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        **optim_params
        },
    )



def train(model, optim_params, tokens_processed_total, train_data, batch_size, context_length, device, logger, save_every):
    # Initialize Optimizer
    optim = AdamW(
        model.parameters(), 
        optim_params["lr"],
        optim_params["weight_decay"],
        optim_params["betas"],
        optim_params["eps"]
    )

    running_loss = 0.0
    total_steps = tokens_processed_total // batch_size // context_length
    save_every_steps = int(save_every * total_steps)

    pbar = tqdm(range(total_steps), desc="Training", ncols=100)

    for e in pbar:
        model.train()
        inputs, targets = get_batch(train_data, batch_size, context_length, device)
        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        avg_loss = running_loss / (e + 1)        

        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        if logger is not None:
            logger.log({"loss": avg_loss, "step": e})

        if e % save_every_steps == 0:
            save_path = f"checkpoints/model_step_{e}.pt"
            save_checkpoint(model, optim, e, save_path)
    
    final_save_path = "checkpoints/model_step_final.pt"
    save_checkpoint(model, optim, e, final_save_path)

    return


def test(model, prompt):
    with open("./data/tiny_stories_train.pkl", "rb") as f:  
        vocab, merges = pickle.load(f)

    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


    tokenized_prompt = torch.tensor(tok.encode(prompt))
    tokenized_prompt = torch.tensor(tokenized_prompt, dtype=torch.int64, device='cpu').unsqueeze(0)


    val = decode(model, tok, tokenized_prompt, max_generated_tokens=20, temperature=1, top_p=0.9)
    print(val)


def load_data(path):
    # Loads the data in this lazy way
    return np.load(path, mmap_mode='r')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple argument parser example") 

    # Transformer Parameters
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument("--rope-theta", type=int, default=10_000)
    
    # Dataset Parameters
    parser.add_argument("--dataset-name", default="TinyStories")
    parser.add_argument("--training-dset-path", type=str, default="./data/tiny_stories_train.npy")
    parser.add_argument("--validation-dset-path", type=str)

    # Optimizer Parameters
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # Training Parameters
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tokens-processed-total", type=int)
    parser.add_argument("--no-wandb", action='store_true')
    parser.add_argument("--save-every", type=float)


    # My own testing stuff
    parser.add_argument("--testing", action="store_true")


    return parser.parse_args()


def main():
    args = parse_arguments()
    
    optim_parameters = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps
    }
    logger = None
    if not args.testing or not args.no_wandb:
        logger = init_wandb(args, optim_parameters)

    train_data = load_data(args.training_dset_path)
    model = initialize_lm(args)

    if args.testing:
        test(model, "hello world")
    else:
        train(
            model, 
            optim_parameters, 
            args.tokens_processed_total, 
            train_data, 
            args.batch_size, 
            args.context_length, 
            args.device,
            logger,
            args.save_every
        )


if __name__ == "__main__":
    main()
