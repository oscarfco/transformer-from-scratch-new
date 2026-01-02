import torch 
import torch.nn as nn
from model import BasicsTransformerLM
import copy
from tqdm import tqdm

def main():
    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff = 1024,
        rope_theta=10_000,
        lora=True,
        lora_config={"rank": 8, "d": 512}
    )

    model_copy = copy.deepcopy(model)

    # Freeze all the non LoRA Parameters
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    

    optimizer = torch.optim.SGD(model.parameters())

    num_steps = 5
    for _ in tqdm(range(num_steps)):

        input_data = torch.randint(size=(8, 256), low=0, high=10_000)

        out = model(input_data)

        loss = out.sum()    # dummy loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pass_test = True
    for (name_old, param_old), (name_new, param_new) in zip(model_copy.named_parameters(), model.named_parameters()):
        if "lora" not in name_old:
            if not torch.equal(param_old.data, param_new.data):
                pass_test = False
                print(f"{name_old} and {name_new} don't match")
                break
        else:
            if torch.equal(param_old.data, param_new.data):
                pass_test = False
                print(f"{name_old} and {name_new} match and they shouldn't!")
                break
    
    if pass_test:
        print("✅ LoRA worked!")


        





if __name__ == "__main__":
    main()