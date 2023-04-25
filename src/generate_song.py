
from pathlib import Path
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type = str, required = True)
    parser.add_argument("--entry_length", type = int, default = 100)
    parser.add_argument("--temperature", type = float, default = 1)
    parser.add_argument("--model", type = str, default="mt5")
    
    return parser.parse_args()

def generate(model, tokenizer, prompt:str, entry_length:int = 30, temperature:float = 1.0):
    """

    Parameters
    ----------
    model : 
        The model to generate from
    tokenizer : 
        The tokenizer
    prompt : str
        The prompt from which to generate text
    entry_length : int
        The number of tokens to generate
    temperature : float
        XXXX. Determines how deterministic the model is???

    Returns
    -------
    generated: str
        The prompt followed by the generated text
        
    """
    with torch.no_grad():
        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

        for i in range(entry_length):
            outputs = model(generated, labels=generated)
            loss, logits = outputs[:2]
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            #if next_token in tokenizer.encode("<|endoftext|>"):
            #    entry_finished = True

        output_list = list(generated.squeeze().numpy())
        generated = f"{tokenizer.decode(output_list)}<|endoftext|>" 
                
    return generated

def load_model(model, path):
    path = Path(__file__).parents[1]

    if model.lower() == "gpt-2":
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        
        # load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        checkpoint = torch.load(path / "mdl" / "finetuned_gpt-2.pt")
        model.load_state_dict(checkpoint)

        model.eval()

    elif model.lower() == "mt5":
        from transformers import MT5ForConditionalGeneration, T5Tokenizer

        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
        model = T5Tokenizer.from_pretrained("google/mt5-base")

        checkpoint = torch.load(path / "mdl" / "finetuned_mt5.pt")
        model.load_state_dict(checkpoint)
        
        model.eval()
    
    return model, tokenizer



def main(): 
    args = parse_args()
    path = Path(__file__).parents[1]
    
    
    model, tokenizer = load_model(args.model, path)

    input_sequence = f"<|lyrics|> {args.prompt}"
    x = generate(model, tokenizer, input_sequence, entry_length=args.entry_length, temperature=args.temperature)
    print(x)

if __name__ == "__main__":
    main()

