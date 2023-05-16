
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
    parser.add_argument("--temperature", type = float, default = 0.3)
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
        generated = tokenizer.decode(output_list, skip_special_tokens=True) 
                
    return generated

def generate(model, tokenizer, prompt: str, entry_length: int = 30, temperature: float = 1.0):
    """
    Generate text based on the given prompt using the specified model and tokenizer.

    Parameters
    ----------
    model : torch.nn.Module
        The model used for text generation.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used for text encoding/decoding.
    prompt : str
        The prompt from which to generate text.
    entry_length : int
        The maximum number of tokens to generate.
    temperature : float
        The softmax temperature used for controlling the randomness of the generated text.

    Returns
    -------
    generated : str
        The prompt followed by the generated text.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    decoder_input_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(entry_length):
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[:, -1, :]
            probabilities = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            decoder_input_ids = torch.cat((decoder_input_ids, next_token), dim=-1)

    generated = input_ids.squeeze().tolist()
    generated = tokenizer.decode(generated, skip_special_tokens=True)
    generated = prompt + generated

    return generated

    def generate(model, tokenizer, prompt, entry_length=512, temperature=1.0):
        prompt = prompt.strip()
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(model.device)

        generated = model.generate(
            input_ids=input_ids,
            max_length=entry_length,
            temperature=temperature,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        generated = generated[:, input_ids.shape[-1]:]
        
        return tokenizer.decode(generated[0], skip_special_tokens=True)




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
        from transformers import MT5Tokenizer, MT5ForConditionalGeneration

        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

        checkpoint = torch.load(path / "mdl" / "finetuned_mt5.pt")
        model.load_state_dict(checkpoint)
        
        model.eval()
    
    return model, tokenizer


def main(): 
    args = parse_args()
    path = Path(__file__).parents[1]
    
    model, tokenizer = load_model(args.model, path)
    
    x = generate(model, tokenizer, args.prompt, entry_length=args.entry_length, temperature=args.temperature)
    print(x)

if __name__ == "__main__":
    main()

