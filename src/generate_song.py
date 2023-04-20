
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



if __name__ == "__main__":
    path = Path(__file__).parents[1]
    
    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')


    checkpoint = torch.load(path / "mdl" / "finetuned_gpt-2.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

    model.eval()

    
