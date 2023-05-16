from pathlib import Path
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import pandas as pd
import re
import random
from torch.nn.utils.rnn import pad_sequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default="mt5")
    parser.add_argument("--epochs", type = int, default = 10)

    return parser.parse_args()


def clean_lyrics(lyrics:list):
    for i, lyric in enumerate(lyrics):
        #remove identifiers like chorus, verse, etc
        lyric = re.sub(r'(\[.*?\])*', '', lyric)

        # replace double linebreaks with single linebreaks
        lyric = re.sub('\n\n', '\n', lyric)  # Gaps between verses

        lyric = lyric.lower()
        # Remove everything before the first time it says "lyrics" (title of the song, contributor, etc.)
        start = lyric.find("lyrics")+7
       
        # Remove suggestions at the end
        stop = lyric.find("you might also like")
        
        lyrics[i] = lyric[start:stop]
        
    return lyrics

def load_model(model_name):
    if model_name.lower() == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        # load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

    elif model_name.lower() == "mt5":
        from transformers import MT5Tokenizer, MT5ForConditionalGeneration

        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
        
    return model, tokenizer


def finetune_model(model, dataloader, epochs, batch_size=152):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=-1)

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        total_loss = 0

        for prompt, target in tqdm(dataloader):
            optimizer.zero_grad()

            input_ids = prompt.squeeze().to(model.device)
            labels = target.squeeze().to(model.device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss / len(dataloader)}")

    return model


class SongLyrics(Dataset):
    def __init__(self, lyrics_df, tokenizer):
        self.tokenizer = tokenizer
        self.lyrics_df = lyrics_df
    
    def __len__(self):
        return len(self.lyrics_df)
    
    def __getitem__(self, idx):
        prompt = self.lyrics_df.iloc[idx]["input"]
        target = self.lyrics_df.iloc[idx]["target"]

        prompt_enc = self.tokenizer.encode(prompt, truncation=True)
        target_enc = self.tokenizer.encode(target, truncation=True)

        return torch.tensor(prompt_enc), torch.tensor(target_enc)

def collate_fn(batch):
    prompts, targets = zip(*batch)
    padded_prompts = pad_sequence(prompts, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)
    return padded_prompts, padded_targets



def main():
    batch_size = 24
    args = parse_args()
    
    # output directory
    path = Path(__file__).parents[1]
    lyrics_path = path / 'data' / 'lyrics.csv'

    lyrics = pd.read_csv(lyrics_path)
    #lyrics = lyrics.sample(10000) # just for testing!!
    
    # load model and tokenizer
    model, tokenizer = load_model(args.model)

    dataset = SongLyrics(lyrics, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = finetune_model(model, train_dataloader, args.epochs, batch_size)
    
    # save the model
    torch.save(model.state_dict(), path / "mdl" / f"finetuned_{args.model}.pt")

if __name__ == "__main__":
    main()