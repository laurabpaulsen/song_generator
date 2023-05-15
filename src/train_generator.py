from pathlib import Path
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import pandas as pd
import re

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

class SongLyrics(Dataset):
    def __init__(self, lyrics, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.prompt = []
        self.target = []

        for lyric in lyrics:
            lines = lyric.split("\n")  # Split into lines

            # Generate prompt and target pairs based on line boundaries
            for i in range(len(lines) - 1):
                prompt = " ".join(lines[:i+1])
                target = " ".join(lines[i+1:])

                prompt_enc = self.tokenizer.encode(prompt, max_length=max_length, truncation=True)
                target_enc = self.tokenizer.encode(target, max_length=max_length, truncation=True)

                # Ensure consistent sequence length
                prompt_enc = prompt_enc[:max_length]
                target_enc = target_enc[:max_length]

                # Pad sequences if necessary
                prompt_enc = self._pad_sequence(prompt_enc, max_length)
                target_enc = self._pad_sequence(target_enc, max_length)

                self.prompt.append(torch.tensor(prompt_enc))
                self.target.append(torch.tensor(target_enc))

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, idx):
        return self.prompt[idx], self.target[idx]

    def _pad_sequence(self, sequence, max_length):
        padding_length = max_length - len(sequence)
        return sequence + [self.tokenizer.pad_token_id] * padding_length

class SongLyrics_old(Dataset):  
    def __init__(self, lyrics, tokenizer, max_length=1024):
        """
        Creates a dataset of song lyrics

        Parameters
        ----------
        lyrics : list
            A list of strings containing the lyrics
        tokenizer : transformers tokenizer
            The tokenizer to use
        control_code : str
            The control code to use
        max_length : int, optional
            The maximum length of a sequence, by default 1024
        truncate : bool, optional
            Whether to truncate the dataset, by default False
        """

        self.tokenizer = tokenizer
        self.prompt = []
        self.target = []

        # create a dataset from the list of strings
        df = pd.DataFrame(lyrics, columns=['lyrics'])
        # check that there is atleast 50 words in each song
        df = df[df['lyrics'].apply(lambda x: len(x.split()) > 50)]

        # split into prompt and target (first 20 words are prompt, rest are target)
        df['prompt']=df['lyrics'].apply(lambda x: ' '.join(x.split()[:20]))
        df['target']=df['lyrics'].apply(lambda x: ' '.join(x.split()[20:]))

        for i, row in df.iterrows():
            prompt = torch.tensor(
                self.tokenizer.encode(row['prompt'],  max_length=max_length, truncation=True))

            target = torch.tensor(
                self.tokenizer.encode('target',  max_length=max_length, truncation=True))

            # pad sequences and append
            self.prompt.append(self._pad_sequence(prompt, max_length))
            self.target.append(self._pad_sequence(target, max_length))

    def _pad_sequence(self, sequence, max_length):
        padding_length = max_length - len(sequence)
        padding_sequence = torch.tensor([self.tokenizer.pad_token_id] * padding_length)
        return torch.cat((sequence, padding_sequence), dim=0)


    def __len__(self):
        return len(self.prompt)
        
    def __getitem__(self,idx):
        return self.prompt[idx], self.target[idx]

def load_txts(path: Path):
    """
    Loads all the txt files in a directory

    Parameters
    ----------
    path : Path
        The path to the directory containing the txt files

    Returns
    -------
    list
        A list of the txt files
    """
    txts = []

    # use pathlib to loop over files
    for f in path.iterdir():
        # only load txt files
        if f.suffix == '.txt':
            with open(f, 'r') as f:
                txts.append(f.read())

    return txts


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


def main():
    batch_size = 152
    args = parse_args()
    
    # output directory
    path = Path(__file__).parents[1]
    lyrics = load_txts(path / 'data' / 'lyrics')
    lyrics = clean_lyrics(lyrics)

    # load model and tokenizer
    model, tokenizer = load_model(args.model)

    dataset = SongLyrics(lyrics, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = finetune_model(model, train_dataloader, args.epochs, batch_size)
    
    # save the model
    torch.save(model.state_dict(), path / "mdl" / f"finetuned_{args.model}.pt")

if __name__ == "__main__":
    main()