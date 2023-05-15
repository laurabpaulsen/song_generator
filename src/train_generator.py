from pathlib import Path
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default="mt5")
    parser.add_argument("--epochs", type = int, default = 10)

    return parser.parse_args()


class SongLyrics(Dataset):  
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
            self.prompt.append(torch.tensor(
                self.tokenizer.encode(f"{row['prompt']}<|endoftext|>")))

            self.target.append(torch.tensor(
                self.tokenizer.encode(f"{row['target'][:max_length]}<|endoftext|>")))

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

def clean_lyrics(lyrics:list):
    for i, lyric in enumerate(lyrics):
        # Remove everything before the first time it says "Lyrics" (title of the song, contributor, etc.)
        start = lyric.find("Lyrics")+7
        # Remove suggestions at the end
        stop = lyric.find("You might also like")
        
        lyrics[i] = lyric[start:stop]

    return lyrics


def load_model(model):
    if model.lower() == "gpt-2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        # load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

    elif model.lower() == "mt5":
        from transformers import MT5Tokenizer, TFMT5ForConditionalGeneration

        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
        model = TFMT5ForConditionalGeneration.from_pretrained("google/mt5-base")
        
    return model, tokenizer


def finetune_model(model, dataloader, epochs, batch_size = 24):
    # train the model
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=-1)

    loss=0

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(dataloader)):

            outputs = model(entry[0].squeeze(), labels=entry[0].squeeze())
            loss = outputs[0]
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

    return model


def main():
    args = parse_args()
    
    # output directory
    path = Path(__file__).parents[1]
    lyrics = load_txts(path / 'data' / 'lyrics')
    lyrics = clean_lyrics(lyrics)

    # load model and tokenizer
    model, tokenizer = load_model(args.model)

    dataset = SongLyrics(lyrics, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = finetune_model(model, train_dataloader, args.epochs)
    
    # save the model
    torch.save(model.state_dict(), path / "mdl" / f"finetuned_{args.model}.pt")

if __name__ == "__main__":
    main()