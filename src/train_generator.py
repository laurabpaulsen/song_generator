"""
Finetunes GPT-2 to generate danish song lyrics.

Inspired by https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272

Author: Laura Bock Paulsen (202005791@post.au.dk)
"""

from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SongLyrics(Dataset):  
    def __init__(self, data, tokenizer, control_code, max_length=1024):
        self.tokenizer = tokenizer
        self.lyrics = []

        for dat in data:
            self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{dat[:max_length]}<|endoftext|>")))
        
        
        self.lyrics_count = len(self.lyrics)
            
    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]
    

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
    # Remove everything before the first time it says "Lyrics" (title of the song, contributor, etc.)
    for i, lyric in enumerate(lyrics):
        lyrics[i] = lyric[lyric.find("Lyrics")+7:]
    
    
    return lyrics

def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(dataset, model, batch_size=16, epochs=5, lr=2e-5, warmup_steps=200):
    """
    Finetunes a GPT-2 model on a dataset

    Parameters
    ----------
    dataset : Dataset
        The dataset to finetune on
    model : GPT2LMHeadModel
        The model to finetune
    batch_size : int, optional
        The batch size to use, by default 16
    epochs : int, optional
        The number of epochs to train, by default 5
    lr : float, optional
        The learning rate to use, by default 2e-5
    warmup_steps : int, optional
        The number of warmup steps to use, by default 200

    Returns
    -------
    GPT2LMHeadModel
        The finetuned model
    """

    device = torch.device("cpu")
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        
    return model


if __name__ == '__main__':
    # output directory
    path = Path(__file__).parents[1]
    lyrics = load_txts(path / 'data' / 'lyrics')
    lyrics = clean_lyrics(lyrics)

    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # prep dataset
    dataset = SongLyrics(lyrics, tokenizer, control_code="lyrics")


    model = train(dataset, model)

    # save model
    torch.save(model.state_dict(), path / "mdl" / "finetuned_gpt-2.pt")



