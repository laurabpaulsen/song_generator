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

def extract_n_words(str, n):
    return ' '.join(str.split()[:n])

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

def main():
    args = parse_args()
    
    # output directory
    path = Path(__file__).parents[1]
    lyrics = load_txts(path / 'data' / 'lyrics')
    lyrics = clean_lyrics(lyrics)

    # create a dataset from the list of strings
    df = pd.DataFrame(lyrics, columns=['lyrics'])
    # check that there is atleast 50 words in each song
    df = df[df['lyrics'].apply(lambda x: len(x.split()) > 50)]

    # split into prompt and target (first 20 words are prompt, rest are target)
    df['prompt']=df['lyrics'].apply(lambda x: ' '.join(x.split()[:20]))
    df['target']=df['lyrics'].apply(lambda x: ' '.join(x.split()[20:]))

    # load model and tokenizer
    model, tokenizer = load_model(args.model)

    # tokenize prompt and target
    df['prompt_tokenized'] = df['prompt'].apply(lambda x: tokenizer.encode(x, return_tensors='pt'))
    df['target_tokenized'] = df['target'].apply(lambda x: tokenizer.encode(x, return_tensors='pt'))

    df = df[['prompt_tokenized', 'target_tokenized']]

    # train the model
    model.train()

    train_dataloader = DataLoader(Dataset(df), batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=-1)

    for epoch in range(args.epochs):
        losses = []
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['prompt_tokenized'].squeeze()
            labels = batch['target_tokenized'].squeeze()
            outputs = model(input_ids, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        print(f'epoch {epoch+1}: mean loss: {sum(losses)/len(losses)}')
    
    # save the model
    torch.save(model.state_dict(), path / "mdl" / f"finetuned_{args.model}.pt")

