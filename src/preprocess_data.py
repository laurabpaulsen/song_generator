"""
This script prepares a pandas dataframe for the model training with sentences and the following sentence.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re
import string
from transformers import MT5Tokenizer
from torch.nn.utils.rnn import pad_sequence
import torch



def load_txts(directory: Path):
    """
    Loads all text files in a directory and returns a list of strings

    Parameters
    ----------
    directory : Path
        The directory to load the text files from

    Returns
    -------
    list
        A list of strings
    """
    txts = []

    for file in directory.iterdir():
        with open(file, 'r') as f:
            txts.append(f.read())

    return txts

def tokenize(tokenizer, txt):
    return tokenizer.encode(txt, truncation=True)


def preprocess_lyrics(lyrics: list):
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


def create_df(lyrics: list, tokenizer):
    """
    Creates a pandas dataframe with the a sentence from the lyrics and the following sentence

    Parameters
    ----------
    lyrics : list
        A list of strings
    
    Returns
    -------
    pandas.DataFrame
        A pandas dataframe with the columns "input" and "target"
    """

    df = pd.DataFrame(columns=["input", "target"])

    for lyric in lyrics:
        sentences = lyric.split("\n")

        # remove specified special characters
        sentences = [sent.translate(str.maketrans('', '', string.punctuation)) for sent in sentences]

        sentences = [sent for sent in sentences if len(sent) > 5]

        for i, sentence in enumerate(sentences[:-1]):
            tmp_data = pd.DataFrame.from_dict({"input": [sentence], "target": [sentences[i+1]]})
            df = pd.concat([df, tmp_data], ignore_index=True)

    
    df["input_tkn"] = df["input"].apply(lambda x: tokenize(tokenizer, x))
    df["target_tkn"] = df["target"].apply(lambda x: tokenize(tokenizer, x))

    # pad sequences
    df["input_tkn"] = pad_sequence([torch.tensor(x) for x in df["input_tkn"]], batch_first=True)
    df["target_tkn"] = pad_sequence([torch.tensor(x) for x in df["target_tkn"]], batch_first=True)


    return df

def main():
    path = Path(__file__)
    lyrics_path = path.parents[1] / "data" / "lyrics"
    output_path = path.parents[1] / "data" / "lyrics.csv"

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

    lyrics = load_txts(lyrics_path)

    lyrics = preprocess_lyrics(lyrics)

    df = create_df(lyrics, tokenizer)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()