"""
This script prepares a pandas dataframe for the model training with sentences and the following sentence.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re


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


def create_df(lyrics: list):
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
        for i, sentence in enumerate(sentences[:-1]):
            df = df.append({"input": sentence, "target": sentences[i+1]}, ignore_index=True)

    return df

def main():
    path = Path(__file__)
    lyrics_path = path.parents[1] / "data" / "lyrics"
    output_path = path.parents[1] / "data" / "lyrics.csv"

    lyrics = load_txts(lyrics_path)

    lyrics = preprocess_lyrics(lyrics)

    df = create_df(lyrics)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()