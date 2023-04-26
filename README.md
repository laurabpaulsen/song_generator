# Danish song generator
This repository holds the code for the final assignment for language analytics (S2023). This includes script for scraping danish songs from danish artists on from Genius.com, training a language model on the scraped data, and at last generating new songs based using the trained model.

## Data
The data consists of a collection of songs from danish artists. The data is scraped from Genius.com using the script `scrape_songs.py`. The data is stored in the `data/lyrics` directory. Each song is stored in a separate `.txt` file. 

## Usage
1. Clone the repository
2. Create a virtual environment and install the required packages
```
bash setup.sh
```
3. Activate the environment
```
source env/bin/activate
```

4. Scrape songs from Genius.com
```
python src/scrape_songs.py
```
5. Train a language model (can be either gpt-2 or mt5)
```
python src/train_generator.py --model gpt2 --epochs 10
```
6. Generate new songs
```
python src/generate_songs.py --model gpt2 --prompt "insert first few lines of song here"
```



## Repository structure

```


```