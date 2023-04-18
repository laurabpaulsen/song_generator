from bs4 import BeautifulSoup
import requests
import re
import os
from pathlib import Path
from langdetect import detect

def get_json(path):
    '''Send request and get response in json format.'''

    # Generate request URL
    requrl = '/'.join(["www.api.genius.com", path])
    token = "Bearer {}".format("")
    headers = {"Authorization": token}

    # Get response object from querying genius api
    response = requests.get(url=requrl, params=None, headers=headers)
    response.raise_for_status()
    return response.json()

def get_song_id(artist_id):
    '''Get all the song id from an artist.'''
    current_page = 1
    next_page = True
    songs = [] # to store final song ids

    while next_page:
        path = "artists/{}/songs/".format(artist_id)
        params = {'page': current_page} # the current page
        data = get_json(path=path, params=params) # get json of songs

        page_songs = data['response']['songs']

        if page_songs:
            # Add all the songs of current page
            songs += page_songs
            # Increment current_page value for next loop
            current_page += 1
            print("Page {} finished scraping".format(current_page))

        else:
            # If page_songs is empty, quit
            next_page = False


# Get artist object from Genius API
def request_artist_info(artist_name, page):
    """ 
    Get the artist's information from Genius.com

    Parameters
    ----------
    artist_name : str
        The name of the artist
    page : int
        The page number

    Returns
    -------
    response : requests.Response
        The response object from Genius.com
    """
    
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + "hkpU8WtKgk18vFkrV6sTN_2NgDKDJcPsf_3pOcPIbC0kMGnVMe8HrB-Cj9RhZNDb"}
    search_url = base_url + '/search?per_page=10&page=' + str(page)
    data = {'q': artist_name}
    response = requests.get(search_url, data=data, headers=headers)
    
    return response


def song_urls(artist_name: str, n: int = 10):
    """
    Get the urls of the songs of an artist
    
    Parameters
    ----------
    artist_name : str
        The name of the artist
    n : int
        The number of songs to scrape
    """
    page = 1
    songs = []
    
    while True:
        response = request_artist_info(artist_name, page)
        json = response.json()
        # Collect up to n song objects from artist
        song_info = []
        for hit in json['response']['hits']:
            if artist_name.lower() in hit['result']['primary_artist']['name'].lower():
                song_info.append(hit)
    
        # Collect song URL's from song objects
        for song in song_info:
            if (len(songs) < n):
                url = song['result']['url']
                songs.append(url)
            
        if (len(songs) == n):
            break
        else:
            page += 1
        
    print('Found {} songs by {}'.format(len(songs), artist_name))
    return songs

def scrape_lyrics(song_url):
    """
    Scrape the lyrics from a song url

    Parameters
    ----------
    song_url : str
        The url of the song
    
    Returns
    -------
    lyrics : str
        The lyrics of the song
    """

    page = requests.get(song_url)
    html = BeautifulSoup(page.text, 'html.parser')

    # Scrape the song lyrics from the HTML
    try:
        lyrics = html.find("div", class_=re.compile("^lyrics$|Lyrics__Root")).get_text()
    except:
        print('Lyrics not found for {}'.format(song_url))
        return ''
        
    #remove identifiers like chorus, verse, etc
    lyrics = re.sub(r'(\[.*?\])*', '', lyrics)
    lyrics = re.sub('\n{2}', '\n', lyrics)  # Gaps between verses

    # add a space everytime there is a small letter followed by a capital letter (solves problem with new lines adding two words together)
    # does not work with ÆØÅ, possible future improvement
    lyrics = re.sub(r'([a-z])([A-Z])', r'\1 \2', lyrics)
    
    return lyrics

def scrape_songs(artist_name, n):
    """
    Scrape n songs from an artist

    Parameters
    ----------
    artist_name : str
        The name of the artist
    n : int
        The number of songs to scrape

    Returns
    -------
    lyrics : list
        A list of the lyrics of the songs from the artist
    """
    urls = song_urls(artist_name, n)
    lyrics = []

    for url in urls:
        lyrics.append(scrape_lyrics(url))

    return lyrics

def check_lyrics(lyrics: str):
    """
    Checks that language is danish and that the lyrics are not empty

    Parameters
    ----------
    lyrics : str
        The lyrics to check
    
    Returns
    -------
    bool
        True if the lyrics are danish and not empty
    """
    if lyrics != '':
        if detect(lyrics) == 'da':
            return True
    
    return False

def main_scraper(artists, n_songs, save_path):
    """
    Scrapes songs from a list of artists and saves them as as separate text files

    Parameters
    ----------
    artists : list
        A list of artists
    n_songs : int
        The number of songs to scrape
    save_path : str
        The path to save the songs to

    Returns
    -------
    None
    """
    for artist in artists:
        lyrics = scrape_songs(artist, n_songs)
        
        for i, lyric in enumerate(lyrics):
            # check that language is danish and that the lyrics are not empty
            if check_lyrics(lyric):
                # save lyrics as text file
                filename = artist + '_' + str(i) + '.txt'
                with open(os.path.join(save_path, filename), 'w') as f:
                    f.write(lyric)

if __name__ == '__main__':
    # output directory
    path = Path(__file__)
    output_dir = path.parents[1] / 'data' / 'lyrics'

    # ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # artists to scrape
    artists = ["Kim Larsen", "Sanne Salomonsen", "Thomas Helmig", "Lis Sørensen", "Natasja"]

    # number of songs to scrape per artist
    n_songs = 30

    main_scraper(artists, n_songs, output_dir)
