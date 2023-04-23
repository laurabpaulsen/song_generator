from bs4 import BeautifulSoup
import requests
import re
import os
from pathlib import Path
from langdetect import detect

def get_json(path, genius_token):
    '''Send request and get response in json format.'''

    # Generate request URL
    requrl = '/'.join(["www.api.genius.com", path])
    token = "Bearer {}".format(genius_token)
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
def request_artist_info(artist_name, page, genius_token):
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
    headers = {'Authorization': 'Bearer ' + genius_token}
    search_url = base_url + '/search?per_page=10&page=' + str(page)
    data = {'q': artist_name}
    response = requests.get(search_url, data=data, headers=headers)
    
    return response


def song_urls(artist_name: str, genius_token: str, n: int = 10):
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
        response = request_artist_info(artist_name, page, genius_token)
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
       
       # Search through a maximum of 20 pages 
        if page > 20:
            break
        
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

def scrape_songs(artist_name, genius_token, n):
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
    urls = song_urls(artist_name, genius_token, n)
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

def clean_lyrics(lyrics:list):
    for i, lyric in enumerate(lyrics):
        # Remove everything before the first time it says "Lyrics" (title of the song, contributor, etc.)
        start = lyric.find("Lyrics")+7
        # Remove suggestions at the end
        stop = lyric.find("You might also like")
        
        lyrics[i] = lyric[start:stop]

    return lyrics

def main_scraper(artists, n_songs, save_path, genius_token):
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
        lyrics = scrape_songs(artist, genius_token, n_songs)
        lyrics = clean_lyrics(lyrics)
        
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

    # get token from txt
    with open(path.parents[1] / "TOKEN.txt") as f:
        genius_token = f.read()


    # artists to scrape
    artists = [
        "Specktors", "Gilli", "Flødeklinikken", "Klumben", "Rent mel",
        "PIND", "UNG-SKAB", "BJØRN", "JOSVA", "KIDD", "Jimilian", "KATO",
        "Cyd Williams",  "Franske piger", "EaggerStunn", "Danser med piger", "Hong Kong"
        "Uro", "Mekdes", "Olivver", "Malte Ebert", "Thor Farlov",  "Xander Linnet",
        "Mas", "knægt", "Marcus.wav", "Angående mig", "FRAADS", "Fastpoholmen", "Iiris",
        "Malk De Koijn", "Jooks", "Den gale pose", "Per Vers","UFO yepha", "Johnson", "USO",
        "Andreas Odbjerg", "Kim Larsen", "Sanne Salomonsen", "Thomas Helmig", "Lis Sørensen", 
        "The Minds Of 99", "Medina", "Burhan G", "Peter Sommer", "Katinka Band", "Tessa", 
        "De Danske Hyrder", "Thomas Buttenschøn", "Rigmor", "Marie Key", "Nephew", 
        "Magtens Korridorer", "Bogfinkevej", "Tobias Rahim",  "Ukendt Kunstner",
        "Guldimund", "Ulige numre", "Gulddreng", "Pharfar",  "Natasja", "Iomfro",
        "Benjamin Hav", "Benal", "KESI", "Rasmus Seebach", "Søren Huss", "Barselona",
        "Carl Emil Petersen", "Karl William", "Danseorkesteret", "Kim Larsen", "Hjalmer",  
        "Folkeklubben", "Sys Bjerre", "C.V. Jørgensen", "Mads Langer", "Shaka Loveless",
        "Emil Kruse", "TopGunn", "Citybois", "Joey Moe",  "Jokeren", "Blak", "Hans Philip",
        "Anne Linnet", "Lars Lilholt Band", "Shu-Bi-Dua", "Tim Christensen", "Lizzie", 
        "Hej Matematik", "Medina", "Anna David",  "Johnny Deluxe", "Rasmus Walter",
        "Gnags", "John Mogensen", "Suspekt", "guldimund", "Gasolin",  "Tøsedrengene", 
        "Poul Krebs", "Blæst",  "Tue West", "Suspekt", "Szhirley", "Brødrene Olsen",
        "Ray Dee Ohh", "Back to Back", "Dodo & the Dodos",  "Bamses Venner", "Tørfisk",
        "Jakob Sveistrup", "Rocazino", "Danser med Drenge", "Laban", "Tommy Seebach",
        "Panamah", "Tobias Rahim", "Birthe Kjær", "Jung", "Zar Paulo", "PATINA", "Joyce", 
        "Kalaset", "Statisk", "Pauline", "Dusin", "Liss", "Kind mod kind", "Annika Aakjær", 
        "Lars H.U.G.", "Sort sol", "Juncker","Undertekst", "Love shop", "Claus Hempler",
        "Pil", "Simon Kvamm", "Niels Brandt", "Ussel", "Hugorm" , "Artigeardit", "Lamin",
        "Lord Siva", "Sivas", "Emil Stabil", "ude af kontrol", "Page Four", "Gulddreng",
        "Vild Smith", "Molo", "Soleima", "Bisse", "De eneste to", "Mikael Simpson", 
        "Hymns from Nineveh", "Rasmus Nøhr", "Big Fat Snake", "Caroline Henderson", "Søs Fenger",
        "Stig Rossen", "Niarn", "Østkyst Hustlers",  "Jøden", "Clemens", "Raske Penge"
        ]

    # number of songs to scrape per artist
    n_songs = 20

    main_scraper(artists, n_songs, output_dir, genius_token)

