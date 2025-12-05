from spotify_functions import *

def get_album_counts(artist_id, access_token):
    '''Get the number of albums and total tracks for a given artist ID.'''
    albums = get_artist_albums(artist_id, access_token)
    album_count = len(albums)
    total_tracks = 0

    for album in albums:
        album_id = album['id']
        tracks = get_album_tracks(album_id, access_token)
        total_tracks += len(tracks)

    return album_count, total_tracks


# Example usage:
'''
from spotify_functions import *
from get_album_counts import *

artist_name = "Beck"
artist_id = get_artist_id(artist_name, access_token)
if artist_id:
    album_count, total_tracks = get_album_counts(artist_id, access_token)
    print(f"Artist: {artist_name}")
    print(f"Number of Albums: {album_count}")
    print(f"Total Number of Tracks: {total_tracks}")
else:
    print(f"Artist {artist_name} not found.")
'''