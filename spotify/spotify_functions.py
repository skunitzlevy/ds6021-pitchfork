import json
import requests
import os

url = "https://accounts.spotify.com/api/token"
headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}
data = {
    "grant_type": "client_credentials",
    "client_id": "66f124b9979b4f5ebf458d45baada331",
    "client_secret": "0d69736f761447859ed094b32818b0b0"
}

response = requests.post(url, headers=headers, data=data)
#print(response.json())

# Extract the access token from the response
access_token = response.json().get("access_token")

def get_artist_id(artist_name, access_token):
    '''Get the Spotify artist ID for a given artist name.'''
    url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "q": artist_name,
        "type": "artist",
        "limit": 1
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    if data['artists']['items']:
        return data['artists']['items'][0]['id']
    else:
        return None


# Example usage
# artist_name = "Beck"
# artist_id = get_artist_id(artist_name, access_token)
# print(f"Artist ID for {artist_name}: {artist_id}")

# Get Artist's Albums
def get_artist_albums(artist_id, access_token):
    '''Get a list of albums for a given artist ID.'''
    url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "include_groups": "album,single",
        "limit": 50 # Spotify API max limit is 50
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    return data.get('items', [])


# List the tracks in each album
def get_album_tracks(album_id, access_token):
    '''Get a list of tracks for a given album ID.'''
    url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "limit": 50 # Spotify API max limit is 50
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    return data.get('items', [])


# Get album details including popularity
def get_album_details(album_id, access_token):
    '''Get detailed information for a given album ID.'''
    url = f"https://api.spotify.com/v1/albums/{album_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    return data


def get_related_artists(artist_id, access_token):
    '''Get a list of related artists for a given artist ID.'''
    url = f"https://api.spotify.com/v1/artists/{artist_id}/related-artists"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    return data.get('artists', [])