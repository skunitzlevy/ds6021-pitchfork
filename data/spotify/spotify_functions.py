import json
import requests
import os
import pandas as pd
from dotenv import load_dotenv
import time
import random
from tqdm import tqdm

load_dotenv()

# Get new get_access_token every 30 minutes to avoid timeout
def new_access_token():
    for _minutes in range(30, 181, 30): # Refresh every 30 minutes up to 3 hours
        time.sleep(_minutes * 60) # Sleep for _minutes
        global access_token
        access_token = get_access_token()
        # New access token refereshed

def get_access_token():
    '''Get a fresh Spotify access token.'''
    url = "https://accounts.spotify.com/api/token"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "client_credentials",
        "client_id": os.getenv("SPOTIFY_CLIENT_ID"),
        "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET")
    }

    response = requests.post(url, headers=headers, data=data)
    return response.json().get("access_token")

# Get initial access token
access_token = get_access_token()

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
    response = requests.get(url, headers=headers, params=params, timeout=10)

    # Check HTTP status code
    if not response.ok:
        if response.status_code == 401:
            raise RuntimeError("Unauthorized: access token may be invalid or expired")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RuntimeError(f"Rate limited (HTTP 429). Retry-After: {retry_after} seconds")
        else:
            raise RuntimeError(f"HTTP {response.status_code} error: {response.text[:200]}")

    # Check if response body is empty
    if not response.text or not response.text.strip():
        return None

    # Parse JSON safely
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        snippet = response.text[:200] if len(response.text) > 200 else response.text
        raise RuntimeError(f"Failed to parse JSON: {e}. Response: {snippet}")

    if data.get('artists', {}).get('items'):
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
    response = requests.get(url, headers=headers, params=params, timeout=10)

    # Check HTTP status code
    if not response.ok:
        if response.status_code == 401:
            raise RuntimeError("Unauthorized: access token may be invalid or expired")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RuntimeError(f"Rate limited (HTTP 429). Retry-After: {retry_after} seconds")
        else:
            raise RuntimeError(f"HTTP {response.status_code} error: {response.text[:200]}")

    # Check if response body is empty
    if not response.text or not response.text.strip():
        return []

    # Parse JSON safely
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        snippet = response.text[:200] if len(response.text) > 200 else response.text
        raise RuntimeError(f"Failed to parse JSON: {e}. Response: {snippet}")

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
    response = requests.get(url, headers=headers, params=params, timeout=10)

    # Check HTTP status code
    if not response.ok:
        if response.status_code == 401:
            raise RuntimeError("Unauthorized: access token may be invalid or expired")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RuntimeError(f"Rate limited (HTTP 429). Retry-After: {retry_after} seconds")
        else:
            raise RuntimeError(f"HTTP {response.status_code} error: {response.text[:200]}")

    # Check if response body is empty
    if not response.text or not response.text.strip():
        return []

    # Parse JSON safely
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        snippet = response.text[:200] if len(response.text) > 200 else response.text
        raise RuntimeError(f"Failed to parse JSON: {e}. Response: {snippet}")

    return data.get('items', [])


# Get album details including popularity
def get_album_details(album_id, access_token):
    '''Get detailed information for a given album ID.'''
    url = f"https://api.spotify.com/v1/albums/{album_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers, timeout=10)

    # Check HTTP status code
    if not response.ok:
        if response.status_code == 401:
            raise RuntimeError("Unauthorized: access token may be invalid or expired")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RuntimeError(f"Rate limited (HTTP 429). Retry-After: {retry_after} seconds")
        else:
            raise RuntimeError(f"HTTP {response.status_code} error: {response.text[:200]}")

    # Check if response body is empty
    if not response.text or not response.text.strip():
        return {}

    # Parse JSON safely
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        snippet = response.text[:200] if len(response.text) > 200 else response.text
        raise RuntimeError(f"Failed to parse JSON: {e}. Response: {snippet}")

    return data


def get_related_artists(artist_id, access_token):
    '''Get a list of related artists for a given artist ID.'''
    url = f"https://api.spotify.com/v1/artists/{artist_id}/related-artists"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers, timeout=10)

    # Check HTTP status code
    if not response.ok:
        if response.status_code == 401:
            raise RuntimeError("Unauthorized: access token may be invalid or expired")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RuntimeError(f"Rate limited (HTTP 429). Retry-After: {retry_after} seconds")
        else:
            raise RuntimeError(f"HTTP {response.status_code} error: {response.text[:200]}")

    # Check if response body is empty
    if not response.text or not response.text.strip():
        return []

    # Parse JSON safely
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        snippet = response.text[:200] if len(response.text) > 200 else response.text
        raise RuntimeError(f"Failed to parse JSON: {e}. Response: {snippet}")

    return data.get('artists', [])

# Heywood was here

def get_artist_info(artist_id, access_token):
    '''Get detailed information for a given artist ID.'''
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers, timeout=10)

    # Check HTTP status code
    if not response.ok:
        if response.status_code == 401:
            raise RuntimeError("Unauthorized: access token may be invalid or expired")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RuntimeError(f"Rate limited (HTTP 429). Retry-After: {retry_after} seconds")
        else:
            raise RuntimeError(f"HTTP {response.status_code} error: {response.text[:200]}")

    # Check if response body is empty
    if not response.text or not response.text.strip():
        return pd.DataFrame()

    # Parse JSON safely
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        snippet = response.text[:200] if len(response.text) > 200 else response.text
        raise RuntimeError(f"Failed to parse JSON: {e}. Response: {snippet}")

    # Save to a dataframe
    artist_info_df = pd.DataFrame([data])
    return artist_info_df


def get_artist_followers(artist_id, access_token):
    '''Get the number of followers for a given artist ID.'''
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers, timeout=10)

    # Check HTTP status code
    if not response.ok:
        if response.status_code == 401:
            raise RuntimeError("Unauthorized: access token may be invalid or expired")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RuntimeError(f"Rate limited (HTTP 429). Retry-After: {retry_after} seconds")
        else:
            raise RuntimeError(f"HTTP {response.status_code} error: {response.text[:200]}")

    # Check if response body is empty
    if not response.text or not response.text.strip():
        return None

    # Parse JSON safely
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        snippet = response.text[:200] if len(response.text) > 200 else response.text
        raise RuntimeError(f"Failed to parse JSON: {e}. Response: {snippet}")

    return data.get('followers', {}).get('total', None)


def get_all_artists_followers(csv_path, min_pause=0.5, max_pause=2.0, token_refresh_minutes=45):
    '''
    Get the number of followers for each artist in the CSV file.
    Automatically refreshes access token every 45 minutes to prevent timeout.

    Args:
        csv_path: Path to the CSV file containing artist names and IDs
        min_pause: Minimum pause time between requests in seconds (default: 0.5)
        max_pause: Maximum pause time between requests in seconds (default: 2.0)
        token_refresh_minutes: Minutes before refreshing access token (default: 45)

    Returns:
        DataFrame with artist names, IDs, and follower counts
    '''
    # Read the CSV file
    artists_df = pd.read_csv(csv_path)

    # Initialize list to store results
    followers_data = []

    # Get initial access token and track time
    access_token = get_access_token()
    token_start_time = time.time()
    token_refresh_seconds = token_refresh_minutes * 60

    # Iterate through each artist with progress bar
    for idx, row in tqdm(artists_df.iterrows(), total=len(artists_df), desc="Fetching followers"):
        # Check if we need to refresh the access token
        if time.time() - token_start_time > token_refresh_seconds:
            print(f"\nRefreshing access token after {token_refresh_minutes} minutes...")
            access_token = get_access_token()
            token_start_time = time.time()
            print("Access token refreshed successfully!")

        artist_name = row['artist']
        artist_id = row['artist_id']

        # Get follower count
        try:
            followers = get_artist_followers(artist_id, access_token)
            followers_data.append({
                'artist': artist_name,
                'artist_id': artist_id,
                'followers': followers
            })
        except Exception as e:
            print(f"Error fetching followers for {artist_name}: {e}")
            followers_data.append({
                'artist': artist_name,
                'artist_id': artist_id,
                'followers': None
            })

        # Random pause between requests (skip on last iteration)
        if idx < len(artists_df) - 1:
            pause_time = random.uniform(min_pause, max_pause)
            time.sleep(pause_time)

    # Create DataFrame from results
    followers_df = pd.DataFrame(followers_data)

    return followers_df


def get_album_count(artist_id):
    '''Get the number of albums for a given artist ID.'''
    albums = get_artist_albums(artist_id, access_token)
    # Count unique albums for each artist (by name)
    album_count = len(albums)
    pd.DataFrame

    return len(albums)
