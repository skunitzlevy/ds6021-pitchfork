#!/usr/bin/env python3
"""
Fetch album count and song count for artists from CSV.

This script loops through artist IDs and fetches:
- Number of albums
- Total number of songs/tracks across all albums

RATE LIMITING:
- Pauses 0.5-1.0 seconds between fetching tracks for different albums
- Pauses 2.0-4.0 seconds between processing different artists
- Implements exponential backoff (60s, 120s, 240s) if rate limited
- These values can be adjusted in the configuration section of main()

FEATURES:
- Automatic token refresh every 30 minutes
- Progress saved every 20 artists (automatic backup to artist_counts_progress.csv)
- Automatic resume from last saved artist if interrupted or error occurs
- Validates artist IDs before processing
- Handles errors gracefully with retry logic
- Continues processing even if individual artists fail
"""

import os
import sys
import pandas as pd
import time
import random
from tqdm import tqdm
from spotify_functions import (
    get_access_token,
    get_artist_albums,
    get_album_tracks
)


def get_album_and_song_count(artist_id, access_token, min_album_pause=0.5, max_album_pause=1.0):
    """
    Get the number of albums and total songs for an artist.

    Args:
        artist_id: Spotify artist ID
        access_token: Spotify API access token
        min_album_pause: Minimum pause between album requests (seconds)
        max_album_pause: Maximum pause between album requests (seconds)

    Returns:
        tuple: (album_count, total_songs)
    """
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Get all albums for the artist
            albums = get_artist_albums(artist_id, access_token)
            album_count = len(albums)
            total_songs = 0

            # Count tracks across all albums
            for album in albums:
                album_id = album['id']
                # Randomized pause between album requests to avoid rate limits
                pause_time = random.uniform(min_album_pause, max_album_pause)
                time.sleep(pause_time)

                tracks = get_album_tracks(album_id, access_token)
                total_songs += len(tracks)

            return album_count, total_songs

        except RuntimeError as e:
            error_msg = str(e)
            # Handle rate limiting with exponential backoff
            if "429" in error_msg or "Rate limited" in error_msg:
                retry_count += 1
                wait_time = 30 * (2 ** retry_count)  # Exponential backoff: 60s, 120s, 240s
                print(f"\n⚠️  Rate limited! Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"Error fetching data for artist {artist_id}: {e}")
                return None, None
        except Exception as e:
            print(f"Error fetching data for artist {artist_id}: {e}")
            return None, None

    print(f"Failed after {max_retries} retries for artist {artist_id}")
    return None, None


def main():
    # ============================================================================
    # RATE LIMITING CONFIGURATION - Adjust these to avoid API limits
    # ============================================================================
    MIN_PAUSE_BETWEEN_ALBUMS = 0.5    # Minimum pause between album requests (seconds)
    MAX_PAUSE_BETWEEN_ALBUMS = 1.0    # Maximum pause between album requests (seconds)
    MIN_PAUSE_BETWEEN_ARTISTS = 2.0   # Minimum pause between artists (seconds)
    MAX_PAUSE_BETWEEN_ARTISTS = 4.0   # Maximum pause between artists (seconds)

    # Progress Configuration
    SAVE_INTERVAL = 20  # Save progress every N artists (reduced from 50 for better backup)

    # File Configuration
    INPUT_CSV = 'artist_ids.csv'  # Input file with artist_id column
    OUTPUT_CSV = 'artist_album_song_counts.csv'  # Output file

    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found!")
        print("Please make sure the file exists in the current directory.")
        sys.exit(1)

    # Load CSV
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Validate columns
    if 'artist_id' not in df.columns:
        print("Error: CSV must have 'artist_id' column")
        sys.exit(1)

    # Filter out invalid artist IDs
    df = df.dropna(subset=['artist_id'])
    df = df[df['artist_id'].astype(str).str.len() == 22]
    print(f"Found {len(df)} valid artists to process")

    # Check for progress file (resume capability)
    progress_file = 'artist_counts_progress.csv'
    processed_ids = set()
    results = []

    if os.path.exists(progress_file):
        print(f"\n Found progress file - resuming from previous run")
        existing_df = pd.read_csv(progress_file)
        processed_ids = set(existing_df['artist_id'].values)
        results = existing_df.to_dict('records')
        print(f"  Already processed: {len(processed_ids)} artists")

    # Get Spotify access token
    print("\nAuthenticating with Spotify API...")
    access_token = get_access_token()
    token_start_time = time.time()
    print(" Authentication successful\n")

    # Statistics
    success_count = 0
    failed_count = 0
    artists_processed_this_session = 0

    print("Processing artists...")
    print("This will take a while as we fetch album and track data for each artist.")
    print("-" * 70)

    # Create progress bar with custom format
    pbar = tqdm(total=len(df), desc="Processing",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    # Process each artist
    for idx, row in df.iterrows():
        artist_id = str(row['artist_id']).strip()

        # Skip if already processed
        if artist_id in processed_ids:
            pbar.update(1)
            continue

        # Refresh token every 30 minutes
        if time.time() - token_start_time > 1800:  # 30 minutes
            print("\n= Refreshing access token...")
            access_token = get_access_token()
            token_start_time = time.time()
            print(" Token refreshed")

        artist_name = row.get('artist', 'Unknown')

        # Update progress bar to show current artist
        pbar.set_description(f"Processing: {artist_name[:35]:<35}")

        # Wrap in try-except to ensure we continue even if one artist fails
        try:
            # Get album and song counts with configured rate limits
            album_count, song_count = get_album_and_song_count(
                artist_id,
                access_token,
                MIN_PAUSE_BETWEEN_ALBUMS,
                MAX_PAUSE_BETWEEN_ALBUMS
            )

            # Store result
            result = {
                'artist_id': artist_id,
                'artist': artist_name if 'artist' in df.columns else None,
                'album_count': album_count,
                'song_count': song_count
            }
            results.append(result)
            processed_ids.add(artist_id)

            if album_count is not None:
                success_count += 1
            else:
                failed_count += 1

        except Exception as e:
            # If any error occurs, save None and continue
            tqdm.write(f"\n⚠️  Unexpected error for {artist_name}: {e}")
            result = {
                'artist_id': artist_id,
                'artist': artist_name if 'artist' in df.columns else None,
                'album_count': None,
                'song_count': None
            }
            results.append(result)
            processed_ids.add(artist_id)
            failed_count += 1

        # Update progress bar
        pbar.update(1)

        # Save progress every SAVE_INTERVAL artists (automatic backup)
        if len(results) % SAVE_INTERVAL == 0 and len(results) > 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(progress_file, index=False)
            tqdm.write(f"💾 Progress saved: {len(results)} artists processed (Success: {success_count}, Failed: {failed_count})")

        # Rate limiting - pause between artists
        # Longer pause since we make multiple API calls per artist
        if idx < len(df) - 1:
            pause = random.uniform(MIN_PAUSE_BETWEEN_ARTISTS, MAX_PAUSE_BETWEEN_ARTISTS)
            time.sleep(pause)

    # Close progress bar
    pbar.close()

    # Save final results
    print(f"\n{'=' * 70}")
    print("Processing complete!")
    print(f"\nStatistics:")
    print(f"  Total processed: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Results saved to: {OUTPUT_CSV}")

    # Show sample results
    print(f"\nSample results:")
    print(results_df.head(10).to_string(index=False))

    # Clean up progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"\n Progress file cleaned up")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n�  Process interrupted by user")
        print("Progress has been saved. Run the script again to resume.")
        sys.exit(0)
    except Exception as e:
        print(f"\nL Error: {e}")
        sys.exit(1)