import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

TIMEOUT = 25
RETRIES = 3
BACKOFF = 2.0  # seconds, exponential backoff
BATCH_SIZE = 500
HEADERS = {
    "User-Agent": f"ds6021-project/0.0 (jhb7ek@virginia.edu) python-requests/{requests.__version__}"
}


def parse_pitchfork_review(html):
    mysoup = BeautifulSoup(html, "html.parser")

    # Helper: safely extract element text
    def safe_text(selector, class_name=None, index=0, attr=None):
        try:
            if attr:
                el = mysoup.find(selector, attrs=attr)
            elif class_name:
                el = mysoup.find_all(selector, class_name)[index]
            else:
                el = mysoup.find_all(selector)[index]
            return el.get_text(strip=True)
        except Exception:
            return None

    # Artist & Album
    album = safe_text("h1", attr={"data-testid": "ContentHeaderHed"})
    artist = safe_text("div", "SplitScreenContentHeaderArtist-fyEeJx")

    # Score (float)
    score = safe_text("p", "Rating-iQoWYo")
    score = float(score) if score else None

    # Genre (first info-slice value)
    genre = safe_text("p", "InfoSliceValue-yycxB", 0)

    # Label (find by "Label:" key)
    label = None
    for key_tag in mysoup.find_all("p", "InfoSliceKey-gsmHBp"):
        if key_tag.get_text(strip=True) == "Label:":
            label = key_tag.find_next_sibling("p").get_text(strip=True)
            break

    # Reviewer
    reviewer = None
    try:
        reviewer = (
            mysoup.find("span", attrs={"data-testid": "BylineName"})
            .find("a")
            .get_text(strip=True)
        )
    except Exception:
        pass

    # Year (int)
    year = safe_text(
        "time", attr={"data-testid": "SplitScreenContentHeaderReleaseYear"}
    )
    year = int(year) if year and year.isdigit() else None

    # Review date (datetime)
    review_date = None
    try:
        review_date_str = None
        for key_tag in mysoup.find_all("p", "InfoSliceKey-gsmHBp"):
            if key_tag.get_text(strip=True) == "Reviewed:":
                review_date_str = key_tag.find_next_sibling("p").get_text(strip=True)
                break
        if review_date_str:
            review_date = datetime.strptime(review_date_str, "%B %d, %Y")
    except Exception:
        pass

    # Review length (word count)
    review_length = None
    try:
        ld_json = mysoup.find("script", type="application/ld+json")
        data = json.loads(ld_json.string)
        review_length = len(data["reviewBody"].split())
    except Exception:
        pass

    # Build structured output
    review_data = {
        "artist": artist,
        "album": album,
        "score": score,
        "genre": genre,
        "label": label,
        "reviewer": reviewer,
        "year": year,
        "review_date": review_date,
        "length": review_length,
    }

    return review_data


def fetch_html(session: requests.Session, url):
    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            if 200 <= resp.status_code < 300:
                return resp.text
            # backoff on non-200s as well
        except requests.RequestException:
            pass
        time.sleep(BACKOFF * attempt)
    return None


def main():
    with open("album_urls.json", "r") as f:
        urls = json.load(f)

    rows = []
    failures = []

    last_cp = 0

    Path("out").mkdir(exist_ok=True)

    with requests.Session() as s:
        try:
            for i, url in enumerate(urls, 1):
                html = fetch_html(s, url)
                if not html:
                    failures.append({"url": url, "reason": "fetch_failed"})
                    continue
                try:
                    data = parse_pitchfork_review(html)
                    data["url"] = url  # keep the source URL
                    rows.append(data)
                except Exception as e:
                    failures.append({"url": url, "reason": f"parse_error: {e}"})
                # polite tiny pause to reduce load
                if i % 25 == 0:
                    time.sleep(0.5)

                # Checkpoint of 500 urls outputs to dataframe
                if i % BATCH_SIZE == 0:
                    pd.DataFrame(rows[last_cp:i]).to_csv(
                        f"out/pitchfork_reviews_batch_{i}.csv", index=False
                    )
                    pd.DataFrame(failures).to_csv(
                        f"out/pitchfork_failures_batch_{i}.csv", index=False
                    )
                    last_cp = i
                    print(f"Checkpoint: saved {i} reviews so far.")
        finally:
            df = pd.DataFrame(rows)

            # Save outputs
            df.to_csv("out/pitchfork_reviews.csv", index=False)
            pd.DataFrame(failures).to_csv("out/pitchfork_failures.csv", index=False)
            print(f"Done. {len(df)} succeeded, {len(failures)} failed.")


if __name__ == "__main__":
    main()
