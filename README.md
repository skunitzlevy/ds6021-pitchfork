# Pitchfork Review Score Prediction

## Overview

Our project analyzed Pitchfork music reviews to understand what factors influence album ratings. By combining web-scraped Pitchfork review data with Spotify artist metrics, we built and evaluated multiple machine learning models to predict review scores and classify music genres based on bias music reviews.

**Research Question:** What influences music critics' ratings? Can we predict Pitchfork scores based on artist popularity, genre, album metadata, and reviewer tendencies?

## Team Members

- Rameez Ali (mus8hp@virginia.edu)
- Sam Kunitz-Levy (jhb7ek@virginia.edu)
- Finn Sjue (afc9jk@virginia.edu)
- Mauricio Torres (kbj2xn@virginia.edu)
- Heywood Williams-Tracy (mtk9va@virginia.edu)

## Data Sources

### Pitchfork Reviews
- ~8,900 album reviews web-scraped from Pitchfork.com
- Fields: artist, album, score (0-10), genre, label, reviewer, publication date, release year

### Spotify API
- ~5,200 unique artists matched to Pitchfork reviews
- Artist follower counts
- Track information
- Used to enrich Pitchfork data with popularity metrics

## Technologies

- **Language:** Python 
- **Data Engineering & Machine Learning:** pandas, numpy, scikit-learn
- **Visualization:** plotly, Dash
- **Web Scraping:** BeautifulSoup, requests
- **API Integration:** Spotify Web API, python-dotenv

## Media Coverage

The project was featured by the University of Virginia's School of Data Science and on Instagram:
- [University of Virginia, School of Data Science (Biased Music Reviews Give AI an Edge in Genre Prediction)](https://datascience.virginia.edu/news/biased-music-reviews-give-ai-edge-genre-prediction)
- [@uvadatascience - Happy Spotify Wrapped Day Instagram Reel](https://www.instagram.com/reel/DRzo9G3kTYQ/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA%3D%3D)

## Project Structure

```
ds6021-pitchfork/
├── Dashboard/             # Interactive Dashboard
│   ├── app.py             # Main dashboard interface
│   ├── linreg.py          # Linear regression model
│   ├── KNN.py             # K-Nearest Neighbors classifier
│   ├── kmeans.py          # K-Means clustering
│   ├── pca_model.py       # Principal Component Analysis
│   ├── elastic.py         # Elastic Net regression
│   ├── spline.py          # Spline regression
│   └── summary_charts.py  # Visualization utilities
├── data/
│   ├── pitchfork/         # Pitchfork scraping notebooks
│   ├── spotify/           # Spotify API scripts
│   ├── clean/             # Cleaned datasets (Cleaned_Data.csv)
│   └── preliminary/       # Raw and intermediate data
├── notebooks/             # Exploratory analysis notebooks
├── Visualizations/        # Static visualization outputs
├── models.ipynb           # Comprehensive modeling notebook
└── requirements.txt       # Python dependencies
```

## Installation & Setup

### Dependencies

```bash
pip install -r requirements.txt
```

### Spotify API Credentials

1. Create a Spotify Developer account and register an application
2. Create a `.env` file in the project root:
   ```
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   ```

### Running the Dashboard

```bash
python Dashboard/app.py
```

## Models

### Supervised Learning (Predicting Review Scores)
- **Linear Regression** - Baseline linear relationship modeling
- **Elastic Net** - Regularized regression (L1 + L2 penalties)
- **Spline Regression** - Non-parametric fitting with knots
- **K-Nearest Neighbors** - Genre classification

### Unsupervised Learning
- **Principal Component Analysis (PCA)** - Dimensionality reduction and feature analysis
- **K-Means Clustering** - Genre clustering and latent feature extraction

## Key Features

### Numerical Features
- score, album_year, length, followers_count
- reviewer_reviews, artist_reviews, genre_len
- review_release_difference

### Categorical Features
- main_genre, label, reviewer

### Transformations
- Log transformations for right-skewed distributions
- Square root transformations
- Feature standardization

## Results / Findings

- Albums reviewed further in the future received higher scores
- Lesser-known indie artists consistently got higher ratings than popular artists
- Review scores dropped noticeably from 2016 onwards (when their dataset began)
- Pitchfork shows preference for niche genres, possibly to identify trends early or generate attention

## Visualizations

The `Visualizations/` folder contains static charts:
- `pitchfork_score_distribution.png` - Distribution of review scores
- `pitchfork_followers_by_artist.png` - Follower counts analysis
- `pitchfork_album_age.png` - Album age distribution
- `pitchfork_review_counts.png` - Review frequency analysis

The interactive dashboard provides real-time model exploration and parameter tuning.

## Usage

### Dashboard

**https://ds6021-pitchfork.onrender.com/**

The Dash application allows you to:
- Explore different model types
- Adjust hyperparameters
- Visualize feature relationships
- Compare model performance

### Reproducing Analysis
1. Run notebooks in `data/pitchfork/` to scrape review data
2. Run scripts in `data/spotify/` to fetch artist metrics
3. Use notebooks in `data/clean/code/` to merge and clean data
4. Explore analysis in `notebooks/` or run `models.ipynb`

## References

- [University of Virginia, School of Data Science (Biased Music Reviews Give AI an Edge in Genre Prediction)](https://datascience.virginia.edu/news/biased-music-reviews-give-ai-edge-genre-prediction)
- [@uvadatascience - Happy Spotify Wrapped Day Instagram Reel](https://www.instagram.com/reel/DRzo9G3kTYQ/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA%3D%3D)
- [Pitchfork](https://pitchfork.com/)
- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api/)
