import pandas as pd
import numpy as np
import warnings
import re
import logging
from transformers import pipeline
from constants import PATHS, Config, Col

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Utility Functions ---
def normalize_text(text: str) -> str:
    """Normalize text for consistency in trend matching."""
    return str(text).lower().strip() if pd.notna(text) else ""

def count_emojis(text: str) -> int:
    """Count number of emojis in a given text."""
    if pd.isna(text): return 0
    return sum(len(match) for match in Config.EMOJI_PATTERN.findall(str(text)))

def extract_username(url: str) -> str:
    """Extract author username from TikTok media URL."""
    if pd.isna(url): return Config.DEFAULT_USERNAME
    match = re.search(Config.USERNAME_REGEX, str(url))
    return match.group(1) if match else Config.DEFAULT_USERNAME

class TikTokDataProcessor:
    def __init__(self):
        # Trend containers
        self.trend_keywords = set()
        self.trend_hashtags = set()
        self.trend_songs = set()
        self.gameshow_hashtags = []
        self.famous_hashtags = [] 
        
        # Initialize PhoBERT Sentiment Analysis
        logging.info("Loading PhoBERT Sentiment Pipeline...")
        self.sentiment_analyzer = pipeline(
            Config.SENTIMENT_TASK, 
            model=Config.PHOBERT_MODEL, 
            tokenizer=Config.PHOBERT_MODEL
        )

    # --- Internal Helper Methods ---
    def _update_set_from_csv(self, filepath: str, columns: list, target_set: set):
        """Helper function to load CSV and update a trend set to avoid code duplication."""
        try:
            df = pd.read_csv(filepath)
            for col in columns:
                if col in df.columns:
                    target_set.update(df[col].dropna().apply(normalize_text).tolist())
        except Exception as e:
            logging.error(f"Failed to load {filepath}: {e}")

    def _check_overlap(self, text: str, reference_set: set) -> int:
        """Check if any item in the reference set exists in the normalized text."""
        if pd.isna(text): return 0
        norm_text = normalize_text(text)
        return 1 if any(item in norm_text for item in reference_set if item) else 0

    def _get_phobert_score(self, text: str) -> float:
        """Calculate sentiment score using PhoBERT (-1 to 1)."""
        if pd.isna(text) or str(text).strip() == "": return 0.0
        try:
            result = self.sentiment_analyzer(str(text), truncation=True, max_length=256)[0]
            label, confidence = result['label'], result['score']
            return float(confidence) if label == Config.LABEL_POS else float(-confidence if label == Config.LABEL_NEG else 0.0)
        except Exception: 
            return 0.0

    # --- Main Pipeline Methods ---
    def load_trends(self):
        """Load and consolidate all trend data from CSV files."""
        logging.info("Loading trend data from files...")
        
        # 1. Load Trends using DRY helper
        self._update_set_from_csv(PATHS["keyword_trend"], Config.TREND_TIME_COLS, self.trend_keywords)
        self._update_set_from_csv(PATHS["hashtag_trend"], Config.TREND_TIME_COLS, self.trend_hashtags)
        self._update_set_from_csv(PATHS["song_trend"], Config.SONG_TREND_COLS, self.trend_songs)

        # 2. Load Gameshow and KOL lists
        try:
            self.gameshow_hashtags = pd.read_csv(PATHS["gameshow"])[Col.HT_GAMESHOW].dropna().astype(str).str.lower().tolist()
            self.famous_hashtags = pd.read_csv(PATHS["kol_to_gameshow"])[Col.HT_FAMOUS].dropna().astype(str).str.lower().tolist()
        except Exception as e:
            logging.error(f"Missing Gameshow/KOL files: {e}")

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline to merge old and new features."""
        df = df.copy()
        logging.info("Extracting basic features...")
        
        # --- Basic Content Features ---
        df[Col.AUTHOR] = df[Col.MEDIA_URL].apply(extract_username)
        df[Col.EMOJI_COUNT] = df[Col.CAPTION].apply(count_emojis)
        df[Col.HT_DENSITY] = df[Col.HASHTAG_COUNT].fillna(0) / df[Col.WORD_COUNT].fillna(1).clip(lower=1)
        
        # --- Trend Matching (Old Features) ---
        logging.info("Matching trends...")
        df[Col.HAS_TREND_KW] = df[Col.CAPTION_CLEAN].apply(lambda x: self._check_overlap(x, self.trend_keywords))
        df[Col.HAS_TREND_HT] = df[Col.HASHTAG_STR].apply(lambda x: self._check_overlap(x, self.trend_hashtags))
        df[Col.HAS_TREND_SONG] = df[Col.MUSIC_NAME].apply(lambda x: 1 if normalize_text(x) in self.trend_songs else 0)

        # --- Special Tags (New Features) ---
        logging.info("Tagging special content...")
        df[Col.IS_ORIG_SOUND] = df[Col.MUSIC_NAME].apply(lambda x: 1 if pd.notna(x) and any(k in str(x).lower() for k in Config.ORIGINAL_SOUND_KWS) else 0)
        df[Col.IS_GAMESHOW] = df[Col.HASHTAG_STR].apply(lambda x: self._check_overlap(x, set(self.gameshow_hashtags)))
        df[Col.COUNT_FAMOUS_HT] = df[Col.HASHTAG_STR].apply(lambda x: sum(1 for t in self.famous_hashtags if t in str(x).lower()) if pd.notna(x) else 0)

        # --- NLP Sentiment (Advanced) ---
        logging.info("Running PhoBERT sentiment analysis...")
        df[Col.SCORE_CAPTION] = df[Col.CAPTION_CLEAN].apply(self._get_phobert_score)

        # --- Time Engineering (Cyclical & Categorical) ---
        logging.info("Processing time variables...")
        df[Col.CREATED_AT] = pd.to_datetime(df[Col.CREATED_AT], utc=True, errors=Config.PANDAS_COERCE)
        df_vn = df[Col.CREATED_AT].dt.tz_convert(Config.TIMEZONE_VN)
        
        df[Col.HOUR] = df_vn.dt.hour 
        df[Col.TIME_SIN] = np.sin(2 * np.pi * df_vn.dt.hour / 24) 
        df[Col.TIME_COS] = np.cos(2 * np.pi * df_vn.dt.hour / 24)
        df[Col.IS_WEEKEND] = (df_vn.dt.dayofweek >= 5).astype(int)
        
        # --- Dynamic Momentum (Grouped by Author) ---
        logging.info("Calculating author momentum...")
        df = df.sort_values(by=[Col.AUTHOR, Col.CREATED_AT])
        
        # Helper variables for grouping
        author_group = df.groupby(Col.AUTHOR)
        
        df[Col.AVG_VIEWS_3] = author_group[Col.VIEWS].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        df[Col.EMA_VIEWS_3] = author_group[Col.VIEWS].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
        df[Col.AVG_LIKES_3] = author_group[Col.LIKES].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        df[Col.EMA_LIKES_3] = author_group[Col.LIKES].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
        df[Col.AVG_SHARES_3] = author_group[Col.SHARES].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        df[Col.EMA_SHARES_3] = author_group[Col.SHARES].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
        df[Col.AVG_COMMENTS_3] = author_group[Col.COMMENTS].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        df[Col.EMA_COMMENTS_3] = author_group[Col.COMMENTS].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
        df[Col.AVG_COLLECTS_3] = author_group[Col.COLLECTS].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        df[Col.EMA_COLLECTS_3] = author_group[Col.COLLECTS].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
        df[Col.LIKE_RATE_TEMP] = df[Col.LIKES] / df[Col.VIEWS].replace(0, 1)
        df[Col.HIST_LIKE_RATE] = author_group[Col.LIKE_RATE_TEMP].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        df[Col.DAYS_SINCE_POST] = author_group[Col.CREATED_AT].diff().dt.total_seconds().fillna(0) / 86400
        
        # Cleanup
        df = df.drop(columns=[Col.LIKE_RATE_TEMP])
        logging.info("Feature extraction complete.")
        
        return df