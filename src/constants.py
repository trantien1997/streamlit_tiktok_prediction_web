import re
from pathlib import Path

# Define the project's root directory (assuming this script is inside a subdirectory)
# Path(__file__).resolve().parent.parent automatically moves up one directory level.
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    """Management of all constant values and configurations."""
    # Models
    PHOBERT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"
    SENTIMENT_TASK = "sentiment-analysis"
    LABEL_POS = "POS"
    LABEL_NEG = "NEG"
    
    # Regex & Matching
    USERNAME_REGEX = r'video-(.*?)-\d{14}'
    DEFAULT_USERNAME = "unknown"
    ORIGINAL_SOUND_KWS = ["nhạc nền -", "original sound"]
    EMOJI_PATTERN = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
        r"\U00002600-\U000026FF]+", 
        flags=re.UNICODE
    )
    
    # Trend columns
    TREND_TIME_COLS = ["last_7_days", "last_30_days", "last_120_days"]
    SONG_TREND_COLS = ["song_last_7_days", "song_last_30_days", "song_last_120_days"]

    # Time & Settings
    TIMEZONE_VN = "Asia/Ho_Chi_Minh"
    PANDAS_COERCE = "coerce"

class Col:
    """Management of all column names for input and output data"""
    # --- Input columns ---
    MEDIA_URL = "media_url"
    CAPTION = "caption"
    CAPTION_CLEAN = "caption_clean"
    CAPTION_LEN = "caption_length"
    HASHTAG_COUNT = "hashtag_count"
    WORD_COUNT = "word_count"
    HASHTAG_STR = "hashtag_str"
    MUSIC_NAME = "music_name"
    CREATED_AT = "created_at"
    VIEWS = "views"
    LIKES = "likes"
    SHARES = "shares"
    COMMENTS = "comments"
    COLLECTS = "collects"
    FOLLOWERS = "followers"
    HT_GAMESHOW = "hashtag_gameshow"
    HT_FAMOUS = "hashtag_famous"

    # --- Generated columns ---
    AUTHOR = "author_username"
    EMOJI_COUNT = "emoji_count"
    HT_DENSITY = "hashtag_density"
    HAS_TREND_KW = "has_trend_keyword"
    HAS_TREND_HT = "has_trend_hashtag"
    HAS_TREND_SONG = "has_trend_song"
    IS_ORIG_SOUND = "is_original_sound"
    IS_GAMESHOW = "is_related_gameshow"
    COUNT_FAMOUS_HT = "count_hashtag_famous"
    SCORE_CAPTION = "score_caption"
    
    # --- Time columns ---
    HOUR = "hour"
    TIME_SIN = "time_sin"
    TIME_COS = "time_cos"
    IS_WEEKEND = "is_weekend"
    
    # --- Momentum columns ---
    AVG_VIEWS_3 = "avg_views_last_3_videos"
    EMA_VIEWS_3 = "ema_views_last_3"
    AVG_LIKES_3 = "avg_like_last_3_videos"
    EMA_LIKES_3 = "ema_like_last_3"
    AVG_SHARES_3 = "avg_shares_last_3_videos"
    EMA_SHARES_3 = "ema_shares_last_3"
    AVG_COMMENTS_3 = "avg_comments_last_3_videos"
    EMA_COMMENTS_3 = "ema_comments_last_3"
    AVG_COLLECTS_3 = "avg_collects_last_3_videos"
    EMA_COLLECTS_3 = "ema_collects_last_3"
    LIKE_RATE_TEMP = "like_rate_temp"
    HIST_LIKE_RATE = "hist_like_rate"
    DAYS_SINCE_POST = "days_since_last_post"

    # --- Target columns ---        
    TARGET_LIKES = "likes_log1p"
    TARGET_VIEWS = "views_log1p"
    TARGET_SHARES = "shares_log1p"


# Use pathlib.Path for cross-platform compatibility and safe path joining
PATHS = {
    "main_data": BASE_DIR / "DB_tiktok" / "TikTok_preprocessed_final.csv",
    "keyword_trend": BASE_DIR / "DB_Trend_keywords_tiktok" / "tiktok_keyword_insights_vn_rank_keyword_7_30_120_21-04-2026.csv",
    "song_trend": BASE_DIR / "DB_trend_song_tiktok" / "songs_rank_7_30_120_12-01-2026.csv",
    "hashtag_trend": BASE_DIR / "DB_trend_hastag_tiktok" / "hashtags_rank_7_30_120_21-04-2026.csv",
    "gameshow": BASE_DIR / "Gameshow" / "Data_gameshow.csv",
    "kol_to_gameshow": BASE_DIR / "Gameshow" / "kol_to_gameshows.csv",
    "output_train": BASE_DIR / "DB_tiktok" / "Tiktok_train.csv",
    "output_val": BASE_DIR / "DB_tiktok" / "Tiktok_validate.csv",
    "base_dir_models": BASE_DIR / "models",
    "output_model_linear_regression": BASE_DIR / "models" / "tiktok_linear_regression_multi.pkl",
    "output_model_random_forest": BASE_DIR / "models" / "tiktok_random_forest_multi.pkl",
    "output_model_xgboost": BASE_DIR / "models" / "tiktok_xgboost_multi.pkl",
    "output_result_xgboost": BASE_DIR / "Result" / "Result_xgboost.csv",
    "output_result_random_forest": BASE_DIR / "Result" / "Result_random_forest.csv",
    "output_result_linear_regression": BASE_DIR / "Result" / "Result_linear_regression.csv",
    "output_feature_importance_lr": BASE_DIR / "Result" / "Feature_Importance_Linear_Regression.png",
    "output_feature_importance_rf": BASE_DIR / "Result" / "Feature_Importance_Random_Forest.png",
    "output_feature_importance_xgb": BASE_DIR / "Result" / "Feature_Importance_XGBoost.png"
}

# Reference variables from the Col class to define Features and Targets dynamically
FEATURES = [
    Col.FOLLOWERS,
    Col.EMA_VIEWS_3,
    Col.AVG_VIEWS_3,
    Col.EMA_LIKES_3,
    Col.AVG_LIKES_3,
    Col.EMA_SHARES_3,
    Col.AVG_SHARES_3,
    Col.EMA_COMMENTS_3,
    Col.AVG_COMMENTS_3,
    Col.EMA_COLLECTS_3,
    Col.AVG_COLLECTS_3,
    Col.HIST_LIKE_RATE,
    Col.DAYS_SINCE_POST,
    Col.IS_GAMESHOW,
    Col.COUNT_FAMOUS_HT, 
    Col.IS_ORIG_SOUND,
    Col.HASHTAG_COUNT,
    Col.EMOJI_COUNT,
    Col.HT_DENSITY, 
    Col.SCORE_CAPTION,
    Col.TIME_SIN,
    Col.TIME_COS,
    Col.IS_WEEKEND,
    Col.CAPTION_LEN,
    Col.WORD_COUNT,
    Col.HAS_TREND_KW,
    Col.HAS_TREND_HT, 
    Col.HAS_TREND_SONG,
    Col.HOUR
]

TARGETS = [
    Col.TARGET_LIKES,
    Col.TARGET_VIEWS,
    Col.TARGET_SHARES
]
