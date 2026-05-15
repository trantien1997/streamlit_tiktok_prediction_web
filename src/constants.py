import re
import pytz
from pathlib import Path

# Thư mục gốc của project
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
    ORIGINAL_SOUND_KWS = ["nhạc nền -", "original sound", "original"]
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
    TIMEZONE_VN = pytz.timezone("Asia/Ho_Chi_Minh")
    PANDAS_COERCE = "coerce"

class Col:
    """Management of all column names for input and output data"""
    POST_ID = "post_id"
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
    
    HOUR = "hour"
    TIME_SIN = "time_sin"
    TIME_COS = "time_cos"
    IS_WEEKEND = "is_weekend"
    
    AVG_VIEWS_3 = "avg_views_last_3_videos"
    EMA_VIEWS_3 = "ema_views_last_3"
    LIKE_RATE_TEMP = "like_rate_temp"
    HIST_LIKE_RATE = "hist_like_rate"
    DAYS_SINCE_POST = "days_since_last_post"

    TARGET_LIKES = "likes_log1p"
    TARGET_VIEWS = "views_log1p"
    TARGET_SHARES = "shares_log1p"

# Đường dẫn tự động tương thích đa nền tảng
print ("DEBUG: BASE_DIR =", BASE_DIR)
PATHS = {
    "keyword_trend": BASE_DIR / "DB_Trend_keywords_tiktok" / "tiktok_keyword_insights_vn_rank_keyword_7_30_120_21-04-2026.csv",
    "song_trend": BASE_DIR / "DB_trend_song_tiktok" / "songs_rank_7_30_120_12-01-2026.csv",
    "hashtag_trend": BASE_DIR / "DB_trend_hastag_tiktok" / "hashtags_rank_7_30_120_21-04-2026.csv",
    "gameshow": BASE_DIR / "Gameshow" / "Data_gameshow.csv",
    "kol_to_gameshow": BASE_DIR / "Gameshow" / "kol_to_gameshows.csv",
    "models": {
        "XGBoost": BASE_DIR / "models" / "tiktok_xgboost_multi.pkl",
        "Random Forest": BASE_DIR / "models" / "tiktok_random_forest_multi.pkl",
        "Linear Regression": BASE_DIR / "models" / "tiktok_linear_regression_multi.pkl"
    }
}

# Đảm bảo danh sách FEATURES khớp hoàn toàn với lúc bạn train model
FEATURES = [
    Col.FOLLOWERS, Col.EMA_VIEWS_3, Col.AVG_VIEWS_3, Col.HIST_LIKE_RATE,
    Col.DAYS_SINCE_POST, Col.IS_GAMESHOW, Col.COUNT_FAMOUS_HT, Col.IS_ORIG_SOUND,
    Col.HASHTAG_COUNT, Col.EMOJI_COUNT, Col.HT_DENSITY, Col.SCORE_CAPTION,
    Col.TIME_SIN, Col.TIME_COS, Col.IS_WEEKEND, Col.CAPTION_LEN, Col.WORD_COUNT,
    Col.HAS_TREND_KW, Col.HAS_TREND_HT, Col.HAS_TREND_SONG, Col.HOUR
]
