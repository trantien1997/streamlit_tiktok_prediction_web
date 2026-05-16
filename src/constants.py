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
    POST_ID = "post_id"

    # --- Generated columns ---
    AUTHOR = "author_username"
    EMOJI_COUNT = "emoji_count"
    HT_DENSITY = "hashtag_density"
    HAS_TREND_KW = "has_trend_keyword"
    HAS_TREND_HT = "has_trend_hashtag"
    HAS_TREND_SONG = "has_trend_song"
    IS_ORIG_SOUND = "is_original_sound"
    IS_GAMESHOW = "is_related_gameshow"
    FAMOUS_HT_COUNT = "hashtag_famous_count"
    SCORE_CAPTION = "score_caption"
    GS_HEAT_SCORE = "gs_heat_score"
    GS_KOL_SYNC = "kol_gs_sync_score"
    GS_SENTIMENT_IMPACT = "gs_sentiment_impact"
    
    # --- Time columns ---
    HOUR = "hour"
    TIME_SIN = "time_sin"
    TIME_COS = "time_cos"
    IS_WEEKEND = "is_weekend"
    
    # --- Momentum columns ---
    AVG_VIEWS_3 = "avg_views_last_3_videos"
    EMA_VIEWS_3 = "ema_views_last_3"
    LIKE_RATE_TEMP = "like_rate_temp"
    HIST_LIKE_RATE = "hist_like_rate"
    DAYS_SINCE_POST = "days_since_last_post"

    # --- Video content features ---
    VIDEO_SIZE_MB = "video_file_size_mb"
    VIDEO_DURATION = "video_duration_sec"
    VIDEO_FPS = "video_fps"
    VIDEO_FRAME_COUNT = "video_frame_count"
    VIDEO_WIDTH = "video_width"
    VIDEO_HEIGHT = "video_height"
    VIDEO_ASPECT_RATIO = "video_aspect_ratio"
    
    # --- Video quality features ---
    BRIGHTNESS_MEAN = "video_frame_brightness_mean"
    BRIGHTNESS_STD = "video_frame_brightness_std"
    BRIGHTNESS_MIN = "video_frame_brightness_min"
    BRIGHTNESS_MAX = "video_frame_brightness_max"
    
    CONTRAST_MEAN = "video_frame_contrast_mean"
    CONTRAST_STD = "video_frame_contrast_std"
    CONTRAST_MIN = "video_frame_contrast_min"
    CONTRAST_MAX = "video_frame_contrast_max"
    
    SHARPNESS_MEAN = "video_frame_sharpness_mean"
    SHARPNESS_STD = "video_frame_sharpness_std"
    SHARPNESS_MIN = "video_frame_sharpness_min"
    SHARPNESS_MAX = "video_frame_sharpness_max"
    
    COLORFULNESS_MEAN = "video_frame_colorfulness_mean"
    COLORFULNESS_STD = "video_frame_colorfulness_std"
    COLORFULNESS_MIN = "video_frame_colorfulness_min"
    COLORFULNESS_MAX = "video_frame_colorfulness_max"
    
    # --- Video content features ---
    OCR_TEXT = "ocr_text" 
    AVG_CONFIDENCE_MEAN = "video_frame_avg_confidence_mean"
    AVG_CONFIDENCE_MAX = "video_frame_avg_confidence_max"
    
    # --- Video person detection features ---
    HAS_PERSON_MEAN = "video_frame_has_person_mean"
    HAS_PERSON_MAX = "video_frame_has_person_max"
    PERSON_COUNT_MEAN = "video_frame_person_count_mean"
    PERSON_COUNT_MAX = "video_frame_person_count_max"
    OBJ_PERSON_COUNT_MEAN = "video_frame_obj_person_count_mean"
    OBJ_PERSON_COUNT_MAX = "video_frame_obj_person_count_max"
    
    # --- Video object detection features ---
    OBJECT_COUNT_MEAN = "video_frame_object_count_mean"
    OBJECT_COUNT_MAX = "video_frame_object_count_max"
    
    # --- Specific object detection features ---
    OBJ_BOOK_MEAN = "video_frame_obj_book_count_mean"
    OBJ_BOOK_MAX = "video_frame_obj_book_count_max"
    OBJ_BOTTLE_MEAN = "video_frame_obj_bottle_count_mean"
    OBJ_BOTTLE_MAX = "video_frame_obj_bottle_count_max"
    OBJ_CAR_MEAN = "video_frame_obj_car_count_mean"
    OBJ_CAR_MAX = "video_frame_obj_car_count_max"
    OBJ_CAT_MEAN = "video_frame_obj_cat_count_mean"
    OBJ_CAT_MAX = "video_frame_obj_cat_count_max"
    OBJ_PHONE_MEAN = "video_frame_obj_cell_phone_count_mean"
    OBJ_PHONE_MAX = "video_frame_obj_cell_phone_count_max"
    OBJ_CHAIR_MEAN = "video_frame_obj_chair_count_mean"
    OBJ_CHAIR_MAX = "video_frame_obj_chair_count_max"
    OBJ_CUP_MEAN = "video_frame_obj_cup_count_mean"
    OBJ_CUP_MAX = "video_frame_obj_cup_count_max"
    OBJ_DOG_MEAN = "video_frame_obj_dog_count_mean"
    OBJ_DOG_MAX = "video_frame_obj_dog_count_max"
    OBJ_HANDBAG_MEAN = "video_frame_obj_handbag_count_mean"
    OBJ_HANDBAG_MAX = "video_frame_obj_handbag_count_max"
    OBJ_LAPTOP_MEAN = "video_frame_obj_laptop_count_mean"
    OBJ_LAPTOP_MAX = "video_frame_obj_laptop_count_max"
    OBJ_SPORTS_BALL_MEAN = "video_frame_obj_sports_ball_count_mean"
    OBJ_SPORTS_BALL_MAX = "video_frame_obj_sports_ball_count_max"
    OBJ_TV_MEAN = "video_frame_obj_tv_count_mean"
    OBJ_TV_MAX = "video_frame_obj_tv_count_max"

    ENGAGEMENT_POWER = "engagement_power"
    RECENT_MOMENTUM = "recent_momentum"
    CONTENT_STRENGTH = "content_strength"

    # --- Target columns ---        
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
    Col.FOLLOWERS,
    Col.EMA_VIEWS_3,
    Col.HIST_LIKE_RATE,
    Col.DAYS_SINCE_POST,
    Col.FAMOUS_HT_COUNT,
    Col.IS_ORIG_SOUND,
    Col.HASHTAG_COUNT,
    Col.EMOJI_COUNT,
    Col.HT_DENSITY, 
    Col.SCORE_CAPTION,
    Col.GS_HEAT_SCORE,
    Col.GS_KOL_SYNC,
    Col.GS_SENTIMENT_IMPACT,
    Col.TIME_SIN,
    Col.TIME_COS,
    Col.IS_WEEKEND,
    Col.WORD_COUNT,
    Col.HAS_TREND_HT,
    Col.VIDEO_DURATION,
    Col.VIDEO_FPS,
    Col.VIDEO_FRAME_COUNT,
    Col.VIDEO_WIDTH,
    Col.VIDEO_HEIGHT,
    Col.VIDEO_ASPECT_RATIO,
    Col.BRIGHTNESS_MEAN,
    Col.BRIGHTNESS_STD,
    Col.CONTRAST_MEAN,
    Col.CONTRAST_STD,
    Col.SHARPNESS_MEAN,
    Col.SHARPNESS_STD,
    Col.COLORFULNESS_MEAN,
    Col.COLORFULNESS_STD,
    Col.AVG_CONFIDENCE_MEAN,
    Col.AVG_CONFIDENCE_MAX,
    Col.HAS_PERSON_MAX,
    Col.OBJECT_COUNT_MEAN,
    Col.OBJECT_COUNT_MAX,
    Col.ENGAGEMENT_POWER,
    Col.RECENT_MOMENTUM,
    Col.CONTENT_STRENGTH
]
