import pandas as pd
import numpy as np
import warnings
import re
import logging
from transformers import pipeline
import torch
from constants import Config, Col, PATHS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class TikTokDataProcessor:
    def __init__(self):
        self.trend_keywords = set()
        self.trend_hashtags = set()
        self.trend_songs = set()
        self.gameshow_hashtags = []
        self.famous_hashtags = [] 
        
        device = 0 if torch.cuda.is_available() else -1
        logging.info("Loading PhoBERT Sentiment Pipeline...")
        try:
            self.sentiment_analyzer = pipeline(
                Config.SENTIMENT_TASK, 
                model=Config.PHOBERT_MODEL, 
                tokenizer=Config.PHOBERT_MODEL,
                device=device
            )
        except Exception as e:
            logging.error(f"Failed to load PhoBERT: {e}")
            self.sentiment_analyzer = None

    def _update_set_from_csv(self, filepath, columns, target_set):
        try:
            df = pd.read_csv(filepath)
            for col in columns:
                if col in df.columns:
                    target_set.update(df[col].dropna().apply(lambda x: str(x).lower().strip()).tolist())
        except Exception as e:
            logging.error(f"Failed to load {filepath}: {e}")

    def load_trends(self):
        """Load trend and gameshow data from files[cite: 4, 7]."""
        logging.info("Loading trend and gameshow data from files...")
        
        # 1. Load basic trend data
        self._update_set_from_csv(PATHS["keyword_trend"], Config.TREND_TIME_COLS, self.trend_keywords)
        self._update_set_from_csv(PATHS["hashtag_trend"], Config.TREND_TIME_COLS, self.trend_hashtags)
        self._update_set_from_csv(PATHS["song_trend"], Config.SONG_TREND_COLS, self.trend_songs)

        # 2. Process Gameshow: Calculate Heat Score based on the 'view' column in the CSV file[cite: 4]
        try:
            df_gs = pd.read_csv(PATHS["gameshow"])
            df_gs['weight'] = np.log1p(df_gs['view']) # Log view để làm mượt dữ liệu
            self.gs_weights = dict(zip(df_gs[Col.HT_GAMESHOW].str.lower(), df_gs['weight']))
            self.gameshow_hashtags = df_gs[Col.HT_GAMESHOW].dropna().astype(str).str.lower().tolist()
            logging.info(f"✅ Loaded {len(self.gs_weights)} gameshow hashtags.")
        except Exception as e:
            logging.error(f"Error processing Data_gameshow.csv: {e}")

        # 3. Process KOL: Prepare list of famous artists[cite: 4]
        try:
            df_kol = pd.read_csv(PATHS["kol_to_gameshow"])
            self.famous_hashtags = df_kol[Col.HT_FAMOUS].dropna().str.lower().tolist()
            self.kol_set = set(self.famous_hashtags)
            logging.info(f"✅ Loaded {len(self.kol_set)} famous KOL hashtags.")
        except Exception as e:
            logging.error(f"Error processing kol_to_gameshows.csv: {e}")

    def extract_caption_features(self, caption):
        if pd.isna(caption): caption = ""
        caption_str = str(caption).strip() 
        
        hashtags = re.findall(r'#(\w+)', caption_str)
        hashtag_str = " ".join([f"#{h}" for h in hashtags]).lower()
        
        clean_text = re.sub(r'#\w+', '', caption_str)
        emojis = Config.EMOJI_PATTERN.findall(clean_text)
            
        clean_text = Config.EMOJI_PATTERN.sub('', clean_text)
        clean_text = re.sub(r'[^\w\s]', '', clean_text) 
        clean_text = " ".join(clean_text.split()) 
        words = clean_text.split()
        
        return {
            Col.HASHTAG_COUNT: len(hashtags),
            Col.WORD_COUNT: len(words), 
            Col.CAPTION_CLEAN: clean_text.lower(),
            Col.HASHTAG_STR: hashtag_str,
            Col.EMOJI_COUNT: len(emojis), 
            Col.CAPTION_LEN: len(clean_text)
        }

    def _get_phobert_score(self, text: str) -> float:
        if not self.sentiment_analyzer or pd.isna(text) or str(text).strip() == "": return 0.0
        try:
            result = self.sentiment_analyzer(str(text)[:256], truncation=True)[0]
            label, confidence = result['label'], result['score']
            return float(confidence) if label == Config.LABEL_POS else float(-confidence if label == Config.LABEL_NEG else 0.0)
        except Exception: return 0.0

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Áp dụng trích xuất Caption Features cho toàn bộ DataFrame
        if Col.HASHTAG_COUNT not in df.columns:
            features_df = df[Col.CAPTION].apply(self.extract_caption_features).apply(pd.Series)
            for c in features_df.columns: 
                df[c] = features_df[c]

        df[Col.AUTHOR] = df.get(Col.AUTHOR, Config.DEFAULT_USERNAME)
        df[Col.HT_DENSITY] = df[Col.HASHTAG_COUNT].fillna(0) / df[Col.WORD_COUNT].fillna(1).clip(lower=1)
        
        # --- Trend Matching ---
        df[Col.HAS_TREND_KW] = df[Col.CAPTION_CLEAN].apply(lambda x: 1 if any(kw in str(x).lower() for kw in self.trend_keywords) else 0)
        df[Col.HAS_TREND_HT] = df[Col.HASHTAG_STR].apply(lambda x: 1 if any(ht in str(x).lower() for ht in self.trend_hashtags) else 0)
        df[Col.HAS_TREND_SONG] = df[Col.MUSIC_NAME].apply(lambda x: 1 if any(k in str(x).lower() for k in self.trend_songs) else 0)
        df[Col.IS_ORIG_SOUND] = df[Col.MUSIC_NAME].apply(lambda x: 1 if pd.notna(x) and any(k in str(x).lower() for k in Config.ORIGINAL_SOUND_KWS) else 0)
        df[Col.IS_GAMESHOW] = df[Col.HASHTAG_STR].apply(lambda x: 1 if any(g in str(x).lower() for g in self.gameshow_hashtags) else 0)
        df[Col.COUNT_FAMOUS_HT] = df[Col.HASHTAG_STR].apply(lambda x: sum(1 for t in self.famous_hashtags if t in str(x).lower()) if pd.notna(x) else 0)

        # --- NLP Sentiment ---
        df[Col.SCORE_CAPTION] = df[Col.CAPTION_CLEAN].apply(self._get_phobert_score)

        # --- Time Engineering ---
        df[Col.CREATED_AT] = pd.to_datetime(df[Col.CREATED_AT], utc=True, errors=Config.PANDAS_COERCE)
        df_vn = df[Col.CREATED_AT].dt.tz_convert(Config.TIMEZONE_VN)
        df[Col.HOUR] = df_vn.dt.hour 
        df[Col.TIME_SIN] = np.sin(2 * np.pi * df_vn.dt.hour / 24) 
        df[Col.TIME_COS] = np.cos(2 * np.pi * df_vn.dt.hour / 24)
        df[Col.IS_WEEKEND] = (df_vn.dt.dayofweek >= 5).astype(int)
        
        # --- Momentum ---
        df = df.sort_values(by=[Col.AUTHOR, Col.CREATED_AT])
        author_group = df.groupby(Col.AUTHOR)
        
        df[Col.AVG_VIEWS_3] = author_group[Col.VIEWS].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        df[Col.EMA_VIEWS_3] = author_group[Col.VIEWS].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
        
        df[Col.LIKE_RATE_TEMP] = df[Col.LIKES] / df[Col.VIEWS].replace(0, 1)
        df[Col.HIST_LIKE_RATE] = author_group[Col.LIKE_RATE_TEMP].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        df[Col.DAYS_SINCE_POST] = author_group[Col.CREATED_AT].diff().dt.total_seconds().fillna(0) / 86400
        
        df = df.drop(columns=[Col.LIKE_RATE_TEMP])
        return df
