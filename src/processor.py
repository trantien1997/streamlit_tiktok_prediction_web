import pandas as pd
import numpy as np
import warnings
import re
import logging
from transformers import pipeline
import torch
import cv2
from PIL import Image
import gc
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
        
        # --- 1. INIT NLP MODEL (PHOBERT) ---
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

        # --- 2. INIT VISUAL MODELS (YOLO & CLIP) ---
        self.yolo_model = None
        self.clip_model = None
        self.clip_processor = None
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.selected_yolo_classes = [
            "person", "cell phone", "handbag", "bottle", "cup", "chair", 
            "tv", "laptop", "book", "sports ball", "car", "dog", "cat"
        ]
        self._load_visual_models()

    def _load_visual_models(self):
        """Khởi tạo mô hình YOLO và CLIP cho xử lý hình ảnh."""
        logging.info("Loading Visual Models (YOLO & CLIP)...")
        # Load YOLO
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            logging.info("✅ YOLO loaded.")
        except Exception as e:
            logging.warning(f"YOLO load failed: {e}")

        # Load CLIP
        try:
            from transformers import CLIPProcessor, CLIPModel
            model_name = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.clip_device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model.eval()
            logging.info("✅ CLIP loaded.")
        except Exception as e:
            logging.warning(f"CLIP load failed: {e}")

    def _update_set_from_csv(self, filepath, columns, target_set):
        try:
            df = pd.read_csv(filepath)
            for col in columns:
                if col in df.columns:
                    target_set.update(df[col].dropna().apply(lambda x: str(x).lower().strip()).tolist())
        except Exception as e:
            logging.error(f"Failed to load {filepath}: {e}")

    def load_trends(self):
        """Load trend and gameshow data from files."""
        logging.info("Loading trend and gameshow data from files...")
        
        # 1. Load basic trend data
        self._update_set_from_csv(PATHS["keyword_trend"], Config.TREND_TIME_COLS, self.trend_keywords)
        self._update_set_from_csv(PATHS["hashtag_trend"], Config.TREND_TIME_COLS, self.trend_hashtags)
        self._update_set_from_csv(PATHS["song_trend"], Config.SONG_TREND_COLS, self.trend_songs)

        # 2. Process Gameshow: Calculate Heat Score based on the 'view' column in the CSV file
        try:
            df_gs = pd.read_csv(PATHS["gameshow"])
            df_gs['weight'] = np.log1p(df_gs['view']) # Log view để làm mượt dữ liệu
            self.gs_weights = dict(zip(df_gs[Col.HT_GAMESHOW].astype(str).str.lower(), df_gs['weight']))
            self.gameshow_hashtags = df_gs[Col.HT_GAMESHOW].dropna().astype(str).str.lower().tolist()
            logging.info(f"✅ Loaded {len(self.gs_weights)} gameshow hashtags.")
        except Exception as e:
            logging.error(f"Error processing Data_gameshow.csv: {e}")

        # 3. Process KOL: Prepare list of famous artists
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

    # ============================================================
    # VIDEO VISUAL FEATURES EXTRACTION LOGIC
    # ============================================================
    def _image_colorfulness(self, img_rgb):
        img = img_rgb.astype("float")
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        return std_root + 0.3 * mean_root

    def _image_quality_features(self, img_rgb, prefix):
        if img_rgb is None: return {}
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = img_rgb.shape[:2]
        return {
            f"{prefix}_width": float(w),
            f"{prefix}_height": float(h),
            f"{prefix}_aspect_ratio": float(w / h) if h > 0 else 0.0,
            f"{prefix}_brightness": float(np.mean(gray)),
            f"{prefix}_contrast": float(np.std(gray)),
            f"{prefix}_sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            f"{prefix}_colorfulness": float(self._image_colorfulness(img_rgb)),
        }

    def _extract_video_metadata_and_frames(self, video_path, n_frames=5):
        features = {
            "video_found": 0, "video_duration_sec": np.nan, "video_fps": np.nan,
            "video_frame_count": np.nan, "video_width": np.nan, "video_height": np.nan,
            "video_aspect_ratio": np.nan,
        }
        frames = []
        if not video_path or not isinstance(video_path, str): return features, frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return features, frames

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        duration = frame_count / fps if fps and fps > 0 else np.nan

        features.update({
            "video_found": 1,
            "video_duration_sec": float(duration) if pd.notna(duration) else np.nan,
            "video_fps": float(fps) if fps else np.nan,
            "video_frame_count": float(frame_count) if frame_count else np.nan,
            "video_width": float(width) if width else np.nan,
            "video_height": float(height) if height else np.nan,
            "video_aspect_ratio": float(width / height) if height and height > 0 else np.nan,
        })

        if frame_count and frame_count > 0:
            start = int(frame_count * 0.1)
            end = int(frame_count * 0.9)
            if end <= start:
                start = 0
                end = max(int(frame_count) - 1, 0)
            indices = np.linspace(start, end, n_frames).astype(int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame_bgr = cap.read()
                if ret and frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
        cap.release()
        return features, frames

    def _aggregate_frame_quality(self, frames):
        if not frames: return {}
        rows = [self._image_quality_features(fr, prefix=f"frame{i}") for i, fr in enumerate(frames)]
        agg = {}
        metrics = ["brightness", "contrast", "sharpness", "colorfulness"]
        for m in metrics:
            vals = [v for d in rows for k, v in d.items() if k.endswith("_" + m)]
            if vals:
                agg[f"video_frame_{m}_mean"] = float(np.mean(vals))
                agg[f"video_frame_{m}_std"] = float(np.std(vals))
                agg[f"video_frame_{m}_min"] = float(np.min(vals))
                agg[f"video_frame_{m}_max"] = float(np.max(vals))
        return agg

    def _yolo_detect_features(self, image_rgb, prefix="frame"):
        features = {f"{prefix}_person_count": 0, f"{prefix}_has_person": 0, f"{prefix}_object_count": 0, f"{prefix}_avg_confidence": 0.0}
        for cls in self.selected_yolo_classes: features[f"{prefix}_obj_{cls.replace(' ', '_')}_count"] = 0
        
        if self.yolo_model is None or image_rgb is None: return features
        try:
            results = self.yolo_model(image_rgb, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0: return features

            names = results[0].names
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            class_names = [names[i] for i in cls_ids]

            features[f"{prefix}_object_count"] = int(len(class_names))
            features[f"{prefix}_avg_confidence"] = float(np.mean(confs)) if len(confs) else 0.0
            
            person_count = sum(1 for c in class_names if c == "person")
            features[f"{prefix}_person_count"] = int(person_count)
            features[f"{prefix}_has_person"] = int(person_count > 0)

            for cls in self.selected_yolo_classes:
                features[f"{prefix}_obj_{cls.replace(' ', '_')}_count"] = int(sum(1 for c in class_names if c == cls))
            return features
        except Exception: return features

    def _aggregate_yolo_features(self, frames):
        if not frames: return {}
        rows = [self._yolo_detect_features(fr, prefix="frame") for fr in frames]
        agg = {}
        keys = sorted(set(k for d in rows for k in d.keys()))
        for k in keys:
            vals = [d.get(k, 0) for d in rows]
            agg[f"video_{k}_mean"] = float(np.mean(vals))
            agg[f"video_{k}_max"] = float(np.max(vals))
        return agg

    def _clip_embed_images(self, images_rgb):
        if self.clip_model is None or self.clip_processor is None or not images_rgb:
            return np.zeros(512, dtype=float)
        
        pil_images = []
        for img in images_rgb:
            try: pil_images.append(Image.fromarray(img).convert("RGB"))
            except Exception: pass
        if not pil_images: return np.zeros(512, dtype=float)

        try:
            inputs = self.clip_processor(images=pil_images, return_tensors="pt", padding=True).to(self.clip_device)
            with torch.no_grad():
                feats = self.clip_model.get_image_features(**inputs)
                if getattr(feats, "pooler_output", None) is not None: feats = feats.pooler_output
                elif getattr(feats, "image_embeds", None) is not None: feats = feats.image_embeds
                if not isinstance(feats, torch.Tensor): feats = feats[0]
            
            feats = feats.detach().cpu().numpy()
            del inputs
            gc.collect()

            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
            feats = feats / norms
            mean_feat = feats.mean(axis=0)
            return (mean_feat / (np.linalg.norm(mean_feat) + 1e-9)).astype(float)
        except Exception as e:
            logging.error(f"CLIP feature extraction error: {e}")
            return np.zeros(512, dtype=float)

    def extract_visual_features(self, video_path, n_frames=5):
        """Hàm công khai: Truyền đường dẫn video vào để lấy mọi Feature Hình ảnh."""
        out_features = {}
        # 1. Lấy Video Meta & Trích xuất Frames
        meta, frames = self._extract_video_metadata_and_frames(video_path, n_frames)
        out_features.update(meta)

        if meta.get("video_found", 0) == 1 and frames:
            # 2. Xử lý chất lượng độ sáng, tương phản
            out_features.update(self._aggregate_frame_quality(frames))
            # 3. Chạy YOLO
            out_features.update(self._aggregate_yolo_features(frames))
            # 4. Chạy CLIP (Lưu thô 512 Vector)
            emb = self._clip_embed_images(frames)
            for j, v in enumerate(emb):
                out_features[f"clip_raw_{j:03d}"] = float(v)
            
            del frames
            gc.collect()
        else:
            # Điền mặc định nếu không có video
            for j in range(512): out_features[f"clip_raw_{j:03d}"] = 0.0

        return out_features

# ============================================================
    # TEXT, MOMENTUM & VISUAL FEATURES INTEGRATION
    # ============================================================
    def process_features(self, df: pd.DataFrame, video_path: str = None, debug_csv_path: str = "debug_process_features.csv") -> pd.DataFrame:
        df = df.copy()
        import os
        
        # --- 1. Xử lý Text & Trending (Giữ nguyên) ---
        if Col.HASHTAG_COUNT not in df.columns:
            features_df = df[Col.CAPTION].apply(self.extract_caption_features).apply(pd.Series)
            for c in features_df.columns: 
                df[c] = features_df[c]

        df[Col.AUTHOR] = df.get(Col.AUTHOR, Config.DEFAULT_USERNAME)
        df[Col.HT_DENSITY] = df[Col.HASHTAG_COUNT].fillna(0) / df[Col.WORD_COUNT].fillna(1).clip(lower=1)
        
        df[Col.HAS_TREND_KW] = df[Col.CAPTION_CLEAN].apply(lambda x: 1 if any(kw in str(x).lower() for kw in self.trend_keywords) else 0)
        df[Col.HAS_TREND_HT] = df[Col.HASHTAG_STR].apply(lambda x: 1 if any(ht in str(x).lower() for ht in self.trend_hashtags) else 0)
        df[Col.HAS_TREND_SONG] = df[Col.MUSIC_NAME].apply(lambda x: 1 if any(k in str(x).lower() for k in self.trend_songs) else 0)
        df[Col.IS_ORIG_SOUND] = df[Col.MUSIC_NAME].apply(lambda x: 1 if pd.notna(x) and any(k in str(x).lower() for k in Config.ORIGINAL_SOUND_KWS) else 0)
        df[Col.IS_GAMESHOW] = df[Col.HASHTAG_STR].apply(lambda x: 1 if any(g in str(x).lower() for g in self.gameshow_hashtags) else 0)
        df[Col.FAMOUS_HT_COUNT] = df[Col.HASHTAG_STR].apply(lambda x: sum(1 for t in self.famous_hashtags if t in str(x).lower()) if pd.notna(x) else 0)

        df[Col.SCORE_CAPTION] = df[Col.CAPTION_CLEAN].apply(self._get_phobert_score)

        # --- Rich Gameshow and KOL Data [cite: 4] ---
        def calculate_dynamic_gs_kol(hashtag_str):
            if pd.isna(hashtag_str): return 0.0, 0.0, 0
            tags = str(hashtag_str).lower().strip()
            
            found_gs = [tag for tag in self.gs_weights.keys() if tag in tags]
            found_kol = [tag for tag in self.kol_set if tag in tags]
            
            # Heat Score: Sum log(view) of all matched Gameshow hashtags[cite: 4]
            gs_heat = sum(self.gs_weights[tag] for tag in found_gs)
            
            # Sync Score: Cumulative score for each matched KOL if the video is associated with a Gameshow[cite: 4]
            sync_score = len(found_kol) * 5.0 if found_gs else 0.0
            
            return gs_heat, sync_score, (1 if found_gs else 0)

        gs_results = df[Col.HASHTAG_STR].apply(calculate_dynamic_gs_kol)
        
        # Assign results to DataFrame using defined columns in constants[cite: 7]
        df[Col.GS_HEAT_SCORE] = gs_results.apply(lambda x: x[0])
        df[Col.GS_KOL_SYNC] = gs_results.apply(lambda x: x[1])
        df[Col.IS_GAMESHOW] = gs_results.apply(lambda x: x[2])

        # Sentiment Impact: Heat Score combined with positive sentiment[cite: 4]
        df[Col.GS_SENTIMENT_IMPACT] = df[Col.GS_HEAT_SCORE] * df[Col.SCORE_CAPTION].apply(lambda x: max(0, x))

        # --- 2. Xử lý Thời gian & Momentum (Giữ nguyên) ---
        df[Col.CREATED_AT] = pd.to_datetime(df[Col.CREATED_AT], utc=True, errors=Config.PANDAS_COERCE)
        df_vn = df[Col.CREATED_AT].dt.tz_convert(Config.TIMEZONE_VN)
        df[Col.HOUR] = df_vn.dt.hour 
        df[Col.TIME_SIN] = np.sin(2 * np.pi * df_vn.dt.hour / 24) 
        df[Col.TIME_COS] = np.cos(2 * np.pi * df_vn.dt.hour / 24)
        df[Col.IS_WEEKEND] = (df_vn.dt.dayofweek >= 5).astype(int)
        
        df = df.sort_values(by=[Col.AUTHOR, Col.CREATED_AT])
        author_group = df.groupby(Col.AUTHOR)
        
        df[Col.AVG_VIEWS_3] = author_group[Col.VIEWS].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        df[Col.EMA_VIEWS_3] = author_group[Col.VIEWS].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
        
        df[Col.LIKE_RATE_TEMP] = df[Col.LIKES] / df[Col.VIEWS].replace(0, 1)
        df[Col.HIST_LIKE_RATE] = author_group[Col.LIKE_RATE_TEMP].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        df[Col.DAYS_SINCE_POST] = author_group[Col.CREATED_AT].diff().dt.total_seconds().fillna(0) / 86400
        # Feature interactions
        df[Col.ENGAGEMENT_POWER] = df[Col.FOLLOWERS] * df[Col.HIST_LIKE_RATE]
        df[Col.RECENT_MOMENTUM] = df[Col.EMA_VIEWS_3] * df[Col.FOLLOWERS]
        df[Col.CONTENT_STRENGTH] = df[Col.SCORE_CAPTION] * df[Col.HT_DENSITY]
        
        df = df.drop(columns=[Col.LIKE_RATE_TEMP])

        # --- 3. MỚI: Tích hợp Visual Features ---
        if video_path and os.path.exists(video_path):
            logging.info(f"Đang trích xuất hình ảnh từ video: {video_path}")
            # Gọi hàm trích xuất hình ảnh/YOLO/CLIP
            visual_features = self.extract_visual_features(video_path, n_frames=5)
            
            # Gộp từng thuộc tính hình ảnh thành các cột mới trong DataFrame
            for col_name, value in visual_features.items():
                df[col_name] = value
        elif video_path:
            logging.warning(f"❌ Không tìm thấy video tại đường dẫn: {video_path}")

        # --- 4. MỚI: Xuất file Debug ---
        if debug_csv_path:
            try:
                # Lưu file CSV với utf-8-sig để Excel đọc không lỗi font tiếng Việt
                df.to_csv(debug_csv_path, index=False, encoding='utf-8-sig')
                logging.info(f"✅ Đã lưu file debug dữ liệu (gồm Text + Video) tại: {debug_csv_path}")
            except Exception as e:
                logging.error(f"❌ Không thể lưu file debug CSV: {e}")

        return df