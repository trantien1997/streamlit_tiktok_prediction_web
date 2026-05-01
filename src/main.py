import subprocess
import json
import pandas as pd
import numpy as np
import joblib
import os
import re
import time
import warnings
import logging
from datetime import datetime

# ==========================================
# CẤU HÌNH LOGGING & CẢNH BÁO (DỌN RÁC LOG)
# ==========================================
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from constants import PATHS, FEATURES, Config, Col
from processor import TikTokDataProcessor

YTDLP_PATH = r"yt-dlp.exe"

def extract_basic_info(caption):
    """Trích xuất các đặc trưng cơ bản để Processor không bị lỗi KeyError."""
    if pd.isna(caption): caption = ""
    hashtags = re.findall(r'#(\w+)', str(caption))
    clean_text = re.sub(r'#\w+', '', str(caption))
    clean_text = Config.EMOJI_PATTERN.sub('', clean_text)
    words = clean_text.split()
    return {
        Col.HASHTAG_COUNT: len(hashtags),
        Col.WORD_COUNT: len(words) if words else 0,
        Col.HASHTAG_STR: " ".join([f"#{h}" for h in hashtags]).lower(),
        Col.CAPTION_CLEAN: clean_text.lower().strip()
    }

def get_follower_count(kol_name):
    """Cào số lượng follower mới nhất của KOL."""
    profile_url = f"https://www.tiktok.com/@{kol_name}"
    cmd = [
        YTDLP_PATH, profile_url,
        "--cookies-from-browser", "chrome",
        "--dump-single-json",
        "--playlist-items", "0",
        "--quiet", "--no-warnings"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Kiểm tra đồng thời các trường có thể chứa số follower
            follower_count = (
                data.get('follower_count') or 
                data.get('subscribers') or 
                data.get('channel_follower_count') or 0
            )
            
            print(f"📊 Đã cập nhật Follower cho {kol_name}: {follower_count:,}")
            return follower_count
    except Exception as e:
        print(f"⚠️ Không thể lấy follower tự động: {e}")
    return 0

def get_latest_history_videos(kol_name, limit=3):
    """Cào dữ liệu 3 video mới nhất và hiển thị số Save thật."""
    profile_url = f"https://www.tiktok.com/@{kol_name}"
    cmd = [
        YTDLP_PATH, profile_url,
        "--cookies-from-browser", "chrome",
        "--dump-single-json",
        "--playlist-end", str(limit),
        "--quiet", "--no-warnings"
    ]
    
    history_data = []
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=200)
        if result.returncode == 0:
            entries = json.loads(result.stdout).get('entries', [])
            print(f"\n{'='*35} 3 VIDEO GẦN NHẤT CỦA {kol_name.upper()} {'='*35}")
            print(f"{'Post ID':<20} | {'Ngày đăng':<18} | {'Like':<10} | {'View':<10} | {'Share':<10} | {'Cmt':<8} | {'Save':<8}")
            print("-" * 115)
            
            for entry in entries:
                dt_str = datetime.fromtimestamp(entry.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M')
                stats = {
                    "post_id": entry.get("id"),
                    Col.VIEWS: entry.get("view_count", 0),
                    Col.LIKES: entry.get("like_count", 0),
                    Col.SHARES: entry.get("repost_count", 0),
                    "comments": entry.get("comment_count", 0),
                    # Sử dụng save_count để lấy đúng số 80 thay vì 0
                    "collects": entry.get("save_count") or entry.get("favorite_count") or 0,
                    "timestamp": entry.get("timestamp", time.time()),
                    Col.CAPTION: entry.get("description", ""),
                    Col.MUSIC_NAME: entry.get("track", "Original sound")
                }
                print(f"{stats['post_id']:<20} | {dt_str:<18} | {stats[Col.LIKES]:<10,} | {stats[Col.VIEWS]:<10,} | {stats[Col.SHARES]:<10,} | {stats['comments']:<8,} | {stats['collects']:<8,}")
                history_data.append(stats)
            print("-" * 115)
        return history_data
    except Exception:
        return []

def run_prediction_for_new_video(my_input, output_file="result.csv", debug_file="debug_feature.csv"):
    # 1. Thu thập 3 video lịch sử
    history = get_latest_history_videos(my_input['kol_name'], limit=3)
    if not history:
        print("[!] Lỗi: Không thể lấy dữ liệu từ TikTok.")
        return

    # 2. Tạo danh sách dữ liệu (Gán author_username cố định để tránh 'unknown')
    rows = []
    for h in reversed(history):
        row = {
            Col.AUTHOR: my_input['kol_name'],
            Col.MEDIA_URL: f"https://www.tiktok.com/@{my_input['kol_name']}/video/{h['post_id']}",
            Col.CREATED_AT: datetime.fromtimestamp(h['timestamp']).strftime('%Y-%m-%d %H:%M:%S+00:00'),
            Col.CAPTION: h[Col.CAPTION],
            Col.MUSIC_NAME: h[Col.MUSIC_NAME],
            Col.VIEWS: h[Col.VIEWS], Col.LIKES: h[Col.LIKES], Col.SHARES: h[Col.SHARES],
            "comments": h['comments'], "collects": h['collects'],
            Col.FOLLOWERS: my_input['followers'], "post_id": h['post_id']
        }
        row.update(extract_basic_info(h[Col.CAPTION]))
        rows.append(row)

    # Video cần dự đoán
    new_video_row = {
        Col.AUTHOR: my_input['kol_name'],
        Col.MEDIA_URL: f"https://www.tiktok.com/@{my_input['kol_name']}/video/UNPUBLISHED",
        Col.CREATED_AT: my_input['created_at'],
        Col.CAPTION: my_input['caption'],
        Col.MUSIC_NAME: my_input['music_name'],
        Col.VIEWS: 0, Col.LIKES: 0, Col.SHARES: 0, "comments": 0, "collects": 0,
        Col.FOLLOWERS: my_input['followers'], "post_id": "TARGET_PREDICT"
    }
    new_video_row.update(extract_basic_info(my_input['caption']))
    rows.append(new_video_row)

    # 3. Chạy Processor tính toán Momentum
    df_all = pd.DataFrame(rows)
    processor = TikTokDataProcessor()
    processor.load_trends()
    processed_df = processor.process_features(df_all)
    
    # Tính Momentum theo tên cột Model yêu cầu
    group = processed_df.groupby(Col.AUTHOR)
    processed_df['avg_comments_last_3_videos'] = group['comments'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    processed_df['ema_comments_last_3'] = group['comments'].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
    processed_df['avg_collects_last_3_videos'] = group['collects'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    processed_df['ema_collects_last_3'] = group['collects'].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
    processed_df['caption_length'] = processed_df[Col.CAPTION].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

    # 4. Tách dữ liệu của video Predict để lưu Debug và dự đoán
    target_data = processed_df[processed_df['post_id'] == "TARGET_PREDICT"].copy()
    
    # Lưu toàn bộ thông tin đặc trưng của riêng video predict vào file debug
    target_data.to_csv(debug_file, mode='a', index=False, header=not os.path.exists(debug_file), encoding='utf-8-sig')
    print(f"📁 Đặc trưng chi tiết đã lưu vào: {debug_file}")

    # 5. Chạy Model dự đoán
    X_input = target_data[FEATURES].reindex(columns=FEATURES)
    models_path = {
        "Linear Regression": r"C:\Users\Admin\OneDrive\Desktop\InvestigateLV\models\tiktok_linear_regression_multi.pkl",
        "Random Forest": r"C:\Users\Admin\OneDrive\Desktop\InvestigateLV\models\tiktok_random_forest_multi.pkl",
        "XGBoost": r"C:\Users\Admin\OneDrive\Desktop\InvestigateLV\models\tiktok_xgboost_multi.pkl"
    }

    print(f"\n🚀 KẾT QUẢ DỰ ĐOÁN CHO VIDEO MỚI:")
    print(f"{'Model Name':<20} | {'Views':<12} | {'Likes':<12} | {'Shares':<12}")
    print("-" * 65)

    prediction_results = []
    for name, path in models_path.items():
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                preds = np.expm1(model.predict(X_input).flatten())
                v, l, s = int(preds[1]), int(preds[0]), int(preds[2])
                print(f"{name:<20} | {v:<12,} | {l:<12,} | {s:<12,}")
                
                prediction_results.append({
                    "author_username": my_input['kol_name'],
                    "media_url": "UNPUBLISHED",
                    "created_at": my_input['created_at'],
                    "caption": my_input['caption'],
                    "model": name,
                    "pred_views": v, "pred_likes": l, "pred_shares": s
                })
            except Exception:
                continue

    # Lưu kết quả dự đoán vào result.csv
    if prediction_results:
        pd.DataFrame(prediction_results).to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file), encoding='utf-8-sig')
        print(f"\n✅ Hoàn tất! Kết quả lưu tại: {output_file}")

if __name__ == "__main__":
    KOL_NAME = "tranthanh123" # Định nghĩa tên KOL ở đây
    current_followers = 4400000
    my_input = {
        "caption": "#sponsored Nhảy 1 chút ở SG, cực nhiều sp như tui mặc đag sale ở Cotton On nhé😆 #mycottonon #cottononvietnam #endofseasonsale #streetfashion #fashion",
        "music_name": "Original sound",
        "kol_name": KOL_NAME,
        "followers": current_followers,
        "created_at": "2026-05-01 10:00:00+07:00"
    }
    run_prediction_for_new_video(my_input)