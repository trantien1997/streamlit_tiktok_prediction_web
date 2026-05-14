import pandas as pd
import numpy as np
import joblib
import os
import re
from datetime import datetime
import yt_dlp

# --- IMPORT TỪ PROJECT COMPONENTS ---
from constants import PATHS, FEATURES, TARGETS, Col, Config
from processor import TikTokDataProcessor

# Lấy pattern từ constants
EMOJI_PATTERN = Config.EMOJI_PATTERN

# --- SECTION 1: UTILS & PREPROCESSING ---

def force_clean_id(x):
    """Giữ ID dưới dạng chuỗi để tránh sai lệch số lớn (19 chữ số)."""
    if pd.isna(x) or x == "" or x is None: return ""
    return re.sub(r'\D', '', str(x))

def extract_features_from_caption(caption):
    """Trích xuất đặc trưng cơ bản từ caption."""
    hashtags = re.findall(r'#(\w+)', caption)
    hashtag_str = " ".join([f"#{h}" for h in hashtags]).lower()
    
    clean_text = re.sub(r'#\w+', '', caption) 
    clean_text = EMOJI_PATTERN.sub('', clean_text) 
    clean_text = re.sub(r'[^\w\s]', '', clean_text) 
    clean_text = " ".join(clean_text.split()) 
    
    return {
        "hashtag_count": len(hashtags),
        "word_count": len(clean_text.split()) if clean_text else 0,
        "caption_clean": clean_text.lower(),
        "hashtag_str": hashtag_str,
        "emoji_count": len(EMOJI_PATTERN.findall(caption)),
        "caption_clean_len": len(clean_text),
        "caption_length": len(clean_text)
    }

# --- SECTION 2: CRAWLER LOGIC ---

def get_tiktok_data(username, target_post_id):
    """
    Crawl Video Target và 3 video đăng trước đó (cũ hơn).
    """
    ydl_opts = {'quiet': True, 'extract_flat': True, 'force_generic_extractor': False}
    target_id_clean = force_clean_id(target_post_id)
    profile_url = f"https://www.tiktok.com/@{username}"
    
    target_info = None
    history_list = []
    
    print(f"\n🔍 Đang truy cập profile KOL: @{username}...")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            p_info = ydl.extract_info(profile_url, download=False)
            if 'entries' not in p_info:
                print("❌ Không tìm thấy danh sách video.")
                return None, None
            
            all_entries = list(p_info['entries'])
            
            # 1. Định vị Video Target
            target_idx = -1
            for i, v in enumerate(all_entries):
                if force_clean_id(v.get('id')) == target_id_clean:
                    target_idx = i
                    break
            
            if target_idx != -1:
                t_vid = all_entries[target_idx]
                # Lấy metadata
                music = t_vid.get('track') or t_vid.get('alt_title') or t_vid.get('title') or "original sound"
                ts = t_vid.get('timestamp')
                
                target_info = {
                    'post_id': target_id_clean,
                    'created_at': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S+00:00') if ts else None,
                    'views': int(t_vid.get('view_count') or 0),
                    'likes': int(t_vid.get('like_count') or 0),
                    'shares': int(t_vid.get('repost_count') or 0),
                    'music_name': music
                }
                print(f"✅ Đã xác định Video Target (Vị trí: {target_idx + 1})")
            else:
                print(f"⚠️ Video {target_id_clean} không có trong danh sách video gần đây.")
                return None, None

            # 2. Lấy 3 video CŨ HƠN (Đăng trước Target)
            history_start_idx = target_idx + 1
            for i in range(history_start_idx, min(history_start_idx + 3, len(all_entries))):
                v = all_entries[i]
                ts = v.get('timestamp')
                history_list.append({
                    'post_id': force_clean_id(v.get('id')),
                    'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
                    'views': int(v.get('view_count') or 0),
                    'likes': int(v.get('like_count') or 0),
                    'shares': int(v.get('repost_count') or 0)
                })
        except Exception as e:
            print(f"❌ Lỗi Crawler: {e}")

    return target_info, history_list

# --- SECTION 3: PREDICTION ENGINE ---

def predict_and_save(raw_input, output_file="results.csv"):
    print("\n" + "="*95)
    print("🚀 TIKTOK PREDICTION SYSTEM (FINAL COMPLETE VERSION)")
    print("="*95)

    post_id = force_clean_id(raw_input.get('post_id', ""))
    username = raw_input.get('author_username', "")

    # 1. Thu thập dữ liệu
    target_data, history = get_tiktok_data(username, post_id)
    if not target_data:
        return

    # 2. Tính toán ngày đăng (days_since_last_post)
    days_since_last = 0
    if len(history) > 0:
        try:
            t_time = pd.to_datetime(target_data['created_at']).tz_localize(None)
            prev_time = pd.to_datetime(history[0]['date'])
            days_since_last = (t_time - prev_time).total_seconds() / 86400.0
        except: pass

    # 3. Tính toán Momentum (AVG/EMA)
    if len(history) > 0:
        v_list = [v['views'] for v in history]
        l_list = [v['likes'] for v in history]
        avg_v = np.mean(v_list)
        ema_v = pd.Series(v_list[::-1]).ewm(span=3).mean().iloc[-1]
        hist_l_rate = sum(l_list) / sum(v_list) if sum(v_list) > 0 else 0
    else:
        avg_v, ema_v, hist_l_rate = 0, 0, 0

    # 4. Hiển thị Bảng Dữ liệu Thực tế
    print("\n📊 BẢNG TỔNG HỢP DỮ LIỆU THỰC TẾ:")
    display_rows = [{
        'ID': f"{target_data['post_id']} (TARGET)", 
        'Date': target_data['created_at'].split('+')[0], 
        'Views': target_data['views'], 'Likes': target_data['likes'], 'Shares': target_data['shares']
    }]
    for v in history:
        display_rows.append({
            'ID': v['post_id'], 'Date': v['date'],
            'Views': v['views'], 'Likes': v['likes'], 'Shares': v['shares']
        })
    print(pd.DataFrame(display_rows).to_string(index=False, justify='left', 
          formatters={'Views': '{:,.0f}'.format, 'Likes': '{:,.0f}'.format, 'Shares': '{:,.0f}'.format}))
    print(f"\n⏱️ Days since last post: {days_since_last:.2f} | Momentum AVG: {avg_v:,.0f}")
    print("-" * 95)

    # 5. Xử lý qua Processor
    caption_feats = extract_features_from_caption(raw_input.get('caption', ""))
    input_dict = {**raw_input, **caption_feats}
    input_dict.update({
        Col.MEDIA_URL: f"https://www.tiktok.com/@{username}/video/{post_id}",
        Col.MUSIC_NAME: target_data['music_name'],
        Col.CREATED_AT: target_data['created_at'],
        Col.VIEWS: target_data['views'],
        Col.LIKES: target_data['likes'], 
        Col.SHARES: target_data['shares'],
        Col.AVG_VIEWS_3: avg_v,
        Col.EMA_VIEWS_3: ema_v,
        Col.HIST_LIKE_RATE: hist_l_rate,
        'days_since_last_post': days_since_last
    })

    try:
        processor = TikTokDataProcessor()
        processor.load_trends()
        processed_df = processor.process_features(pd.DataFrame([input_dict]))

        # Áp đặt giá trị thực tế sau xử lý
        processed_df['days_since_last_post'] = days_since_last
        processed_df[Col.AVG_VIEWS_3] = avg_v
        processed_df[Col.EMA_VIEWS_3] = ema_v
        processed_df[Col.HIST_LIKE_RATE] = hist_l_rate

        # Lưu Debug
        processed_df.to_csv("debug_feature.csv", index=False, encoding="utf-8-sig")

        # 6. Dự đoán
        X = processed_df[FEATURES].reindex(columns=FEATURES).fillna(0)
        models_paths = {
            "Linear Regression": r"C:\Users\Admin\OneDrive\Desktop\LinearRegression\models\tiktok_linear_regression_multi.pkl",
            "Random Forest": r"C:\Users\Admin\OneDrive\Desktop\LinearRegression\models\tiktok_random_forest_multi.pkl",
            "XGBoost": r"C:\Users\Admin\OneDrive\Desktop\LinearRegression\models\tiktok_xgboost_multi.pkl"
        }

        actual_vals = [target_data['likes'], target_data['views'], target_data['shares']]
        for name, path in models_paths.items():
            if os.path.exists(path):
                model = joblib.load(path)
                preds = np.expm1(model.predict(X).flatten())
                
                print(f"\n🔹 KẾT QUẢ DỰ ĐOÁN TỪ: {name}")
                print(f"{'Metric':<10} | {'Actual (Thực)':<15} | {'Predict (Dự)':<15} | {'Sai số (%)'}")
                print("-" * 65)
                metrics_labels = ["Likes", "Views", "Shares"]
                for i in range(len(metrics_labels)):
                    act, pre = actual_vals[i], preds[i]
                    err = (abs(act - pre) / act * 100) if act > 0 else 0
                    print(f"{metrics_labels[i]:<10} | {int(act):<15,} | {int(pre):<15,} | {err:.2f}%")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    my_input = {
        "post_id": "7588504551695584530",
        "author_username": "quangtuanofficial0302",
        "caption": "Chạy Ngay Đi tập tối nay có thể cừi tới kíp sao =))) @Huỳnh Lập Official ơi!!! Anh nhắc em 😌 #quangtuan #runningmanvietnamseason3 #congchua #runningman #huynhlap",
        "followers": 3300000, 
    }
    predict_and_save(my_input)