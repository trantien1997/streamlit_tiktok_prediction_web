import os
import re
import json
import subprocess
import base64
from datetime import datetime
import pytz

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from fpdf import FPDF
except ImportError:
    pass  # Thông báo lỗi sẽ được hiển thị ở app.py

from constants import PATHS, FEATURES, Config, Col
from processor import TikTokDataProcessor

# ─── Constants for Backend ──────────────────────────────────────────────────
YTDLP_PATH = r"yt-dlp.exe"
EMOJI_PATTERN = Config.EMOJI_PATTERN

# ════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Đang tải mô hình AI (YOLO, CLIP, PhoBERT)…")
def get_processor() -> TikTokDataProcessor:
    p = TikTokDataProcessor()
    p.load_trends()
    return p

@st.cache_resource(show_spinner="⏳ Đang tải mô hình ML…")
def load_ml_models() -> dict:
    return {
        name: joblib.load(path)
        for name, path in PATHS["models"].items()
        if os.path.exists(path)
    }

# ════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ════════════════════════════════════════════════════════════════════════════
def extract_caption_features(caption: str) -> dict:
    """Basic NLP features from raw caption string."""
    hashtags    = re.findall(r'#(\w+)', caption)
    hashtag_str = " ".join(f"#{h}" for h in hashtags).lower()
    clean = re.sub(r'#\w+', '', caption)
    clean = EMOJI_PATTERN.sub('', clean)
    clean = re.sub(r'[^\w\s]', '', clean)
    clean = " ".join(clean.split())
    return {
        "hashtag_count":    len(hashtags),
        "word_count":       len(clean.split()) if clean else 0,
        "caption_clean":    clean.lower(),
        "hashtag_str":      hashtag_str,
        "emoji_count":      len(EMOJI_PATTERN.findall(caption)),
        "caption_clean_len": len(clean),
        "caption_length":   len(clean),
    }

def generate_recommendations(raw_input: dict, prediction: dict, processor_obj) -> list[str]:
    """Return actionable content improvement tips."""
    caption    = raw_input.get("caption", "")
    music_name = raw_input.get("music_name", "")
    created_at = raw_input.get("created_at", datetime.now().isoformat())
    followers  = int(raw_input.get("followers", 0) or 0)
    info       = processor_obj.extract_caption_features(caption)
    recs       = []

    cap_len = info.get(Col.CAPTION_LEN, 0)
    if cap_len < 15:
        recs.append("Caption đang khá ngắn — hãy thêm hook cảm xúc hoặc thông tin lợi ích ngay đầu.")
    elif cap_len > 180:
        recs.append("Caption hơi dài — nên rút gọn, đưa ý chính lên đầu để giữ sự chú ý.")
    else:
        recs.append("Độ dài caption tốt. Có thể tối ưu thêm bằng CTA rõ hơn.")

    ht_count = info.get(Col.HASHTAG_COUNT, 0)
    if ht_count < 3:
        recs.append("Hashtag còn ít — thêm 3–5 hashtag: 1 ngành, 1–2 trend, 1 thương hiệu.")
    elif ht_count > 8:
        recs.append("Hashtag hơi nhiều — giữ lại những hashtag liên quan nhất để tránh spam.")
    else:
        recs.append("Số lượng hashtag hợp lý.")

    generic_tags = ["#fyp", "#video", "#viral"]
    if any(tag in info.get(Col.HASHTAG_STR, "") for tag in generic_tags):
        recs.append("Hashtag #fyp/#video khá chung — bổ sung hashtag cụ thể theo sản phẩm/ngành.")

    emoji_cnt = info.get(Col.EMOJI_COUNT, 0)
    if emoji_cnt == 0:
        recs.append("Chưa có emoji — thêm 1–3 emoji phù hợp để tăng độ nổi bật.")
    elif emoji_cnt > 6:
        recs.append("Emoji hơi nhiều — nên giảm để caption chuyên nghiệp hơn.")

    try:
        hour = pd.to_datetime(created_at).hour
        if 0 <= hour <= 5:
            recs.append("Giờ đăng khuya/sáng sớm — nên thử 18h–22h hoặc 11h–13h.")
        elif 18 <= hour <= 22:
            recs.append("Khung giờ 18h–22h rất tốt — đang gần thời điểm người dùng hoạt động cao.")
        else:
            recs.append("Có thể thử thêm khung 18h–22h để so sánh hiệu quả.")
    except Exception:
        pass

    if "original" in music_name.lower():
        recs.append("Đang dùng Original sound — nếu cần reach rộng hơn, hãy thử âm thanh đang trend.")
    else:
        recs.append("Âm thanh không phải Original — kiểm tra xem audio có đang trend không.")

    if followers > 0:
        view_rate = prediction.get("pred_views", 0) / followers
        if view_rate < 1:
            recs.append("View dự đoán thấp hơn followers — cải thiện hook 3 giây đầu và thumbnail.")
        elif view_rate > 3:
            recs.append("View dự đoán cao so với followers — nội dung có tiềm năng phân phối rộng!")

    return recs

# ════════════════════════════════════════════════════════════════════════════
# TikTok CRAWLER (via yt-dlp)
# ════════════════════════════════════════════════════════════════════════════
def get_latest_history_videos(kol_name: str, limit: int = 3) -> list[dict]:
    """Crawl the most recent `limit` videos from a TikTok profile."""
    profile_url = f"https://www.tiktok.com/@{kol_name}"
    cmd = [
        YTDLP_PATH, profile_url,
        "--cookies-from-browser", "chrome",
        "--dump-single-json",
        "--playlist-end", str(limit),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", timeout=200,
        )
        if result.returncode != 0:
            print(f"yt-dlp error: {result.stderr}")
            return []

        entries = json.loads(result.stdout).get("entries", [])
        history = []
        for entry in entries:
            ts = entry.get("timestamp")
            if ts:
                dt_vn = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(Config.TIMEZONE_VN)
                created_at = dt_vn.strftime("%Y-%m-%d %H:%M:%S+07:00")
            else:
                created_at = datetime.now(Config.TIMEZONE_VN).strftime("%Y-%m-%d %H:%M:%S+07:00")

            history.append({
                "post_id":     entry.get("id"),
                Col.VIEWS:     entry.get("view_count", 0),
                Col.LIKES:     entry.get("like_count", 0),
                Col.SHARES:    entry.get("repost_count", 0),
                Col.COMMENTS:  entry.get("comment_count", 0),
                Col.COLLECTS:  entry.get("save_count") or entry.get("favorite_count") or 0,
                Col.CREATED_AT: created_at,
                Col.CAPTION:   entry.get("description", ""),
                Col.MUSIC_NAME: entry.get("track", "Original sound"),
            })
        return history
    except Exception as e:
        print(f"Crawler error: {e}")
        return []

# ════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ════════════════════════════════════════════════════════════════════════════
def run_prediction(my_input: dict, processor_obj, models: dict) -> dict | None:
    """
    Core prediction pipeline.
    Returns a dict with keys: preds_dict, target_data, prediction_results, raw_history.
    """
    kol_name = my_input["kol_name"]
    history  = get_latest_history_videos(kol_name, limit=3)

    # --- Compute timing features ---
    latest_ts = history[0].get(Col.CREATED_AT) if history else None
    if not latest_ts:
        latest_ts = datetime.now(Config.TIMEZONE_VN).strftime("%Y-%m-%d %H:%M:%S+07:00")

    def _aware(ts_str):
        dt = pd.to_datetime(ts_str)
        return Config.TIMEZONE_VN.localize(dt) if dt.tzinfo is None else dt.astimezone(Config.TIMEZONE_VN)

    target_time = _aware(my_input["created_at"])
    last_time   = _aware(latest_ts)
    diff_days   = max(0, (target_time - last_time).total_seconds() / 86400) if history else 0

    # --- Build DataFrame for processor ---
    rows = []
    for h in reversed(history):
        rows.append({
            Col.AUTHOR:     kol_name,
            Col.MEDIA_URL:  f"https://www.tiktok.com/@{kol_name}/video/{h.get('post_id', '')}",
            Col.CREATED_AT: h[Col.CREATED_AT],
            Col.CAPTION:    h.get(Col.CAPTION, ""),
            Col.MUSIC_NAME: h.get(Col.MUSIC_NAME, ""),
            Col.VIEWS:      h.get(Col.VIEWS, 0),
            Col.LIKES:      h.get(Col.LIKES, 0),
            Col.SHARES:     h.get(Col.SHARES, 0),
            Col.COMMENTS:   h.get(Col.COMMENTS, 0),
            Col.COLLECTS:   h.get(Col.COLLECTS, 0),
            Col.FOLLOWERS:  my_input["followers"],
            "post_id":      h.get("post_id", ""),
        })

    # Target (new unpublished video)
    rows.append({
        Col.AUTHOR:     kol_name,
        Col.MEDIA_URL:  f"https://www.tiktok.com/@{kol_name}/video/UNPUBLISHED",
        Col.CREATED_AT: my_input["created_at"],
        Col.CAPTION:    my_input["caption"],
        Col.MUSIC_NAME: my_input["music_name"],
        Col.VIEWS: 0, Col.LIKES: 0, Col.SHARES: 0,
        Col.COMMENTS: 0, Col.COLLECTS: 0,
        Col.FOLLOWERS:  my_input["followers"],
        "post_id":      "TARGET_PREDICT",
    })

    df_all = pd.DataFrame(rows)
    df_all[Col.CREATED_AT] = pd.to_datetime(df_all[Col.CREATED_AT])

    # Process features (video path optional)
    video_p = my_input.get("video_path", "").strip('"\'')
    if video_p and os.path.exists(video_p):
        processed_df = processor_obj.process_features(df_all, video_path=video_p)
    else:
        processed_df = processor_obj.process_features(df_all)

    target_data = processed_df[processed_df["post_id"] == "TARGET_PREDICT"].copy()
    target_data[Col.DAYS_SINCE_POST] = diff_days
    target_data[Col.CREATED_AT]      = my_input["created_at"]

    # Đã sửa lỗi KeyError bằng cách sử dụng reindex trực tiếp lên toàn bộ bảng
    X_input = target_data.reindex(columns=FEATURES).fillna(0)

    # --- Run each model ---
    preds_dict        = {}
    prediction_rows   = []
    for name, model in models.items():
        try:
            raw = np.expm1(model.predict(X_input).flatten())
            v, l, s = int(raw[1]), int(raw[0]), int(raw[2])
            preds_dict[name] = {"views": v, "likes": l, "shares": s}
            prediction_rows.append({
                Col.AUTHOR:  kol_name,
                Col.CREATED_AT: my_input["created_at"],
                Col.CAPTION:    my_input["caption"],
                "model":     name,
                "pred_views": v, "pred_likes": l, "pred_shares": s,
            })
        except Exception as exc:
            print(f"Model {name} error: {exc}")

    return {
        "preds_dict":         preds_dict,
        "target_data":        target_data,
        "prediction_results": pd.DataFrame(prediction_rows) if prediction_rows else None,
        "raw_history":        history,
    }

# ════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ════════════════════════════════════════════════════════════════════════════
def build_pdf_report(campaign: dict, result_df: pd.DataFrame, chart_path: str = None) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "BAO CAO DU DOAN HIEU QUA CHIEN DICH TIKTOK", ln=True, align="C")
    pdf.ln(8)

    # Campaign info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "1. THONG TIN CHIEN DICH:", ln=True)
    pdf.set_font("Arial", "", 11)
    for label, val in [
        ("Thoi gian du kien:", f"{campaign['date']} {campaign['time']}"),
        ("Am nhac:",           campaign["music"].encode("latin-1", "replace").decode("latin-1")),
        ("Mo hinh:",           campaign["model"]),
    ]:
        pdf.cell(55, 10, label)
        pdf.cell(145, 10, val, ln=True)
    pdf.ln(4)

    # Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "2. KET QUA DU DOAN & XEP HANG KOL:", ln=True)
    widths = [45, 25, 30, 30, 30, 25]
    headers = ["KOL", "Followers", "Pred Views", "Pred Likes", "Pred Shares", "Score"]
    pdf.set_font("Arial", "B", 10)
    for w, h in zip(widths, headers):
        pdf.cell(w, 10, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Arial", "", 10)
    for _, row in result_df.iterrows():
        pdf.cell(widths[0], 10, str(row["name_of_creator"]),     border=1)
        pdf.cell(widths[1], 10, f"{int(row['followers']):,}",    border=1, align="R")
        pdf.cell(widths[2], 10, f"{int(row['pred_views']):,}",   border=1, align="R")
        pdf.cell(widths[3], 10, f"{int(row['pred_likes']):,}",   border=1, align="R")
        pdf.cell(widths[4], 10, f"{int(row['pred_shares']):,}",  border=1, align="R")
        pdf.cell(widths[5], 10, f"{float(row['kol_score']):.1f}", border=1, align="R")
        pdf.ln()
    pdf.ln(8)

    # Chart
    if chart_path and os.path.exists(chart_path):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "3. BIEU DO SO SANH:", ln=True)
        pdf.image(chart_path, x=15, w=180)
        pdf.ln(4)

    # Conclusion
    title_idx = "4." if (chart_path and os.path.exists(chart_path or "")) else "3."
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, f"{title_idx} DE XUAT:", ln=True)
    pdf.set_font("Arial", "I", 11)
    pdf.cell(200, 10, f"-> He thong de xuat chon Creator '{result_df.iloc[0]['name_of_creator']}'.", ln=True)

    return pdf.output(dest="S").encode("latin1")

def pdf_download_link(pdf_bytes: bytes, filename: str) -> str:
    b64 = base64.b64encode(pdf_bytes).decode()
    style = (
        "display:inline-flex;align-items:center;gap:6px;"
        "background:#fff;border:1.5px solid #E2E6EF;border-radius:8px;"
        "padding:8px 18px;color:#1A1F36;font-weight:600;font-size:.87rem;"
        "text-decoration:none;transition:all .2s;"
    )
    return (
        f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="{style}">'
        f'📄 Tải báo cáo PDF</a>'
    )
