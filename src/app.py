import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import subprocess
import json
import joblib
import re
import time
import pytz
import warnings
import logging
from datetime import datetime, time as dt_time
import base64

try:
    from fpdf import FPDF
except ImportError:
    st.error("Vui lòng cài đặt thư viện fpdf bằng lệnh: pip install fpdf")

# =====================================================================
# 1. CẤU HÌNH TRANG & LOAD MODELS
# =====================================================================
st.set_page_config(page_title="Hệ thống Dự đoán TikTok - UIT", layout="wide")

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from constants import PATHS, FEATURES, Config, Col
from processor import TikTokDataProcessor

YTDLP_PATH = r"C:\Users\Admin\OneDrive\Desktop\streamlit_tiktok_prediction_web\src\yt-dlp.exe"

@st.cache_resource
def get_ai_processor():
    p = TikTokDataProcessor()
    p.load_trends()
    return p

@st.cache_resource
def load_ml_models():
    loaded_models = {}
    for name, path in PATHS["models"].items():
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
    return loaded_models

# Khởi tạo toàn cục để dùng chung
processor = get_ai_processor()
models_dict = load_ml_models()

# =====================================================================
# KHỞI TẠO BỘ NHỚ TẠM (SESSION STATE)
# =====================================================================
if 'all_preds' not in st.session_state:
    st.session_state.all_preds = None  
if 'is_predicted' not in st.session_state:
    st.session_state.is_predicted = False
if 'target_data_df' not in st.session_state:
    st.session_state.target_data_df = None
if 'result_data_df' not in st.session_state:
    st.session_state.result_data_df = None
if 'raw_history_tab2' not in st.session_state:
    st.session_state.raw_history_tab2 = []
if 'raw_history_tab3' not in st.session_state:
    st.session_state.raw_history_tab3 = []
    
if 'biz_kol_df' not in st.session_state:
    st.session_state.biz_kol_df = pd.DataFrame({
        "name_of_creator": ["tranthanh123", "hariwonday", "khoailangthang"],
        "followers": [7500000, 4400000, 3300000],
        "like_avg": [0, 0, 0],
        "view_avg": [0, 0, 0],
        "share_avg": [0, 0, 0],
        "comment_avg": [0, 0, 0],
        "collects_avg": [0, 0, 0]
    })

# =====================================================================
# CÁC HÀM XỬ LÝ REPORT & BACKEND
# =====================================================================
def create_pdf_report(campaign_info, result_df, chart_image_path=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    pdf.cell(200, 10, txt="BAO CAO DU DOAN HIEU QUA CHIEN DICH TIKTOK", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. THONG TIN CHIEN DICH:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(50, 10, txt="Thoi gian du kien:")
    pdf.cell(150, 10, txt=f"{campaign_info['date']} {campaign_info['time']}", ln=True)
    pdf.cell(50, 10, txt="Am nhac su dung:")
    pdf.cell(150, 10, txt=campaign_info['music'].encode('latin-1', 'replace').decode('latin-1'), ln=True)
    
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. KET QUA DU DOAN VA XEP HANG KOL:", ln=True)
    pdf.set_font("Arial", 'B', 10)
    
    col_widths = [45, 25, 30, 30, 30, 25]
    headers = ['KOL', 'Followers', 'Pred Views', 'Pred Likes', 'Pred Shares', 'Score']
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, txt=header, border=1, align='C')
    pdf.ln()
    
    pdf.set_font("Arial", '', 10)
    for index, row in result_df.iterrows():
        pdf.cell(col_widths[0], 10, txt=str(row['name_of_creator']), border=1, align='L')
        pdf.cell(col_widths[1], 10, txt=f"{int(row['followers']):,}", border=1, align='R')
        pdf.cell(col_widths[2], 10, txt=f"{int(row['pred_views']):,}", border=1, align='R')
        pdf.cell(col_widths[3], 10, txt=f"{int(row['pred_likes']):,}", border=1, align='R')
        pdf.cell(col_widths[4], 10, txt=f"{int(row['pred_shares']):,}", border=1, align='R')
        pdf.cell(col_widths[5], 10, txt=f"{float(row['kol_score']):.1f}", border=1, align='R')
        pdf.ln()
    
    pdf.ln(10)

    if chart_image_path and os.path.exists(chart_image_path):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="3. BIEU DO SO SANH:", ln=True)
        pdf.image(chart_image_path, x=15, w=180)
        pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    conclusion_title = "4. DE XUAT:" if chart_image_path else "3. DE XUAT:"
    pdf.cell(200, 10, txt=conclusion_title, ln=True)
    pdf.set_font("Arial", 'I', 11)
    best_kol = result_df.iloc[0]['name_of_creator']
    pdf.cell(200, 10, txt=f"-> He thong de xuat chon Creator '{best_kol}' cho chien dich nay.", ln=True)
    
    return pdf.output(dest='S').encode('latin1')

def get_pdf_download_link(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration: none;"><button style="background-color: white; color: #31333F; border: 1px solid #d3d3d3; border-radius: 8px; padding: 0.5rem 1rem; font-weight: 400; font-size: 14px; cursor: pointer; transition: 0.3s;" onmouseover="this.style.borderColor=\'#FF2C55\'; this.style.color=\'#FF2C55\'" onmouseout="this.style.borderColor=\'#d3d3d3\'; this.style.color=\'#31333F\'">📄 Tải báo cáo PDF</button></a>'
    return href

def generate_post_recommendations(raw_input: dict, prediction: dict) -> list[str]:
    caption = raw_input.get("caption", "")
    music_name = raw_input.get("music_name", "")
    created_at = raw_input.get("created_at", datetime.now().isoformat())
    followers = int(raw_input.get("followers", 0) or 0)
    info = processor.extract_caption_features(caption)

    recs = []

    if info[Col.CAPTION_LEN] < 15:
        recs.append("Caption đang khá ngắn; nên thêm thông tin lợi ích, cảm xúc hoặc câu hook ở đầu caption.")
    elif info[Col.CAPTION_LEN] > 180:
        recs.append("Caption hơi dài; nên rút gọn, đưa ý chính lên đầu để người xem nắm nhanh.")
    else:
        recs.append("Độ dài caption tương đối ổn, có thể tiếp tục tối ưu bằng CTA rõ hơn.")

    if info[Col.HASHTAG_COUNT] < 3:
        recs.append("Hashtag còn ít; nên thêm 3–5 hashtag gồm: hashtag ngành, hashtag trend và hashtag thương hiệu.")
    elif info[Col.HASHTAG_COUNT] > 8:
        recs.append("Hashtag hơi nhiều; nên giữ lại hashtag liên quan nhất để tránh cảm giác spam.")
    else:
        recs.append("Số lượng hashtag tương đối hợp lý.")

    generic_tags = ["#fyp", "#video", "#viral"]
    if any(tag in info[Col.HASHTAG_STR] for tag in generic_tags):
        recs.append("Hashtag như #fyp/#video khá chung; nên bổ sung hashtag cụ thể theo sản phẩm, ngành hoặc nhóm khách hàng.")

    if info[Col.EMOJI_COUNT] == 0:
        recs.append("Caption chưa có emoji; có thể thêm 1–3 emoji phù hợp để tăng độ nổi bật.")
    elif info[Col.EMOJI_COUNT] > 6:
        recs.append("Emoji hơi nhiều; nên giảm để caption chuyên nghiệp và dễ đọc hơn.")

    hour = pd.to_datetime(created_at).hour
    if 0 <= hour <= 5:
        recs.append("Thời gian đăng hiện tại là khuya/sáng sớm; nên thử khung 18h–22h hoặc 11h–13h.")
    elif 18 <= hour <= 22:
        recs.append("Khung giờ đăng đang tốt vì thường gần thời điểm người dùng hoạt động cao.")
    else:
        recs.append("Có thể thử thêm khung giờ 18h–22h để so sánh hiệu quả.")

    if "original" in music_name.lower():
        recs.append("Đang dùng Original sound; nếu mục tiêu là tăng reach, có thể thử âm thanh đang trend.")
    else:
        recs.append("Âm thanh không phải Original sound; nên kiểm tra xem audio có đang trend hay không.")

    if followers > 0:
        view_rate = prediction.get("pred_views", 0) / followers if followers else 0
        if view_rate < 1:
            recs.append("View dự đoán thấp hơn followers; nên cải thiện hook 3 giây đầu và thumbnail/video mở đầu.")
        elif view_rate > 3:
            recs.append("View dự đoán khá tốt so với followers; nội dung có tiềm năng phân phối rộng.")

    return recs

def get_latest_history_videos(kol_name, limit=3):
    profile_url = f"https://www.tiktok.com/@{kol_name}"
    cmd = [
        YTDLP_PATH, profile_url,
        "--cookies-from-browser", "chrome",
        "--dump-single-json",
        "--playlist-end", str(limit)
        # TÔI ĐÃ XÓA "--quiet" VÀ "--no-warnings" ĐỂ XEM LỖI THẬT
    ]
    history_data = []
    
    try:
        print(f"🔄 Đang cào dữ liệu từ: {profile_url}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=200)
        
        # ---> NẾU CHẠY THẤT BẠI, IN LỖI RA MÀN HÌNH <---
        if result.returncode != 0:
            print(f"❌ LỖI YT-DLP: {result.stderr}")
            return []
            
        if result.returncode == 0:
            entries = json.loads(result.stdout).get('entries', [])
            for entry in entries:
                ts = entry.get("timestamp")
                if ts:
                    dt_vn = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(Config.TIMEZONE_VN)
                    created_at_val = dt_vn.strftime('%Y-%m-%d %H:%M:%S+07:00')
                else:
                    created_at_val = datetime.now(Config.TIMEZONE_VN).strftime('%Y-%m-%d %H:%M:%S+07:00')

                stats = {
                    Col.POST_ID: entry.get("id"),
                    Col.VIEWS: entry.get("view_count", 0),
                    Col.LIKES: entry.get("like_count", 0),
                    Col.SHARES: entry.get("repost_count", 0),
                    Col.COMMENTS: entry.get("comment_count", 0),
                    Col.COLLECTS: entry.get("save_count") or entry.get("favorite_count") or 0,
                    Col.CREATED_AT: created_at_val, 
                    Col.CAPTION: entry.get("description", ""),
                    Col.MUSIC_NAME: entry.get("track", "Original sound")
                }
                history_data.append(stats)
            print("✅ Đã cào xong:", history_data)
        return history_data
    except Exception as e:
        print(f"❌ LỖI HỆ THỐNG TRONG HÀM: {e}")
        return []

def run_prediction_for_new_video(my_input):
    history = get_latest_history_videos(my_input['kol_name'], limit=3)

    latest_video_timestamp_str = history[0].get(Col.CREATED_AT) if history else None
    if not latest_video_timestamp_str:
        latest_video_timestamp_str = datetime.now(Config.TIMEZONE_VN).strftime('%Y-%m-%d %H:%M:%S+07:00')
    
    rows = []
    for h in reversed(history):
        row = {
            Col.AUTHOR: my_input['kol_name'],
            Col.MEDIA_URL: f"https://www.tiktok.com/@{my_input['kol_name']}/video/{h.get('post_id', '')}",
            Col.CREATED_AT: h[Col.CREATED_AT],
            Col.CAPTION: h.get(Col.CAPTION, ""),
            Col.MUSIC_NAME: h.get(Col.MUSIC_NAME, ""),
            Col.VIEWS: h.get(Col.VIEWS, 0), 
            Col.LIKES: h.get(Col.LIKES, 0), 
            Col.SHARES: h.get(Col.SHARES, 0),
            Col.COMMENTS: h.get(Col.COMMENTS, 0), 
            Col.COLLECTS: h.get(Col.COLLECTS, 0),
            Col.FOLLOWERS: my_input['followers'], 
            Col.POST_ID: h.get(Col.POST_ID, '')
        }
        rows.append(row)

    target_post_time = pd.to_datetime(my_input['created_at'])
    if target_post_time.tzinfo is None:
        target_post_time = Config.TIMEZONE_VN.localize(target_post_time)
    else:
        target_post_time = target_post_time.astimezone(Config.TIMEZONE_VN)

    last_post_time = pd.to_datetime(latest_video_timestamp_str)
    if last_post_time.tzinfo is None:
        last_post_time = Config.TIMEZONE_VN.localize(last_post_time)
    else:
        last_post_time = last_post_time.astimezone(Config.TIMEZONE_VN)

    diff_days = max(0, (target_post_time - last_post_time).total_seconds() / 86400) if history else 0

    new_video_row = {
        Col.AUTHOR: my_input['kol_name'],
        Col.MEDIA_URL: f"https://www.tiktok.com/@{my_input['kol_name']}/video/UNPUBLISHED",
        Col.CREATED_AT: my_input['created_at'],
        Col.CAPTION: my_input['caption'],
        Col.MUSIC_NAME: my_input['music_name'],
        Col.VIEWS: 0, Col.LIKES: 0, Col.SHARES: 0, Col.COMMENTS: 0, Col.COLLECTS: 0,
        Col.FOLLOWERS: my_input['followers'], 
        Col.POST_ID: "TARGET_PREDICT"
    }
    rows.append(new_video_row)

    df_all = pd.DataFrame(rows)
    df_all[Col.CREATED_AT] = pd.to_datetime(df_all[Col.CREATED_AT])
    processor.load_trends()
    
    # --- Truyền video_path vào process_features ---
    video_p = my_input.get('video_path', "").strip('\"').strip('\'')
    if video_p and os.path.exists(video_p):
        processed_df = processor.process_features(df_all, video_path=video_p)
    else:
        processed_df = processor.process_features(df_all)
        
    target_data = processed_df[processed_df[Col.POST_ID] == "TARGET_PREDICT"].copy()
    
    # BẮT BUỘC GHI ĐÈ BẰNG KẾT QUẢ ĐÃ TÍNH ĐÚNG BÊN TRÊN
    target_data[Col.DAYS_SINCE_POST] = diff_days
    target_data[Col.CREATED_AT] = my_input['created_at']

    target_data.to_csv("debug_feature.csv", index=False, encoding='utf-8-sig')
    X_input = target_data[FEATURES].reindex(columns=FEATURES).fillna(0)

    prediction_results = []
    return_dict = {}
    for name, model in models_dict.items():
        try:
            preds = np.expm1(model.predict(X_input).flatten())
            v, l, s = int(preds[1]), int(preds[0]), int(preds[2])
            
            return_dict[name] = {"views": v, "likes": l, "shares": s}
            prediction_results.append({
                Col.AUTHOR: my_input['kol_name'],
                Col.MEDIA_URL: "UNPUBLISHED",
                Col.CREATED_AT: my_input['created_at'],
                Col.CAPTION: my_input['caption'],
                "model": name,
                "pred_views": v, "pred_likes": l, "pred_shares": s
            })
        except Exception as e: 
            print(f"Lỗi dự đoán model {name}: {e}")

    if prediction_results:
        pd.DataFrame(prediction_results).to_csv("result.csv", index=False, encoding='utf-8-sig')

    return {
        "preds_dict": return_dict,
        "target_data": target_data,
        "prediction_results": pd.DataFrame(prediction_results) if prediction_results else None,
        "raw_history": history 
    }

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FB; }
    
    .uit-header { 
        background-color: #003366; 
        color: white; 
        padding: 25px; 
        border-radius: 12px; 
        text-align: center; 
        margin-bottom: 20px;
    }
    
    .info-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 25px;
        padding: 0 10px;
        border-bottom: 2px solid #EEE;
        padding-bottom: 15px;
    }
    .topic-box { width: 70%; }
    .student-box { 
        width: 25%; 
        text-align: right; 
        font-size: 1.05em; 
        line-height: 1.6;
    }

    .stMetric { 
        background-color: #ffffff; 
        border-radius: 12px; 
        border: 1px solid #E6E8F0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    
    [data-testid="stDataFrame"] { width: 100%; }

    div[data-testid="stButton"] button[kind="secondary"], 
    div[data-testid="stDownloadButton"] button[kind="secondary"] {
        background-color: white !important;
        border: 1px solid #d3d3d3 !important;
        color: #31333F !important;
        transition: all 0.2s ease-in-out;
    }
    
    div[data-testid="stButton"] button[kind="secondary"]:hover,
    div[data-testid="stDownloadButton"] button[kind="secondary"]:hover {
        background-color: #007BFF !important; 
        border-color: #007BFF !important;
        color: white !important;
    }
    
    div[data-testid="stTextInput"] input:disabled {
        -webkit-text-fill-color: #003366 !important; 
        background-color: #E8F0FE !important; 
        font-weight: bold !important; 
        opacity: 1 !important; 
        cursor: default !important; 
        border: 1px solid #B6D4FE !important; 
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="uit-header">
        <h2 style='margin:0;'>ĐẠI HỌC QUỐC GIA TP.HỒ CHÍ MINH</h2>
        <h2 style='margin:5px 0;'>TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN</h2>
        <h3 style='margin:0; font-weight:normal;'>KHOA KHOA HỌC MÁY TÍNH</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <div class="info-container">
        <div class="topic-box">
            <h3 style='margin:0; color:#003366;'>Đề tài: Xây dựng hệ thống dự đoán mức độ tương tác video TikTok dựa trên các thuật toán Machine Learning</h3>
        </div>
        <div class="student-box">
            <b>Sinh viên thực hiện:</b> Nguyễn Thị Yến<br>
            <b>Ngày báo cáo:</b> {datetime.now().strftime('%d/%m/%Y')}
        </div>
    </div>
    """, unsafe_allow_html=True)

tab_overview, tab_personal, tab_business = st.tabs([
    "📊 Tổng quan mô hình Machine Learning", 
    "👤 Ứng dụng cá nhân", 
    "🏢 Ứng dụng doanh nghiệp"
])

# --- TAB 1: OVERVIEW WEB ---
with tab_overview:
    st.header("📈 Đánh giá hiệu năng các mô hình huấn luyện")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "models", "model_metrics_report.csv")

    try:
        raw_metrics_df = pd.read_csv(csv_path)

        target_options = raw_metrics_df['Target_Variable'].unique()
        selected_target = st.selectbox(
            "📍 Chọn biến mục tiêu cần xem đánh giá:", 
            target_options,
            index=0
        )

        filtered_df = raw_metrics_df[raw_metrics_df['Target_Variable'] == selected_target].copy()

        display_df = filtered_df.rename(columns={
            "Model_Name": "Mô hình",
            "R2_Score": "R² Score",
            "Avg_Error_Pct": "Sai số TB (%)"
        })
        
        table_df = display_df[["Mô hình", "MAE", "RMSE", "R² Score", "Sai số TB (%)"]]
        st.table(table_df)

        fig = go.Figure(go.Bar(
            x=display_df["Mô hình"],
            y=display_df["R² Score"],
            marker_color=['#003366', '#FF2C55', '#00F2EA'],
            text=display_df["R² Score"],
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f"So sánh chỉ số R-Squared cho biến '{selected_target}' (Càng cao càng tốt)", 
            height=450,
            yaxis=dict(range=[0, 1]) 
        )
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error(f"Không tìm thấy file {csv_path}. Vui lòng chạy lại file main.py để xuất báo cáo.")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc dữ liệu: {e}")

# --- TAB 2: PERSONAL APPLICATION ---
with tab_personal:
    st.header("👤 Dự đoán hiệu quả bài post cá nhân")
    
    with st.expander("📝 Nhập thông tin Video cần dự đoán", expanded=True):
        col_left, col_right = st.columns(2)
        
        with col_left:
            kol_name = st.text_input("KOL Username", placeholder="Nhập username của KOL tại đây...", value="")
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            followers = st.number_input("Số lượng Followers hiện tại", value=0)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            video_path_input = st.text_input("🎬 Đường dẫn Video MP4 (Tùy chọn)", placeholder=r"C:\Video\tiktok.mp4", value="")
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            c_date, c_time = st.columns(2)
            with c_date:
                d_date = st.date_input("Ngày đăng dự kiến", datetime.now())
            with c_time:
                d_time = st.time_input("Giờ đăng dự kiến", dt_time(10, 0))
                
        with col_right:
            caption = st.text_area("Nội dung Caption dự kiến", height=68, placeholder="Nhập caption kèm hashtag tại đây...")
            
            c_log_v, c_log_l, c_log_s = st.columns(3)
            ph_v = c_log_v.empty()
            ph_l = c_log_l.empty()
            ph_s = c_log_s.empty()
                
            c_music, c_model = st.columns(2)
            with c_music:
                music_name = st.text_input("🎵 Âm nhạc sử dụng", placeholder="Nhập tên bài hát tại đây...",value="")
            with c_model:
                selected_model = st.selectbox(
                    "🤖 Chọn mô hình dự đoán:", 
                    ["Linear Regression", "Random Forest", "XGBoost"],
                    index=2 
                )
            
            v_val, l_val, s_val = "0", "0", "0"
            if st.session_state.is_predicted and st.session_state.all_preds and selected_model in st.session_state.all_preds:
                data = st.session_state.all_preds[selected_model]
                v_val = f"{data['views']:,}"
                l_val = f"{data['likes']:,}"
                s_val = f"{data['shares']:,}"
                
            ph_v.text_input("👀 View", value=v_val, disabled=False)
            ph_l.text_input("❤️ Like", value=l_val, disabled=False)
            ph_s.text_input("🔗 Share", value=s_val, disabled=False)
        
        st.markdown("<br>", unsafe_allow_html=True) 
        
        c_btn1, c_btn2, c_btn3 = st.columns(3)
        
        with c_btn1:
            if st.session_state.is_predicted and st.session_state.target_data_df is not None:
                df_feature = st.session_state.target_data_df.copy()
                
                # 1. Định nghĩa các cột thông tin cơ bản và text cần giữ lại
                info_cols = [
                    "post_id", Col.AUTHOR, Col.CREATED_AT, 
                    Col.CAPTION_CLEAN, Col.HASHTAG_STR
                ]
                
                # 2. Gộp danh sách: Cột cơ bản + Cột dùng để train (FEATURES)
                cols_to_keep = info_cols + FEATURES
                
                # 3. Chỉ lọc lấy những cột thực sự tồn tại để tránh lỗi KeyError
                valid_cols = [col for col in cols_to_keep if col in df_feature.columns]
                
                # 4. Xóa bỏ các tên cột bị trùng lặp (nếu có) nhưng vẫn giữ đúng thứ tự
                valid_cols = list(dict.fromkeys(valid_cols))
                
                # 5. Ép kiểu và xuất CSV
                csv_feature = df_feature[valid_cols].to_csv(index=False).encode('utf-8-sig')
                
                st.download_button(
                    label="📥 Save Feature (CSV)",
                    data=csv_feature, file_name="debug_feature_filtered.csv", mime="text/csv",
                    use_container_width=True, type="secondary"
                )
            else:
                if st.button("📥 Save Feature (Trống)", use_container_width=True, type="secondary"):
                    st.warning("⚠️ Vui lòng nhấn nút 'Predict' để phân tích dữ liệu trước khi tải xuống!")
                
        with c_btn2:
            if st.session_state.is_predicted and st.session_state.result_data_df is not None:
                csv_result = st.session_state.result_data_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Save Result (CSV)",
                    data=csv_result, file_name="result.csv", mime="text/csv",
                    use_container_width=True, type="secondary"
                )
            else:
                if st.button("📥 Save Result (Trống)", use_container_width=True, type="secondary"):
                    st.warning("⚠️ Vui lòng nhấn nút 'Predict' để phân tích dữ liệu trước khi tải xuống!")
                
        with c_btn3:
            predict_btn = st.button("🚀 Predict", use_container_width=True, type="secondary")

    if predict_btn:
        with st.spinner(f"Đang thu thập dữ liệu và phân tích mô hình..."):
            vn_tz = Config.TIMEZONE_VN
            dt_combined = datetime.combine(d_date, d_time)
            dt_aware = vn_tz.localize(dt_combined)
            created_at_str = dt_aware.strftime('%Y-%m-%d %H:%M:%S+07:00')
            
            my_input = {
                "caption": caption,
                "music_name": music_name, 
                "kol_name": kol_name,
                "followers": followers,
                "created_at": created_at_str,
                "video_path": video_path_input
            }
            res_payload = run_prediction_for_new_video(my_input)
            
            if res_payload:
                st.session_state.all_preds = res_payload["preds_dict"]
                st.session_state.target_data_df = res_payload["target_data"]
                st.session_state.result_data_df = res_payload["prediction_results"]
                st.session_state.is_predicted = True
                
                st.session_state.raw_history_tab2 = res_payload.get("raw_history", [])
                
                st.success("✅ Phân tích xong! Số liệu đã được cập nhật.")
                time.sleep(0.5)
                st.rerun() 
            else:
                st.error("❌ Không thể lấy dữ liệu từ TikTok hoặc xảy ra lỗi model.")

    if st.session_state.is_predicted and st.session_state.all_preds and selected_model in st.session_state.all_preds:
        st.markdown("---")
        st.subheader("📈 Kết quả dự đoán")
        
        current_pred = st.session_state.all_preds[selected_model]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Views", f"{current_pred['views']:,}")
        m2.metric("Predicted Likes", f"{current_pred['likes']:,}")
        m3.metric("Predicted Shares", f"{current_pred['shares']:,}")
        
        if st.session_state.raw_history_tab2:
            with st.expander("🛠 Debug Log: Thông tin chi tiết các video đã cào"):
                log2_df = pd.DataFrame(st.session_state.raw_history_tab2)
                if not log2_df.empty:
                    log2_df['Ngày đăng'] = pd.to_datetime(log2_df[Col.CREATED_AT]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    cols_to_show = ["post_id", 'Ngày đăng', Col.VIEWS, Col.LIKES, Col.SHARES, Col.COMMENTS, Col.COLLECTS]
                    st.dataframe(log2_df[cols_to_show], use_container_width=True, hide_index=True)
        elif st.session_state.is_predicted:
            st.info("⚠️ Không cào được video cũ từ TikTok. Dự đoán dựa hoàn toàn vào nội dung bạn vừa nhập.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 Nhận xét cải thiện nội dung")
        
        raw_input_for_rec = {
            "caption": caption,
            "music_name": music_name,
            "created_at": f"{d_date.strftime('%Y-%m-%d')} {d_time.strftime('%H:%M:%S')}",
            "followers": followers
        }
        pred_for_rec = {"pred_views": current_pred['views']}
        
        recs = generate_post_recommendations(raw_input_for_rec, pred_for_rec)
        for rec in recs:
            st.markdown(f"- {rec}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 🔍 Feature được trích xuất")
        extracted_info = processor.extract_caption_features(caption)
        df_features = pd.DataFrame([extracted_info])
        st.dataframe(df_features, use_container_width=True, hide_index=True)

# --- TAB 3: ỨNG DỤNG DOANH NGHIỆP ---
with tab_business:
    st.header("🏢 So sánh & Lựa chọn KOL cho Doanh nghiệp")
    st.info("Doanh nghiệp có thể nhập thông tin chiến dịch và danh sách KOL để hệ thống tự động tính điểm, gợi ý Creator phù hợp nhất.")
    
    with st.container():
        col_b_left, col_b_right = st.columns(2)
        
        with col_b_left:
            biz_music = st.text_input("🎵 Âm nhạc sử dụng", value="Original sound", key="biz_music")
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            c_b_date, c_b_time = st.columns(2)
            with c_b_date:
                biz_date = st.date_input("Ngày đăng dự kiến", datetime.now(), key="biz_date")
            with c_b_time:
                biz_time = st.time_input("Giờ đăng dự kiến", dt_time(19, 0), key="biz_time")
                
        with col_b_right:
            biz_caption = st.text_area("Nội dung Caption dự kiến", height=110, value="Sản phẩm mới cực chất! #trending #review", key="biz_caption")
            
            biz_model = st.selectbox(
                "🤖 Chọn mô hình dự đoán:", 
                ["Linear Regression", "Random Forest", "XGBoost"],
                index=2,
                key="biz_model"
            )

    st.markdown("### 📋 Danh sách Creator đầu vào")
    
    edited_biz_df = st.data_editor(st.session_state.biz_kol_df, num_rows="dynamic", use_container_width=True)
    st.session_state.biz_kol_df = edited_biz_df
    
    if st.button("🔄 Tự động tính toán chỉ số trung bình (từ TikTok)", use_container_width=True, type="secondary"):
        with st.spinner("Đang cào dữ liệu từ TikTok cho các KOL... Vui lòng đợi..."):
            temp_df = st.session_state.biz_kol_df.copy()
            all_logs_tab3 = []
            
            for idx, row in temp_df.iterrows():
                kol = str(row['name_of_creator']).strip()
                if kol and kol.lower() not in ['none', 'nan', '']:
                    history = get_latest_history_videos(kol, limit=3)
                    if history:
                        temp_df.at[idx, 'view_avg'] = int(np.mean([h.get(Col.VIEWS, 0) for h in history]))
                        temp_df.at[idx, 'like_avg'] = int(np.mean([h.get(Col.LIKES, 0) for h in history]))
                        temp_df.at[idx, 'share_avg'] = int(np.mean([h.get(Col.SHARES, 0) for h in history]))
                        temp_df.at[idx, 'comment_avg'] = int(np.mean([h.get(Col.COMMENTS, 0) for h in history]))
                        temp_df.at[idx, 'collects_avg'] = int(np.mean([h.get(Col.COLLECTS, 0) for h in history]))
                        
                        for h in history: h['KOL'] = kol
                        all_logs_tab3.extend(history)
            
            st.session_state.biz_kol_df = temp_df
            st.session_state.raw_history_tab3 = all_logs_tab3
            st.rerun() 
            
    if st.session_state.raw_history_tab3:
        with st.expander("🛠 Debug Log: Thông tin chi tiết các video đã cào"):
            log3_df = pd.DataFrame(st.session_state.raw_history_tab3)
            if not log3_df.empty:
                log3_df['Ngày đăng'] = pd.to_datetime(log3_df[Col.CREATED_AT]).dt.strftime('%Y-%m-%d %H:%M:%S')
                cols_to_show_3 = ['KOL', "post_id", 'Ngày đăng', Col.VIEWS, Col.LIKES, Col.SHARES, Col.COMMENTS, Col.COLLECTS]
                st.dataframe(log3_df[cols_to_show_3], use_container_width=True, hide_index=True)

    if st.button("📊 Dự đoán & Đề xuất KOL", type="primary", use_container_width=True):
        with st.spinner("Đang chạy mô hình dự đoán cho từng KOL. Vui lòng chờ..."):
            
            vn_tz = Config.TIMEZONE_VN
            dt_combined_biz = datetime.combine(biz_date, biz_time)
            dt_aware_biz = vn_tz.localize(dt_combined_biz)
            created_at_str = dt_aware_biz.strftime('%Y-%m-%d %H:%M:%S+07:00')
            
            results_list = []
            pred_logs_tab3 = []
            
            for idx, row in st.session_state.biz_kol_df.iterrows():
                kol_name = str(row['name_of_creator']).strip()
                followers = int(row['followers'])
                
                if kol_name and kol_name.lower() not in ['none', 'nan', '']:
                    my_input = {
                        "caption": biz_caption,
                        "music_name": biz_music, 
                        "kol_name": kol_name,
                        "followers": followers,
                        "created_at": created_at_str 
                    }
                    
                    res_payload = run_prediction_for_new_video(my_input)
                    
                    if res_payload and biz_model in res_payload["preds_dict"]:
                        preds = res_payload["preds_dict"][biz_model]
                        v, l, s = preds['views'], preds['likes'], preds['shares']
                        
                        h_data = res_payload.get('raw_history', [])
                        for h in h_data: h['KOL'] = kol_name
                        pred_logs_tab3.extend(h_data)
                    else:
                        v, l, s = int(row['view_avg']), int(row['like_avg']), int(row['share_avg'])
                    
                    score = (v * 0.7) + (l * 1.5) + (s * 2.0)
                    
                    row_data = row.to_dict()
                    row_data['pred_views'] = int(v)
                    row_data['pred_likes'] = int(l)
                    row_data['pred_shares'] = int(s)
                    row_data['kol_score'] = round(score, 2)
                    
                    results_list.append(row_data)
            
            if pred_logs_tab3:
                st.session_state.raw_history_tab3 = pred_logs_tab3
                
            if results_list:
                result_df = pd.DataFrame(results_list)
                result_df = result_df.sort_values("kol_score", ascending=False).reset_index(drop=True)
                
                st.markdown("---")
                st.markdown("### 📌 Kết quả dự đoán theo từng KOL")
                st.dataframe(result_df, use_container_width=True)
                
                st.markdown("### 📊 Biểu đồ so sánh")
                fig = go.Figure(data=[
                    go.Bar(
                        name='Dự đoán Views', x=result_df['name_of_creator'], y=result_df['pred_views'], 
                        marker_color='#FF2C55', text=result_df['pred_views'], textposition='outside', texttemplate='%{text:.2s}'
                    ), 
                    go.Bar(
                        name='Dự đoán Likes', x=result_df['name_of_creator'], y=result_df['pred_likes'], 
                        marker_color='#0066CC', text=result_df['pred_likes'], textposition='outside', texttemplate='%{text:.2s}'
                    ),   
                    go.Bar(
                        name='Dự đoán Shares', x=result_df['name_of_creator'], y=result_df['pred_shares'], 
                        marker_color='#00F2EA', text=result_df['pred_shares'], textposition='outside', texttemplate='%{text:,}' 
                    ) 
                ])
                
                fig.update_layout(
                    barmode='group', 
                    xaxis_title="Tên KOL (Creator)",
                    yaxis_title="Số lượng tương tác dự đoán",
                    height=550, 
                    margin=dict(t=80, b=0), 
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                    yaxis=dict(automargin=True)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                best_creator = result_df.iloc[0]['name_of_creator']
                best_score = result_df.iloc[0]['kol_score']
                
                st.markdown("### ✅ Kết luận")
                st.success(f"Doanh nghiệp nên ưu tiên chọn **{best_creator}** vì có điểm tổng hợp cao nhất ({best_score:,.2f}).")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                campaign_data = {
                    "date": biz_date.strftime("%d/%m/%Y"),
                    "time": biz_time.strftime("%H:%M"),
                    "music": biz_music,
                    "model": biz_model,
                    "caption": biz_caption
                }
                
                temp_img_path = "temp_chart.png"
                try:
                    fig.write_image(temp_img_path)
                    pdf_bytes = create_pdf_report(campaign_data, result_df, chart_image_path=temp_img_path)
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                except Exception as e:
                    st.warning("Không thể đính kèm biểu đồ vào PDF. Đảm bảo đã cài đặt 'kaleido'.")
                    pdf_bytes = create_pdf_report(campaign_data, result_df)

                pdf_filename = f"BaoCao_KOL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.markdown(get_pdf_download_link(pdf_bytes, pdf_filename), unsafe_allow_html=True)
            else:
                st.error("Không có KOL nào hợp lệ để tiến hành dự đoán!")