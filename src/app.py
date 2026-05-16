"""
TikTok Engagement Prediction System — UIT
==========================================
Main Entry Point
"""

import os
import time
import logging
import warnings
from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from fpdf import FPDF
except ImportError:
    st.error("Vui lòng cài đặt: pip install fpdf")

# ─── Suppress noisy logs ────────────────────────────────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════
PAGE_TITLE  = "TikTok Engagement Predictor — UIT"
st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Hàm nạp file CSS ───────────────────────────────────────────────────────
def load_css(file_name: str):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ─── Project local imports ──────────────────────────────────────────────────
from constants import Col, FEATURES, Config
from backend import (
    get_processor, load_ml_models,
    generate_recommendations, get_latest_history_videos, 
    run_prediction, build_pdf_report, pdf_download_link
)

# ─── Constants ──────────────────────────────────────────────────────────────
MODELS_LIST = ["Linear Regression", "Random Forest", "XGBoost"]

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════
def init_session_state() -> None:
    defaults = {
        "all_preds":        None,
        "is_predicted":     False,
        "target_data_df":   None,
        "result_data_df":   None,
        "raw_history_tab2": [],
        "raw_history_tab3": [],
        "biz_kol_df": pd.DataFrame({
            "name_of_creator": ["tranthanh123", "hariwonday", "khoailangthang"],
            "followers":       [7_500_000, 4_400_000, 3_300_000],
            "like_avg":        [0, 0, 0],
            "view_avg":        [0, 0, 0],
            "share_avg":       [0, 0, 0],
            "comment_avg":     [0, 0, 0],
            "collects_avg":    [0, 0, 0],
        }),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════
def render_header() -> None:
    today = datetime.now().strftime("%d/%m/%Y")
    st.markdown(f"""
    <div class="app-header">
        <div class="logo-group">
            <h2>ĐẠI HỌC QUỐC GIA TP.HỒ CHÍ MINH &nbsp;·&nbsp; TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN</h2>
            <p>Khoa Khoa học Máy tính</p>
        </div>
        <div class="badge-group">
            <span class="badge">📅 {today}</span>
            <span class="badge">🤖 ML System</span>
        </div>
    </div>
    <div class="topic-strip">
        <h4>📌 Đề tài: Xây dựng hệ thống dự đoán mức độ tương tác video TikTok dựa trên các thuật toán Machine Learning</h4>
        <div class="meta"><b>Sinh viên:</b> Nguyễn Thị Yến</div>
    </div>
    """, unsafe_allow_html=True)

def section_header(icon_name: str, title: str) -> None:
    st.markdown(f"""
    <div class="section-header" style="display: flex; align-items: center; gap: 8px;">
        <span class="material-icons-round" style="color: var(--primary-mid); font-size: 1.4rem;">{icon_name}</span>
        <h3 style="margin: 0; padding: 0;">{title}</h3>
    </div>""", unsafe_allow_html=True)

def pred_cards(views: int, likes: int, shares: int) -> None:
    c1, c2, c3 = st.columns(3)
    for col, icon, label, val, color in [
        (c1, "visibility", "Predicted Views",  views,  "#003E8A"),
        (c2, "favorite", "Predicted Likes",  likes,  "#FF2C55"),
        (c3, "share", "Predicted Shares", shares, "#00C9C3"),
    ]:
        with col:
            st.markdown(f"""
            <div class="pred-card">
                <div class="icon" style="margin-bottom: 6px;"><span class="material-icons-round" style="font-size: 2rem; color:{color};">{icon}</span></div>
                <div class="label">{label}</div>
                <div class="value" style="color:{color}">{val:,}</div>
            </div>""", unsafe_allow_html=True)

def history_debug_expander(history_list: list, extra_col: str = None) -> None:
    if not history_list:
        return
    with st.expander("🛠 Debug Log — Chi tiết video đã cào từ TikTok"):
        df = pd.DataFrame(history_list)
        if df.empty:
            st.info("Không có dữ liệu.")
            return
        df["Ngày đăng"] = pd.to_datetime(df[Col.CREATED_AT]).dt.strftime("%Y-%m-%d %H:%M")
        show_cols = ([extra_col] if extra_col and extra_col in df.columns else []) + \
                    ["post_id", "Ngày đăng", Col.VIEWS, Col.LIKES, Col.SHARES, Col.COMMENTS, Col.COLLECTS]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
def tab_overview_content() -> None:
    section_header("analytics", "Đánh giá hiệu năng mô hình huấn luyện")

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "models", "model_metrics_report.csv",
    )

    try:
        raw_df = pd.read_csv(csv_path)
        targets = raw_df["Target_Variable"].unique()

        selected = st.selectbox("📍 Chọn biến mục tiêu:", targets)
        filt_df  = raw_df[raw_df["Target_Variable"] == selected].copy()

        display_df = filt_df.rename(columns={
            "Model_Name":    "Mô hình",
            "R2_Score":      "R² Score",
            "Avg_Error_Pct": "Sai số TB (%)",
        })

        col_tbl, col_chart = st.columns([1, 1.6], gap="large")
        with col_tbl:
            section_header("📋", "Bảng chỉ số")
            st.table(display_df[["Mô hình", "MAE", "RMSE", "R² Score", "Sai số TB (%)"]])

        with col_chart:
            section_header("📊", f"R² Score — {selected}")
            COLORS = ["#003E8A", "#FF2C55", "#00C9C3"]
            fig = go.Figure(go.Bar(
                x=display_df["Mô hình"],
                y=display_df["R² Score"],
                marker_color=COLORS[:len(display_df)],
                text=display_df["R² Score"].round(4),
                textposition="outside",
            ))
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=0, l=0, r=0),
                yaxis=dict(range=[0, 1.05], title="R² Score"),
                xaxis_title="Mô hình",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Be Vietnam Pro"),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#E2E6EF")
            st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error(f"Không tìm thấy `{csv_path}`. Hãy chạy lại `main.py` để xuất báo cáo.")
    except Exception as exc:
        st.error(f"Lỗi đọc dữ liệu: {exc}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERSONAL PREDICTION
# ════════════════════════════════════════════════════════════════════════════
def tab_personal_content(processor_obj, models: dict) -> None:
    section_header("account_circle", "Dự đoán hiệu quả bài post cá nhân")

    with st.expander("📝 Nhập thông tin video cần dự đoán", expanded=True):
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #0057BE; font-size: 1.15rem; vertical-align: middle;">account_circle</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">KOL Username</span></div>', unsafe_allow_html=True)
            kol_name   = st.text_input("KOL Username", placeholder="vd: quangtuanofficial0302", label_visibility="collapsed")
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #6366F1; font-size: 1.15rem; vertical-align: middle;">groups</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Số Followers hiện tại</span></div>', unsafe_allow_html=True)
            followers  = st.number_input("Số Followers hiện tại", min_value=0, value=0, step=1000, label_visibility="collapsed")
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #FF2C55; font-size: 1.15rem; vertical-align: middle;">movie</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Đường dẫn MP4 (tuỳ chọn)</span></div>', unsafe_allow_html=True)
            video_path = st.text_input("Đường dẫn MP4", placeholder=r"C:\Video\tiktok.mp4", label_visibility="collapsed")
            
            c_d, c_t = st.columns(2)
            with c_d:
                st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #0CAF60; font-size: 1.15rem; vertical-align: middle;">calendar_month</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Ngày đăng dự kiến</span></div>', unsafe_allow_html=True)
                post_date = st.date_input("Ngày đăng dự kiến", datetime.now(), label_visibility="collapsed")

            with c_t:
                st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #F59E0B; font-size: 1.15rem; vertical-align: middle;">schedule</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Giờ đăng dự kiến</span></div>', unsafe_allow_html=True)
                post_time = st.time_input("Giờ đăng dự kiến",  dt_time(19, 0), label_visibility="collapsed")

        with col_r:
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #0072E5; font-size: 1.15rem; vertical-align: middle;">edit_note</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Nội dung Caption (kèm hashtag)</span></div>', unsafe_allow_html=True)
            caption    = st.text_area("Nội dung Caption (kèm hashtag)", height=100, placeholder="Nhập caption…", label_visibility="collapsed")
            
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #E91E63; font-size: 1.15rem; vertical-align: middle;">music_note</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Âm nhạc sử dụng</span></div>', unsafe_allow_html=True)
            music_name = st.text_input("Âm nhạc sử dụng", placeholder="Tên bài hát…", label_visibility="collapsed")
            
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #9C27B0; font-size: 1.15rem; vertical-align: middle;">smart_toy</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Mô hình dự đoán</span></div>', unsafe_allow_html=True)
            sel_model  = st.selectbox("Mô hình dự đoán", MODELS_LIST, index=2, label_visibility="collapsed")

            v_val = l_val = s_val = "—"
            if (st.session_state.is_predicted
                    and st.session_state.all_preds
                    and sel_model in st.session_state.all_preds):
                d = st.session_state.all_preds[sel_model]
                v_val = f"{d['views']:,}"
                l_val = f"{d['likes']:,}"
                s_val = f"{d['shares']:,}"

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #003E8A; font-size: 1.15rem; vertical-align: middle;">visibility</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Views</span></div>', unsafe_allow_html=True)
                st.text_input("Views", value=v_val, disabled=False, label_visibility="collapsed")

            with c2:
                st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #FF2C55; font-size: 1.15rem; vertical-align: middle;">favorite</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Likes</span></div>', unsafe_allow_html=True)
                st.text_input("Likes", value=l_val, disabled=False, label_visibility="collapsed")

            with c3:
                st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #00C9C3; font-size: 1.15rem; vertical-align: middle;">share</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Shares</span></div>', unsafe_allow_html=True)
                st.text_input("Shares", value=s_val, disabled=False, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    btn_feat, btn_res, btn_pred = st.columns([1, 1, 1], gap="small")

    with btn_feat:
        if st.session_state.is_predicted and st.session_state.target_data_df is not None:
            df_feat = st.session_state.target_data_df.copy()
            keep_cols = list(dict.fromkeys(
                ["post_id", Col.AUTHOR, Col.CREATED_AT, Col.CAPTION_CLEAN, Col.HASHTAG_STR] + FEATURES
            ))
            keep_cols = [c for c in keep_cols if c in df_feat.columns]
            
            # Thay thế bằng icon file_download tròn mịn của Google nằm trong nút
            st.download_button(
                ":material/file_download: Tải Feature CSV", 
                data=df_feat[keep_cols].to_csv(index=False).encode("utf-8-sig"),
                file_name="debug_feature.csv", 
                mime="text/csv",
                use_container_width=True, 
                type="secondary",
            )
        else:
            # Thay thế bằng icon file_download_off để báo nút đang bị khóa/trống
            st.button(
                ":material/file_download_off: Feature CSV (trống)", 
                use_container_width=True, 
                type="secondary", 
                disabled=True
            )

    with btn_res:
        if st.session_state.is_predicted and st.session_state.result_data_df is not None:
            # Thay thế bằng icon file_download chuẩn Google Material cho trạng thái có dữ liệu
            st.download_button(
                ":material/file_download: Tải Result CSV",
                data=st.session_state.result_data_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="result.csv", mime="text/csv",
                use_container_width=True, type="secondary",
            )
        else:
            # Thay thế bằng icon file_download_off cho trạng thái bảng trống
            st.button(
                ":material/file_download_off: Result CSV (trống)", 
                use_container_width=True, 
                type="secondary", 
                disabled=True
            )

    with btn_pred:
        # Thay thế emoji tên lửa cũ bằng icon vector rocket_launch tròn mịn của Google
        predict_clicked = st.button(
            ":material/rocket_launch: Dự đoán", 
            use_container_width=True, 
            type="primary"
        )

    if predict_clicked:
        if not kol_name:
            st.warning("Vui lòng nhập KOL Username.")
        else:
            vn_tz = Config.TIMEZONE_VN
            created_at_str = vn_tz.localize(
                datetime.combine(post_date, post_time)
            ).strftime("%Y-%m-%d %H:%M:%S+07:00")

            my_input = {
                "caption":    caption,
                "music_name": music_name,
                "kol_name":   kol_name,
                "followers":  followers,
                "created_at": created_at_str,
                "video_path": video_path,
            }

            with st.spinner("Đang thu thập dữ liệu từ TikTok và chạy mô hình…"):
                payload = run_prediction(my_input, processor_obj, models)

            if payload:
                st.session_state.all_preds        = payload["preds_dict"]
                st.session_state.target_data_df   = payload["target_data"]
                st.session_state.result_data_df   = payload["prediction_results"]
                st.session_state.raw_history_tab2 = payload.get("raw_history", [])
                st.session_state.is_predicted     = True
                st.success("Phân tích hoàn tất!", icon=":material/check_circle:")
                time.sleep(0.4)
                st.rerun()
            else:
                st.error("Không thể lấy dữ liệu hoặc xảy ra lỗi mô hình.", icon=":material/error:")

    if (st.session_state.is_predicted
            and st.session_state.all_preds
            and sel_model in st.session_state.all_preds):

        st.markdown("---")
        cur = st.session_state.all_preds[sel_model]
        section_header("trending_up", f"Kết quả dự đoán — {sel_model}")
        pred_cards(cur["views"], cur["likes"], cur["shares"])

        history_debug_expander(st.session_state.raw_history_tab2)

        section_header("psychology", "Nhận xét & Gợi ý cải thiện nội dung")
        recs = generate_recommendations(
            {"caption": caption, "music_name": music_name,
             "created_at": f"{post_date} {post_time}", "followers": followers},
            {"pred_views": cur["views"]},
            processor_obj,
        )
        for rec in recs:
            st.markdown(f'<div class="rec-item">💡 {rec}</div>', unsafe_allow_html=True)

        section_header("manage_search", "Feature đã trích xuất từ caption")
        extracted = processor_obj.extract_caption_features(caption)
        st.dataframe(pd.DataFrame([extracted]), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — BUSINESS / KOL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
def tab_business_content(processor_obj, models: dict) -> None:
    section_header("business", "So sánh & Lựa chọn KOL cho Doanh nghiệp")
    st.info("Nhập thông tin chiến dịch và danh sách KOL — hệ thống tự động tính điểm và gợi ý Creator phù hợp nhất.")

    with st.expander(":material/settings: Cấu hình chiến dịch", expanded=True):
        c_l, c_r = st.columns(2, gap="large")
        with c_l:
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #E91E63; font-size: 1.15rem; vertical-align: middle;">music_note</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Âm nhạc</span></div>', unsafe_allow_html=True)
            biz_music = st.text_input("Âm nhạc", placeholder="Original sound", key="biz_music", label_visibility="collapsed")
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #FF2C55; font-size: 1.15rem; vertical-align: middle;">movie</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Đường dẫn MP4 (tuỳ chọn)</span></div>', unsafe_allow_html=True)
            biz_video_path = st.text_input("Đường dẫn MP4 (tuỳ chọn)", placeholder=r"C:\Video\tiktok.mp4", key="biz_video_path", label_visibility="collapsed")
            c_d, c_t = st.columns(2)
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #0CAF60; font-size: 1.15rem; vertical-align: middle;">calendar_month</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Ngày đăng dự kiến</span></div>', unsafe_allow_html=True)
            biz_date  = st.date_input("Ngày đăng dự kiến", datetime.now(), key="biz_date", label_visibility="collapsed")
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #F59E0B; font-size: 1.15rem; vertical-align: middle;">schedule</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Giờ đăng dự kiến</span></div>', unsafe_allow_html=True)
            biz_time  = st.time_input("Giờ đăng dự kiến",  dt_time(19, 0), key="biz_time", label_visibility="collapsed")
        with c_r:
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #0072E5; font-size: 1.15rem; vertical-align: middle;">edit_note</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Nội dung Caption dự kiến</span></div>', unsafe_allow_html=True)
            biz_caption = st.text_area("Nội dung Caption dự kiến", height=112, placeholder="Sản phẩm mới cực chất! #trending #review", key="biz_caption", label_visibility="collapsed")
            st.markdown('<div class="custom-label" style="display: flex; align-items: center; gap: 6px;"><span class="material-icons-round" style="color: #9C27B0; font-size: 1.15rem; vertical-align: middle;">smart_toy</span><span style="font-weight: 600; font-size: 0.85rem; color: var(--text); vertical-align: middle;">Mô hình dự đoán</span></div>', unsafe_allow_html=True)
            biz_model = st.selectbox("Mô hình dự đoán", MODELS_LIST, index=2, key="biz_model", label_visibility="collapsed")

    section_header("groups", "Danh sách Creator")
    edited_df = st.data_editor(
        st.session_state.biz_kol_df, 
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name_of_creator": st.column_config.TextColumn(
                "Tên Creator", 
                help="Tên tài khoản người sáng tạo nội dung"
            ),
            "followers": st.column_config.NumberColumn(
                "Followers", 
                format="%d"
            ),
            "like_avg": st.column_config.NumberColumn(
                "Likes Avg", 
                format="%d"
            ),
            "view_avg": st.column_config.NumberColumn(
                "Views Avg", 
                format="%d"
            ),
            "share_avg": st.column_config.NumberColumn(
                "Shares Avg", 
                format="%d"
            ),
            "comment_avg": st.column_config.NumberColumn(
                "Comments Avg", 
                format="%d"
            ),
            "collects_avg": st.column_config.NumberColumn(
                "Collects Avg", 
                format="%d"
            ),
        }
    )
    st.session_state.biz_kol_df = edited_df

    col_auto, col_pred = st.columns(2, gap="small")

    with col_auto:
        if st.button(":material/sync: Tự động lấy chỉ số trung bình từ TikTok", use_container_width=True, type="secondary"):
            with st.spinner("Đang cào dữ liệu TikTok…"):
                tmp_df = st.session_state.biz_kol_df.copy()
                all_logs: list = []
                for idx, row in tmp_df.iterrows():
                    kol = str(row["name_of_creator"]).strip()
                    if kol and kol.lower() not in ("none", "nan", ""):
                        hist = get_latest_history_videos(kol, limit=3)
                        if hist:
                            def _avg(key): return int(np.mean([h.get(key, 0) for h in hist]))
                            tmp_df.at[idx, "view_avg"]    = _avg(Col.VIEWS)
                            tmp_df.at[idx, "like_avg"]    = _avg(Col.LIKES)
                            tmp_df.at[idx, "share_avg"]   = _avg(Col.SHARES)
                            tmp_df.at[idx, "comment_avg"] = _avg(Col.COMMENTS)
                            tmp_df.at[idx, "collects_avg"]= _avg(Col.COLLECTS)
                            for h in hist: h["KOL"] = kol
                            all_logs.extend(hist)
                st.session_state.biz_kol_df       = tmp_df
                st.session_state.raw_history_tab3 = all_logs
            st.rerun()

    with col_pred:
        run_biz = st.button(":material/leaderboard: Dự đoán & Xếp hạng KOL",  use_container_width=True, type="primary")

    history_debug_expander(st.session_state.raw_history_tab3, extra_col="KOL")

    if run_biz:
        with st.spinner("Đang chạy mô hình cho từng KOL…"):
            vn_tz = Config.TIMEZONE_VN
            created_at_str = vn_tz.localize(
                datetime.combine(biz_date, biz_time)
            ).strftime("%Y-%m-%d %H:%M:%S+07:00")

            results_list: list = []
            pred_logs: list    = []

            for _, row in st.session_state.biz_kol_df.iterrows():
                kol = str(row["name_of_creator"]).strip()
                if not kol or kol.lower() in ("none", "nan", ""):
                    continue

                inp = {
                    "caption":    biz_caption,
                    "music_name": biz_music,
                    "kol_name":   kol,
                    "followers":  int(row["followers"]),
                    "created_at": created_at_str,
                    "video_path": biz_video_path,
                }
                payload = run_prediction(inp, processor_obj, models)

                if payload and biz_model in payload["preds_dict"]:
                    p = payload["preds_dict"][biz_model]
                    v, l, s = p["views"], p["likes"], p["shares"]
                    for h in payload.get("raw_history", []):
                        h["KOL"] = kol
                        pred_logs.append(h)
                else:
                    v, l, s = int(row["view_avg"]), int(row["like_avg"]), int(row["share_avg"])

                score = v * 0.7 + l * 1.5 + s * 2.0
                row_d = row.to_dict()
                row_d.update(pred_views=int(v), pred_likes=int(l), pred_shares=int(s),
                             kol_score=round(score, 2))
                results_list.append(row_d)

            if pred_logs:
                st.session_state.raw_history_tab3 = pred_logs

        if not results_list:
            st.error("Không có KOL hợp lệ nào.")
            return

        result_df = (
            pd.DataFrame(results_list)
            .sort_values("kol_score", ascending=False)
            .reset_index(drop=True)
        )

        section_header("emoji_events", "Kết quả xếp hạng KOL")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        section_header("bar_chart", "Biểu đồ so sánh")
        fig = go.Figure([
            go.Bar(name="Views",  x=result_df["name_of_creator"], y=result_df["pred_views"],
                   marker_color="#003E8A", text=result_df["pred_views"],
                   textposition="outside", texttemplate="%{text:.2s}"),
            go.Bar(name="Likes",  x=result_df["name_of_creator"], y=result_df["pred_likes"],
                   marker_color="#FF2C55", text=result_df["pred_likes"],
                   textposition="outside", texttemplate="%{text:.2s}"),
            go.Bar(name="Shares", x=result_df["name_of_creator"], y=result_df["pred_shares"],
                   marker_color="#00C9C3", text=result_df["pred_shares"],
                   textposition="outside", texttemplate="%{text:,}"),
        ])
        fig.update_layout(
            barmode="group", height=500,
            margin=dict(t=60, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Creator", yaxis_title="Lượt tương tác dự đoán",
            font=dict(family="Be Vietnam Pro"),
        )
        fig.update_yaxes(showgrid=True, gridcolor="#E2E6EF")
        st.plotly_chart(fig, use_container_width=True)

        best = result_df.iloc[0]
        st.success(
            f"✅ Hệ thống gợi ý chọn **{best['name_of_creator']}** "
            f"với điểm tổng hợp cao nhất: **{best['kol_score']:,.2f}**"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        campaign_meta = {
            "date":    biz_date.strftime("%d/%m/%Y"),
            "time":    biz_time.strftime("%H:%M"),
            "music":   biz_music,
            "model":   biz_model,
            "caption": biz_caption,
        }
        tmp_chart = "temp_chart.png"
        try:
            fig.write_image(tmp_chart)
            pdf_bytes = build_pdf_report(campaign_meta, result_df, tmp_chart)
            if os.path.exists(tmp_chart):
                os.remove(tmp_chart)
        except Exception:
            st.warning("Không thể đính kèm biểu đồ vào PDF (cài `kaleido` để hỗ trợ).")
            pdf_bytes = build_pdf_report(campaign_meta, result_df)

        fname = f"BaoCao_KOL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.markdown(pdf_download_link(pdf_bytes, fname), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # --- THAY ĐỔI Ở ĐÂY: Gọi hàm nạp file style.css ---
    load_css("style.css")
    
    init_session_state()
    render_header()

    processor_obj = get_processor()
    models        = load_ml_models()

    tab1, tab2, tab3 = st.tabs([
        ":material/analytics: Tổng quan mô hình",
        ":material/person: Ứng dụng cá nhân",
        ":material/business: Ứng dụng doanh nghiệp",
    ])

    with tab1:
        tab_overview_content()

    with tab2:
        tab_personal_content(processor_obj, models)

    with tab3:
        tab_business_content(processor_obj, models)

    st.markdown("""
    <div class="app-footer">
        Hệ thống dự đoán tương tác TikTok &nbsp;·&nbsp; UIT — Đại học Quốc gia TP.HCM &nbsp;·&nbsp;
        Built với Streamlit &amp; Machine Learning
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
