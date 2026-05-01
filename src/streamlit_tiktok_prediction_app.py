import os
import re
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Optional: nếu bạn có sẵn project hiện tại thì import các thành phần này
# Code demo vẫn chạy được bằng mock prediction nếu chưa có model.
try:
    import joblib
    from constants import FEATURES, TARGETS, EMOJI_PATTERN
    from processor import TikTokDataProcessor
    PROJECT_AVAILABLE = True
except Exception:
    PROJECT_AVAILABLE = False
    TARGETS = ["likes", "views", "shares"]
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="TikTok Engagement Prediction",
    page_icon="📊",
    layout="wide",
)

MODEL_PATHS = {
    "Linear Regression": r"C:\Users\Admin\OneDrive\Desktop\LinearRegression\src\models\tiktok_linear_regression_multi_model.pkl",
    "Random Forest": r"C:\Users\Admin\OneDrive\Desktop\LinearRegression\src\models\tiktok_random_forest_multi_model.pkl",
    "XGBoost": r"C:\Users\Admin\OneDrive\Desktop\LinearRegression\src\models\xgboost_multioutput_model.pkl",
}


# =========================
# FEATURE FUNCTIONS
# =========================

def extract_features_from_caption(caption: str) -> dict:
    """Trích xuất hashtag, số từ, emoji, caption clean từ caption."""
    caption = caption or ""

    hashtags = re.findall(r"#(\w+)", caption)
    hashtag_str = " ".join([f"#{h}" for h in hashtags]).lower()
    hashtag_count = len(hashtags)
    emoji_count = len(EMOJI_PATTERN.findall(caption))

    clean_text = re.sub(r"#\w+", "", caption)
    clean_text = EMOJI_PATTERN.sub("", clean_text)
    clean_text = re.sub(r"[^\w\s]", "", clean_text)
    clean_text = " ".join(clean_text.split())

    word_count = len(clean_text.split()) if clean_text else 0
    clean_len = len(clean_text)

    return {
        "hashtag_count": hashtag_count,
        "word_count": word_count,
        "caption_clean": clean_text.lower(),
        "hashtag_str": hashtag_str,
        "emoji_count": emoji_count,
        "caption_clean_len": clean_len,
    }


def mock_predict(caption: str, music_name: str, followers: int, created_at: str, kol_stats: dict | None = None) -> dict:
    """Dự đoán giả lập để app chạy được khi chưa load được model thật."""
    features = extract_features_from_caption(caption)

    base_followers = max(followers, 1)
    hashtag_bonus = min(features["hashtag_count"] * 0.08, 0.4)
    emoji_bonus = min(features["emoji_count"] * 0.05, 0.25)
    trend_bonus = 0.25 if any(tag in features["hashtag_str"] for tag in ["#fyp", "#trend", "#viral"]) else 0
    sound_bonus = 0.15 if "original" not in music_name.lower() else 0.05

    hour = pd.to_datetime(created_at).hour
    time_bonus = 0.2 if 18 <= hour <= 22 else (-0.15 if 0 <= hour <= 5 else 0)

    kol_multiplier = 1.0
    if kol_stats:
        view_avg = float(kol_stats.get("view_avg", 0) or 0)
        like_avg = float(kol_stats.get("like_avg", 0) or 0)
        share_avg = float(kol_stats.get("share_avg", 0) or 0)
        kol_multiplier += min(view_avg / max(base_followers * 10, 1), 1.5)
        kol_multiplier += min(like_avg / max(base_followers, 1), 0.8)
        kol_multiplier += min(share_avg / max(base_followers, 1), 0.4)

    engagement_factor = 1 + hashtag_bonus + emoji_bonus + trend_bonus + sound_bonus + time_bonus

    pred_views = int(base_followers * 2.8 * engagement_factor * kol_multiplier)
    pred_likes = int(pred_views * 0.08)
    pred_shares = int(pred_views * 0.012)

    return {
        "pred_views": max(pred_views, 0),
        "pred_likes": max(pred_likes, 0),
        "pred_shares": max(pred_shares, 0),
    }


@st.cache_resource
def load_processor():
    if not PROJECT_AVAILABLE:
        return None
    processor = TikTokDataProcessor()
    processor.load_trends()
    return processor


@st.cache_resource
def load_model(model_name: str):
    path = MODEL_PATHS.get(model_name)
    if not path or not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_with_real_model(raw_input: dict, model_name: str) -> dict | None:
    """Dự đoán bằng model thật nếu project/model tồn tại."""
    if not PROJECT_AVAILABLE:
        return None

    model = load_model(model_name)
    if model is None:
        return None

    try:
        caption_info = extract_features_from_caption(raw_input.get("caption", ""))
        full_case = {**raw_input, **caption_info}

        # Các cột mặc định để tránh KeyError khi process 1 dòng input
        full_case["media_url"] = full_case.get("media_url", "")
        full_case["author_username"] = full_case.get("name_of_creator", "unknown_user")
        full_case["views"] = float(full_case.get("view_avg", 0) or 0)
        full_case["likes"] = float(full_case.get("like_avg", 0) or 0)
        full_case["shares"] = float(full_case.get("share_avg", 0) or 0)

        df_input = pd.DataFrame([full_case])
        processor = load_processor()
        processed_df = processor.process_features(df_input)

        processed_df["caption_length"] = processed_df.get("caption_clean_len", caption_info["caption_clean_len"])

        # Với single post, nếu không có lịch sử author thì set 0
        for col in [
            "avg_views_last_3_videos",
            "ema_views_last_3",
            "hist_like_rate",
            "days_since_last_post",
        ]:
            if col in processed_df.columns:
                processed_df[col] = float(raw_input.get(col, 0) or 0)

        X_input = processed_df.reindex(columns=FEATURES, fill_value=0)
        log_pred = model.predict(X_input).flatten()

        pred_likes = int(np.expm1(log_pred[0]))
        pred_views = int(np.expm1(log_pred[1]))
        pred_shares = int(np.expm1(log_pred[2]))

        return {
            "pred_views": max(pred_views, 0),
            "pred_likes": max(pred_likes, 0),
            "pred_shares": max(pred_shares, 0),
        }
    except Exception as e:
        st.warning(f"Không chạy được model thật, chuyển sang demo prediction. Lỗi: {e}")
        return None


def predict_engagement(raw_input: dict, model_name: str = "Demo Mock") -> dict:
    if model_name != "Demo Mock":
        real_result = predict_with_real_model(raw_input, model_name)
        if real_result is not None:
            return real_result

    return mock_predict(
        caption=raw_input.get("caption", ""),
        music_name=raw_input.get("music_name", ""),
        followers=int(raw_input.get("followers", 0) or 0),
        created_at=raw_input.get("created_at", datetime.now().isoformat()),
        kol_stats=raw_input,
    )


# =========================
# RECOMMENDATION FUNCTIONS
# =========================

def generate_post_recommendations(raw_input: dict, prediction: dict) -> list[str]:
    caption = raw_input.get("caption", "")
    music_name = raw_input.get("music_name", "")
    created_at = raw_input.get("created_at", datetime.now().isoformat())
    followers = int(raw_input.get("followers", 0) or 0)
    info = extract_features_from_caption(caption)

    recs = []

    if info["caption_clean_len"] < 30:
        recs.append("Caption đang khá ngắn; nên thêm thông tin lợi ích, cảm xúc hoặc câu hook ở đầu caption.")
    elif info["caption_clean_len"] > 180:
        recs.append("Caption hơi dài; nên rút gọn, đưa ý chính lên đầu để người xem nắm nhanh.")
    else:
        recs.append("Độ dài caption tương đối ổn, có thể tiếp tục tối ưu bằng CTA rõ hơn.")

    if info["hashtag_count"] < 3:
        recs.append("Hashtag còn ít; nên thêm 3–5 hashtag gồm: hashtag ngành, hashtag trend và hashtag thương hiệu.")
    elif info["hashtag_count"] > 8:
        recs.append("Hashtag hơi nhiều; nên giữ lại hashtag liên quan nhất để tránh cảm giác spam.")
    else:
        recs.append("Số lượng hashtag tương đối hợp lý.")

    generic_tags = ["#fyp", "#video", "#viral"]
    if any(tag in info["hashtag_str"] for tag in generic_tags):
        recs.append("Hashtag như #fyp/#video khá chung; nên bổ sung hashtag cụ thể theo sản phẩm, ngành hoặc nhóm khách hàng.")

    if info["emoji_count"] == 0:
        recs.append("Caption chưa có emoji; có thể thêm 1–3 emoji phù hợp để tăng độ nổi bật.")
    elif info["emoji_count"] > 6:
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
        view_rate = prediction["pred_views"] / followers
        if view_rate < 1:
            recs.append("View dự đoán thấp hơn followers; nên cải thiện hook 3 giây đầu và thumbnail/video mở đầu.")
        elif view_rate > 3:
            recs.append("View dự đoán khá tốt so với followers; nội dung có tiềm năng phân phối rộng.")

    return recs


def calculate_kol_score(row: pd.Series) -> float:
    views = float(row.get("pred_views", 0) or 0)
    likes = float(row.get("pred_likes", 0) or 0)
    shares = float(row.get("pred_shares", 0) or 0)

    # Score có thể đổi theo mục tiêu chiến dịch
    return views * 0.5 + likes * 3 + shares * 8


def recommend_best_kol(df_result: pd.DataFrame, campaign_goal: str) -> str:
    if df_result.empty:
        return "Chưa có dữ liệu KOL để kết luận."

    if campaign_goal == "Tăng nhận diện thương hiệu":
        best = df_result.sort_values("pred_views", ascending=False).iloc[0]
        metric = "Views dự đoán"
        value = best["pred_views"]
    elif campaign_goal == "Tăng tương tác":
        df_result = df_result.copy()
        df_result["interaction_score"] = df_result["pred_likes"] + df_result["pred_shares"] * 3
        best = df_result.sort_values("interaction_score", ascending=False).iloc[0]
        metric = "Like + Share dự đoán"
        value = int(best["interaction_score"])
    else:
        best = df_result.sort_values("kol_score", ascending=False).iloc[0]
        metric = "điểm tổng hợp"
        value = int(best["kol_score"])

    return f"Doanh nghiệp nên ưu tiên chọn {best['name_of_creator']} vì có {metric} cao nhất ({value:,})."


# =========================
# PDF EXPORT
# =========================

def create_pdf_report(df_result: pd.DataFrame, conclusion: str) -> BytesIO:
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        # Page 1: Table summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("TikTok KOL Prediction Report", fontsize=18, pad=20)

        summary_text = f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nConclusion:\n{conclusion}"
        ax.text(0.02, 0.92, summary_text, fontsize=11, va="top", wrap=True)

        table_cols = ["name_of_creator", "followers", "pred_views", "pred_likes", "pred_shares", "kol_score"]
        table_df = df_result[table_cols].copy()
        for col in ["followers", "pred_views", "pred_likes", "pred_shares", "kol_score"]:
            table_df[col] = table_df[col].apply(lambda x: f"{int(x):,}")

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Chart
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(df_result))
        width = 0.25

        ax.bar(x - width, df_result["pred_views"], width, label="Views")
        ax.bar(x, df_result["pred_likes"], width, label="Likes")
        ax.bar(x + width, df_result["pred_shares"], width, label="Shares")

        ax.set_title("Predicted Engagement by KOL")
        ax.set_xticks(x)
        ax.set_xticklabels(df_result["name_of_creator"], rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer


# =========================
# UI
# =========================

st.title("📊 TikTok Engagement Prediction System")
st.caption("Demo dự đoán View / Like / Share và hỗ trợ tối ưu nội dung/KOL cho marketing.")

with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_options = ["Demo Mock", "Linear Regression", "Random Forest", "XGBoost"]
    selected_model = st.selectbox("Chọn model", model_options)

    if selected_model != "Demo Mock" and not PROJECT_AVAILABLE:
        st.warning("Chưa import được project/model, app sẽ tự fallback sang Demo Mock.")

    st.markdown("---")
    st.write("**Gợi ý:** Dùng Demo Mock để chạy UI trước. Khi có model `.pkl`, chọn model thật ở đây.")


tab_personal, tab_business = st.tabs(["👤 Ứng dụng cá nhân", "🏢 Ứng dụng doanh nghiệp"])


# =========================
# TAB 1: PERSONAL
# =========================

with tab_personal:
    st.subheader("👤 Dự đoán hiệu quả bài post cá nhân")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        caption = st.text_area(
            "Caption",
            value="Video mới nè mọi người xem nha #fyp #video",
            height=120,
        )
        music_name = st.text_input("Music name", value="Original sound")
        followers = st.number_input("Followers", min_value=0, value=8500, step=100)
        created_at_date = st.date_input("Ngày đăng", value=datetime(2026, 4, 25).date())
        created_at_time = st.time_input("Giờ đăng", value=datetime.strptime("03:15", "%H:%M").time())

        created_at = f"{created_at_date} {created_at_time}+07:00"

        predict_btn = st.button("🚀 Dự đoán bài post", type="primary")

    with col2:
        st.info(
            "Hệ thống sẽ trích xuất caption, hashtag, emoji, music, thời gian đăng và followers để dự đoán tương tác."
        )

    if predict_btn:
        raw_input = {
            "caption": caption,
            "music_name": music_name,
            "followers": followers,
            "created_at": created_at,
        }

        result = predict_engagement(raw_input, selected_model)
        recs = generate_post_recommendations(raw_input, result)

        st.markdown("### 📈 Kết quả dự đoán")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Views", f"{result['pred_views']:,}")
        m2.metric("Predicted Likes", f"{result['pred_likes']:,}")
        m3.metric("Predicted Shares", f"{result['pred_shares']:,}")

        st.markdown("### 🧠 Nhận xét cải thiện nội dung")
        for rec in recs:
            st.write(f"- {rec}")

        st.markdown("### 🔍 Feature được trích xuất")
        st.dataframe(pd.DataFrame([extract_features_from_caption(caption)]), use_container_width=True)


# =========================
# TAB 2: BUSINESS
# =========================

with tab_business:
    st.subheader("🏢 So sánh KOL/Creator cho doanh nghiệp")

    st.write(
        "Doanh nghiệp có thể nhập thủ công hoặc upload Excel danh sách KOL gồm: "
        "`name_of_creator`, `followers`, `like_avg`, `view_avg`, `share_avg`."
    )

    campaign_goal = st.selectbox(
        "Mục tiêu chiến dịch",
        ["Cân bằng", "Tăng nhận diện thương hiệu", "Tăng tương tác"],
    )

    common_caption = st.text_area(
        "Caption chiến dịch",
        value="Video mới nè mọi người xem nha #fyp #video",
        height=100,
        key="business_caption",
    )
    common_music = st.text_input("Music name cho chiến dịch", value="Original sound", key="business_music")
    common_created_at_date = st.date_input("Ngày đăng chiến dịch", value=datetime(2026, 4, 25).date(), key="business_date")
    common_created_at_time = st.time_input("Giờ đăng chiến dịch", value=datetime.strptime("03:15", "%H:%M").time(), key="business_time")
    common_created_at = f"{common_created_at_date} {common_created_at_time}+07:00"

    st.markdown("---")
    st.markdown("### Cách 1: Upload Excel")

    uploaded_file = st.file_uploader("Upload file Excel KOL", type=["xlsx", "xls", "csv"])

    sample_df = pd.DataFrame(
        {
            "name_of_creator": ["Creator A", "Creator B", "Creator C"],
            "followers": [8500, 20000, 12000],
            "like_avg": [1200, 2500, 1800],
            "view_avg": [30000, 60000, 45000],
            "share_avg": [150, 300, 220],
        }
    )

    with st.expander("Xem mẫu format Excel"):
        st.dataframe(sample_df, use_container_width=True)

    st.markdown("### Cách 2: Nhập thủ công nhanh")
    manual_data = st.data_editor(
        sample_df,
        num_rows="dynamic",
        use_container_width=True,
    )

    kol_df = None
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            kol_df = pd.read_csv(uploaded_file)
        else:
            kol_df = pd.read_excel(uploaded_file)
    else:
        kol_df = manual_data.copy()

    required_cols = {"name_of_creator", "followers", "like_avg", "view_avg", "share_avg"}
    missing_cols = required_cols - set(kol_df.columns)

    if missing_cols:
        st.error(f"File/dữ liệu thiếu cột: {', '.join(missing_cols)}")
    else:
        st.markdown("### Danh sách KOL đầu vào")
        st.dataframe(kol_df, use_container_width=True)

        if st.button("📊 Dự đoán & chọn KOL", type="primary"):
            results = []

            for _, row in kol_df.iterrows():
                raw_input = {
                    "caption": common_caption,
                    "music_name": common_music,
                    "name_of_creator": row["name_of_creator"],
                    "followers": int(row["followers"] or 0),
                    "like_avg": float(row["like_avg"] or 0),
                    "view_avg": float(row["view_avg"] or 0),
                    "share_avg": float(row["share_avg"] or 0),
                    "created_at": common_created_at,
                }

                pred = predict_engagement(raw_input, selected_model)
                results.append({**raw_input, **pred})

            df_result = pd.DataFrame(results)
            df_result["kol_score"] = df_result.apply(calculate_kol_score, axis=1)
            df_result = df_result.sort_values("kol_score", ascending=False).reset_index(drop=True)

            conclusion = recommend_best_kol(df_result, campaign_goal)

            st.session_state["df_result"] = df_result
            st.session_state["conclusion"] = conclusion

    if "df_result" in st.session_state:
        df_result = st.session_state["df_result"]
        conclusion = st.session_state["conclusion"]

        st.markdown("### 📌 Kết quả dự đoán theo từng KOL")
        st.dataframe(
            df_result[
                [
                    "name_of_creator",
                    "followers",
                    "like_avg",
                    "view_avg",
                    "share_avg",
                    "pred_views",
                    "pred_likes",
                    "pred_shares",
                    "kol_score",
                ]
            ],
            use_container_width=True,
        )

        st.markdown("### 📊 Biểu đồ so sánh")
        chart_df = df_result.set_index("name_of_creator")[["pred_views", "pred_likes", "pred_shares"]]
        st.bar_chart(chart_df)

        st.markdown("### ✅ Kết luận")
        st.success(conclusion)

        pdf_buffer = create_pdf_report(df_result, conclusion)
        st.download_button(
            label="📄 Tải báo cáo PDF",
            data=pdf_buffer,
            file_name="tiktok_kol_prediction_report.pdf",
            mime="application/pdf",
        )
