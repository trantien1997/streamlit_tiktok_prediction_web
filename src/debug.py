import pandas as pd
import os
from processor import TikTokDataProcessor

def main():
    print("⏳ Đang tải các mô hình AI (YOLO, CLIP, PhoBERT)... Vui lòng đợi...")
    processor = TikTokDataProcessor()
    processor.load_trends()
    print("✅ Đã tải xong mô hình!\n")

    # 1. Đường dẫn video của bạn
    video_path = r"D:\Video\tiktok_video_cache\videos\6.946757260609687e_18.mp4"
    video_path = video_path.strip('\"').strip('\'')

    if not os.path.exists(video_path):
        print(f"❌ Lỗi: Không tìm thấy file tại đường dẫn '{video_path}'")
        return

    # 2. Tạo một dòng dữ liệu (Text/Chỉ số) giả lập để test việc gộp cột
    print("📝 Đang tạo dữ liệu Text/Chỉ số để đưa vào gộp chung...")
    data = [{
        "author_username": "test_user",
        "created_at": "2024-05-15 10:00:00+07:00",
        "caption": "Video đi chơi quá đã mọi người ơi #dulich #trending",
        "music_name": "Original sound",
        "views": 10000, 
        "likes": 2000, 
        "shares": 150, 
        "comments": 80, 
        "collects": 40
    }]
    df_input = pd.DataFrame(data)

    # 3. Chạy hàm process_features (Tiến hành GỘP Text + Video)
    print(f"\n🔄 Đang phân tích tổng hợp (Text + Hình ảnh) và gộp cột...")
    try:
        output_file = "merged_features_debug.csv"
        
        # Hàm này sẽ vừa trích xuất ảnh, vừa phân tích text, sau đó đính kèm vào nhau
        df_merged = processor.process_features(
            df_input, 
            video_path=video_path, 
            debug_csv_path=output_file
        )
        
        # 4. Hiển thị thông tin kiểm tra xem gộp ok chưa
        print("\n🎉 PHÂN TÍCH VÀ GỘP CỘT THÀNH CÔNG!")
        print(f" ├─ Kích thước bảng dữ liệu cuối: {df_merged.shape[0]} dòng x {df_merged.shape[1]} cột")
        
        # Kiểm tra sự tồn tại của các nhóm đặc trưng
        has_text = "hashtag_count" in df_merged.columns
        has_phobert = "score_caption" in df_merged.columns
        has_visual = "video_duration_sec" in df_merged.columns
        has_clip = "clip_raw_000" in df_merged.columns
        
        print("\n🔍 KIỂM TRA CHẤT LƯỢNG GỘP CỘT:")
        print(f" ├─ Cột NLP/Text (Hashtag, Emoji): {'✅ OK' if has_text else '❌ Lỗi'}")
        print(f" ├─ Cột PhoBERT (Cảm xúc): {'✅ OK' if has_phobert else '❌ Lỗi'}")
        print(f" ├─ Cột Video/YOLO (Thời lượng, Đếm người): {'✅ OK' if has_visual else '❌ Lỗi'}")
        print(f" ├─ Cột Vector CLIP (512 chiều): {'✅ OK' if has_clip else '❌ Lỗi'}")
        
        if has_visual:
            max_person = df_merged['video_frame_person_count_max'].iloc[0] if 'video_frame_person_count_max' in df_merged.columns else 0
            print(f" ├─ Số người xuất hiện tối đa (YOLO) trong video này: {int(max_person)} người")
        
        print(f"\n📁 File kết quả tổng quát (gồm tất cả các cột) đã được lưu tại: {output_file}")
        print("👉 Bạn hãy mở file CSV này bằng Excel, cuộn từ trái sang phải để xem Text và Ảnh đã nằm trên cùng 1 hàng chưa nhé!")
        
    except Exception as e:
        print(f"\n❌ Có lỗi xảy ra trong quá trình xử lý: {e}")

if __name__ == "__main__":
    main()