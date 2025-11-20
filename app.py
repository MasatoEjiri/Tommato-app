import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# --- 設定 ---
st.set_page_config(page_title="GG-TomatoAI β版", layout="wide")

# --- クラシックで洗練されたデザインのCSS ---
st.markdown("""
    <style>
    /* ファイルアップローダーのドロップゾーン全体 */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #4a4a4a !important; /* 落ち着いたダークグレーの枠線 */
        border-radius: 4px; /* 角丸を少し減らしてシャープに */
        background-color: #f9f9f9; /* 無機質なライトグレー */
        padding: 40px 20px;
        transition: all 0.3s ease;
    }
    
    /* マウスを乗せた時の動き */
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #000000 !important; /* ホバー時は真っ黒に */
        background-color: #f0f0f0; /* 少しだけ濃く */
        cursor: pointer;
    }

    /* テキストの色 */
    [data-testid="stFileUploaderDropzone"] div, 
    [data-testid="stFileUploaderDropzone"] span {
        color: #333 !important; /* 黒に近いグレー */
        font-family: "Helvetica Neue", Arial, sans-serif; /* 定番フォント */
        letter-spacing: 0.05em; /* 文字間隔を少し開けて上品に */
    }
    
    /* 「Browse files」ボタンのデザイン (モノトーン) */
    button[data-testid="stBaseButton-secondary"] {
        border: 1px solid #4a4a4a !important;
        color: #4a4a4a !important;
        background-color: transparent !important;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: normal;
        text-transform: uppercase; /* 大文字にしてクラシック感を出す */
        font-size: 0.9em;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #4a4a4a !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# タイトル
st.title("🍅 GG-TomatoAI β版")

# 案内テキスト（シンプルに）
st.markdown("""
    <div style='text-align: center; margin-bottom: 15px; color: #666; font-size: 0.9em;'>
        PLEASE DROP YOUR IMAGE HERE
    </div>
    """, unsafe_allow_html=True)

# --- モデルの読み込み ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    st.sidebar.success("AIモデルの読み込みに成功しました！")
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました。'best.pt'が同じフォルダにあるか確認してください。\nエラー: {e}")
    st.stop()

# --- サイドバー設定 ---
st.sidebar.header("検出設定")
conf_threshold = st.sidebar.slider("AIの確信度(Confidence)", 0.1, 1.0, 0.25, 0.05, help="数値を上げると、自信があるものだけ検出します。下げると見逃しが減りますが誤検出が増えます。")

# --- メイン処理 ---
uploaded_file = st.file_uploader(
    "画像をアップロード", 
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # 画像変換処理
    image_pil = Image.open(uploaded_file).convert("RGB")
    img_cv2 = np.array(image_pil)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

    # AI推論
    results = model(img_cv2, conf=conf_threshold, verbose=False)
    result = results[0]
    n_tomatoes = len(result.boxes)
    
    if n_tomatoes > 0:
        st.success(f"{n_tomatoes} 個のトマトを検出しました！")
        
        measurement_data = []
        display_img = img_cv2.copy()
        
        # 座標順（左上から右下）にソート
        sorted_boxes = sorted(result.boxes, key=lambda b: b.xywh[0][1] * 10 + b.xywh[0][0])

        for i, box in enumerate(sorted_boxes):
            # 座標取得
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # サイズ計算
            width = x2 - x1
            height = y2 - y1
            long_axis = max(width, height)
            short_axis = min(width, height)
            ratio = short_axis / long_axis
            ratio_text = f"1:{ratio:.2f}"
            
            measurement_data.append({
                "ID": i + 1,
                "長軸(px)": round(long_axis, 1),
                "短軸(px)": round(short_axis, 1),
                "縦:横": ratio_text,
                "確信度": f"{box.conf[0]:.2f}"
            })

            # --- 描画処理 ---
            
            # 中心座標を計算
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # テキストの設定
            label = str(i + 1)
            font_scale = 1.1  # ★サイズアップ (0.7 -> 1.1)
            thickness = 3     # ★太くして視認性を確保 (2 -> 3)
            color = (0, 255, 0) # 緑色
            
            # 文字のサイズを取得して配置調整
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = center_x - int(text_w / 2)
            text_y = center_y + int(text_h / 2)
            
            # 文字を描画
            cv2.putText(display_img, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            
        # --- 表示エリア ---
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("検出画像")
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption=f"検出結果 ({n_tomatoes}個)", use_container_width=True)
            
        with col2:
            st.subheader("計測データ")
            df = pd.DataFrame(measurement_data)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="CSVをダウンロード",
                data=csv,
                file_name='ai_tomato_result.csv',
                mime='text/csv',
            )
    else:
        st.warning("トマトが検出されませんでした。設定の「確信度」を下げてみてください。")
        st.image(image_pil, caption="アップロード画像", use_container_width=True)
