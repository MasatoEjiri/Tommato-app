import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# --- 設定 ---
st.set_page_config(page_title="GG-TomatoAI β版", layout="wide")

# --- デザイン設定 (強制ダークモード & モダンUI) ---
st.markdown("""
    <style>
    /* 1. アプリ全体の強制ダークモード化 */
    .stApp {
        background-color: #0E1117; /* 深い黒に近いグレー */
        color: #FAFAFA; /* 白文字 */
    }
    
    /* サイドバーもダークに */
    [data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #464b5f;
    }

    /* ヘッダーやテキストの色を強制的に白くする */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #FAFAFA !important;
    }

    /* 2. モダンなドロップエリアのデザイン */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1E1E1E !important; /* カードっぽい背景色 */
        border: 2px dashed #4B4B4B !important; /* 控えめな枠線 */
        border-radius: 12px; /* 今風の丸み */
        padding: 40px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); /* 滑らかなアニメーション */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); /* 浮き上がるような影 */
    }
    
    /* ホバー時のエフェクト */
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #FF4B4B !important; /* トマト色のアクセント */
        background-color: #252525 !important;
        box-shadow: 0 10px 20px rgba(255, 75, 75, 0.1); /* 赤い光の漏れ */
        transform: scale(1.01); /* ほんの少し拡大 */
    }

    /* ドロップエリア内の文字色修正 */
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] small {
        color: #A0A0A0 !important; /* 少し暗めのグレー文字 */
    }
    
    /* ボタンのデザイン (ダークモード仕様) */
    button[data-testid="stBaseButton-secondary"] {
        border: 1px solid #555 !important;
        background-color: #2b2b2b !important;
        color: #eee !important;
        border-radius: 8px;
        font-weight: 600;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        border-color: #FF4B4B !important;
        color: #FF4B4B !important;
        background-color: #2b2b2b !important;
    }
    
    /* データフレーム(表)の文字色対応 */
    [data-testid="stDataFrame"] {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# タイトル
st.title("🍅 GG-TomatoAI β版")

# 案内テキスト（モダンで控えめに）
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px; opacity: 0.8; font-size: 0.9rem; letter-spacing: 1px;'>
        UPLOAD IMAGE FOR ANALYSIS
    </div>
    """, unsafe_allow_html=True)

# --- モデルの読み込み ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    st.sidebar.success("AI System Online") # 文言も少しかっこよく
except Exception as e:
    st.error(f"Error loading model. Check 'best.pt'.\n{e}")
    st.stop()

# --- サイドバー設定 ---
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# --- メイン処理 ---
uploaded_file = st.file_uploader(
    "Upload Image", 
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
        st.markdown(f"""
        <div style="background-color: #1E1E1E; border-left: 5px solid #FF4B4B; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin:0; padding:0;">Detected: <span style="color:#FF4B4B;">{n_tomatoes}</span> Tomatoes</h3>
        </div>
        """, unsafe_allow_html=True)
        
        measurement_data = []
        display_img = img_cv2.copy()
        
        # 座標順ソート
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
                "Long axis (px)": round(long_axis, 1),
                "Short axis (px)": round(short_axis, 1),
                "Ratio": ratio_text,
                "Conf": f"{box.conf[0]:.2f}"
            })

            # --- 描画処理 ---
            
            # 中心座標
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # テキスト設定 (大きく見やすく)
            label = str(i + 1)
            font_scale = 1.1  # 大きいまま維持
            thickness = 3     # 太いまま維持
            color = (0, 255, 0) # 緑色 (黒背景によく映えます)
            
            # 配置調整
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = center_x - int(text_w / 2)
            text_y = center_y + int(text_h / 2)
            
            # 描画
            cv2.putText(display_img, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            
        # --- 表示エリア ---
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("##### Detection Result")
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col2:
            st.markdown("##### Measurement Data")
            df = pd.DataFrame(measurement_data)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="DOWNLOAD CSV",
                data=csv,
                file_name='ai_tomato_result.csv',
                mime='text/csv',
            )
    else:
        st.warning("No tomatoes detected. Try lowering the confidence threshold.")
        st.image(image_pil, caption="Original Image", use_container_width=True)
