import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# --- 設定 ---
st.set_page_config(page_title="AIトマト計測アプリ", layout="wide")

# カスタムCSSでファイルアップローダーのスタイルを変更
st.markdown("""
    <style>
    .stFileUploader > div > button {
        visibility: hidden;
        height: 0;
        width: 0;
    }
    .stFileUploader > div > div {
        border: 2px dashed #999999; /* 点線で囲む */
        border-radius: 8px; /* 角を丸くする */
        padding: 20px;
        text-align: center;
        background-color: #f0f2f6; /* 少し背景色をつける */
        color: #666666;
        font-size: 1.2em;
        font-weight: bold;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 150px; /* 最小高さを設定 */
    }
    .stFileUploader > div > div:hover {
        border-color: #007bff; /* ホバーで色を変える */
        color: #007bff;
    }
    .stFileUploader > div > div > p {
        margin-top: 10px;
        font-size: 0.9em;
        color: #888888;
    }
    </style>
    """, unsafe_allow_html=True)


# タイトル変更
st.title("🍅 GG-TomatoAI")
st.markdown("学習済みAIモデル（YOLOv8）を使用して、トマトを自動検出し計測します。")

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
# ファイルアップローダーのラベルを非表示にし、ヘルプテキストで指示
uploaded_file = st.file_uploader(
    "画像をアップロードしてください", 
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed", # デフォルトのラベルを非表示
    help="ここに画像をドラッグ＆ドロップしてください" # ヘルプテキストをヒントとして表示
)


if uploaded_file is not None:
    # PIL画像をNumPy配列（OpenCV形式）に変換
    image_pil = Image.open(uploaded_file).convert("RGB")
    img_cv2 = np.array(image_pil)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR) # OpenCVはBGR形式

    # AIで推論実行！
    results = model(img_cv2, conf=conf_threshold, verbose=False) # verbose=Falseでログを抑制

    result = results[0]
    
    n_tomatoes = len(result.boxes)
    
    if n_tomatoes > 0:
        st.success(f"{n_tomatoes} 個のトマトを検出しました！")
        
        # --- 計測データと描画 ---
        measurement_data = []
        
        # 描画用の画像を用意
        display_img = img_cv2.copy()
        
        # ID順にソート（左上から右下の順）
        sorted_boxes = sorted(result.boxes, key=lambda b: b.xywh[0][1] * display_img.shape[1] + b.xywh[0][0])

        for i, box in enumerate(sorted_boxes):
            # バウンディングボックスの座標 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # 幅と高さを計算
            width = x2 - x1
            height = y2 - y1
            
            # 長軸・短軸の判定（長い方を長軸とする）
            long_axis = max(width, height)
            short_axis = min(width, height)
            
            # 比率計算
            ratio = short_axis / long_axis
            ratio_text = f"1:{ratio:.2f}"
            
            measurement_data.append({
                "ID": i + 1,
                "長軸(px)": round(long_axis, 1),
                "短軸(px)": round(short_axis, 1),
                "縦:横": ratio_text,
                "確信度": f"{box.conf[0]:.2f}"
            })

            # --- 画像にシンプルに描画 ---
            # 緑色のボックス
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 緑色のID番号 (2枚目の画像のように)
            cv2.putText(display_img, str(i + 1), (x1 + 5, y1 + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
        # --- 表示エリア ---
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("検出画像")
            # OpenCV画像をStreamlit表示用にRGBに変換
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
