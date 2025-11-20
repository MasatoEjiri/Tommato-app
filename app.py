import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# --- 設定 ---
st.set_page_config(page_title="GG-TomatoAI β版", layout="wide")

# --- 洗練されたデザインのCSS ---
st.markdown("""
    <style>
    /* ファイルアップローダーのドロップゾーン全体 */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #ff7f7f !important; /* 優しいトマト色の枠線 */
        border-radius: 16px; /* 角を丸く */
        background-color: #fffbfb; /* ほんのり赤い白背景 */
        padding: 40px 20px; /* 余白をたっぷりとる */
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* うっすら影をつけて浮き上がらせる */
        transition: all 0.3s ease; /* アニメーション */
    }
    
    /* マウスを乗せた時の動き */
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ff4b4b !important; /* 枠線を濃く */
        background-color: #ffefef; /* 背景を少し濃く */
        transform: translateY(-2px); /* 少し上に浮く */
        box-shadow: 0 8px 16px rgba(255, 75, 75, 0.15); /* 赤い影を強める */
    }

    /* 中にある「Drag and drop...」などの文字色 */
    [data-testid="stFileUploaderDropzone"] div, 
    [data-testid="stFileUploaderDropzone"] span {
        color: #666 !important; /* 落ち着いたグレー */
        font-family: "Helvetica Neue", Arial, sans-serif;
    }
    
    /* 「Browse files」ボタンのデザイン */
    button[data-testid="stBaseButton-secondary"] {
        border: 1px solid #ff4b4b !important;
        color: #ff4b4b !important;
        background-color: white !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# タイトル
st.title("🍅 GG-TomatoAI β版")

# 案内テキストをアイコン付きで表示（ここを目立たせる）
st.markdown("""
    <div style='text-align: center; margin-bottom: 10px; color: #444;'>
        <h5>👇 トマトの画像をここにドロップしてください</h5>
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
# label_visibility="collapsed" で標準のラベルを消し、上のカスタム案内を目立たせる
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

            # --- 描画処理（シンプル緑枠） ---
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # テキスト描画
            label = str(i + 1)
            font_scale = 0.6
            thickness = 2
            
            # 文字位置調整
            text_y = y1 - 5 if y1 - 5 > 10 else y1 + 20
            
            cv2.putText(display_img, label, (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            
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
