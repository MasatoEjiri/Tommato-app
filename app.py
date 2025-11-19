import streamlit as st
import cv2
import numpy as np
import pandas as pd

def process_tomatoes(image_file, min_area, h_min, s_min, v_min):
    # 画像の読み込み
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 画像を少しぼかしてノイズを減らす
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # サイドバーで設定した値を使ってマスクを作成
    # 赤色はHueが0付近と170~180付近に分かれるため2回マスク作成
    
    # 範囲1: 0 〜 h_min (例: 0~10)
    lower_red1 = np.array([0, s_min, v_min])
    upper_red1 = np.array([h_min, 255, 255])
    
    # 範囲2: (180-h_min) 〜 180 (例: 170~180)
    lower_red2 = np.array([180 - h_min, s_min, v_min])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # 穴埋め処理（モルフォロジー演算）
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            valid_contours.append(cnt)

    # 座標順ソート
    boxes = [cv2.boundingRect(c) for c in valid_contours]
    if not boxes:
        return img, mask, []

    contours_boxes = zip(valid_contours, boxes)
    sorted_contours = sorted(contours_boxes, key=lambda b: b[1][1] * 3 + b[1][0])

    results = []
    img_out = img.copy()
    
    for i, (cnt, box) in enumerate(sorted_contours):
        idx = i + 1
        if len(cnt) < 5:
            continue
            
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img_out, ellipse, (0, 255, 0), 2)

        axis_lengths = ellipse[1]
        long_axis = max(axis_lengths)
        short_axis = min(axis_lengths)
        
        ratio = short_axis / long_axis
        ratio_text = f"1:{ratio:.2f}"

        results.append({
            "ID": idx,
            "長軸(px)": round(long_axis, 1),
            "短軸(px)": round(short_axis, 1),
            "縦:横": ratio_text
        })

        # テキスト描画
        x, y, w, h = box
        cv2.putText(img_out, str(idx), (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return img_out, mask, results

# --- Streamlit UI ---

st.title("🍅 トマト形状解析ツール (調整版)")

st.sidebar.header("1. 検出感度の調整")
st.sidebar.markdown("右の「二値化画像」を見ながら、トマトだけが白くなるように調整してください。")

# スライダーの設定
s_min = st.sidebar.slider("彩度(鮮やかさ)の下限", 0, 255, 100, help="値を上げると、薄い色のもの（ダンボールなど）を除外します")
v_min = st.sidebar.slider("明度(明るさ)の下限", 0, 255, 50, help="値を上げると、暗い影などを除外します")
h_min = st.sidebar.slider("色相(赤の範囲)", 1, 30, 10, help="赤色の幅を広げます")
min_area = st.sidebar.slider("最小サイズ除去", 0, 5000, 500, help="小さいゴミを除去します")

uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 解析実行
    processed_img, mask_img, data = process_tomatoes(uploaded_file, min_area, h_min, s_min, v_min)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("検出結果")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("調整用モニター(二値化)")
        st.markdown("※ここが重要です。トマトだけが白く見えるようにスライダーを動かしてください。")
        st.image(mask_img, caption="白=トマトと認識している部分", use_container_width=True)

    st.subheader("計測データ")
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("CSVをダウンロード", data=csv, file_name='tomato_v2.csv', mime='text/csv')
    else:
        st.warning("トマトが見つかりません。スライダーを調整してください。")
