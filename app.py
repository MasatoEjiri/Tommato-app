import streamlit as st
import cv2
import numpy as np
import pandas as pd

def process_tomatoes(image_file, min_area, h_min, s_min, v_min, separation_strength):
    # 画像読み込み
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # ノイズ除去（ブラー）
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 1. 色による抽出（前回と同じ）
    lower_red1 = np.array([0, s_min, v_min])
    upper_red1 = np.array([h_min, 255, 255])
    lower_red2 = np.array([180 - h_min, s_min, v_min])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # ノイズ処理（穴埋め）
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # 膨張させすぎるとくっつくので、ここは控えめに
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- ここから新しい分離ロジック (Watershed) ---
    
    # 背景を確実にする（膨張）
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # 前景（トマトの中心）を確実にする（距離変換）
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # スライダーで分離強度を調整
    # separation_strength は 0.0 ~ 1.0。高いほど中心だけを厳密に取る（分離しやすくなる）
    ret, sure_fg = cv2.threshold(dist_transform, separation_strength * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 不明領域（境界線候補）
    unknown = cv2.subtract(sure_bg, sure_fg)

    # マーカー作成
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 # 背景を1にする
    markers[unknown == 255] = 0 # 不明領域を0にする

    # Watershed実行
    markers = cv2.watershed(img, markers)
    
    # 境界線を描画（黄色）
    img[markers == -1] = [0, 255, 255]

    # 解析結果の収集
    results = []
    img_out = img.copy()
    
    # マーカーごとにループ（ラベル1は背景なのでスキップ）
    unique_markers = np.unique(markers)
    obj_count = 0

    for marker_id in unique_markers:
        if marker_id <= 1: # 背景または境界線
            continue

        # このマーカーのマスクを作成
        obj_mask = np.zeros_like(mask, dtype=np.uint8)
        obj_mask[markers == marker_id] = 255

        # 輪郭検出
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            obj_count += 1
            
            # 楕円フィッティング
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(img_out, ellipse, (0, 255, 0), 2)
                
                axis_lengths = ellipse[1]
                long_axis = max(axis_lengths)
                short_axis = min(axis_lengths)
                
                ratio = short_axis / long_axis
                ratio_text = f"1:{ratio:.2f}"
                
                results.append({
                    "ID": obj_count,
                    "長軸(px)": round(long_axis, 1),
                    "短軸(px)": round(short_axis, 1),
                    "縦:横": ratio_text
                })

                # テキスト描画
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(img_out, str(obj_count), (cX - 10, cY), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # モニター用に距離変換画像を可視化
    dist_display = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    dist_display = np.uint8(dist_display)
    dist_display = cv2.cvtColor(dist_display, cv2.COLOR_GRAY2BGR)

    return img_out, dist_display, results

# --- Streamlit UI ---

st.title("🍅 トマト形状解析ツール (分離機能付き)")

st.sidebar.header("1. 検出感度 & 分離")

# スライダー
st.sidebar.subheader("基本検出")
s_min = st.sidebar.slider("彩度(S) 下限", 0, 255, 60)
v_min = st.sidebar.slider("明るさ(V) 下限", 0, 255, 60)
h_min = st.sidebar.slider("色相(H) 幅", 1, 30, 8)
min_area = st.sidebar.slider("最小サイズ除去", 0, 5000, 200)

st.sidebar.subheader("くっつき分離")
separation_strength = st.sidebar.slider("分離強度", 0.1, 0.9, 0.5, 0.05, help="値を上げると、くっついたトマトを強く切り離そうとします。上げすぎるとトマトが消えます。")

uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    processed_img, dist_img, data = process_tomatoes(uploaded_file, min_area, h_min, s_min, v_min, separation_strength)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("検出結果")
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("分離判定用モニター")
        st.markdown("トマトの「芯（中心）」が明るく光ります。ここが離れていれば分離できます。")
        st.image(dist_img, caption="距離変換画像", use_container_width=True)

    st.subheader("計測データ")
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("CSVをダウンロード", data=csv, file_name='tomato_v3.csv', mime='text/csv')
    else:
        st.warning("トマトが見つかりません。調整してください。")
