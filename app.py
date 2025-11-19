import streamlit as st
import cv2
import numpy as np
import pandas as pd

def process_tomatoes(image_file, min_area_threshold):
    # 画像の読み込み
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # BGRからHSVへ変換（赤色検出のため）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色の範囲定義（トマト用）
    # 赤色はHueが0付近と180付近に分かれるため2回マスク作成
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # ノイズ除去（オープニング・クロージング）
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積でソートまたはフィルタリングし、座標順（左上から右下）に並べ替え
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area_threshold: # 小さすぎるゴミを除去
            valid_contours.append(cnt)

    # 並べ替えロジック：Y座標（上から下）優先、次にX座標
    # boundingBox = (x, y, w, h)
    boxes = [cv2.boundingRect(c) for c in valid_contours]
    
    if not boxes:
        return img, []

    # zipでまとめてソート（y座標 + x座標の重み付けで並び替え）
    # ここでは単純に「y * 10 + x」のようなスコアで左上から順に並ぶように簡易ソート
    contours_boxes = zip(valid_contours, boxes)
    sorted_contours = sorted(contours_boxes, key=lambda b: b[1][1] * 3 + b[1][0])

    results = []
    
    # 解析と描画
    for i, (cnt, box) in enumerate(sorted_contours):
        idx = i + 1
        
        # 回転を考慮した楕円フィッティング
        # ((center_x, center_y), (width, height), angle)
        # ここでのwidth, heightは楕円の長軸・短軸（回転含む）
        if len(cnt) < 5:
            continue # 点が少なすぎると楕円フィットできない
            
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)

        # 楕円の長軸と短軸を取得
        # fitEllipseは (MA, ma) を返すが、どっちが縦かは角度による
        # 今回はシンプルに「長い方を縦(Long Axis)」「短い方を横(Short Axis)」として比率を出す
        # ※ヘタの位置が画像解析だけでは特定困難なため
        
        axis_lengths = ellipse[1]
        long_axis = max(axis_lengths)
        short_axis = min(axis_lengths)
        
        # 比率計算 (縦を1とした場合)
        # ユーザー要望: 結果は「X:Y」
        # ここでは 長軸(縦と仮定):短軸(横と仮定) で計算します
        ratio = short_axis / long_axis
        ratio_text = f"1:{ratio:.2f}"

        # 結果リストに追加
        results.append({
            "ID": idx,
            "長軸(px)": round(long_axis, 1),
            "短軸(px)": round(short_axis, 1),
            "縦:横": ratio_text
        })

        # 画像にIDを描画
        # 重心を計算して文字を配置
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = int(box[0]), int(box[1])
            
        cv2.putText(img, str(idx), (cX - 10, cY + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img, results

# --- Streamlit アプリ部分 ---

st.title("🍅 トマト形状解析ツール")
st.markdown("""
画像のトマトを検出し、それぞれの形状比率（長軸:短軸）を計測します。
""")

# サイドバー設定
st.sidebar.header("設定")
min_area = st.sidebar.slider("最小検出面積 (ノイズ除去)", 100, 5000, 1000)

# ファイルアップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 解析実行
    processed_img, data = process_tomatoes(uploaded_file, min_area)

    # カラム分けして表示
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("解析画像")
        # OpenCVはBGR、StreamlitはRGBなので変換
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="検出結果(緑枠=近似楕円)", use_container_width=True)

    with col2:
        st.subheader("計測データ")
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df)
            
            # CSVダウンロードボタン
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="CSVをダウンロード",
                data=csv,
                file_name='tomato_analysis.csv',
                mime='text/csv',
            )
        else:
            st.warning("トマトが検出されませんでした。設定の「最小検出面積」を調整してみてください。")

    st.markdown("""
    **補足:**
    * IDの順番は画像の左上から自動的に採番されます（画像の手書き番号とは一致しない場合があります）。
    * ヘタの位置を自動検出するのは難易度が高いため、**「最も長い軸を縦」**と仮定して計算しています。
    * 7番のような横倒しのトマトも、長い方を縦軸として計測されます。
    """)