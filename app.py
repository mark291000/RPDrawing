
import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
import time

st.set_page_config(page_title="RPs Drawing Extractor Tool", layout="centered")
st.title("RPs Drawing Extractor Tool")

uploaded_files = st.file_uploader("Choose at least one drawing to begin", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

@st.cache_resource
def get_reader():
    return easyocr.Reader(['en'])

if uploaded_files:
    reader = get_reader()
    results = []

    def color_distance(c1, c2):
        return np.linalg.norm(np.array(c1, dtype=int) - np.array(c2, dtype=int))

    def get_center_color(image, bbox):
        (x1, y1), _, (x3, y3), _ = bbox
        x_center = int((x1 + x3) / 2)
        y_center = int((y1 + y3) / 2)
        if y_center >= image.shape[0] or x_center >= image.shape[1]:
            return None
        return image[y_center, x_center]

    def fix_missing_parenthesis(text):
        if '(' in text and ')' not in text:
            last_comma = text.rfind(',')
            if last_comma != -1 and len(text) > last_comma + 3:
                return text[:last_comma+3] + ')' + text[last_comma+4:]
        return text

    progress_bar = st.progress(0, text="Scanning...")
    total = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        file_stem = uploaded_file.name.split('.')[0]
        prefix = file_stem[:3].upper()

        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue

        if image.shape[1] > 1600:
            scale = 1600 / image.shape[1]
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        ocr_results = reader.readtext(image)
        candidates = []

        for bbox, text, _ in ocr_results:
            clean_text = text.strip().replace(' ', '')
            if clean_text.upper().startswith(prefix):
                color = get_center_color(image, bbox)
                if color is not None:
                    candidates.append((clean_text, color, bbox))

        threshold = 35
        main_groups = []
        visited = set()

        for i, (text_i, color_i, _) in enumerate(candidates):
            if i in visited:
                continue
            group = [text_i]
            visited.add(i)
            for j, (text_j, color_j, _) in enumerate(candidates):
                if j != i and j not in visited:
                    if color_distance(color_i, color_j) <= threshold:
                        group.append(text_j)
                        visited.add(j)
            main_groups.append((file_stem, group))

        for file, group in main_groups:
            for line in group:
                line = re.sub(r'\s+', '', line)
                line = fix_missing_parenthesis(line)

                match = re.match(r"([A-Za-z0-9]+)\(([^)]+)\)([A-Za-z0-9]+)", line)
                if match:
                    prefix_code, middle, suffix = match.groups()
                    for part in middle.split(','):
                        if '-' in part:
                            continue
                        code = f"{prefix_code}{part}{suffix}"
                        if len(code) >= 10:
                            results.append((file, code))
                else:
                    if '-' not in line and len(line) >= 10:
                        results.append((file, line))

        percent = (idx + 1) / total
        progress_bar.progress(percent, text=f"Processing {idx + 1}/{total} Drawings ({int(percent * 100)}%)")
        time.sleep(0.1)

    progress_bar.empty()

    df = pd.DataFrame(results, columns=["Drawing", "RPs Code"])
    st.subheader("Result:")
    st.dataframe(df)

st.markdown("---")
st.caption("📌 For any issues related to the app, please contact Mark Dang.")
