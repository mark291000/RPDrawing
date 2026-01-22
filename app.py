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

def extract_fg_code(text):
    """
    Lấy toàn bộ chuỗi FG nhưng loại bỏ phần text phía sau dấu -
    Ví dụ: '102-08 Greg' -> '102-08'
    Ví dụ: 'ABC-123-XYZ More Text' -> 'ABC-123-XYZ'
    Ví dụ: '102-08' -> '102-08'
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Tìm vị trí dấu - cuối cùng
    last_dash_index = text.rfind('-')
    
    if last_dash_index == -1:
        # Không có dấu -, trả về text gốc
        return text
    
    # Tìm vị trí ký tự không phải số/chữ sau dấu - cuối cùng
    after_dash = text[last_dash_index + 1:]
    
    # Tìm phần số/chữ ngay sau dấu - (bỏ qua khoảng trắng)
    match = re.match(r'^\s*([A-Za-z0-9]+)', after_dash)
    
    if match:
        # Lấy phần từ đầu đến hết phần số/chữ sau dấu - cuối cùng
        end_position = last_dash_index + 1 + match.end()
        return text[:end_position].strip()
    
    # Nếu không match được, trả về đến dấu - cuối
    return text[:last_dash_index + 1].strip()

def calculate_fg_from_rps(fg_original, rps_code):
    """
    Tính toán FG dựa trên logic mới:
    
    1. Nếu phần trước dấu - có từ 5 ký tự trở lên:
       - Lấy toàn bộ FG, bỏ phần text và dấu -
       - Ví dụ: "12345-08" -> "1234508"
       
    2. Nếu phần trước dấu - có từ 4 ký tự trở xuống:
       - Lấy toàn bộ phần trước dấu -
       - Thay dấu - bằng 2 ký tự từ RPs Code (vị trí số đầu + 4)
       - Thêm 2 ký tự sau dấu - từ FG gốc
       - Ví dụ: "123-08" + RPs "ABC1234567" -> "123" + "45" + "08" = "12345 08"
    
    Logic lấy 2 ký tự từ RPs Code:
    - ABC1234567: Số đầu = vị trí 3 (số 1), lấy vị trí 3+4=7 -> "45"
    - 1236789: Số đầu = vị trí 0 (số 1), lấy vị trí 0+4=4 -> "67"
    """
    if not fg_original or not rps_code or '-' not in fg_original:
        return fg_original
    
    # Tách FG theo dấu - đầu tiên
    parts = fg_original.split('-', 1)
    prefix = parts[0]  # Phần trước dấu -
    suffix = parts[1] if len(parts) > 1 else ""  # Phần sau dấu -
    
    # Lấy 2 ký tự đầu từ suffix (bỏ qua khoảng trắng và text)
    suffix_digits = re.match(r'^\s*([A-Za-z0-9]{0,2})', suffix)
    suffix_2chars = suffix_digits.group(1) if suffix_digits else ""
    
    # Kiểm tra độ dài phần trước dấu -
    if len(prefix) >= 5:
        # TH1: Từ 5 ký tự trở lên - Lấy toàn bộ, bỏ dấu -
        result = prefix + suffix_2chars
        return result
    else:
        # TH2: Từ 4 ký tự trở xuống - Thay dấu - bằng 2 ký tự từ RPs Code
        
        # Tìm vị trí đầu tiên của số (0-9) trong RPs Code
        first_digit_pos = None
        for i, char in enumerate(rps_code):
            if char.isdigit():
                first_digit_pos = i
                break
        
        if first_digit_pos is None:
            return fg_original
        
        # Tính vị trí cần lấy: first_digit_pos + 4 (lấy ký tự thứ 5 và 6 sau số đầu tiên)
        extract_pos = first_digit_pos + 3
        
        # Kiểm tra xem có đủ ký tự không
        if extract_pos + 2 > len(rps_code):
            return fg_original
        
        # Lấy 2 ký tự từ RPs Code
        rps_2chars = rps_code[extract_pos:extract_pos + 2]
        
        # Ghép: prefix + 2 ký tự từ RPs + 2 ký tự sau dấu - từ FG gốc
        result = prefix + rps_2chars + suffix_2chars
        return result

def find_ashley_fg(ocr_results):
    """Tìm text nằm dưới chữ ASHLEY và trích xuất FG code"""
    ashley_boxes = []
    
    # Tìm tất cả các vị trí có chữ ASHLEY
    for bbox, text, _ in ocr_results:
        if 'ASHLEY' in text.upper():
            ashley_boxes.append(bbox)
    
    if not ashley_boxes:
        return None
    
    # Với mỗi ASHLEY, tìm text ngay bên dưới
    fg_candidates = []
    for ashley_bbox in ashley_boxes:
        (x1_a, y1_a), _, (x3_a, y3_a), _ = ashley_bbox
        ashley_bottom = y3_a
        ashley_x_center = (x1_a + x3_a) / 2
        
        # Tìm text nằm dưới ASHLEY (trong khoảng hợp lý)
        min_distance = float('inf')
        best_fg = None
        
        for bbox, text, _ in ocr_results:
            (x1, y1), _, (x3, y3), _ = bbox
            text_top = y1
            text_x_center = (x1 + x3) / 2
            
            # Kiểm tra text có nằm dưới ASHLEY không
            if text_top > ashley_bottom:
                # Kiểm tra căn chỉnh theo chiều ngang (có nằm gần cùng cột không)
                horizontal_distance = abs(text_x_center - ashley_x_center)
                vertical_distance = text_top - ashley_bottom
                
                # Text phải nằm gần dưới ASHLEY và căn chỉnh theo cột
                if horizontal_distance < 100 and vertical_distance < 150:
                    total_distance = vertical_distance + horizontal_distance
                    if total_distance < min_distance:
                        min_distance = total_distance
                        best_fg = text.strip()
        
        if best_fg:
            fg_candidates.append(best_fg)
    
    # Trả về FG đầu tiên tìm được, đã được trích xuất
    if fg_candidates:
        return extract_fg_code(fg_candidates[0])
    return None

if uploaded_files:
    reader = get_reader()
    results = []

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
        
        # Tìm FG code (text dưới ASHLEY)
        fg_code_original = find_ashley_fg(ocr_results)
        
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
                    for part in re.split(r'[.,]', middle):
                        if '-' in part:
                            continue
                        code = f"{prefix_code}{part}{suffix}"
                        if len(code) >= 10:
                            # Tính FG từ RPs Code
                            fg_final = calculate_fg_from_rps(fg_code_original if fg_code_original else "", code)
                            results.append((file, fg_final, code))
                else:
                    if '-' not in line and len(line) >= 10:
                        # Tính FG từ RPs Code
                        fg_final = calculate_fg_from_rps(fg_code_original if fg_code_original else "", line)
                        results.append((file, fg_final, line))

        percent = (idx + 1) / total
        progress_bar.progress(percent, text=f"Processing {idx + 1}/{total} Drawings ({int(percent * 100)}%)")
        time.sleep(0.1)

    progress_bar.empty()

    # Tạo DataFrame với thứ tự cột: Drawing, FG, RPs Code
    df = pd.DataFrame(results, columns=["Drawing", "FG", "RPs Code"])
    
    st.subheader("Result:")
    st.dataframe(df, use_container_width=True)
    
    # Nút download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name='rps_extraction_results.csv',
        mime='text/csv',
    )

st.markdown("---")
st.caption("📌 For any issues related to the app, please contact Mark Dang.")
