import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
import time
from pdf2image import convert_from_bytes
from PIL import Image

st.set_page_config(page_title="RPs Drawing Extractor Tool", layout="centered")
st.title("RPs Drawing Extractor Tool")

uploaded_files = st.file_uploader(
    "Choose at least one drawing to begin", 
    type=['png', 'jpg', 'jpeg', 'pdf'], 
    accept_multiple_files=True
)

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
    Tính toán FG dựa trên logic:
    
    1. Nếu phần trước dấu - có từ 5 ký tự trở lên:
       - Lấy toàn bộ FG, bỏ phần text và dấu -
       - Ví dụ: "12345-08" -> "1234508"
       
    2. Nếu phần trước dấu - có từ 4 ký tự trở xuống:
       - Lấy toàn bộ phần trước dấu -
       - Thay dấu - bằng 2 ký tự từ RPs Code (vị trí số đầu + 4)
       - Thêm 2 ký tự sau dấu - từ FG gốc
       - Ví dụ: "123-08" + RPs "ABC1234567" -> "123" + "45" + "08" = "12345 08"
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

def pdf_to_images(pdf_bytes):
    """Convert PDF bytes to list of PIL images"""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        return images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        return []

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    # Convert PIL Image to RGB (if not already)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    numpy_image = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    return opencv_image

def process_image(image, file_stem, prefix, reader):
    """Process a single image and return results"""
    results = []
    
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
    
    return results

if uploaded_files:
    reader = get_reader()
    all_results = []
    
    # Count total pages/images
    total_pages = 0
    file_info = []
    
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == 'pdf':
            pdf_bytes = uploaded_file.read()
            images = pdf_to_images(pdf_bytes)
            total_pages += len(images)
            file_info.append({
                'name': uploaded_file.name,
                'type': 'pdf',
                'images': images,
                'page_count': len(images)
            })
        else:
            total_pages += 1
            uploaded_file.seek(0)  # Reset file pointer
            file_info.append({
                'name': uploaded_file.name,
                'type': 'image',
                'data': uploaded_file.read(),
                'page_count': 1
            })

    progress_bar = st.progress(0, text="Scanning...")
    processed_pages = 0

    for file_data in file_info:
        file_name = file_data['name']
        file_stem = file_name.split('.')[0]
        prefix = file_stem[:3].upper()
        
        if file_data['type'] == 'pdf':
            # Process each page of PDF
            for page_num, pil_image in enumerate(file_data['images'], 1):
                page_file_stem = f"{file_stem}_page_{page_num}"
                
                # Convert PIL to OpenCV
                image = pil_to_cv2(pil_image)
                
                # Process the image
                page_results = process_image(image, page_file_stem, prefix, reader)
                all_results.extend(page_results)
                
                processed_pages += 1
                percent = processed_pages / total_pages
                progress_bar.progress(
                    percent, 
                    text=f"Processing {file_name} - Page {page_num}/{file_data['page_count']} ({int(percent * 100)}%)"
                )
                time.sleep(0.05)
        else:
            # Process single image
            image = cv2.imdecode(np.frombuffer(file_data['data'], np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                page_results = process_image(image, file_stem, prefix, reader)
                all_results.extend(page_results)
            
            processed_pages += 1
            percent = processed_pages / total_pages
            progress_bar.progress(
                percent, 
                text=f"Processing {file_name} ({int(percent * 100)}%)"
            )
            time.sleep(0.05)

    progress_bar.empty()

    # Tạo DataFrame với thứ tự cột: Drawing, FG, RPs Code
    df = pd.DataFrame(all_results, columns=["Drawing", "FG", "RPs Code"])
    
    st.subheader("Result:")
    st.dataframe(df, use_container_width=True)
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files", len(file_info))
    with col2:
        st.metric("Total Pages Processed", total_pages)
    with col3:
        st.metric("Total Results", len(all_results))
    
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
