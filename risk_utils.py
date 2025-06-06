import fitz
from PIL import Image, ImageOps
import pytesseract
import io
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ✅ Tesseract 경로 설정 (Mac 기준)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# ✅ 이미지 전처리
def preprocess_image(img):
    gray = img.convert("L")
    enhanced = ImageOps.autocontrast(gray)
    binarized = enhanced.point(lambda x: 0 if x < 180 else 255, '1')
    return binarized

# ✅ 한글만 추출
def clean_korean_text(text):
    return ''.join(re.findall(r'[가-힣]', text))

# ✅ PDF에서 주소 및 건물명 추출
def extract_address_and_building_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=400)
        img = Image.open(io.BytesIO(pix.tobytes()))
        img = preprocess_image(img)
        page_text = pytesseract.image_to_string(img, lang="kor+eng", config="--psm 6 --oem 1")
        text += page_text + "\n"

    match = re.search(r"\[\s*집\s*합\s*건\s*물\s*\]\s*([^\n]+)", text)
    if not match:
        print("❌ '[집합건물]' 줄 없음")
        return None, None

    line = match.group(1).strip()
    words = line.split()

    dong_end_idx = -1
    for i, word in enumerate(words):
        if '동' in word:
            dong_end_idx = i
        if re.match(r"\d{1,4}-?\d*", word):
            dong_end_idx = i - 1
            break

    if dong_end_idx < 3:
        return None, None

    region = words[0]
    citygu = words[1] + words[2]
    dong = ''.join(words[3: dong_end_idx + 1])
    address = f"{region} {citygu} {dong}"

    building_words = words[dong_end_idx + 2:]
    cleaned = [w for w in building_words if not re.search(r"(제|\d+호|층|동|\d+)", w)]
    building_name = clean_korean_text(''.join(cleaned))

    return address.strip(), building_name

# ✅ 주소 → 법정동코드
def get_lawd_cd(address, csv_path):
    parts = address.split()
    if len(parts) < 3:
        print("❌ 주소 형식이 올바르지 않습니다.")
        return None

    시도, 시군구, 읍면동 = parts[0], parts[1], parts[2]
    df = pd.read_csv(csv_path, dtype=str)

    match = df[
        (df["시도명"] == 시도) &
        (df["시군구명"] == 시군구) &
        (df["읍면동명"] == 읍면동)
    ]

    if not match.empty:
        return match.iloc[0]["법정동코드"][:5]
    else:
        print("❌ 법정동코드를 찾을 수 없습니다.")
        return None

# ✅ PDF 전체 OCR 텍스트 추출
def extract_text_from_pdf_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=400)
        img = Image.open(io.BytesIO(pix.tobytes()))
        img = preprocess_image(img)
        text = pytesseract.image_to_string(img, lang="kor+eng", config="--psm 6 --oem 1")
        full_text += text + "\n"
    return full_text

# ✅ 실거래가 조회
def get_latest_officetel_trade(lawd_cd, building_name, area, service_key):
    url = "http://apis.data.go.kr/1613000/RTMSDataSvcOffiTrade/getRTMSDataSvcOffiTrade"
    result = []

    for month in range(1, 13):
        deal_ymd = f"2024{month:02d}"
        print(f"📦 {deal_ymd} 조회 중...")
        params = {
            "serviceKey": service_key,
            "LAWD_CD": lawd_cd,
            "DEAL_YMD": deal_ymd,
            "pageNo": "1",
            "numOfRows": "100"
        }

        response = requests.get(url, params=params)
        soup = BeautifulSoup(response.text, "xml")
        items = soup.find_all("item")

        for item in items:
            try:
                result.append({
                    "단지명": item.find("offiNm").text,
                    "전용면적": float(item.find("excluUseAr").text),
                    "계약일": f"{item.find('dealYear').text}-{item.find('dealMonth').text}-{item.find('dealDay').text}",
                    "거래금액(만원)": item.find("dealAmount").text
                })
            except:
                continue

    df = pd.DataFrame(result)
    df_filtered = df[df["단지명"].str.contains(building_name, na=False)].copy()
    df_filtered["면적차이"] = abs(df_filtered["전용면적"] - area)
    df_filtered["계약일"] = pd.to_datetime(df_filtered["계약일"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["계약일"])
    if df_filtered.empty:
        print("❌ 조건에 맞는 거래 정보가 없습니다.")
        return None
    latest = df_filtered.sort_values(["면적차이", "계약일"], ascending=[True, False]).head(1).squeeze()

    return latest

# ✅ OCR 텍스트 → 특징 추출
def parse_ocr_text_to_features(text, jeonse_ratio):
    text = text.replace(" ", "")
    
    def contains(keyword):
        return int(keyword in text)

    def normalize_mortgage(text):
        matches = re.findall(r'근저당권설정금(\d+,\d+|\d+)', text)
        values = [int(m.replace(',', '')) for m in matches]
        if not values:
            return 0
        max_value = max(values)
        return max_value / 1_0000_0000

    return {
        "전세가율": jeonse_ratio,
        "신탁": contains("신탁"),
        "근저당정규화": normalize_mortgage(text),
        "가압류": contains("가압류"),
        "압류": contains("압류"),
        "소유권이전": contains("소유권이전"),
        "임차권등기명령": contains("임차권등기명령")
    }

# ✅ 위험 점수 해석
def interpret_risk_score(score):
    if score >= 80:
        return "매우 높음", "⚠️ 보증금 반환이 어려울 가능성이 매우 높습니다."
    elif score >= 60:
        return "높음", "⚠️ 보증금 반환에 위험 요소가 존재합니다."
    elif score >= 40:
        return "보통", "⚠️ 일부 위험 요소가 있습니다."
    elif score >= 20:
        return "낮음", "✅ 위험 요소는 적은 편입니다."
    else:
        return "매우 낮음", "✅ 위험 요소가 거의 없습니다."
