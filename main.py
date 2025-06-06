from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import pandas as pd
import joblib
import shap
from dotenv import load_dotenv
import google.generativeai as genai

from risk_utils import (
    extract_address_and_building_from_pdf,
    get_lawd_cd,
    get_latest_officetel_trade,
    extract_text_from_pdf_with_ocr,
    parse_ocr_text_to_features,
    interpret_risk_score
)

# ✅ .env 로드 및 API 키 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERVICE_KEY = os.getenv("PUBLIC_DATA_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ✅ FastAPI 앱
app = FastAPI()

# ✅ CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 분석 설명 생성 (Gemini 사용)
def generate_llm_explanation(score, level, shap_list):
    top_factors = sorted(shap_list, key=lambda x: abs(x["impact"]), reverse=True)[:3]
    factor_text = "\n".join([
        f"- {f['feature']}: 영향도 {abs(f['impact']):.2f}, 위험 {('상승' if f['direction'] == 'up' else '완화')}"
        for f in top_factors
    ])

    prompt = f"""
전세사기 분석 결과입니다:

• 위험 점수: {round(score, 2)}점
• 등급: {level}
• 주요 영향 요인:
{factor_text}

이 내용을 바탕으로 사용자에게 간결하고 이해하기 쉬운 분석 요약을 작성해주세요.
"""

    response = model.generate_content(prompt)
    return response.text.strip()

# ✅ 분석 API
@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile,
    area: float = Form(...),
    jeonse_price: int = Form(...)
):
    temp_path = f"./temp/{file.filename}"
    os.makedirs("./temp", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    address, building_name = extract_address_and_building_from_pdf(temp_path)
    if not address or not building_name:
        return JSONResponse(status_code=400, content={"error": "주소 또는 건물명 추출 실패"})

    lawd_cd = get_lawd_cd(address, "./법정동코드.csv")
    if not lawd_cd:
        return JSONResponse(status_code=400, content={"error": "법정동코드 조회 실패"})

    latest_info = get_latest_officetel_trade(lawd_cd, building_name, area, SERVICE_KEY)
    if latest_info is None:
        return JSONResponse(status_code=404, content={"error": "실거래가 조회 실패"})

    sale_price = int(latest_info["거래금액(만원)"].replace(",", "")) * 10000
    jeonse_ratio = jeonse_price / sale_price

    ocr_text = extract_text_from_pdf_with_ocr(temp_path)
    features = parse_ocr_text_to_features(ocr_text, jeonse_ratio)
    df = pd.DataFrame([features])

    model_file = joblib.load("./risk_score_model_by_ratio.pkl")
    score = model_file.predict(df)[0]
    level, message = interpret_risk_score(score)

    explainer = shap.Explainer(model_file)
    shap_values = explainer(df)
    shap_result = sorted(
        zip(df.columns, shap_values[0].values),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    shap_result = [
        {"feature": k, "impact": float(v), "direction": "up" if v > 0 else "down"}
        for k, v in shap_result
    ]

    # ✅ Gemini 기반 요약
    llm_summary = generate_llm_explanation(score, level, shap_result)

    return {
        "address": address,
        "building": building_name,
        "sale_price": int(sale_price),
        "jeonse_price": int(jeonse_price),
        "jeonse_ratio": round(float(jeonse_ratio) * 100, 2),
        "risk_score": round(float(score), 2),
        "risk_level": level,
        "risk_message": message,
        "shap": shap_result,
        "llm_explanation": llm_summary
    }
