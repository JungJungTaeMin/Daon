import os
import uuid
import json
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import ContentType
import uvicorn

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # 실행 시 API 키가 없으면 에러 발생. .env 파일을 확인하세요.
    # raise ValueError("GEMINI_API_KEY environment variable not set")
    print("Warning: GEMINI_API_KEY not set. AI features will fail.")

else:
    genai.configure(api_key=api_key)

# ---------------------------------------------------------
# Prompt & Model Setup
# ---------------------------------------------------------

# 기본 시스템 프롬프트 (Fallback용)
DEFAULT_SYSTEM_PROMPT = """
당신은 '마음 건강 관리 플랫폼'의 전문 AI 심리 상담사입니다.
사용자의 감정 상태를 파악하여 따뜻한 위로와 조언을 제공하세요.
"""

# JSON 응답을 위한 설정 (분석 및 추천용)
json_generation_config = {
    "response_mime_type": "application/json",
}

# 분석용 모델 (설문 분석 등 JSON 출력용)
analysis_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=json_generation_config,
    system_instruction="You are a psychological analyst."
)

app = FastAPI()

# 데모를 위한 간단한 인메모리 저장소
fake_db = {
    "keywords": [],
    "quests": {},  # quest_id: status 매핑 저장
    "sessions": {}  # session_id: system_prompt 매핑 저장
}


# ---------------------------------------------------------
# Data Models
# ---------------------------------------------------------

# 1. 설문조사 모델
class SurveyItem(BaseModel):
    number: int
    question: str
    answer: str


class AnalysisDetail(BaseModel):
    strong: List[str]
    weakness: List[str]
    keyword: List[str]


class SurveyAnalysisResponse(BaseModel):
    message: str
    result: List[AnalysisDetail]


# 2. 키워드/페르소나 모델
class PersonaData(BaseModel):
    description: str
    keywords: str


class PersonaResponse(BaseModel):
    message: str
    data: PersonaData


# 3. 퀘스트 모델
class Mission(BaseModel):
    question: str
    state: str = "NOT"


# 4. 채팅 관련 모델 (신규/수정됨)

class ChatInitRequest(BaseModel):
    system_prompt: str


class ChatInitResponse(BaseModel):
    session_id: str
    welcome_message: str


class ChatHistoryItem(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    history: List[ChatHistoryItem]
    new_message: str


class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.post("/api/survey", response_model=SurveyAnalysisResponse)
async def analyze_survey(items: List[SurveyItem]):
    """
    설문 답변 분석 API
    """
    conversation_text = "\n".join([f"Q: {item.question}\nA: {item.answer}" for item in items])
    prompt = f"""
    아래는 사용자의 설문 답변입니다. 심리학적으로 분석하여 JSON으로 출력하세요.

    [답변 데이터]
    {conversation_text}

    [출력 형식]
    {{
        "strong": ["강점1", "강점2", "강점3"],
        "weakness": ["약점1", "약점2", "약점3"],
        "keyword": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"]
    }}
    """
    try:
        response = await analysis_model.generate_content_async(prompt)
        result_json = json.loads(response.text)

        if isinstance(result_json, list):
            result_data = result_json[0]
        else:
            result_data = result_json

        return SurveyAnalysisResponse(
            message="분석 완료 및 저장 성공",
            result=[
                AnalysisDetail(
                    strong=result_data.get("strong", []),
                    weakness=result_data.get("weakness", []),
                    keyword=result_data.get("keyword", [])
                )
            ]
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post("/api/survey/key", response_model=PersonaResponse)
async def select_keywords(keywords: List[str]):
    """
    키워드 저장 및 페르소나 생성 API
    """
    fake_db["keywords"] = keywords
    keywords_str = ", ".join(keywords)

    prompt = f"""
    키워드: [{keywords_str}]
    위 키워드를 바탕으로 희망차고 동기부여가 되는 '페르소나 설명(description)'을 3문장 내외로 작성하세요.
    JSON 형식: {{ "description": "..." }}
    """
    try:
        response = await analysis_model.generate_content_async(prompt)
        data = json.loads(response.text)
        return PersonaResponse(
            message="최종 페르소나 저장 완료",
            data=PersonaData(
                description=data.get("description", ""),
                keywords=keywords_str
            )
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")


@app.get("/api/questions", response_model=List[Mission])
async def get_missions():
    """
    미션 생성 API
    """
    saved_keywords = fake_db["keywords"] or ["긍정", "행복"]
    keywords_str = ", ".join(saved_keywords)

    prompt = f"""
    키워드: [{keywords_str}]
    이 키워드를 달성하기 위한 구체적인 미션 10가지를 JSON 배열로 생성하세요.
    각 항목 형식: {{ "question": "미션내용", "state": "NOT" }}
    """
    try:
        response = await analysis_model.generate_content_async(prompt)
        missions = json.loads(response.text)

        # 데이터 검증 및 변환
        validated_missions = []
        for i, m in enumerate(missions):
            if isinstance(m, dict) and "question" in m:
                # DB에 초기 상태 저장 (Mock)
                fake_db["quests"][str(i + 1)] = "NOT"
                validated_missions.append(Mission(question=m["question"], state="NOT"))

        if not validated_missions:
            return [Mission(question="생성된 미션이 없습니다.", state="NOT")]

        return validated_missions
    except Exception as e:
        return [Mission(question="미션 생성 실패 (기본 미션)", state="NOT")]


@app.patch("/api/questions")
async def update_quest_status(status_update: Dict[str, str]):
    """
    퀘스트 상태 일괄 변경 API
    Request Body 예시: { "1": "NOT", "2": "SUCCESS" }
    """
    # 실제 DB 업데이트 로직이 들어갈 자리
    for quest_id, state in status_update.items():
        fake_db["quests"][quest_id] = state
        print(f"Quest {quest_id} updated to {state}")

    return {
        "status": "success",
        "updated_count": len(status_update),
        "data": status_update
    }


# ---------------------------------------------------------
# Chat Endpoints (New Logic)
# ---------------------------------------------------------

@app.post("/api/conversation/init", response_model=ChatInitResponse)
async def init_chat(request: ChatInitRequest):
    """
    챗봇 세션 시작: 시스템 프롬프트를 받아 세션을 초기화합니다.
    """
    new_session_id = str(uuid.uuid4())

    # 세션별 시스템 프롬프트 저장 (메모리 DB)
    fake_db["sessions"][new_session_id] = request.system_prompt

    return ChatInitResponse(
        session_id=new_session_id,
        welcome_message="안녕하세요! 대화를 시작할 준비가 되었습니다."
    )


@app.post("/api/conversation/chat", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """
    채팅 주고받기: 히스토리를 포함하여 AI 응답을 생성합니다.
    Request 예시:
    {
      "history": [
        { "role": "system", "content": "너는 추구미야..." },
        { "role": "user", "content": "안녕" },
        { "role": "assistant", "content": "반가워" }
      ],
      "new_message": "요즘 너무 힘들어"
    }
    """
    try:
        # 1. 시스템 프롬프트 추출
        # 히스토리의 첫 번째가 system이면 그것을 사용, 아니면 기본값
        system_instruction = DEFAULT_SYSTEM_PROMPT
        chat_history = []

        for item in request.history:
            if item.role == "system":
                system_instruction = item.content
            elif item.role == "user":
                chat_history.append({"role": "user", "parts": [item.content]})
            elif item.role == "assistant":
                chat_history.append({"role": "model", "parts": [item.content]})

        # 2. 모델 동적 생성 (요청별 시스템 프롬프트 적용을 위해)
        chat_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_instruction
        )

        # 3. 채팅 세션 시작 (히스토리 주입)
        chat_session = chat_model.start_chat(history=chat_history)

        # 4. 메시지 전송 및 응답 생성
        response = await chat_session.send_message_async(request.new_message)

        return ChatResponse(reply=response.text)

    except Exception as e:
        print(f"Chat Error: {e}")
        return ChatResponse(reply="죄송합니다. 잠시 오류가 발생했습니다. 다시 말씀해 주시겠어요?")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8083, reload=True)