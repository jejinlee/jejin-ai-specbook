import streamlit as st
st.set_page_config(page_title="시방서 AI 검색 시스템", layout="centered")

from sentence_transformers import SentenceTransformer, util
import torch

# 정제된 항목별 시방서 데이터 (예시)
paragraphs = [
    "경량 천장 공사는 주로 경량 철골(바탕틀)을 이용한 천장 마감공사로, 반자틀은 일반적으로 1000mm 간격, 인서트(앵커)는 900~1200mm 간격으로 시공한다. 틀은 수평을 유지해야 하며, 전기설비나 환기 덕트와 간섭이 없도록 한다. 석고보드 마감 시, 이음부는 퍼티 등으로 평활 처리 후 도장 마감한다.",
    "석고보드 시공은 규격에 따라 적절한 간격으로 나사못을 체결하며, 이음부에는 조인트 테이프와 퍼티를 적용하여 매끄러운 표면을 확보한다. 보드 마감 후에는 도장 또는 도배 마감이 일반적이다.",
    "경량 칸막이 공사는 스터드 및 트랙 구조로 이루어지며, 수직재 간격은 600mm 이하로 한다. 배선·배관 시 고려 사항을 반영하고, 방화 성능 확보를 위해 마감재와 충진재 사용 기준을 따른다."
]

@st.cache_resource
def load_embeddings():
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    embeddings = model.encode(paragraphs, convert_to_tensor=True)
    return model, embeddings, paragraphs

model, corpus_embeddings, paragraphs = load_embeddings()

st.title("📘 인테리어 시방서 AI 검색")

query = st.text_input("시방서 관련 질문을 입력하세요", placeholder="예: 경량 천장 반자틀 간격은?")

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    top_k = min(3, len(paragraphs))
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)

    st.subheader("🔍 AI 응답")
    top_result = paragraphs[hits[0][0]['corpus_id']]
    st.write(top_result)

    with st.expander("📄 관련 시방서 문단"):
        for hit in hits[0]:
            st.markdown(f"- {paragraphs[hit['corpus_id']]}")
