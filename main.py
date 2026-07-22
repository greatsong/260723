import streamlit as st
from supabase import create_client
import pandas as pd

# ── Supabase 연결 ──────────────────────────────
# secrets.toml (또는 Streamlit Cloud의 Secrets 설정)에 저장된 값을 불러와요
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)

# ── 화면 기본 설정 ──────────────────────────────
st.set_page_config(page_title="약속 잡기", page_icon="📅")
st.title("📅 약속 잡기")
st.write("모두가 편한 시간을 함께 찾아봐요 😊")

# 선택 가능한 시간 목록 (필요하면 자유롭게 수정하세요)
시간목록 = [
    "월요일 오전", "월요일 오후", "월요일 저녁",
    "화요일 오전", "화요일 오후", "화요일 저녁",
    "수요일 오전", "수요일 오후", "수요일 저녁",
    "목요일 오전", "목요일 오후", "목요일 저녁",
    "금요일 오전", "금요일 오후", "금요일 저녁",
]

# ── 입력 폼 ────────────────────────────────────
이름 = st.text_input("이름을 입력해주세요")
선택시간 = st.multiselect("가능한 시간을 모두 골라주세요", 시간목록)

if st.button("제출"):
    if not 이름:
        st.warning("이름을 입력해주세요!")
    elif not 선택시간:
        st.warning("가능한 시간을 하나 이상 선택해주세요!")
    else:
        # 여러 개 선택한 시간을 콤마(,)로 이어 하나의 문자열로 저장해요
        시간문자열 = ", ".join(선택시간)
        supabase.table("260722").insert({"name": 이름, "times": 시간문자열}).execute()
        st.success(f"{이름}님, 제출 완료되었어요! 감사합니다 🙌")

st.divider()

# ── 전체 명단 표시 ──────────────────────────────
st.subheader("📋 지금까지 제출된 명단")

응답 = supabase.table("260722").select("*").execute()
데이터 = 응답.data

if 데이터:
    df = pd.DataFrame(데이터)
    st.dataframe(df[["name", "times"]], use_container_width=True)
else:
    st.info("아직 제출된 내용이 없어요. 첫 번째로 참여해보세요!")
