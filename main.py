import datetime

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ------------------------------
# 페이지 기본 설정
# ------------------------------
st.set_page_config(page_title="영화 흥행 예측기", page_icon="🎬")
st.title("🎬 영화 흥행 예측기")
st.write("영화를 고르거나 숫자를 직접 넣으면 **최종 누적 관객수**를 예측해드려요!")

# ------------------------------
# 1) 데이터 불러오기 (캐시로 속도 향상)
# ------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# ------------------------------
# 1-2) KOBIS API로 현재 상영 중인 영화(어제 박스오피스 상위 10편) 불러오기
# ------------------------------
KOBIS_KEY = st.secrets["KOBIS_KEY"]

@st.cache_data(ttl=3600)
def load_boxoffice(target_dt):
    """target_dt(YYYYMMDD) 하루치 일별 박스오피스 상위 10편을 가져와요."""
    url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
    res = requests.get(url, params={"key": KOBIS_KEY, "targetDt": target_dt}, timeout=10)
    res.raise_for_status()
    return res.json()["boxOfficeResult"]["dailyBoxOfficeList"]

# ------------------------------
# 2) 학습에 쓸 컬럼만 뽑고, 결측치 제거
#    영화명은 나중에 '영화 골라서 예측'에 쓰려고 같이 남겨둬요
# ------------------------------
features = ["스크린수", "상영횟수", "관객수"]  # 관객수 = 개봉 첫날 관객
target = "최종관객"

data = df[["영화명"] + features + [target]].dropna()

# ------------------------------
# 3) 로그 변환 (큰 편차 완화)
#    log1p = log(1+x)  → 0이 있어도 안전하게 로그 처리 가능
# ------------------------------
X_log = np.log1p(data[features])
y_log = np.log1p(data[target])

# ------------------------------
# 4) 선형회귀 모델 학습
# ------------------------------
model = LinearRegression()
model.fit(X_log, y_log)

# 학습 데이터로 예측 후 R² 계산 (로그 스케일 기준)
y_pred_log = model.predict(X_log)
r2 = r2_score(y_log, y_pred_log)

# ------------------------------
# 5) R² 카드로 보여주기
# ------------------------------
st.metric(label="모델의 R² (결정계수)", value=f"{r2:.3f}")
st.caption("📌 R²는 모델이 실제 관객수 변화를 얼마나 잘 설명하는지 나타내는 지표예요 (1에 가까울수록 좋아요).")

st.divider()

# ------------------------------
# 6) 예측 방법 고르기
# ------------------------------
st.subheader("🎛️ 예측할 영화 정보를 입력해주세요")

mode = st.radio("예측 방법", ["현재 상영작 골라서 예측", "숫자 직접 입력"], horizontal=True)

current_acc = None  # 현재까지 누적 관객수 (상영작을 골랐을 때만 채워짐)

if mode == "현재 상영작 골라서 예측":
    # 어제 날짜 기준 박스오피스 = 지금 극장에서 상영 중인 영화들
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y%m%d")
    boxoffice = load_boxoffice(yesterday)

    labels = [f"{m['rank']}위 · {m['movieNm']}" for m in boxoffice]
    picked = st.selectbox("현재 상영 중인 영화를 골라주세요 (어제 박스오피스 상위 10편)", labels)
    selected = boxoffice[labels.index(picked)]

    # 예측에는 '개봉 첫날' 데이터가 필요해요 → 개봉일 박스오피스를 다시 조회
    open_dt = selected["openDt"].replace("-", "")
    first = None
    if open_dt and open_dt <= yesterday:
        opening_list = load_boxoffice(open_dt)
        first = next((m for m in opening_list if m["movieCd"] == selected["movieCd"]), None)

    if first is not None:
        screen = float(first["scrnCnt"])
        showings = float(first["showCnt"])
        first_day = float(first["audiCnt"])
        st.caption(f"개봉일 {selected['openDt']} 기준 · 스크린수 {screen:,.0f} · 상영횟수 {showings:,.0f} · 첫날 관객 {first_day:,.0f}")
    else:
        # 개봉일 기록을 못 찾으면(시사회 상영 등) 어제 데이터로 대신해요
        screen = float(selected["scrnCnt"])
        showings = float(selected["showCnt"])
        first_day = float(selected["audiCnt"])
        st.caption(f"개봉일 첫날 기록이 없어 어제 데이터로 예측해요 · 스크린수 {screen:,.0f} · 상영횟수 {showings:,.0f} · 관객 {first_day:,.0f}")

    current_acc = float(selected["audiAcc"])
else:
    screen = st.slider("스크린수", min_value=1, max_value=3000, value=500, step=10)
    showings = st.slider("상영횟수", min_value=1, max_value=50000, value=5000, step=100)
    first_day = st.number_input("첫날 관객수", min_value=0, value=10000, step=1000)

# ------------------------------
# 7) 예측하기 (로그 → 원래 스케일로 복원)
# ------------------------------
input_log = np.log1p(pd.DataFrame(
    [[screen, showings, first_day]], columns=features
))

pred_log = model.predict(input_log)[0]
pred_final = np.expm1(pred_log)  # 로그 되돌리기 (expm1 = log1p의 역함수)

# 음수 방지 (혹시 모를 경우 대비)
pred_final = max(pred_final, 0)

st.divider()
st.subheader("🔮 예측 결과")

if current_acc is not None:
    # 상영작을 골랐을 때는 예측과 현재 누적을 나란히 비교
    col1, col2 = st.columns(2)
    col1.metric("예측 최종관객", f"{pred_final:,.0f} 명",
                delta=f"{pred_final - current_acc:,.0f} 명 (현재 누적과의 차이)")
    col2.metric("현재까지 누적 관객", f"{current_acc:,.0f} 명")
    st.caption("※ 아직 상영 중이라 최종 관객수는 알 수 없어요. 예측이 현재 누적보다 작다면, 이미 예측을 넘어선 거예요!")
else:
    st.markdown(f"## 예상 최종 누적 관객수: **{pred_final:,.0f} 명**")

st.info("💡 이 예측은 과거 영화 데이터를 바탕으로 한 추정치이며, 실제 흥행은 다양한 변수에 영향을 받아요!")

# ------------------------------
# 8) 예측 원리를 초보자 눈높이로 설명하기
# ------------------------------
st.divider()
st.subheader("🧑‍🏫 어떤 원리로 예측했나요?")

with st.expander("예측 원리 4단계로 알아보기", expanded=False):
    # 모델이 배운 규칙에서 어떤 재료의 영향이 가장 큰지 계산
    influence = pd.Series(model.coef_, index=features)
    friendly_names = {"스크린수": "스크린수", "상영횟수": "상영횟수", "관객수": "첫날 관객수"}
    strongest = friendly_names[influence.abs().idxmax()]

    st.markdown(f"""
**1단계. 과거 영화의 기록을 공부해요**

과거 한국 영화 {len(data):,}편의 기록을 모았어요.
영화마다 개봉 첫날의 세 가지 숫자(스크린수, 상영횟수, 첫날 관객수)와
최종 누적 관객수가 함께 적혀 있어요.
"첫날 성적이 좋았던 영화는 최종 관객도 많았을까?"를 데이터로 확인하는 거예요.

**2단계. 숫자의 폭을 고르게 만들어요 (로그 변환)**

관객수는 영화마다 차이가 아주 커요. 몇천 명짜리 영화도 있고 천만 명짜리 영화도 있죠.
이렇게 차이가 크면 큰 영화 몇 편이 규칙을 독차지해요.
그래서 큰 수를 눌러 주는 로그 변환으로 숫자의 폭을 줄인 뒤 규칙을 찾았어요.

**3단계. 선형회귀로 계산식을 만들어요**

선형회귀는 "첫날 성적이 이 정도면 최종 관객은 이 정도"라는
가장 그럴듯한 계산식을 과거 데이터에서 찾아 주는 방법이에요.
이 식이 과거 기록을 얼마나 잘 설명하는지 나타낸 값이 맨 위의 R²({r2:.3f})예요.
이번에 배운 규칙에서는 세 재료 가운데 **{strongest}**의 영향이 가장 컸어요.

**4단계. 새 영화의 숫자를 식에 넣어요**

지금 고른 영화의 첫날 숫자
(스크린수 {screen:,.0f} · 상영횟수 {showings:,.0f} · 첫날 관객 {first_day:,.0f})를
그 계산식에 넣어 나온 값이 바로 예측 결과 **{pred_final:,.0f}명**이에요.
과거 영화들이 걸어간 길을 근거로 "이 영화도 비슷하게 가면 이쯤 도착한다"고 어림하는 거죠.
""")
