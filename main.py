import streamlit as st
import pandas as pd
import numpy as np
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

mode = st.radio("예측 방법", ["영화 골라서 예측", "숫자 직접 입력"], horizontal=True)

actual_final = None  # 실제 최종관객 (영화를 골랐을 때만 채워짐)

if mode == "영화 골라서 예측":
    # 드롭다운에 제목을 입력하면 목록이 걸러져요 (검색되는 드롭다운)
    movie = st.selectbox("영화를 골라주세요", data["영화명"].tolist())
    row = data[data["영화명"] == movie].iloc[0]
    screen = row["스크린수"]
    showings = row["상영횟수"]
    first_day = row["관객수"]
    actual_final = row["최종관객"]
    st.caption(f"스크린수 {screen:,.0f} · 상영횟수 {showings:,.0f} · 첫날 관객 {first_day:,.0f}")
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

if actual_final is not None:
    # 영화를 골랐을 때는 예측과 실제를 나란히 비교
    col1, col2 = st.columns(2)
    col1.metric("예측 최종관객", f"{pred_final:,.0f} 명",
                delta=f"{pred_final - actual_final:,.0f} 명 (실제와의 차이)")
    col2.metric("실제 최종관객", f"{actual_final:,.0f} 명")
    st.caption("※ 이미 학습에 쓴 영화라서, 처음 보는 영화보다 잘 맞는 편이에요.")
else:
    st.markdown(f"## 예상 최종 누적 관객수: **{pred_final:,.0f} 명**")

st.info("💡 이 예측은 과거 영화 데이터를 바탕으로 한 추정치이며, 실제 흥행은 다양한 변수에 영향을 받아요!")
