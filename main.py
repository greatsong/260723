import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# ── 기본 화면 설정 ──────────────────────────────
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌤️", layout="centered")
st.title("🌤️ 서울 기온 예측기")
st.write("1907년부터 오늘까지 서울 기온 데이터를 가지고, 미래 기온을 예측해보는 앱이에요!")

# ── 데이터 불러오기 (캐시로 재요청 방지) ──────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(url, encoding="cp949")
    return df

df = load_data()

# 날짜 컬럼을 진짜 날짜 타입으로 바꾸고, 연도만 뽑아내기
df["날짜"] = pd.to_datetime(df["날짜"])
df["연도"] = df["날짜"].dt.year

# ── 연도별 평균기온 계산 ──────────────────────────
# 연도별로 묶어서 평균기온의 평균과, 그 해 관측된 일수(count)를 같이 구함
yearly = df.groupby("연도")["평균기온"].agg(["mean", "count"]).reset_index()
yearly.columns = ["연도", "평균기온", "관측일수"]

# 올해(진행 중이라 데이터가 덜 쌓인 해)는 제외
current_year = datetime.now().year
yearly = yearly[yearly["연도"] != current_year]

# 관측일수가 너무 적은 해(첫 해, 전쟁 시기 등)는 평균이 왜곡되니 제외
# 1년은 보통 365일 → 300일 미만이면 신뢰하기 어렵다고 보고 뺌
yearly_filtered = yearly[yearly["관측일수"] >= 300].reset_index(drop=True)

# ── 선형회귀 학습 ────────────────────────────────
X = yearly_filtered[["연도"]].values           # 입력: 연도
y = yearly_filtered["평균기온"].values          # 정답: 평균기온

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)  # 모델이 실제 데이터를 얼마나 잘 설명하는지 나타내는 값

# 학습에 사용한 데이터의 연도 범위 (이 밖은 미래 예측이라 조심해야 함)
min_year = int(yearly_filtered["연도"].min())
max_year = int(yearly_filtered["연도"].max())

# ── R² 카드 표시 ─────────────────────────────────
st.subheader("📊 모델 성능")
st.metric("R² (결정계수)", f"{r2:.3f}")
st.caption("R²는 우리 직선이 실제 기온 변화를 얼마나 잘 설명하는지 나타내는 값이에요 (1에 가까울수록 잘 맞아요).")

# ── 슬라이더로 연도 선택 → 예측 ───────────────────
st.subheader("🔮 연도를 골라 기온을 예측해보세요")
selected_year = st.slider("연도 선택", 1900, 2100, value=current_year)

predicted_temp = model.predict([[selected_year]])[0]

st.metric(f"{selected_year}년 예측 평균기온", f"{predicted_temp:.1f} °C")

# 학습 데이터 범위 밖이면 조심하라는 안내
if selected_year < min_year or selected_year > max_year:
    st.warning(f"⚠️ 참고용, 조심! 이 예측은 학습 데이터 범위({min_year}~{max_year}년)를 벗어난 값이에요.")

# ── plotly 그래프: 실제 데이터 + 회귀 직선 ─────────
st.subheader("📈 연도별 평균기온과 예측 직선")

fig = go.Figure()

# 실제 관측된 연도별 평균기온 (점)
fig.add_trace(go.Scatter(
    x=yearly_filtered["연도"], y=yearly_filtered["평균기온"],
    mode="markers", name="실제 연도별 평균기온",
    marker=dict(color="royalblue")
))

# 학습한 회귀 직선 (1900~2100년까지 쭉 그려보기)
line_years = np.arange(1900, 2101)
line_pred = model.predict(line_years.reshape(-1, 1))
fig.add_trace(go.Scatter(
    x=line_years, y=line_pred,
    mode="lines", name="회귀 직선(예측선)",
    line=dict(color="orange")
))

# 내가 고른 연도는 빨간 점으로 강조
fig.add_trace(go.Scatter(
    x=[selected_year], y=[predicted_temp],
    mode="markers", name="선택한 연도",
    marker=dict(color="red", size=13, symbol="star")
))

fig.update_layout(
    xaxis_title="연도", yaxis_title="평균기온(°C)",
    legend=dict(orientation="h", y=-0.2)
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"※ 학습에는 관측일수가 충분한 {min_year}~{max_year}년 데이터만 사용했어요.")
