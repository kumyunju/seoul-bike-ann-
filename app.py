from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="자전거 대여량 예측",
    page_icon="🚲",
)

st.title("인공신경망을 이용한 데이터 예측 프로그램")
st.write("기온, 습도, 풍속을 입력하여 서울시 자전거 대여량을 예측합니다.")

# app.py와 같은 폴더에서 CSV 파일 찾기
csv_files = list(
    Path(__file__).parent.glob("SeoulBikeData_1000_3inputs*.csv")
)

if len(csv_files) != 1:
    st.error(
        "SeoulBikeData_1000_3inputs CSV 파일을 "
        "app.py와 같은 폴더에 한 개만 올려 주세요."
    )
    st.stop()

data = pd.read_csv(csv_files[0], encoding="utf-8-sig")

input_columns = [
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
]
target_column = "Rented Bike Count"


@st.cache_resource
def train_model(data):
    X = data[input_columns]
    y = data[target_column]

    # 학습용 80%, 평가용 20%로 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 입력값 표준화 + 인공신경망
    neural_network = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=2000,
            early_stopping=True,
            random_state=42,
        ),
    )

    # 대여량도 표준화하여 학습하고 예측 시 원래 단위로 복원
    model = TransformedTargetRegressor(
        regressor=neural_network,
        transformer=StandardScaler(),
    )

    model.fit(X_train, y_train)

    # 대여량이 음수가 되지 않도록 보정
    predictions = model.predict(X_test).clip(min=0)

    return model, y_test, predictions


with st.spinner("인공신경망이 데이터를 학습하고 있습니다..."):
    model, y_test, predictions = train_model(data)

st.subheader("예측 조건 입력")

with st.form("prediction_form"):
    temperature = st.number_input(
        "기온 (°C)",
        min_value=float(data[input_columns[0]].min()),
        max_value=float(data[input_columns[0]].max()),
        value=20.0,
        step=0.1,
    )

    humidity = st.number_input(
        "습도 (%)",
        min_value=float(data[input_columns[1]].min()),
        max_value=float(data[input_columns[1]].max()),
        value=50.0,
        step=1.0,
    )

    wind_speed = st.number_input(
        "풍속 (m/s)",
        min_value=float(data[input_columns[2]].min()),
        max_value=float(data[input_columns[2]].max()),
        value=1.5,
        step=0.1,
    )

    submitted = st.form_submit_button("자전거 대여량 예측")

if submitted:
    new_data = pd.DataFrame(
        [[temperature, humidity, wind_speed]],
        columns=input_columns,
    )
    result = max(0.0, float(model.predict(new_data)[0]))
    st.success(f"예측 자전거 대여량: 약 {result:,.0f}대")

st.subheader("모델 평가")

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

col1, col2 = st.columns(2)
col1.metric("평균 절대 오차 (MAE)", f"{mae:,.1f}대")
col2.metric("결정계수 (R²)", f"{r2:.3f}")

st.caption(
    "학습에 사용하지 않은 평가용 데이터 20%로 계산했습니다. "
    "MAE는 낮을수록 좋고, R²는 1에 가까울수록 좋습니다. "
    "R²는 음수가 될 수도 있으며 정확도 백분율이 아닙니다."
)

st.subheader("실제 대여량과 예측 대여량 비교")

comparison = pd.DataFrame({
    "실제 대여량": y_test.to_numpy(),
    "예측 대여량": predictions,
})

st.scatter_chart(
    comparison,
    x="실제 대여량",
    y="예측 대여량",
)

st.caption(
    "점들이 실제 대여량과 예측 대여량이 같은 대각선에 "
    "가까울수록 예측 오차가 작습니다."
)

with st.expander("데이터와 인공신경망 구조 보기"):
    st.write(f"전체 데이터: {len(data):,}개")
    st.write("입력층 3개 → 은닉층 32개 → 은닉층 16개 → 출력층 1개")
    st.dataframe(data.head(20), hide_index=True)

st.info(
    "이 프로그램은 과제용 예측 모델입니다. "
    "시간대, 강수량, 공휴일 등은 입력에 포함되지 않아 "
    "실제 대여량과 차이가 날 수 있습니다."
)
