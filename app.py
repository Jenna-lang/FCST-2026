import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="LED Lighting Forecast 2026", layout="wide")

# --- HÀM TÍNH TOÁN ---
def calculate_metrics(df):
    actual = pd.to_numeric(df['Actual'], errors='coerce').fillna(0)
    forecast = pd.to_numeric(df['Forecast'], errors='coerce').fillna(0)
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100 if mask.any() else 0
    return mape, actual, forecast

# --- GIAO DIỆN CHÍNH ---
st.title("🚀 Hệ thống Dự báo Đơn hàng LED - FCST 2026")

# Giả sử bạn load dữ liệu từ file csv trong repo
# df = pd.read_csv('data_lighting.csv')

# Test thử với dữ liệu giả lập (Bạn thay bằng dữ liệu thật của mình nhé)
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Actual': [1200, 1500, 1100, 1700, 1400, 1600],
    'Forecast': [1150, 1450, 1200, 1650, 1380, 1550]
}
df = pd.DataFrame(data)

# Hiển thị Chỉ số
mape_val, act, fcst = calculate_metrics(df)
col1, col2, col3 = st.columns(3)
col1.metric("MAPE (%)", f"{mape_val:.2f}%")
col2.metric("Accuracy", f"{100-mape_val:.2f}%")
col3.metric("Tổng đơn hàng", f"{act.sum():,.0f}")

# Vẽ biểu đồ
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Month'], y=act, name='Thực tế', line=dict(color='#00CC96')))
fig.add_trace(go.Scatter(x=df['Month'], y=fcst, name='Dự báo', line=dict(dash='dash', color='#636EFA')))
st.plotly_chart(fig, use_container_width=True)
