# --- PHẦN CŨ CỦA BẠN (Giữ nguyên) ---
df = pd.read_excel("forecast_led_lighting.xlsx") 
# Giả sử df của bạn đã có cột 'Actual' và 'Forecast'

# --- PHẦN MỚI CHÈN THÊM (Chỉ thêm, không thay thế) ---
import numpy as np
import plotly.graph_objects as go

# Tính toán (không làm thay đổi df gốc)
mask = df['Actual'] != 0
mape = np.mean(np.abs((df['Actual'][mask] - df['Forecast'][mask]) / df['Actual'][mask])) * 100

# Hiển thị thêm lên giao diện
st.divider() # Vạch ngăn cách giữa phần cũ và phần mới
st.subheader("📊 Phân tích độ chính xác")
st.metric("Chỉ số MAPE", f"{mape:.2f}%")

# Vẽ biểu đồ so sánh (Tận dụng dữ liệu cũ để vẽ)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Thực tế'))
fig.add_trace(go.Scatter(x=df.index, y=df['Forecast'], name='Dự báo', line=dict(dash='dash')))
st.plotly_chart(fig)
