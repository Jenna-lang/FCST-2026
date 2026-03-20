import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình & Tải dữ liệu
st.set_page_config(page_title="AI Supply Chain Advisor", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_excel('AICheck.xlsx')
        df.columns = [str(col).strip() for col in df.columns]
        # Chuyển đổi ngày tháng an toàn
        df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
        df = df.dropna(subset=['ds'])
        return df
    except Exception as e:
        st.error(f"Lỗi tải file: {e}")
        return None

@st.cache_resource
def fast_forecast(data_series):
    if len(data_series) < 2: return None
    # Cấu hình Prophet tối ưu tốc độ
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(data_series)
    future = m.make_future_dataframe(periods=12, freq='MS')
    fcst = m.predict(future)
    return fcst[['ds', 'yhat']]

df = load_data()

if df is not None:
    # 2. Sidebar Thiết lập
    st.sidebar.header("⚡ Cấu hình")
    cust_col = st.sidebar.selectbox("Cột khách hàng:", [c for c in df.columns if 'Customer' in c] or df.columns)
    cie_col = st.sidebar.selectbox("Cột CIE:", [c for c in df.columns if 'CIE' in c] or df.columns)
    
    cust_list = sorted(df[cust_col].unique().astype(str))
    selected_cust = st.sidebar.selectbox("1. Chọn khách hàng:", ["-- Chọn --"] + cust_list)

    if selected_cust != "-- Chọn --":
        cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
        
        # Pareto 80/20 tính theo M USD
        sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
        sales['Cum_Pct'] = sales['M USD'].cumsum() / (sales['M USD'].sum() or 1)
        top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
        
        selected_prod = st.selectbox("2. Chọn sản phẩm phân tích:", top_prods if len(top_prods)>0 else df['Material name'].unique()[:5])
        
        # Biến trung gian để truyền dữ liệu giữa 2 Tab
        growth_final = 0.0 

        tab1, tab2 = st.tabs(["📊 Trung bình cùng kỳ (AI)", "📋 Bảng ngang Pareto 2026"])

        with tab1:
            prod_df = cust_df[cust_df['Material name'] == selected_prod].copy()
            prod_df['m'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
            actual_prod = prod_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
            
            res = fast_forecast(actual_prod)
            if res is not None:
                # --- LOGIC TRUNG BÌNH CÙNG KỲ ---
                future_dates = res[res['ds'] > actual_prod['ds'].max()]
                avg_fcst = future_dates['yhat'].mean()
                
                # Tìm tháng tương ứng năm 2025
                rel_months = future_dates['ds'].dt.month.unique()
                act_25_sp = actual_prod[(actual_prod['ds'].dt.year == 2025) & (actual_prod['ds'].dt.month.isin(rel_months))]
                avg_act_25 = act_25_sp['y'].mean() if not act_25_sp.empty else actual_prod[actual_prod['ds'].dt.year == 2025]['y'].mean()
                
                growth_final = ((avg_fcst - avg_act_25) / avg_act_25 * 100) if avg_act_25 > 0 else 0
                color = "green" if growth_final >= 0 else "red"
                
                st.info(f"💡 Dự báo giai đoạn tới so với cùng kỳ 2025 tăng trưởng: **{growth_final:.1f}%**")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual_prod['ds'], y=actual_prod['y'], name="Thực tế (Actual)"))
                fig.add_trace(go.Scatter(x=future_dates['ds'], y=future_dates['yhat'], line=dict(dash='dash', color=color), name="Dự báo 2026 (FCST)"))
                fig.update_layout(title=f"Xu hướng nhu cầu: {selected_prod}", xaxis_title="Thời gian", yaxis_title="Số lượng")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Không đủ dữ liệu lịch sử để AI thực hiện dự báo.")

        with tab2:
            st.subheader("📋 Kế hoạch sản xuất Pareto 2026")
            # Lọc dữ liệu Pareto năm 2025 làm nền tảng
            p_df = cust_df[cust_df['Material name'].isin(top_prods) & (cust_df['ds'].dt.year == 2025)].copy()
            
            if not p_df.empty:
                # Tính trung bình tháng thực tế 2025 cho từng mã CIE
                base_stats = p_df.groupby(['Material name', cie_col])['Order qty.(A)'].mean().reset_index()
                
                # Áp dụng hệ số tăng trưởng từ Tab 1 (Nếu Tab 1 lỗi thì mặc định là 1.0)
                g_factor = 1 + (growth_final / 100) if growth_final != 0 else 1.0
                
                # Tạo danh sách các tháng 2026 làm tiêu đề cột
                months_26 = [m.strftime('%m/%Y') for m in pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')]
                
                pivot_list = []
                for _, r in base_stats.iterrows():
                    row = {'Sản phẩm': r['Material name'], 'CIE': r[cie_col]}
                    for m in months_26:
                        row[m] = round(r['Order qty.(A)'] * g_factor, 0)
                    pivot_list.append(row)
                
                res_df = pd.DataFrame(pivot_list)
                
                # Bộ lọc nhanh sản phẩm
                f_p = st.multiselect("Lọc nhanh theo sản phẩm:", top_prods)
                if f_p: 
                    res_df = res_df[res_df['Sản phẩm'].isin(f_p)]
                
                st.write(f"📌 Hệ số dự báo đang áp dụng: **{g_factor:.2f}x**")
                st.dataframe(res_df.style.format("{:,.0f}", subset=months_26), use_container_width=True)
                
                st.download_button("📥 Tải báo cáo Kế hoạch (CSV)", res_df.to_csv(index=False).encode('utf-8-sig'), "Production_Plan_2026.csv")
            else:
                st.warning("⚠️ Không tìm thấy dữ liệu thực tế năm 2025 của khách hàng này để lập kế hoạch.")
