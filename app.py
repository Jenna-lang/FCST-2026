import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình hệ thống
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- HÀM XỬ LÝ DỮ LIỆU ---
def process_data(uploaded_file):
    try:
        # Đọc toàn bộ các sheet nếu cần, mặc định sheet đầu tiên
        df = pd.read_excel(uploaded_file)
        # Làm sạch tên cột: bỏ khoảng trắng thừa, viết hoa chữ đầu
        df.columns = [str(col).strip() for col in df.columns]
        
        date_col = 'Requested deliv. date'
        if date_col in df.columns:
            # Chuyển đổi ngày tháng, giữ nguyên tính chính xác của năm
            df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            # Loại bỏ các dòng không có ngày tháng hoặc không có số lượng
            df = df.dropna(subset=['ds', 'Order qty.(A)'])
            # Đảm bảo Order qty là số thực
            df['Order qty.(A)'] = pd.to_numeric(df['Order qty.(A)'], errors='coerce').fillna(0)
            return df
        else:
            st.error(f"Không tìm thấy cột '{date_col}'! Hãy kiểm tra lại file Excel.")
            return None
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

def calculate_metrics_all_years(cust_df, prod_name):
    p_df = cust_df[cust_df['Material name'] == prod_name].copy()
    if p_df.empty: return 0.0, 0.0, 0.0
    
    # Gom nhóm chính xác theo tháng, không bỏ sót bất kỳ năm nào (2023, 2024, 2025, 2026)
    monthly_all = p_df.groupby(p_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
    monthly_all['ds_ts'] = monthly_all['ds'].dt.to_timestamp()
    
    # 1. Avg Growth (Dựa trên toàn bộ chuỗi lịch sử từ 2023)
    monthly_all['Pct_Change'] = monthly_all['Order qty.(A)'].pct_change() * 100
    avg_growth = monthly_all['Pct_Change'].mean()
    
    # 2. Số thực tế 2026
    df_2026 = monthly_all[monthly_all['ds_ts'].dt.year == 2026]
    act_2026_sum = df_2026['Order qty.(A)'].sum()
    run_rate = df_2026['Order qty.(A)'].mean() if not df_2026.empty else 0.0
    
    # 3. YoY Growth (So sánh thực tế 2026 hiện tại vs cùng kỳ 2025)
    months_in_26 = df_2026['ds_ts'].dt.month.tolist()
    act_2025_same_period = monthly_all[(monthly_all['ds_ts'].dt.year == 2025) & 
                                       (monthly_all['ds_ts'].dt.month.isin(months_in_26))]['Order qty.(A)'].sum()
    
    yoy_growth = (act_2026_sum - act_2025_same_period) / act_2025_same_period if act_2025_same_period > 0 else 0.0
    
    return avg_growth, yoy_growth, run_rate

# --- SIDEBAR ---
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        # Tự động tìm cột phù hợp
        cust_col = st.sidebar.selectbox("Customer Col:", [c for c in df.columns if 'Customer' in c] or [df.columns[0]])
        cie_col = st.sidebar.selectbox("CIE/Color Col:", [c for c in df.columns if 'CIE' in c] or [df.columns[1]])
        adj_growth = st.sidebar.slider("Manual Adjustment (%)", -50, 50, 0)
        
        selected_cust = st.sidebar.selectbox("Select Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # Pareto 85% Doanh thu (M USD)
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            rev['Cum_Sum'] = rev['M USD'].cumsum() / rev['M USD'].sum()
            top_prods = rev[rev['Cum_Sum'] <= 0.86]['Material name'].unique().tolist()

            tab1, tab2 = st.tabs(["📊 Performance & AI Audit", "📋 2026 Strategic Plan"])

            with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods if top_prods else cust_df['Material name'].unique())
                avg_g, yoy_g, r_rate = calculate_metrics_all_years(cust_df, selected_prod)
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg Growth (History)", f"{avg_g:.1f}%")
                m2.metric("YoY Growth (26 vs 25)", f"{yoy_g*100:.1f}%")
                m3.metric("Run-rate 2026", f"{r_rate:,.0f}")

                # BIỂU ĐỒ ĐA TẦNG
                p_df_plot = cust_df[cust_df['Material name'] == selected_prod].copy()
                df_monthly = p_df_plot.groupby(p_df_plot['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                df_monthly['ds'] = df_monthly['ds'].dt.to_timestamp()
                df_monthly = df_monthly.rename(columns={'Order qty.(A)': 'y'})

                # Huấn luyện AI trên TOÀN BỘ dữ liệu từ file Excel
                m = Prophet(yearly_seasonality=True, interval_width=0.95)
                m.fit(df_monthly)
                future = m.make_future_dataframe(periods=12, freq='MS')
                fcst = m.predict(future)

                fig = go.Figure()
                # 1. Toàn bộ lịch sử (Excel có năm nào hiện năm đó)
                fig.add_trace(go.Scatter(x=df_monthly['ds'], y=df_monthly['y'], name="Actual History", line=dict(color='blue', width=2)))
                # 2. Dự báo tương lai
                fcst_future = fcst[fcst['ds'] > df_monthly['ds'].max()]
                fig.add_trace(go.Scatter(x=fcst_future['ds'], y=fcst_future['yhat'], name="AI Forecast", line=dict(dash='dash', color='orange')))
                
                fig.update_layout(title=f"Full Timeline Audit: {selected_prod}", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("📋 2026 Strategic Plan")
                # Tạo dải thời gian năm 2026
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []

                for p in top_prods:
                    _, yoy_val, rr_val = calculate_metrics_all_years(cust_df, p)
                    final_factor = 1 + yoy_val + (adj_growth / 100)
                    cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    
                    for c in cies:
                        row = {'Product': p, 'CIE': c, 'YoY': f"{yoy_val*100:.1f}%"}
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            # Lấy thực tế từ file
                            act_val = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & 
                                            (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if act_val > 0:
                                row[m_str] = act_val
                            elif m_date > df_monthly['ds'].max():
                                # Dự báo: Ưu tiên YoY từ 2025, nếu không có thì dùng Run-rate 2026
                                h25 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & 
                                              (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2025)]['Order qty.(A)'].sum()
                                row[m_str] = round(h25 * final_factor, 0) if h25 > 0 else round(rr_val * final_factor, 0)
                            else:
                                row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    st.dataframe(res_df.style.format("{:,.0f}", subset=cols_26), use_container_width=True)
                    st.download_button("📥 Export Plan", res_df.to_csv(index=False).encode('utf-8-sig'), "Plan_2026.csv")
