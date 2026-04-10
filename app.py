import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Cấu hình hệ thống
st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")

# --- CORE LOGIC FUNCTIONS ---

def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val):
    """Tính trung bình thực tế của các tháng CÓ SỐ LIỆU (>0) trong một Quý."""
    temp = df[(df['Material name'] == prod) & 
              (df[cie_col] == cie_val) & 
              (df['ds'].dt.year == year) & 
              (df['ds'].dt.quarter == quarter)].copy()
    
    # Gom nhóm theo tháng để lấy tổng từng tháng
    monthly_sum = temp.groupby(temp['ds'].dt.month)['Order qty.(A)'].sum()
    actual_months = monthly_sum[monthly_sum > 0]
    
    if actual_months.empty:
        return 0.0
    return actual_months.mean() # Chia đúng cho số tháng thực tế (1, 2 hoặc 3)

def get_quarterly_growth_logic(cust_df, prod, cie_col, cie_val):
    """Tính tăng trưởng dựa trên so sánh TB Quý gần nhất 2026 vs cùng kỳ 2025."""
    df_26 = cust_df[cust_df['ds'].dt.year == 2026].copy()
    if df_26.empty: return 0.0
    
    valid_26 = df_26[df_26['Order qty.(A)'] > 0]
    if valid_26.empty: return 0.0
    
    # Tìm quý gần nhất của 2026 đã có dữ liệu
    latest_q_26 = valid_26['ds'].dt.quarter.max()
    
    avg_26 = get_actual_avg_qty(cust_df, 2026, latest_q_26, prod, cie_col, cie_val)
    avg_25 = get_actual_avg_qty(cust_df, 2025, latest_q_26, prod, cie_col, cie_val)
    
    if avg_25 > 0:
        growth = (avg_26 / avg_25) - 1
        return min(max(growth, -0.5), 0.5) # Giới hạn biên độ +/- 50%
    return 0.0

def process_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [str(col).strip() for col in df.columns]
        if 'Requested deliv. date' in df.columns and 'Order qty.(A)' in df.columns:
            df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')
            df['Order qty.(A)'] = pd.to_numeric(df['Order qty.(A)'], errors='coerce').fillna(0)
            return df.dropna(subset=['ds'])
        return None
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

# --- GIAO DIỆN CHÍNH ---
st.sidebar.header("📁 Quản lý dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file AICheck.xlsx", type=['xlsx'])

if uploaded_file:
    df = process_data(uploaded_file)
    if df is not None:
        all_cols = df.columns.tolist()
        
        # Lựa chọn cột thông minh
        cust_suggest = next((c for c in all_cols if 'customer' in c.lower()), all_cols[0])
        cust_col = st.sidebar.selectbox("Cột Khách hàng:", all_cols, index=all_cols.index(cust_suggest))
        
        cie_suggest = all_cols[1] if len(all_cols) > 1 else all_cols[0]
        cie_col = st.sidebar.selectbox("Cột CIE / Color Code:", all_cols, index=all_cols.index(cie_suggest))
        
        adj_growth = st.sidebar.slider("Điều chỉnh tăng trưởng (%)", -50, 50, 0)
        
        selected_cust = st.sidebar.selectbox("Chọn khách hàng mục tiêu:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))

        if selected_cust != "-- Select --":
            cust_df = df[df[cust_col] == selected_cust].copy()
            
            # Pareto 85% dựa trên doanh số (M USD)
            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
            top_prods = rev[rev['M USD'].cumsum() / rev['M USD'].sum() <= 0.86]['Material name'].unique()[:20]

            # Khởi tạo Tabs
            tab1, tab2 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan"])

            # --- TAB 1: PERFORMANCE AUDIT ---
            with tab1:
                selected_prod = st.selectbox("Chọn sản phẩm kiểm tra:", top_prods)
                
                # Lấy CIE mẫu để tính toán nhanh
                sample_cies = cust_df[cust_df['Material name'] == selected_prod][cie_col].unique()
                s_cie = sample_cies[0]
                
                q_growth = get_quarterly_growth_logic(cust_df, selected_prod, cie_col, s_cie)
                f_growth = q_growth + (adj_growth/100)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Tăng trưởng Quý", f"{q_growth*100:.1f}%")
                m2.metric("Tăng trưởng sau điều chỉnh", f"{f_growth*100:.1f}%")
                m3.metric("Số lượng mã CIE", len(sample_cies))

                # Biểu đồ xu hướng
                p_plot = cust_df[cust_df['Material name'] == selected_prod].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                p_plot['ds'] = p_plot['ds'].dt.to_timestamp()
                p_plot = p_plot.rename(columns={'Order qty.(A)': 'y'})
                
                if len(p_plot) > 2:
                    model = Prophet(yearly_seasonality=True).fit(p_plot)
                    fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=p_plot['ds'], y=p_plot['y'], name="Thực tế", line=dict(color='blue')))
                    fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                    fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Dự báo", line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bảng Variance với dòng Average Variance
                    st.subheader("🔢 So sánh Thực tế vs AI Dự báo (Variance)")
                    act_26 = p_plot[p_plot['ds'].dt.year == 2026]
                    v_df = pd.merge(act_26, fcst_26[['ds', 'yhat']], on='ds', how='inner')
                    
                    if not v_df.empty:
                        v_df['Variance %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                        avg_v = v_df['Variance %'].mean()
                        
                        # Thêm dòng trung bình
                        avg_row = pd.DataFrame({'ds': [pd.NA], 'y': [v_df['y'].mean()], 'yhat': [v_df['yhat'].mean()], 'Variance %': [avg_v]})
                        v_display = pd.concat([v_df, avg_row], ignore_index=True)
                        
                        st.dataframe(v_display.style.format({
                            'ds': lambda x: x.strftime('%m/%Y') if pd.notna(x) else "AVERAGE VAR",
                            'y': '{:,.0f}', 'yhat': '{:,.0f}', 'Variance %': '{:+.1f}%'
                        }).apply(lambda x: ['background: #f0f2f6; font-weight: bold'] * len(x) if x.name == len(v_display)-1 else [''] * len(x), axis=1), use_container_width=True)
                        
                        st.markdown(f"### 💡 Nhận xét từ AI cho {selected_prod}")
                        if avg_v > 15: st.warning(f"Thực tế cao hơn dự báo {avg_v:.1f}%. Cần kiểm tra nguy cơ thiếu hụt vật tư.")
                        elif avg_v < -15: st.error(f"Thực tế thấp hơn dự báo {abs(avg_v):.1f}%. Cần rà soát hàng tồn kho.")
                        else: st.success(f"Dự báo ổn định. Độ lệch trung bình chỉ {avg_v:.1f}%.")

            # --- TAB 2: 2026 STRATEGIC PLAN (CHI TIẾT) ---
            with tab2:
                st.subheader(f"📋 Kế hoạch 2026 cho {selected_cust}")
                months_26 = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
                cols_26 = [m.strftime('%m/%Y') for m in months_26]
                pivot_list = []
                last_act_date = df[df['Order qty.(A)'] > 0]['ds'].max()

                # Vòng lặp tính toán cho từng mã hàng và từng CIE
                for p in top_prods:
                    product_cies = cust_df[cust_df['Material name'] == p][cie_col].unique()
                    for c in product_cies:
                        q_growth = get_quarterly_growth_logic(cust_df, p, cie_col, c)
                        f_growth = q_growth + (adj_growth/100)
                        
                        row = {'Sản phẩm': p, 'CIE': str(c), 'Tăng trưởng': f"{f_growth*100:.1f}%"}
                        
                        for m_date in months_26:
                            m_idx, m_str = m_date.month, m_date.strftime('%m/%Y')
                            q_idx = (m_idx - 1) // 3 + 1
                            
                            # Lấy số thực tế 2026 nếu đã có
                            act_26 = cust_df[(cust_df['Material name']==p) & (cust_df[cie_col]==c) & (cust_df['ds'].dt.month==m_idx) & (cust_df['ds'].dt.year==2026)]['Order qty.(A)'].sum()
                            
                            if act_26 > 0:
                                row[m_str] = act_26
                            elif m_date > last_act_date:
                                # Tính dự báo: Trung bình cùng quý 2025 x (1 + Tăng trưởng)
                                avg_25 = get_actual_avg_qty(cust_df, 2025, q_idx, p, cie_col, c)
                                row[m_str] = round(avg_25 * (1 + f_growth), 0)
                            else:
                                row[m_str] = 0
                        pivot_list.append(row)
                
                if pivot_list:
                    res_df = pd.DataFrame(pivot_list)
                    
                    # Dòng Tổng cộng cuối bảng
                    total_row = {'Sản phẩm': 'TỔNG CỘNG (GRAND TOTAL)', 'CIE': '', 'Tăng trưởng': ''}
                    for col in cols_26: total_row[col] = res_df[col].sum()
                    res_df = pd.concat([res_df, pd.DataFrame([total_row])], ignore_index=True)
                    
                    # Hiển thị bảng kết quả
                    st.dataframe(res_df.style.apply(lambda x: ['background: #e6f3ff; font-weight: bold' if x['Sản phẩm']=='TỔNG CỘNG (GRAND TOTAL)' else '' for _ in x], axis=1).format("{:,.0f}", subset=cols_26), use_container_width=True)
                    
                    # Nút tải file
                    st.download_button("📥 Tải kế hoạch (CSV)", data=res_df.to_csv(index=False).encode('utf-8-sig'), file_name=f"Strategic_Plan_2026_{selected_cust}.csv")
