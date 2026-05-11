import streamlit as st

import pandas as pd

from prophet import Prophet

import plotly.graph_objects as go



# 1. System Config

st.set_page_config(page_title="AI Supply Chain Advisor 2026", layout="wide")



# --- CORE LOGIC FUNCTIONS ---

def get_actual_avg_qty(df, year, quarter, prod, cie_col, cie_val):

    temp = df[(df['Material name'] == prod) & 

              (df[cie_col] == cie_val) & 

              (df['ds'].dt.year == year) & 

              (df['ds'].dt.quarter == quarter)].copy()

    monthly_sum = temp.groupby(temp['ds'].dt.month)['Order qty.(A)'].sum()

    actual_months = monthly_sum[monthly_sum > 0]

    return actual_months.mean() if not actual_months.empty else 0.0



def get_quarterly_growth_logic(cust_df, prod, cie_col, cie_val):

    df_26 = cust_df[cust_df['ds'].dt.year == 2026].copy()

    if df_26.empty: return 0.0

    valid_26 = df_26[df_26['Order qty.(A)'] > 0]

    if valid_26.empty: return 0.0

    latest_q_26 = valid_26['ds'].dt.quarter.max()

    avg_26 = get_actual_avg_qty(cust_df, 2026, latest_q_26, prod, cie_col, cie_val)

    avg_25 = get_actual_avg_qty(cust_df, 2025, latest_q_26, prod, cie_col, cie_val)

    return min(max((avg_26 / avg_25) - 1, -0.5), 0.5) if avg_25 > 0 else 0.0



def process_data(uploaded_file):

    try:

        df = pd.read_excel(uploaded_file)

        df.columns = [str(col).strip() for col in df.columns]

        if 'Requested deliv. date' in df.columns:

            df['ds'] = pd.to_datetime(df['Requested deliv. date'], errors='coerce')

            df['Order qty.(A)'] = pd.to_numeric(df['Order qty.(A)'], errors='coerce').fillna(0)

            return df.dropna(subset=['ds'])

        return None

    except: return None



# --- MAIN UI ---

st.sidebar.header("📁 Data Management")

uploaded_file = st.sidebar.file_uploader("Upload AICheck.xlsx", type=['xlsx'])



if uploaded_file:

    df = process_data(uploaded_file)

    if df is not None:

        all_cols = df.columns.tolist()

        cust_col = st.sidebar.selectbox("Customer Column:", all_cols, index=0)

        cie_col = st.sidebar.selectbox("CIE / Color Code Column:", all_cols, index=1)

        adj_growth = st.sidebar.slider("Manual Growth Adjustment (%)", -50, 50, 0)

        selected_cust = st.sidebar.selectbox("Select Target Customer:", ["-- Select --"] + sorted(df[cust_col].unique().tolist()))



        if selected_cust != "-- Select --":

            cust_df = df[df[cust_col] == selected_cust].copy()

            rev = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()

            rev['Cum%'] = rev['M USD'].cumsum() / rev['M USD'].sum()

            top_prods = rev[rev['Cum%'] <= 0.86]['Material name'].unique()



            # --- TẠO TAB TẠI ĐÂY ---

            tab1, tab2, tab3 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan", "🧪 Model Testing"])

            auto_adjustments = {}



with tab1:
                st.subheader("🎯 Pareto 85% & Variance Audit")
                selected_prod = st.selectbox("Product Audit:", top_prods)
                
                # Logic dự báo Prophet
                p_plot = cust_df[cust_df['Material name'] == selected_prod].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                p_plot['ds'] = p_plot['ds'].dt.to_timestamp()
                p_plot = p_plot.rename(columns={'Order qty.(A)': 'y'})
                
                if len(p_plot) > 2:
                    model = Prophet(yearly_seasonality=True).fit(p_plot)
                    fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                    fcst_26 = fcst[fcst['ds'].dt.year == 2026].copy()
                    v_df = pd.merge(p_plot[p_plot['ds'].dt.year == 2026], fcst_26[['ds', 'yhat']], on='ds', how='inner')
                    
                    if not v_df.empty:
                        # 1. Tính toán Average Variance (Chỉ số quan trọng nhất)
                        avg_v = ((v_df['y'] - v_df['yhat']) / v_df['yhat']).mean()
                        auto_adjustments[selected_prod] = avg_v
                        
                        # 2. Vẽ biểu đồ đối soát
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=p_plot['ds'], y=p_plot['y'], name="Actual Order", line=dict(color='#1f77b4')))
                        # Đường dự báo có Offset để sếp thấy sự điều chỉnh của AI
                        fcst_26['yhat_adj'] = fcst_26['yhat'] * (1 + avg_v)
                        fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat_adj'], name="AI FCST (Offset applied)", line=dict(dash='dash', color='#ff7f0e')))
                        st.plotly_chart(fig, use_container_width=True)

                        # 3. KHÔI PHỤC BẢNG VARIANCE CHI TIẾT CÓ DÒNG AVERAGE
                        st.subheader("🔢 Actual vs AI Variance Details")
                        
                        v_table = v_df.copy()
                        v_table['Variance %'] = ((v_table['y'] - v_table['yhat']) / v_table['yhat']) * 100
                        v_table = v_table.rename(columns={'ds': 'Month', 'y': 'Actual Qty', 'yhat': 'AI FCST Qty'})
                        
                        # Tạo dòng trung bình cộng (Average)
                        avg_row = pd.DataFrame({
                            'Month': ["AVERAGE"], 
                            'Actual Qty': [v_table['Actual Qty'].mean()], 
                            'AI FCST Qty': [v_table['AI FCST Qty'].mean()], 
                            'Variance %': [v_table['Variance %'].mean()]
                        })
                        
                        full_v_df = pd.concat([v_table, avg_row], ignore_index=True)
                        
                        # Hiển thị bảng và tô màu dòng Average để sếp dễ quan sát
                        st.dataframe(
                            full_v_df.style.format({
                                'Month': lambda x: x.strftime('%m/%Y') if hasattr(x, 'strftime') else x,
                                'Actual Qty': '{:,.0f}', 
                                'AI FCST Qty': '{:,.0f}', 
                                'Variance %': '{:+.1f}%'
                            }).apply(lambda x: ['background-color: #e6f3ff; font-weight: bold; color: #000000']*len(x) 
                                     if x['Month'] == "AVERAGE" else ['']*len(x), axis=1), 
                            use_container_width=True
                        )

                        # Cảnh báo thông minh nếu sai số vượt ngưỡng
                        if abs(avg_v) > 0.20:
                            st.warning(f"⚠️ **Note for Manager:** Mã này lệch **{avg_v*100:+.1f}%** so với thực tế. Hệ thống đã tự động bù sai số (Offset) vào kế hoạch Strategic Plan phía sau.")
