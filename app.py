import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Hệ thống Dự báo Pareto 2026", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Cung Ứng Pareto 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- TỰ ĐỘNG TÌM CỘT TÊN KHÁCH HÀNG (BỎ QUA CỘT MÃ SỐ) ---
    potential_names = [c for c in df.columns if ('cust' in c.lower() or 'name' in c.lower()) 
                      and 'code' not in c.lower() and 'id' not in c.lower()]
    cust_col = potential_names[0] if potential_names else 'End Cust'

    # Phân tích Pareto 85%
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    st.sidebar.header("🔍 Bộ Lọc")
    selected_prod = st.sidebar.selectbox("Chọn mã linh kiện:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        tab1, tab2 = st.tabs(["📊 Tổng quan mã hàng", "🎯 Dự báo & Lời khuyên AI 2026"])

        with tab1:
            st.subheader(f"Phân tích khách hàng cho: {selected_prod}")
            ca, cb = st.columns(2)
            with ca:
                fig_p = px.pie(prod_df.groupby(cust_col)['M USD'].sum().reset_index(), 
                               values='M USD', names=cust_col, hole=0.4, title="Tỷ trọng doanh số")
                st.plotly_chart(fig_p, use_container_width=True)
            with cb:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                fig_b = px.bar(prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index(), 
                               x='Month', y='Order qty.(A)', color=cust_col, title="Lịch sử đặt hàng")
                st.plotly_chart(fig_b, use_container_width=True)

        with tab2:
            customers = sorted([str(x) for x in prod_df[cust_col].unique()])
            selected_cust = st.selectbox("Chọn Tên Khách Hàng cụ thể:", customers)
            
            if selected_cust:
                c_data = prod_df[prod_df[cust_col] == selected_cust].copy()
                c_data['ds_m'] = c_data['ds'].dt.to_period('M').dt.to_timestamp()
                p_df = c_data.groupby('ds_m')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_m':'ds', 'Order qty.(A)':'y'})

                if len(p_df) >= 2:
                    # Chạy mô hình dự báo
                    m = Prophet(yearly_seasonality=True).fit(p_df)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    
                    # Giới hạn không cho dự báo âm
                    for col in ['yhat', 'yhat_lower', 'yhat_upper']: fcst[col] = fcst[col].clip(lower=0)
                    
                    st.divider()
                    col_l, col_r = st.columns([2, 1])
                    
                    with col_l:
                        st.subheader(f"📈 Biểu đồ dự báo cho {selected_cust}")
                        st.pyplot(m.plot(fcst))
                    
                    with col_r:
                        st.subheader("🤖 AI Supply Chain Advisor")
                        f26 = fcst[fcst['ds'].dt.year == 2026]
                        avg26 = f26['yhat'].mean()
                        
                        st.metric("Dự báo TB tháng 2026", f"{avg26:,.0f} Pcs")
                        
                        # LOGIC LỜI KHUYÊN
                        if avg26 > p_df['y'].mean() * 1.15:
                            st.error(f"🚩 **Cảnh báo tăng trưởng:** Nhu cầu tăng mạnh (>15%).")
                            st.write(f"👉 **Hành động:** Liên hệ khách hàng {selected_cust} xác nhận dự án mới.")
                        else:
                            st.success(f"✅ **Nhu cầu ổn định:**")
                            st.write("👉 **Hành động:** Duy trì kế hoạch cung ứng hiện tại.")
                        
                        st.info(f"💡 **Mức dự phòng (Max):** {f26['yhat_upper'].max():,.0f} Pcs")

                    st.subheader("📋 Kế hoạch đặt hàng chi tiết năm 2026")
                    res = f26[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    res.columns = ['Tháng', 'Trung bình', 'Min (An toàn)', 'Max (Kỳ vọng)']
                    res['Tháng'] = res['Tháng'].dt.strftime('%m/%Y')
                    st.dataframe(res.style.format('{:,.0f}', subset=['Trung bình', 'Min (An toàn)', 'Max (Kỳ vọng)']), use_container_width=True)
                else:
                    st.warning(f"⚠️ Dữ liệu của {selected_cust} quá ít để AI đưa ra dự báo.")

except Exception as e:
    st.error(f"Đã xảy ra lỗi hệ thống: {e}")
