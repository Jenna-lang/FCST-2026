import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Dự báo Pareto 2026", layout="wide")
st.title("🚀 Hệ Thống Phân Tích Cung Ứng Pareto 2026")

@st.cache_data
def load_data():
    df = pd.read_excel('AICheck.xlsx')
    df.columns = [str(col).strip() for col in df.columns]
    df['ds'] = pd.to_datetime(df['Requested deliv. date'])
    return df

try:
    df = load_data()
    
    # --- TỰ ĐỘNG TÌM CỘT TÊN KHÁCH HÀNG ---
    # Ưu tiên các cột có chữ 'Cust' hoặc 'Name' và KHÔNG chứa chữ 'Code'
    name_cols = [c for c in df.columns if ('cust' in c.lower() or 'name' in c.lower()) and 'code' not in c.lower()]
    cust_col = name_cols[0] if name_cols else df.columns[0]

    # Lọc Pareto (85%)
    summary = df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    summary['Cum_Pct'] = summary['M USD'].cumsum() / summary['M USD'].sum()
    pareto_list = summary[summary['Cum_Pct'] <= 0.85]['Material name'].unique()

    st.sidebar.header("🔍 Bộ Lọc")
    selected_prod = st.sidebar.selectbox("Chọn mã linh kiện:", pareto_list)

    if selected_prod:
        prod_df = df[df['Material name'] == selected_prod].copy()
        t1, t2 = st.tabs(["📊 Tổng quan", "🎯 Dự báo & Lời khuyên AI"])

        with t1:
            st.subheader(f"Tỷ trọng khách hàng: {selected_prod}")
            ca, cb = st.columns(2)
            with ca:
                fig_p = px.pie(prod_df.groupby(cust_col)['M USD'].sum().reset_index(), values='M USD', names=cust_col, hole=0.4)
                st.plotly_chart(fig_p, use_container_width=True)
            with cb:
                prod_df['Month'] = prod_df['ds'].dt.to_period('M').dt.to_timestamp()
                fig_b = px.bar(prod_df.groupby(['Month', cust_col])['Order qty.(A)'].sum().reset_index(), x='Month', y='Order qty.(A)', color=cust_col)
                st.plotly_chart(fig_b, use_container_width=True)

        with t2:
            customers = sorted([str(x) for x in prod_df[cust_col].unique()])
            selected_cust = st.selectbox("Chọn Tên Khách Hàng:", customers)
            
            if selected_cust:
                c_data = prod_df[prod_df[cust_col] == selected_cust].copy()
                c_data['ds_m'] = c_data['ds'].dt.to_period('M').dt.to_timestamp()
                p_df = c_data.groupby('ds_m')['Order qty.(A)'].sum().reset_index().rename(columns={'ds_m':'ds', 'Order qty.(A)':'y'})

                if len(p_df) >= 2:
                    m = Prophet(yearly_seasonality=True).fit(p_df)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    
                    st.divider()
                    col_l, col_r = st.columns([2, 1])
                    with col_l:
                        st.subheader(f"Dự báo cho {selected_cust}")
                        st.pyplot(m.plot(fcst))
                    with col_r:
                        st.subheader("🤖 AI Advisor")
                        f26 = fcst[fcst['ds'].dt.year == 2026]
                        avg26 = f26['yhat'].mean()
                        st.metric("Dự báo TB 2026", f"{avg26:,.0f} Pcs")
                        
                        if avg26 > p_df['y'].mean() * 1.15:
                            st.error("🚩 Cảnh báo: Nhu cầu tăng mạnh (>15%).")
                        else:
                            st.success("✅ Nhu cầu ổn định.")
                        st.info(f"💡 Dự phòng (Max): {f26['yhat_upper'].max():,.0f} Pcs")

                    st.subheader("📋 Bảng số liệu 2026")
                    res = f26[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    res.columns = ['Tháng', 'Trung bình', 'Min', 'Max']
                    res['Tháng'] = res['Tháng'].dt.strftime('%m/%Y')
                    st.dataframe(res.style.format('{:,.0f}', subset=['Trung bình', 'Min', 'Max']), use_container_width=True)
except Exception as e:
    st.error(f"Lỗi: {e}")
