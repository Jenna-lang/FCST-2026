# --- PHẦN XỬ LÝ DỮ LIỆU TỔNG HỢP ---
if selected_cust:
    cust_df = df[df[cust_col].astype(str) == selected_cust].copy()
    
    # Tính toán danh sách Pareto (80/20) cho khách hàng này
    sales = cust_df.groupby('Material name')['M USD'].sum().sort_values(ascending=False).reset_index()
    sales['Cum_Pct'] = sales['M USD'].cumsum() / sales['M USD'].sum()
    top_prods = sales[sales['Cum_Pct'] <= 0.81]['Material name'].unique()
    
    # Tạo 2 danh sách riêng: 1 để vẽ biểu đồ (1 sản phẩm) và 1 để làm bảng tổng hợp (Tất cả sản phẩm)
    selected_prod = st.selectbox("2. Chọn sản phẩm để xem biểu đồ:", top_prods)
    
    # Khởi tạo danh sách lưu trữ kết quả cho BẢNG TỔNG HỢP TOÀN BỘ PARETO
    all_pareto_results = []
    chart_results = []
    ai_insights = []

    # CHẠY DỰ BÁO CHO TẤT CẢ SẢN PHẨM TRONG LIST PARETO
    with st.spinner('Đang tính toán dự báo cho toàn bộ danh mục Pareto...'):
        for prod in top_prods:
            temp_prod_df = cust_df[cust_df['Material name'] == prod].copy()
            cies = temp_prod_df[cie_col].unique()
            
            for cie in cies:
                cie_df = temp_prod_df[temp_prod_df[cie_col].astype(str) == cie].copy()
                cie_df['m'] = cie_df['ds'].dt.to_period('M').dt.to_timestamp()
                actual = cie_df.groupby('m')['Order qty.(A)'].sum().reset_index().rename(columns={'m':'ds', 'Order qty.(A)':'y'})
                
                if len(actual) >= 2:
                    m = Prophet(yearly_seasonality=True).fit(actual)
                    future = m.make_future_dataframe(periods=12, freq='MS')
                    fcst = m.predict(future)
                    fcst['yhat'] = fcst['yhat'].clip(lower=0)
                    
                    # Lấy dữ liệu 2026 cho bảng tổng hợp
                    f2026 = fcst[fcst['ds'].dt.year == 2026].copy()
                    f2026['Month'] = f2026['ds'].dt.strftime('%m/%Y')
                    f2026['CIE'] = cie
                    f2026['Product'] = prod
                    all_pareto_results.append(f2026[['Month', 'Product', 'CIE', 'yhat']])
                    
                    # Nếu là sản phẩm đang được chọn ở Selectbox thì lưu vào Chart
                    if prod == selected_prod:
                        chart_results.append({'cie': cie, 'actual': actual, 'fcst': fcst})
                        # Thêm Insight cho sản phẩm đang xem biểu đồ
                        avg_past = actual['y'].tail(12).mean()
                        max_f = f2026['yhat'].max()
                        growth = ((max_f - avg_past) / avg_past * 100) if avg_past > 0 else 0
                        ai_insights.append(f"**{prod} - {cie}**: {'Tăng' if growth > 5 else 'Giảm' if growth < -5 else 'Ổn định'} ({growth:.1f}%).")

    # --- HIỂN THỊ TABS ---
    tab1, tab2 = st.tabs(["📊 Analytics Chart", "📋 All Pareto FCST Details"])

    with tab1:
        st.subheader(f"Analysis: {selected_prod}")
        fig = go.Figure()
        for res in chart_results:
            fig.add_trace(go.Scatter(x=res['actual']['ds'], y=res['actual']['y'], name=f"Actual {res['cie']}"))
            f_only = res['fcst'][res['fcst']['ds'] > res['actual']['ds'].max()]
            fig.add_trace(go.Scatter(x=f_only['ds'], y=f_only['yhat'], line=dict(dash='dash'), name=f"FCST {res['cie']}"))
        st.plotly_chart(fig, use_container_width=True)
        for insight in ai_insights: st.info(insight)

    with tab2:
        if all_pareto_results:
            st.subheader(f"Full Pareto Forecast List for {selected_cust}")
            final_df = pd.concat(all_pareto_results)
            # Hiển thị bảng tổng hợp tất cả mã hàng
            st.dataframe(final_df.rename(columns={'yhat': 'Qty (Pcs)'}).style.format("{:,.0f}", subset=['Qty (Pcs)']), use_container_width=True)
            
            csv = final_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Tải toàn bộ danh sách Pareto (CSV)", data=csv, file_name=f"Full_Pareto_{selected_cust}.csv")
