with tab1:
                # Pareto List Expander (Giữ nguyên)
                with st.expander("🎯 85% Revenue Contribution (Pareto List)"):
                    st.dataframe(rev[rev['Cum%'] <= 0.86][['Material name', 'M USD', 'Cum%']].style.format({'M USD': '${:,.2f}', 'Cum%': '{:.1%}'}), use_container_width=True)

                selected_prod = st.selectbox("Product Audit:", top_prods)
                
                # Logic xử lý dữ liệu cho biểu đồ
                p_plot = cust_df[cust_df['Material name'] == selected_prod].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                p_plot['ds'] = p_plot['ds'].dt.to_timestamp()
                p_plot = p_plot.rename(columns={'Order qty.(A)': 'y'})
                
                if len(p_plot) > 2:
                    # Chạy Prophet model
                    model = Prophet(yearly_seasonality=True).fit(p_plot)
                    fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                    
                    # Tính toán Variance thực tế của năm 2026
                    fcst_26 = fcst[fcst['ds'].dt.year == 2026].copy()
                    v_df = pd.merge(p_plot[p_plot['ds'].dt.year == 2026], fcst_26[['ds', 'yhat']], on='ds', how='inner')
                    
                    if not v_df.empty:
                        # Tính % Variance trung bình để làm Offset
                        avg_v = ((v_df['y'] - v_df['yhat']) / v_df['yhat']).mean()
                        auto_adjustments[selected_prod] = avg_v
                        
                        # Vẽ biểu đồ
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=p_plot['ds'], y=p_plot['y'], name="Actual", line=dict(color='blue')))
                        
                        # Đường dự báo có áp dụng Offset (đường cam nét đứt)
                        fcst_26['yhat_adj'] = fcst_26['yhat'] * (1 + avg_v)
                        fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat_adj'], name="AI FCST (Adjusted)", line=dict(dash='dash', color='orange')))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if abs(avg_v) > 0.20:
                            st.info(f"💡 AI Auto-Adjustment active: **{avg_v*100:+.1f}%** (due to Variance > 20%)")

                        # --- KHÔI PHỤC BẢNG VARIANCE CHI TIẾT TẠI ĐÂY ---
                        st.subheader("🔢 Actual vs AI Variance Details")
                        v_df['Variance %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                        v_df = v_df.rename(columns={'ds': 'Month Code', 'y': 'Actual Order Quantity', 'yhat': 'AI FCST Quantity'})
                        
                        # Thêm dòng Average
                        avg_row = pd.DataFrame({
                            'Month Code': ["AVERAGE"], 
                            'Actual Order Quantity': [v_df['Actual Order Quantity'].mean()], 
                            'AI FCST Quantity': [v_df['AI FCST Quantity'].mean()], 
                            'Variance %': [v_df['Variance %'].mean()]
                        })
                        
                        full_v_df = pd.concat([v_df, avg_row], ignore_index=True)
                        
                        st.dataframe(full_v_df.style.format({
                            'Month Code': lambda x: x.strftime('%m/%Y') if hasattr(x, 'strftime') else x,
                            'Actual Order Quantity': '{:,.0f}', 
                            'AI FCST Quantity': '{:,.0f}', 
                            'Variance %': '{:+.1f}%'
                        }).apply(lambda x: ['background: #f0f2f6; font-weight: bold']*len(x) if x['Month Code'] == "AVERAGE" else ['']*len(x), axis=1), use_container_width=True)
