with tab1:
                selected_prod = st.selectbox("Product Audit:", top_prods)
                
                # Hiển thị trung bình quý để audit
                sample_cies = cust_df[cust_df['Material name'] == selected_prod][cie_col].unique()
                s_cie = sample_cies[0]
                
                q_growth = get_quarterly_growth_logic(cust_df, selected_prod, cie_col, s_cie)
                f_growth = q_growth + (adj_growth/100)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Quarterly Growth", f"{q_growth*100:.1f}%")
                m2.metric("Final Adjusted", f"{f_growth*100:.1f}%")
                m3.metric("CIE Count", len(sample_cies))

                # Chart Prophet
                p_plot = cust_df[cust_df['Material name'] == selected_prod].groupby(cust_df['ds'].dt.to_period('M'))['Order qty.(A)'].sum().reset_index()
                p_plot['ds'] = p_plot['ds'].dt.to_timestamp()
                p_plot = p_plot.rename(columns={'Order qty.(A)': 'y'})
                
                if len(p_plot) > 2:
                    model = Prophet(yearly_seasonality=True).fit(p_plot)
                    fcst = model.predict(model.make_future_dataframe(periods=12, freq='MS'))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=p_plot['ds'], y=p_plot['y'], name="Actual", line=dict(color='blue')))
                    fcst_26 = fcst[fcst['ds'].dt.year == 2026]
                    fig.add_trace(go.Scatter(x=fcst_26['ds'], y=fcst_26['yhat'], name="AI Forecast", line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- PHẦN CẢI TIẾN: VARIANCE TABLE VỚI AVERAGE LINE ---
                    st.subheader("🔢 Actual vs AI Forecast Variance")
                    act_26 = p_plot[p_plot['ds'].dt.year == 2026]
                    v_df = pd.merge(act_26, fcst_26[['ds', 'yhat']], on='ds', how='inner')
                    
                    if not v_df.empty:
                        # Tính Variance từng tháng
                        v_df['Variance %'] = ((v_df['y'] - v_df['yhat']) / v_df['yhat']) * 100
                        
                        # Tạo dòng Average Variance
                        avg_v = v_df['Variance %'].mean()
                        avg_row = pd.DataFrame({
                            'ds': [pd.NA], 
                            'y': [v_df['y'].mean()], 
                            'yhat': [v_df['yhat'].mean()], 
                            'Variance %': [avg_v]
                        })
                        
                        # Kết hợp bảng chính và dòng tổng quát
                        v_display = pd.concat([v_df, avg_row], ignore_index=True)
                        
                        # Hiển thị bảng với format và highlight dòng cuối
                        st.dataframe(
                            v_display.style.format({
                                'ds': lambda x: x.strftime('%m/%Y') if pd.notna(x) else "AVERAGE VAR",
                                'y': '{:,.0f}',
                                'yhat': '{:,.0f}',
                                'Variance %': '{:+.1f}%'
                            }).apply(lambda x: ['background: #f0f2f6; font-weight: bold'] * len(x) if x.name == len(v_display)-1 else [''] * len(x), axis=1),
                            use_container_width=True
                        )
                        
                        # Commentary dựa trên Average Variance thay vì chỉ tháng cuối
                        st.markdown(f"### 💡 AI Strategic Commentary for {selected_prod}")
                        if avg_v > 15: 
                            st.warning(f"Trend: Actual is {avg_v:.1f}% higher than AI. System might be under-forecasting.")
                        elif avg_v < -15: 
                            st.error(f"Trend: Actual is {abs(avg_v):.1f}% lower than AI. High risk of overstocking.")
                        else: 
                            st.success(f"System is healthy. Average variance is only {avg_v:.1f}%.")
