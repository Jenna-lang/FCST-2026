with tab2:
            st.subheader("📋 Kế hoạch sản xuất Pareto 2026 (Đồng bộ mùa vụ)")
            
            # 1. Lấy dữ liệu thực tế chi tiết theo từng tháng của năm 2025
            df_25_raw = cust_df[(cust_df['Material name'].isin(top_prods)) & (cust_df['ds'].dt.year == 2025)]
            
            # Tính tổng theo Material, CIE và Month của năm 2025
            # Dùng tổng thực tế từng tháng để giữ đúng biểu đồ hình sin của ngành LED
            act_25_map = df_25_raw.groupby(['Material name', cie_col, df_25_raw['ds'].dt.month])['Order qty.(A)'].sum().to_dict()
            
            # 2. Lấy thực tế 2026 đã có (T1-T3)
            df_26_act = cust_df[(cust_df['Material name'].isin(top_prods)) & (cust_df['ds'].dt.year == 2026)]
            act_26_map = df_26_act.groupby(['Material name', cie_col, df_26_act['ds'].dt.month])['Order qty.(A)'].sum().to_dict()

            # Hệ số tăng trưởng từ Tab 1
            g_factor = 1 + (growth_final / 100)
            
            # Danh sách 12 tháng năm 2026
            months_range = pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
            columns_2026 = [m.strftime('%m/%Y') for m in months_range]
            
            # Lấy danh sách Product-CIE duy nhất từ dữ liệu 2025
            unique_items = df_25_raw[['Material name', cie_col]].drop_duplicates()
            
            pivot_list = []
            for _, r in unique_items.iterrows():
                row = {'Sản phẩm': r['Material name'], 'CIE': r[cie_col]}
                p_name, c_name = r['Material name'], r[cie_col]
                
                for m_date in months_range:
                    m_idx = m_date.month
                    m_str = m_date.strftime('%m/%Y')
                    
                    # ƯU TIÊN 1: Nếu tháng đó ĐÃ CÓ thực tế 2026 (T1, T2, T3)
                    if (p_name, c_name, m_idx) in act_26_map:
                        row[m_str] = act_26_map[(p_name, c_name, m_idx)]
                    
                    # ƯU TIÊN 2: Dự báo cho tương lai (T4-T12) dựa trên CÙNG THÁNG NĂM NGOÁI
                    elif (p_name, c_name, m_idx) in act_25_map:
                        val_25 = act_25_map[(p_name, c_name, m_idx)]
                        # Nhân đúng hệ số tăng trưởng vào số lượng tháng tương ứng
                        row[m_str] = round(val_25 * g_factor, 0)
                    
                    # ƯU TIÊN 3: Nếu tháng đó năm ngoái không có số, dùng trung bình tháng 2025 của mã đó
                    else:
                        avg_25 = df_25_raw[(df_25_raw['Material name']==p_name) & (df_25_raw[cie_col]==c_name)]['Order qty.(A)'].mean()
                        row[m_str] = round(avg_25 * g_factor, 0) if pd.notna(avg_25) else 0
                        
                pivot_list.append(row)
            
            res_df = pd.DataFrame(pivot_list)
            
            if not res_df.empty:
                st.success(f"✅ Đã đồng bộ: Dự báo T4-T12 = (Thực tế cùng tháng 2025) x {g_factor:.2f}")
                
                # Bộ lọc nhanh
                f_p = st.multiselect("Lọc nhanh theo sản phẩm:", top_prods)
                if f_p: res_df = res_df[res_df['Sản phẩm'].isin(f_p)]
                
                # Hiển thị bảng ngang
                st.dataframe(res_df.style.format("{:,.0f}", subset=columns_2026), use_container_width=True)
                
                csv = res_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 Tải kế hoạch đồng bộ 2026", csv, "Supply_Plan_Final.csv")
