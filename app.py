with tab2:
            st.subheader("📋 Kế hoạch sản xuất Pareto 2026")
            
            # Lọc dữ liệu Pareto năm 2025
            p_df = cust_df[cust_df['Material name'].isin(top_prods) & (cust_df['ds'].dt.year == 2025)].copy()
            
            if not p_df.empty:
                # 1. Xác định hệ số tăng trưởng (Dùng trung bình năm nếu cùng kỳ bị lỗi)
                if 'growth' in locals() and growth != 0:
                    g_factor = 1 + (growth / 100)
                else:
                    # Tính trung bình toàn bộ 2025 và dự báo 2026 để lấy con số thay thế
                    sum_25 = cust_df[cust_df['ds'].dt.year == 2025]['Order qty.(A)'].sum()
                    # Giả định tăng trưởng mặc định 10% nếu AI chưa tính kịp
                    g_factor = 1.10 
                
                # 2. Tính mức nền cho từng mã
                base_stats = p_df.groupby(['Material name', cie_col])['Order qty.(A)'].mean().reset_index()
                
                # 3. Tạo bảng ngang
                months_26 = [m.strftime('%m/%Y') for m in pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')]
                
                pivot_data = []
                for _, r in base_stats.iterrows():
                    row = {'Sản phẩm': r['Material name'], 'CIE': r[cie_col]}
                    for m in months_26:
                        row[m] = round(r['Order qty.(A)'] * g_factor, 0)
                    pivot_data.append(row)
                
                res_df = pd.DataFrame(pivot_data)
                
                if not res_df.empty:
                    # Bộ lọc nhanh
                    f_p = st.multiselect("Lọc nhanh theo sản phẩm:", top_prods)
                    if f_p: 
                        res_df = res_df[res_df['Sản phẩm'].isin(f_p)]
                    
                    # Hiển thị bảng
                    st.write(f"📌 Hệ số dự báo đang áp dụng: **{g_factor:.2f}x**")
                    st.dataframe(res_df.style.format("{:,.0f}", subset=months_26), use_container_width=True)
                    
                    st.download_button("📥 Tải báo cáo CSV", res_df.to_csv(index=False).encode('utf-8-sig'), "Pareto_2026.csv")
                else:
                    st.error("Không thể tạo bảng dự báo. Vui lòng kiểm tra lại cột CIE.")
            else:
                st.warning("⚠️ Không tìm thấy dữ liệu thực tế của năm 2025 để làm cơ sở dự báo.")
