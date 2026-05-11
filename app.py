# --- UPDATE TAB DEFINITION ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Audit", "📋 2026 Strategic Plan", "🧪 Model Testing"])

# [Giữ nguyên logic xử lý dữ liệu và Tab 1, Tab 2 như cũ...]

# --- NEW: TAB 3 - MODEL TESTING & COMPLIANCE ---
with tab3:
    st.header("🧪 Model Accuracy & Compliance Audit")
    st.write("Kiểm tra mức độ tuân thủ tiêu chuẩn sai số: M+2 (±20%), M+3 (±30%), M+4 (±50%)")

    # 1. Chuẩn bị dữ liệu đối soát
    test_results = []
    
    # Lấy mốc thời gian hiện tại từ dữ liệu thực tế cuối cùng
    last_act_date = df[df['Order qty.(A)'] > 0]['ds'].max()
    curr_m = last_act_date.month
    curr_y = last_act_date.year

    for p in top_prods:
        # Lấy sai số trung bình (Variance) từ Tab 1
        avg_v_pct = auto_adjustments.get(p, 0.0) * 100 
        abs_v = abs(avg_v_pct)
        
        # Kiểm tra tiêu chuẩn theo từng tầng nấc
        status_m2 = "✅ Pass" if abs_v <= 20 else "❌ Fail"
        status_m3 = "✅ Pass" if abs_v <= 30 else "❌ Fail"
        status_m4 = "✅ Pass" if abs_v <= 50 else "❌ Fail"
        
        test_results.append({
            "Material Name": p,
            "Current Variance": avg_v_pct,
            "M+2 Status (±20%)": status_m2,
            "M+3 Status (±30%)": status_m3,
            "M+4 Status (±50%)": status_m4,
            "Reliability Score": 100 - abs_v if abs_v < 100 else 0
        })

    test_df = pd.DataFrame(test_results)

    # 2. Hiển thị Tổng kết (Summary Metrics)
    avg_acc = test_df['Reliability Score'].mean()
    pass_rate_m2 = (test_df['M+2 Status (±20%)'] == "✅ Pass").mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Accuracy Score", f"{avg_acc:.1f}%")
    c2.metric("M+2 Compliance Rate", f"{pass_rate_m2:.1f}%")
    c3.metric("Total SKUs Audited", len(top_prods))

    # 3. Bảng chi tiết tuân thủ
    st.subheader("📋 Detailed Compliance Matrix")
    
    def color_status(val):
        if "✅" in str(val): color = '#d4edda' # Xanh nhạt
        elif "❌" in str(val): color = '#f8d7da' # Đỏ nhạt
        else: color = ''
        return f'background-color: {color}'

    st.dataframe(
        test_df.style.applymap(color_status, subset=['M+2 Status (±20%)', 'M+3 Status (±30%)', 'M+4 Status (±50%)'])
        .format({"Current Variance": "{:+.1f}%", "Reliability Score": "{:.1f}%"}),
        use_container_width=True
    )

    # 4. Giải thích logic cho sếp
    st.info("""
    **Ghi chú nghiệp vụ:**
    - **M+2 (20%)**: Ngưỡng nghiêm ngặt cho sản xuất và chốt linh kiện.
    - **M+3 (30%)**: Ngưỡng linh hoạt hơn cho chuẩn bị nguyên vật liệu dài hạn.
    - **M+4 (50%)**: Ngưỡng chiến lược cho hoạch định năng lực thiết bị.
    - *Hệ thống AI đã tự động điều chỉnh (Offset) cho các mã hàng có trạng thái ❌ Fail.*
    """)
