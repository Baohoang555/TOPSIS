import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Cấu hình trang
st.set_page_config(
    page_title="TOPSIS - Phân tích Rủi ro Cổ phiếu",
    page_icon="📈",
    layout="wide"
)

# CSS tùy chỉnhC:\python313\python.exe -m streamlit run c:/Users/Administrator/Desktop/topsis_app.py

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4F46E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low {
        background-color: #10B981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #F59E0B;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #EF4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề
st.markdown('<div class="main-header">📈 Phân tích Rủi ro Cổ phiếu - TOPSIS</div>', unsafe_allow_html=True)

# Thông tin thuật toán
with st.expander("ℹ️ Giới thiệu về TOPSIS"):
    st.info("""
    **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) 
    đánh giá cổ phiếu dựa trên độ gần với giải pháp lý tưởng.
    
    **Các bước thực hiện:**
    1. Chuẩn bị ma trận quyết định
    2. Chuẩn hóa ma trận (Vector normalization)
    3. Tính ma trận trọng số chuẩn hóa
    4. Xác định giải pháp lý tưởng dương (A+) và âm (A-)
    5. Tính khoảng cách Euclidean đến A+ và A-
    6. Tính điểm TOPSIS = A-/(A+ + A-)
    
    **Điểm càng cao = Rủi ro càng thấp, Tiềm năng càng tốt**
    """)

# Khởi tạo session state
if 'stocks_data' not in st.session_state:
    st.session_state.stocks_data = pd.DataFrame({
        'Mã CP': ['VNM', 'VIC', 'VHM', 'HPG'],
        'Giá mua': [75000, 42000, 58000, 28000],
        'Giá bán': [82000, 45000, 62000, 30000],
        'KL GD': [1500000, 2800000, 3200000, 5100000],
        'P/E': [18.5, 15.2, 12.8, 9.5],
        'EPS': [4200, 2800, 4500, 2950],
        'ROE (%)': [25.3, 18.7, 22.1, 19.4],
        'Nợ/Vốn': [0.35, 0.52, 0.48, 0.41]
    })

# Sidebar - Cấu hình trọng số
st.sidebar.header("⚖️ Trọng số các tiêu chí")
st.sidebar.markdown("*Tổng trọng số nên bằng 1.0*")

weights = {
    'priceChange': st.sidebar.slider('Biến động giá (%)', 0.0, 1.0, 0.20, 0.05),
    'volume': st.sidebar.slider('Khối lượng GD', 0.0, 1.0, 0.15, 0.05),
    'pe': st.sidebar.slider('P/E', 0.0, 1.0, 0.15, 0.05),
    'eps': st.sidebar.slider('EPS', 0.0, 1.0, 0.20, 0.05),
    'roe': st.sidebar.slider('ROE (%)', 0.0, 1.0, 0.20, 0.05),
    'debt': st.sidebar.slider('Nợ/Vốn', 0.0, 1.0, 0.10, 0.05)
}

total_weight = sum(weights.values())
if abs(total_weight - 1.0) > 0.01:
    st.sidebar.warning(f"⚠️ Tổng trọng số: {total_weight:.2f} (nên = 1.0)")
else:
    st.sidebar.success(f"✅ Tổng trọng số: {total_weight:.2f}")

# Nhập dữ liệu cổ phiếu
st.header("📊 Dữ liệu cổ phiếu")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("➕ Thêm cổ phiếu", use_container_width=True):
        new_row = pd.DataFrame({
            'Mã CP': [f'CP{len(st.session_state.stocks_data)+1}'],
            'Giá mua': [50000],
            'Giá bán': [55000],
            'KL GD': [1000000],
            'P/E': [15.0],
            'EPS': [3000],
            'ROE (%)': [20.0],
            'Nợ/Vốn': [0.4]
        })
        st.session_state.stocks_data = pd.concat([st.session_state.stocks_data, new_row], ignore_index=True)
        st.rerun()

# Hiển thị và chỉnh sửa dữ liệu
edited_df = st.data_editor(
    st.session_state.stocks_data,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Mã CP": st.column_config.TextColumn("Mã CP", width="small"),
        "Giá mua": st.column_config.NumberColumn("Giá mua (VNĐ)", format="%d"),
        "Giá bán": st.column_config.NumberColumn("Giá bán (VNĐ)", format="%d"),
        "KL GD": st.column_config.NumberColumn("KL GD", format="%d"),
        "P/E": st.column_config.NumberColumn("P/E", format="%.2f"),
        "EPS": st.column_config.NumberColumn("EPS", format="%d"),
        "ROE (%)": st.column_config.NumberColumn("ROE (%)", format="%.2f"),
        "Nợ/Vốn": st.column_config.NumberColumn("Nợ/Vốn", format="%.2f")
    }
)
st.session_state.stocks_data = edited_df

# Hàm tính TOPSIS
def calculate_topsis(df, weights):
    # Bước 1: Tính các chỉ số
    df['Biến động giá (%)'] = ((df['Giá bán'] - df['Giá mua']) / df['Giá mua']) * 100
    
    # Bước 2: Chuẩn bị ma trận quyết định
    criteria_cols = ['Biến động giá (%)', 'KL GD', 'P/E', 'EPS', 'ROE (%)', 'Nợ/Vốn']
    decision_matrix = df[criteria_cols].values
    
    # Bước 3: Chuẩn hóa ma trận (Vector normalization)
    normalized = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))
    
    # Bước 4: Tính ma trận trọng số
    weights_array = np.array([weights['priceChange'], weights['volume'], 
                              weights['pe'], weights['eps'], 
                              weights['roe'], weights['debt']])
    weighted_normalized = normalized * weights_array
    
    # Bước 5: Xác định A+ và A-
    # Benefit criteria: Biến động giá, KL GD, EPS, ROE (cao hơn tốt hơn)
    # Cost criteria: P/E, Nợ/Vốn (thấp hơn tốt hơn)
    benefit_idx = [0, 1, 3, 4]
    cost_idx = [2, 5]
    
    ideal_positive = np.zeros(len(criteria_cols))
    ideal_negative = np.zeros(len(criteria_cols))
    
    for i in range(len(criteria_cols)):
        if i in benefit_idx:
            ideal_positive[i] = weighted_normalized[:, i].max()
            ideal_negative[i] = weighted_normalized[:, i].min()
        else:
            ideal_positive[i] = weighted_normalized[:, i].min()
            ideal_negative[i] = weighted_normalized[:, i].max()
    
    # Bước 6: Tính khoảng cách
    distance_positive = np.sqrt(((weighted_normalized - ideal_positive) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((weighted_normalized - ideal_negative) ** 2).sum(axis=1))
    
    # Bước 7: Tính điểm TOPSIS
    topsis_score = distance_negative / (distance_positive + distance_negative)
    
    return topsis_score, distance_positive, distance_negative

# Nút tính toán
if st.button("🧮 Tính toán TOPSIS", type="primary", use_container_width=True):
    if len(st.session_state.stocks_data) < 2:
        st.error("⚠️ Cần ít nhất 2 cổ phiếu để so sánh!")
    else:
        # Tính TOPSIS
        scores, dist_pos, dist_neg = calculate_topsis(st.session_state.stocks_data, weights)
        
        # Tạo DataFrame kết quả
        results_df = st.session_state.stocks_data.copy()
        results_df['Điểm TOPSIS'] = scores
        results_df['Khoảng cách A+'] = dist_pos
        results_df['Khoảng cách A-'] = dist_neg
        results_df['Biến động giá (%)'] = ((results_df['Giá bán'] - results_df['Giá mua']) / results_df['Giá mua']) * 100
        
        # Xếp hạng
        results_df = results_df.sort_values('Điểm TOPSIS', ascending=False).reset_index(drop=True)
        results_df['Xếp hạng'] = range(1, len(results_df) + 1)
        
        # Phân loại rủi ro
        def get_risk_level(score):
            if score >= 0.7:
                return 'Rủi ro thấp'
            elif score >= 0.5:
                return 'Rủi ro trung bình'
            else:
                return 'Rủi ro cao'
        
        results_df['Mức rủi ro'] = results_df['Điểm TOPSIS'].apply(get_risk_level)
        
        # Hiển thị kết quả
        st.header("🏆 Kết quả phân tích")
        
        # Biểu đồ cột điểm TOPSIS
        fig_bar = go.Figure(data=[
            go.Bar(
                x=results_df['Mã CP'],
                y=results_df['Điểm TOPSIS'] * 100,
                text=results_df['Điểm TOPSIS'].apply(lambda x: f"{x*100:.2f}"),
                textposition='auto',
                marker=dict(
                    color=results_df['Điểm TOPSIS'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Điểm TOPSIS")
                )
            )
        ])
        fig_bar.update_layout(
            title="Điểm TOPSIS theo cổ phiếu",
            xaxis_title="Mã cổ phiếu",
            yaxis_title="Điểm TOPSIS (%)",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Hiển thị chi tiết từng cổ phiếu
        for idx, row in results_df.iterrows():
            risk_class = 'risk-low' if row['Mức rủi ro'] == 'Rủi ro thấp' else (
                'risk-medium' if row['Mức rủi ro'] == 'Rủi ro trung bình' else 'risk-high'
            )
            
            with st.container():
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 3, 2])
                
                with col1:
                    st.markdown(f"### 🏅 #{row['Xếp hạng']}")
                    st.markdown(f"## **{row['Mã CP']}**")
                
                with col2:
                    st.markdown(f'<span class="{risk_class}">{row["Mức rủi ro"]}</span>', unsafe_allow_html=True)
                    st.metric("Biến động giá", f"{row['Biến động giá (%)']:.2f}%", 
                             delta=f"{row['Biến động giá (%)']:.2f}%")
                
                with col3:
                    st.metric("Điểm TOPSIS", f"{row['Điểm TOPSIS']*100:.2f}", 
                             help="Điểm càng cao càng tốt")
                
                # Chi tiết các chỉ số
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("P/E", f"{row['P/E']:.2f}")
                with col2:
                    st.metric("EPS", f"{row['EPS']:,.0f}")
                with col3:
                    st.metric("ROE", f"{row['ROE (%)']:.1f}%")
                with col4:
                    st.metric("Nợ/Vốn", f"{row['Nợ/Vốn']:.2f}")
                
                # Khoảng cách
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📏 Khoảng cách A+: {row['Khoảng cách A+']:.4f}")
                with col2:
                    st.info(f"📏 Khoảng cách A-: {row['Khoảng cách A-']:.4f}")
        
        # Biểu đồ radar so sánh
        st.markdown("---")
        st.subheader("📊 Biểu đồ so sánh đa chiều")
        
        # Chuẩn hóa dữ liệu cho radar chart (0-100)
        radar_cols = ['Biến động giá (%)', 'P/E', 'EPS', 'ROE (%)', 'Nợ/Vốn']
        radar_data = []
        
        for idx, row in results_df.iterrows():
            radar_data.append(go.Scatterpolar(
                r=[
                    (row['Biến động giá (%)'] + 10) * 5,  # Chuyển về 0-100
                    100 - (row['P/E'] / results_df['P/E'].max() * 100),  # Đảo ngược vì thấp tốt hơn
                    row['EPS'] / results_df['EPS'].max() * 100,
                    row['ROE (%)'] / results_df['ROE (%)'].max() * 100,
                    100 - (row['Nợ/Vốn'] / results_df['Nợ/Vốn'].max() * 100)  # Đảo ngược
                ],
                theta=['Biến động giá', 'P/E', 'EPS', 'ROE', 'Nợ/Vốn'],
                fill='toself',
                name=row['Mã CP']
            ))
        
        fig_radar = go.Figure(data=radar_data)
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Giải thích
        st.success("""
        ### 📖 Giải thích kết quả:
        
        - **Điểm TOPSIS cao (≥70):** Cổ phiếu có tiềm năng tốt, rủi ro thấp
        - **Điểm TOPSIS trung bình (50-70):** Cổ phiếu ổn định, cân nhắc kỹ trước khi đầu tư
        - **Điểm TOPSIS thấp (<50):** Cổ phiếu có rủi ro cao, cần thận trọng
        - **A+ (Ideal Positive):** Giải pháp lý tưởng dương - các giá trị tốt nhất
        - **A- (Ideal Negative):** Giải pháp lý tưởng âm - các giá trị xấu nhất
        
        **Lưu ý:** Kết quả chỉ mang tính chất tham khảo. Cần kết hợp với phân tích kỹ thuật và cơ bản khác.
        """)