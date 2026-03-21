import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Trỏ đường dẫn để gọi module từ thư mục src
sys.path.append(os.path.abspath('src'))
from dataPreprocessing import load_and_preprocess_raw
from featureEngineering import generate_all_features, create_lag_features_for_item

# ==========================================
# 1. CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Smart Inventory AI", page_icon="📦", layout="wide")
st.title("📦 Hệ Thống Tối Ưu Tồn Kho Thông Minh (AI-Driven)")

# ==========================================
# 2. TẢI DỮ LIỆU (Có Cache để Web chạy siêu nhanh)
# ==========================================
@st.cache_data
def load_data():
    RAW_DIR = 'dataset/raw/'
    df_master = load_and_preprocess_raw(RAW_DIR, nrows=15000)
    df_featured = generate_all_features(df_master, RAW_DIR)
    return df_featured

with st.spinner('Đang tải dữ liệu và khởi động AI...'):
    df = load_data()

# ==========================================
# 3. THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Bảng Điều Khiển")
store_list = df['store_id'].unique()
selected_store = st.sidebar.selectbox("🏠 Chọn Cửa hàng:", store_list)

# Lọc mã hàng theo cửa hàng đã chọn
item_list = df[df['store_id'] == selected_store]['item_id'].unique()
selected_item = st.sidebar.selectbox("🛒 Chọn Mã Sản Phẩm:", item_list)

st.sidebar.markdown("---")
st.sidebar.subheader("Biến số Mô phỏng")
lead_time = st.sidebar.slider("⏱️ Thời gian giao hàng (Lead Time):", 1, 14, 3)
holding_cost = st.sidebar.number_input("💰 Phí lưu kho/sp:", value=0.5)
stockout_cost = st.sidebar.number_input("⚠️ Phí thiệt hại hết hàng/sp:", value=5.0)
initial_inv = st.sidebar.number_input("📦 Tồn kho hiện tại:", value=15)

# Nút bấm chạy AI
run_button = st.sidebar.button("🚀 Chạy Mô Phỏng AI", use_container_width=True)

# ==========================================
# 4. LUỒNG XỬ LÝ CHÍNH KHI BẤM NÚT
# ==========================================
if run_button:
    st.markdown("---")
    st.subheader(f"📊 Kết quả Phân tích cho Sản phẩm: **{selected_item}**")
    
    # Chuẩn bị dữ liệu
    df_item = df[(df['item_id'] == selected_item) & (df['store_id'] == selected_store)].copy()
    df_item = create_lag_features_for_item(df_item)
    
    features = ['day', 'is_weekend', 'sell_price', 'price_discount', 'price_momentum', 'event_name_1', 'lag_7', 'lag_14', 'rolling_mean_7']
    X_train, y_train = df_item[features], df_item['demand']
    
    # Train AI siêu tốc
    params = {'objective': 'tweedie', 'tweedie_variance_power': 1.1, 'metric': 'rmse', 'learning_rate': 0.05, 'verbose': -1}
    train_set = lgb.Dataset(X_train, y_train, categorical_feature=['event_name_1'])
    model = lgb.train(params, train_set, num_boost_round=100)
    
    # Dự báo Cuốn chiếu (28 ngày)
    history_df = df_item.copy()
    future_predictions = []
    
    for i in range(1, 29):
        last_row = history_df.iloc[-1]
        next_date = last_row['date'] + pd.Timedelta(days=1)
        current_demands = history_df['demand'].values
        
        new_features = pd.DataFrame([{
            'day': next_date.day, 'is_weekend': 1 if next_date.dayofweek in [5, 6] else 0,
            'sell_price': last_row['sell_price'], 'price_discount': last_row['price_discount'], 
            'price_momentum': 1.0, 'event_name_1': last_row['event_name_1'],
            'lag_7': current_demands[-7], 'lag_14': current_demands[-14], 'rolling_mean_7': np.mean(current_demands[-7:])
        }])
        
        pred_demand = max(0, model.predict(new_features)[0])
        future_predictions.append(pred_demand)
        
        new_row = pd.DataFrame({'date': [next_date], 'demand': [pred_demand], 'sell_price': last_row['sell_price'], 'price_discount': last_row['price_discount'], 'event_name_1': last_row['event_name_1']})
        history_df = pd.concat([history_df, new_row], ignore_index=True)

    # ĐÂY LÀ BIẾN CHÍNH
    daily_demand_forecast = np.round(future_predictions).astype(int)

    # Mô phỏng Tồn kho
    def simulate_inventory(demand_array, ROP, Q):
        inventory = initial_inv
        total_cost = 0
        days_to_arrival = 0
        order_placed = False
        inv_history = []
        for d in demand_array:
            if order_placed and days_to_arrival == 0:
                inventory += Q
                order_placed = False
            if inventory >= d:
                inventory -= d
            else:
                total_cost += (d - inventory) * stockout_cost
                inventory = 0
            total_cost += inventory * holding_cost
            if inventory <= ROP and not order_placed:
                total_cost += 50.0 # Ordering cost cố định
                order_placed = True
                days_to_arrival = lead_time
            if order_placed:
                days_to_arrival -= 1
            inv_history.append(inventory)
        return total_cost, inv_history

    best_cost, best_ROP, best_Q = float('inf'), 0, 0
    best_history = []
    for rop in range(0, 50, 2):
        for q in range(10, 150, 5):
            cost, history = simulate_inventory(daily_demand_forecast, rop, q)
            if cost < best_cost:
                best_cost, best_ROP, best_Q, best_history = cost, rop, q, history

    # HIỂN THỊ KẾT QUẢ RA WEB
    st.markdown("---")
    
    col_rop, col_eoq, col_cost = st.columns(3)
    col_rop.metric(label="🎯 Ngưỡng ROP tối ưu", value=f"{best_ROP} sp", delta="Chạm là đặt hàng")
    col_eoq.metric(label="✅ Lượng Đặt EOQ tối ưu", value=f"{best_Q} sp", delta="Lô hàng tiết kiệm nhất")
    col_cost.metric(label="💰 Tổng Chi Phí Dự Kiến", value=f"${best_cost:,.2f}", delta="- Chi phí thấp nhất")

    tab_sim, tab_ai = st.tabs(["Mô phỏng Lộ trình Kho (Simulation)", "📊 Đánh giá Sai số AI (Backtest)"])
    
    with tab_sim:
        st.markdown("### 📈 Mô phỏng Tồn kho dựa theo AI (28 ngày tới)")
        fig_inv, ax_inv = plt.subplots(figsize=(12, 4))
        ax_inv.step(range(1, 29), best_history, where='post', color='#27AE60', linewidth=2.5, label='Lượng Tồn Kho (Inventory)')
        ax_inv.axhline(y=best_ROP, color='#E74C3C', linestyle='--', linewidth=2, label=f'Ngưỡng ROP = {best_ROP}')
        ax_inv.set_xlabel("Ngày tương lai", fontweight='bold')
        ax_inv.set_ylabel("Số lượng trong kho", fontweight='bold')
        ax_inv.set_xticks(range(1, 29))
        ax_inv.grid(True, linestyle='-.', alpha=0.5)
        ax_inv.legend(loc='upper right')
        st.pyplot(fig_inv)

    with tab_ai:
        st.markdown("### 🔮 So sánh AI dự báo với Thực tế (Backtest trên 28 ngày cuối)")
        
        df_item_raw = df[(df['item_id'] == selected_item) & (df['store_id'] == selected_store)].copy()
        
        if len(df_item_raw) >= 28:
            daily_demand_actuals = df_item_raw.iloc[-28:]['demand'].values
            
            # Đã sửa lỗi tên biến ở đây: dùng daily_demand_forecast
            mae = mean_absolute_error(daily_demand_actuals, daily_demand_forecast)
            rmse = np.sqrt(mean_squared_error(daily_demand_actuals, daily_demand_forecast))
            
            c1, c2 = st.columns(2)
            c1.metric("MAE (Lỗi Trung bình)", f"{mae:,.2f} sp")
            c2.metric("RMSE (Độ lệch chuẩn)", f"{rmse:,.2f} sp")
            
            fig_comp, ax_comp = plt.subplots(figsize=(12, 4))
            # Sửa tên biến ở đây: daily_demand_forecast
            ax_comp.bar(range(1, 29), daily_demand_forecast, color='#3498DB', edgecolor='black', alpha=0.6, label='Dự báo AI')
            ax_comp.plot(range(1, 29), daily_demand_actuals, color='#D35400', marker='o', linewidth=2.5, label='Nhu cầu Thực tế')
            ax_comp.set_xlabel("Ngày so sánh", fontweight='bold')
            ax_comp.set_ylabel("Số lượng bán", fontweight='bold')
            ax_comp.set_xticks(range(1, 29))
            ax_comp.grid(axis='y', linestyle='--', alpha=0.5)
            ax_comp.legend(loc='upper left')
            st.pyplot(fig_comp)
        else:
            st.warning("⚠️ Dữ liệu quá ngắn để thực hiện Backtest.")
    
    st.success("✅ Phân tích hoàn tất!")

else:
    st.info("👈 Cấu hình ở Menu bên trái và bấm 'Chạy Mô Phỏng AI'.")