import pandas as pd
import numpy as np
from scipy.stats import norm

def inventory_simulation_model(df, review_length, service_level):
    """
    df: DataFrame chứa cột 'item_id', 'actual_demand', 'forecast_demand'
    review_length (R): Khoảng thời gian kiểm tra kho (ngày)
    service_level (alpha): Mức độ phục vụ mong muốn (ví dụ 0.95)
    """
    
    # --- BƯỚC 1: TÍNH CÁC THÔNG SỐ ĐẦU VÀO ---
    # 1. Tính sai số dự báo e = Actual - Forecast [cite: 1, 3]
    df['e'] = df['actual_demand'] - df['forecast_demand']
    # 2. Tính trung bình sai số mu_e [cite: 4, 9]
    mu_e = df['e'].mean()
    
    # 3. Tính Safety Stock (SS) [cite: 5]
    # Giả sử SS tính dựa trên độ lệch chuẩn của sai số e và Service Level
    sigma_e = df['e'].std()
    z_score = norm.ppf(service_level)
    safety_stock = z_score * sigma_e * np.sqrt(review_length) 
    
    # 4. Tính d_f_R: Tổng nhu cầu dự báo trong 1 chu kỳ R [cite: 8, 10]
    # Ở đây lấy trung bình forecast nhân với R để đại diện cho 1 chu kỳ
    d_f_R = df['forecast_demand'].mean() * review_length 
    
    # 5. Tính Mức tồn kho mục tiêu S 
    S = d_f_R + mu_e + safety_stock 
    
    # --- BƯỚC 2: CHẠY MÔ PHỎNG TỪNG NGÀY ---
    inventory_log = []
    on_hand = S  # Tồn đầu kỳ 1 = S 
    lead_time = 0 # Lead time bằng 0 theo yêu cầu 
    
    total_actual_demand = 0
    total_sold = 0
    out_of_stock_days = 0
    
    for t in range(len(df)):
        day_data = df.iloc[t]
        actual = day_data['actual_demand']
        total_actual_demand += actual
        
        start_inv = on_hand
        
        # Kiểm tra đặt hàng vào ngày Review (t chia hết cho review_length)
        order_qty = 0
        if t % review_length == 0:
            order_qty = max(0, S - start_inv)
        
        # Vì Lead time = 0 nên hàng về ngay lập tức 
        on_hand += order_qty
        
        # Tính toán bán hàng và tồn cuối
        if on_hand >= actual:
            sold = actual
        else:
            sold = on_hand
            out_of_stock_days += 1
            
        total_sold += sold
        on_hand -= sold
        end_inv = on_hand
        
        # Lưu nhật ký mô phỏng từng ngày
        inventory_log.append({
            'Ngày': t + 1,
            'Tồn đầu': start_inv,
            'Nhu cầu thực tế': actual,
            'Số lượng đặt': order_qty,
            'Tồn cuối': end_inv
        })

    # --- BƯỚC 3: TÍNH CHỈ SỐ HIỆU QUẢ ---
    # Service Level (Alpha): Tỷ lệ ngày không hết hàng
    actual_alpha = (len(df) - out_of_stock_days) / len(df)
    
    # Fill Rate (Beta): Tỷ lệ nhu cầu được đáp ứng
    beta = total_sold / total_actual_demand if total_actual_demand > 0 else 0
    
    # Kết quả trả về
    df_sim = pd.DataFrame(inventory_log)
    df_metrics = pd.DataFrame([{
        'item_id': df['item_id'].iloc[0],
        'service_level_alpha': actual_alpha,
        'fill_rate_beta': beta,
        'target_level_S': S
    }])
    
    return df_sim, df_metrics