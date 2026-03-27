import pandas as pd
import numpy as np

def optimize_budget_knapsack(df, budget, scale=100):
    """
    scale=100 nghĩa là hỗ trợ đến 2 chữ số thập phân (ví dụ: 1.55)
    Nếu muốn chính xác hơn nữa (1.555), hãy chỉnh scale=1000.
    """
    # 1. TÍNH TOÁN (Giữ nguyên float để chính xác tuyệt đối)
    df['total_cost'] = df['demand'] * df['unit_cost']
    df['gross_profit'] = df['demand'] * (df['sell_price'] - df['unit_cost'])
    df['holding_total'] = df['demand'] * df['holding_cost_per_unit'] * 0.1
    df['shortage_penalty'] = df['demand'] * df['shortage_cost_per_unit']
    
    # Giá trị mang lại (Net Value) - Vẫn là số thập phân
    df['net_value'] = df['gross_profit'] - df['holding_total'] + df['shortage_penalty']

    # 2. CHUYỂN ĐỔI SANG ĐƠN VỊ SCALE (Để lập bảng Knapsack)
    # Chỉ có Chi phí và Ngân sách cần biến thành số nguyên để làm Index
    scaled_costs = (df['total_cost'] * scale).round().astype(int).values
    scaled_budget = int(budget * scale)
    
    # Lợi nhuận (Values) có thể giữ float nếu dùng bảng Dictionary hoặc ép int để tối ưu tốc độ
    values = df['net_value'].values 
    items = df['item_id'].values
    n = len(df)

    # 3. LẬP BẢNG KNAPSACK
    # Dùng float cho bảng K để chứa lợi nhuận lẻ
    K = np.zeros((n + 1, scaled_budget + 1))

    for i in range(1, n + 1):
        cost_i = scaled_costs[i-1]
        val_i = values[i-1]
        for w in range(scaled_budget + 1):
            if cost_i <= w:
                K[i][w] = max(val_i + K[i-1][w-cost_i], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]

    # 4. TRUY VẾT (Backtracking)
    w = scaled_budget
    result_details = []
    
    for i in range(n, 0, -1):
        # Kiểm tra sự chênh lệch (dùng sai số nhỏ 1e-5 thay vì != để tránh lỗi float)
        if K[i][w] > K[i-1][w] + 1e-5:
            idx = i - 1
            result_details.append({
                'item_id': items[idx],
                'quantity_to_buy': df.iloc[idx]['demand'], # Giữ nguyên số gốc
                'total_cost': df.iloc[idx]['total_cost'],    # Giữ nguyên số gốc
                'value_added': df.iloc[idx]['net_value']     # Giữ nguyên số gốc
            })
            w -= scaled_costs[idx]

    df_result = pd.DataFrame(result_details)
    
    # Trả về Tổng lợi nhuận (ô cuối) và bảng chi tiết
    return K[n][scaled_budget], df_result