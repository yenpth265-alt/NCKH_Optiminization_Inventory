import numpy as np
import pandas as pd
def optimize_inventory_knapsack(df, budget, scale=100):
    import numpy as np
    import pandas as pd
    # 1. Tính toán các thành phần cơ bản
    df['total_cost'] = df['demand'] * df['unit_cost']
    df['net_gain_if_buy'] = (df['demand'] * (df['sell_price'] - df['unit_cost'])) - (df['demand'] * df['holding_cost_per_unit'] * 0.1)
    df['shortage_cost'] = df['demand'] * df['shortage_cost_per_unit']
    
    n = len(df)
    costs = (df['total_cost'] * scale).round().astype(int).values
    gains = df['net_gain_if_buy'].values
    penalties = df['shortage_cost'].values
    items = df['item_id'].values
    scaled_budget = int(budget * scale)

    # 2. KHỞI TẠO BẢNG K
    # Thay vì khởi tạo bằng 0, hàng đầu tiên (i=0) sẽ mang giá trị phạt 
    # của tất cả các mặt hàng chưa xét đến.
    K = np.zeros((n + 1, scaled_budget + 1))
    
    # Tính tổng phạt ban đầu (trạng thái tệ nhất: không mua gì cả)
    total_penalty_all = df['shortage_cost'].sum()
    K[0, :] = -total_penalty_all 

    # 3. CHẠY QUY HOẠCH ĐỘNG
    for i in range(1, n + 1):
        cost_i = costs[i-1]
        gain_i = gains[i-1]        # Lãi nếu mua
        penalty_i = penalties[i-1]  # Phí phạt nếu KHÔNG mua
        
        for w in range(scaled_budget + 1):
            # Trường hợp KHÔNG MUA món i: 
            # Giá trị kế thừa từ K[i-1][w] (đã bao gồm penalty của món i)
            res_no_buy = K[i-1][w]
            
            if cost_i <= w:
                # Trường hợp MUA món i:
                # Ta lấy giá trị từ ô trước đó (K[i-1][w-cost_i]) 
                # CỘNG thêm lãi và CỘNG LẠI số tiền phạt đã bị trừ mặc định
                res_buy = K[i-1][w-cost_i] + gain_i + penalty_i
                
                # So sánh trực tiếp 2 trường hợp như bạn muốn
                K[i][w] = max(res_buy, res_no_buy)
            else:
                K[i][w] = res_no_buy

    # 4. TRUY VẾT (Backtracking)
    w = scaled_budget
    selected_items = []
    for i in range(n, 0, -1):
        # Nếu giá trị tại ô hiện tại khác với ô không mua hàng trên nó
        if K[i][w] > K[i-1][w] + 1e-5:
            selected_items.append(items[i-1])
            w -= costs[i-1]

    df_result = df[df['item_id'].isin(selected_items)].copy()
    
    # Kết quả cuối cùng tại ô K[n][scaled_budget] chính là lợi nhuận thực tế sau khi trừ phạt
    return K[n][scaled_budget], df_result