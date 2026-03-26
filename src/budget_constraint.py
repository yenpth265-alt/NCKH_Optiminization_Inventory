def optimize_procurement_knapsack(df, budget):
    """
    Thuật toán Tham lam (Greedy Knapsack) tối ưu cho máy RAM yếu.
    Sắp xếp theo tỷ lệ Lợi nhuận / Chi phí và chọn từ cao xuống thấp.
    """
    # 1. Tính chỉ số hiệu quả: Lợi nhuận trên mỗi đồng vốn bỏ ra
    df['efficiency'] = df['profit'] / df['unit_cost']
    
    # 2. Sắp xếp giảm dần theo hiệu quả
    df = df.sort_values(by='efficiency', ascending=False)
    
    selected_items = {}
    total_profit = 0
    remaining_budget = budget
    
    # 3. Duyệt và hốt những món hời nhất trước
    for _, row in df.iterrows():
        item_id = row['item_id']
        unit_cost = row['unit_cost']
        demand = int(row['demand_forecast']) # Số lượng AI dự báo
        
        if remaining_budget <= 0:
            break
            
        # Tính số lượng tối đa có thể mua cho món này (không quá cầu và không quá tiền)
        max_qty_can_buy = int(remaining_budget // unit_cost)
        qty_to_buy = min(demand, max_qty_can_buy)
        
        if qty_to_buy > 0:
            selected_items[item_id] = qty_to_buy
            total_profit += qty_to_buy * row['profit_per_unit']
            remaining_budget -= qty_to_buy * unit_cost
            
    return total_profit, selected_items