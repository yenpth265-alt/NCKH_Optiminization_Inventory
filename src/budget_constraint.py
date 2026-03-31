import pulp
import pandas as pd
import numpy as np

def solve_bounded_knapsack(df, budget):
    # 1. Khởi tạo bài toán
    prob = pulp.LpProblem("Inventory_Optimization", pulp.LpMaximize)

    # 2. Tạo danh sách chỉ số sản phẩm
    indices = df.index

    # 3. Tạo biến quyết định: Số lượng sản phẩm i cần mua (x_i)
    # lowBound=0: Không mua số âm
    # cat=pulp.LpInteger: Phải là số nguyên (1, 2, 3...)
    x = pulp.LpVariable.dicts("qty", indices, lowBound=0, cat=pulp.LpInteger)

    # 4. Hàm mục tiêu: Tối đa hóa tổng giá trị ròng (Net Value)
    # Net Value ở đây nên là giá trị trên 1 đơn vị sản phẩm
    prob += pulp.lpSum([df.loc[i, 'net_value_per_unit'] * x[i] for i in indices])

    # 5. Các ràng buộc (Constraints)

    # Ràng buộc 1: Tổng chi phí <= Ngân sách
    prob += pulp.lpSum([df.loc[i, 'unit_cost'] * x[i] for i in indices]) <= budget

    # Ràng buộc 2: Số lượng mua không vượt quá nhu cầu (Demand)
    # Đây là điểm khác biệt quan trọng so với bài trước
    for i in indices:
        prob += x[i] <= df.loc[i, 'demand']

    # 6. Giải bài toán
    # msg=0 để tắt log, msg=1 nếu bạn muốn xem quá trình solver tính toán
    status = prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=60))

    # 7. Thu thập kết quả
    results = []
    for i in indices:
        qty = pulp.value(x[i])
        if qty > 0:
            results.append({
                'item_id': df.loc[i, 'item_id'],
                'quantity_to_buy': int(qty),
                'unit_cost': df.loc[i, 'unit_cost'],
                'total_cost': qty * df.loc[i, 'unit_cost'],
                'total_value': qty * df.loc[i, 'net_value_per_unit']
            })

    df_res = pd.DataFrame(results)
    return pulp.value(prob.objective), df_res
