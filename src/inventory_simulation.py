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
    # 1. Tính sai số dự báo e = Actual - Forecast 
    df['e'] = df['actual_demand'] - df['forecast_demand']
    # 2. Tính trung bình sai số mu_e 
    mu_e = df['e'].mean()
    
    # 3. Tính Safety Stock (SS) [cite: 5]
    # Giả sử SS tính dựa trên độ lệch chuẩn của sai số e và Service Level
    sigma_e = df['e'].std()
    z_score = norm.ppf(service_level)
    safety_stock = z_score * sigma_e * np.sqrt(review_length) 
    
    # 4. Tính d_f_R: Tổng nhu cầu dự báo trong 1 chu kỳ R [cite: 8, 10]
    # Ở đây lấy trung bình forecast nhân với R để đại diện cho 1 chu kỳ
    d_f_R = df['forecast_demand'].iloc[:review_length].sum()  
    
    # 5. Tính Mức tồn kho mục tiêu S 
    S = d_f_R + mu_e*review_length + safety_stock 
    
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

def simulate_periodic_review_L0(
    mu_demand,
    std_demand,
    review_period,
    service_level,
    initial_inventory,
    demand_series,
    holding_cost=0.0,
    shortage_cost=0.0,
    order_cost=0.0,
    return_details=True
):
    """
    Mô phỏng hệ thống tồn kho periodic review với:
    - Lead time L = 0
    - Safety stock được tính theo công thức lý thuyết
    - Order-up-to level S được tính theo công thức lý thuyết
    - demand_series dùng để kiểm thử mô phỏng

    Parameters
    ----------
    mu_demand : float
        Nhu cầu trung bình mỗi kỳ.
    std_demand : float
        Độ lệch chuẩn nhu cầu mỗi kỳ.
    review_period : int
        Chu kỳ kiểm tra tồn kho R.
    service_level : float
        Mức độ phục vụ mong muốn, ví dụ 0.95.
    initial_inventory : float
        Tồn kho đầu kỳ.
    demand_series : list, np.ndarray, pd.Series
        Chuỗi nhu cầu thực tế dùng để chạy mô phỏng.
    holding_cost : float, default=0.0
        Chi phí lưu kho trên mỗi đơn vị tồn cuối kỳ.
    shortage_cost : float, default=0.0
        Chi phí thiếu hàng trên mỗi đơn vị thiếu.
    order_cost : float, default=0.0
        Chi phí cố định cho mỗi lần đặt hàng.
    return_details : bool, default=True
        Nếu True trả về bảng chi tiết từng kỳ.

    Returns
    -------
    results_df : pd.DataFrame or None
        Bảng mô phỏng chi tiết theo từng kỳ.
    summary : dict
        Thống kê tổng hợp mô phỏng.
    """

    if review_period <= 0:
        raise ValueError("review_period phải > 0")
    if std_demand < 0:
        raise ValueError("std_demand không được âm")
    if not (0 < service_level < 1):
        raise ValueError("service_level phải nằm trong khoảng (0, 1)")
    if initial_inventory < 0:
        raise ValueError("initial_inventory không được âm")

    demand_series = pd.Series(demand_series, dtype=float).reset_index(drop=True)
    if len(demand_series) == 0:
        raise ValueError("demand_series không được rỗng")

    # =========================
    # 1. TÍNH POLICY THEO LÝ THUYẾT
    # =========================
    z = norm.ppf(service_level)

    # Vì L = 0 nên protection period = R
    sigma_R = std_demand * np.sqrt(review_period)
    safety_stock = z * sigma_R
    expected_demand_over_review = mu_demand * review_period
    order_up_to_level = expected_demand_over_review + safety_stock

    # =========================
    # 2. MÔ PHỎNG
    # =========================
    inventory = float(initial_inventory)

    total_holding_cost = 0.0
    total_shortage_cost = 0.0
    total_order_cost = 0.0
    total_order_qty = 0.0
    total_shortage_units = 0.0
    total_sales = 0.0
    num_orders = 0

    records = []

    for t, demand in enumerate(demand_series, start=1):
        beginning_inventory = inventory
        is_review_period = ((t - 1) % review_period == 0)

        order_qty = 0.0

        # Review đầu kỳ; do L=0 nên nhận ngay
        if is_review_period:
            inventory_position = inventory
            if inventory_position < order_up_to_level:
                order_qty = order_up_to_level - inventory_position
                inventory += order_qty
                total_order_qty += order_qty
                num_orders += 1
                if order_qty > 0:
                    total_order_cost += order_cost

        inventory_after_replenishment = inventory

        # Nhu cầu xảy ra trong kỳ
        sales = min(inventory, demand)
        shortage = max(demand - inventory, 0.0)
        ending_inventory = max(inventory - demand, 0.0)

        # Chi phí
        period_holding_cost = ending_inventory * holding_cost
        period_shortage_cost = shortage * shortage_cost

        # Cập nhật tổng
        total_sales += sales
        total_shortage_units += shortage
        total_holding_cost += period_holding_cost
        total_shortage_cost += period_shortage_cost

        # Tồn kho chuyển sang kỳ sau
        inventory = ending_inventory

        if return_details:
            records.append({
                "period": t,
                "demand": demand,
                "beginning_inventory": beginning_inventory,
                "review_period_flag": is_review_period,
                "z_value": z,
                "safety_stock": safety_stock,
                "order_up_to_level": order_up_to_level,
                "order_qty": order_qty,
                "inventory_after_replenishment": inventory_after_replenishment,
                "sales": sales,
                "shortage": shortage,
                "ending_inventory": ending_inventory,
                "holding_cost": period_holding_cost,
                "shortage_cost": period_shortage_cost
            })

    #Xuất ra dataframe kết quả
    results_df = pd.DataFrame(records) if return_details else None

    # =========================
    # 3. TỔNG HỢP KẾT QUẢ
    # =========================
    total_demand = demand_series.sum()
    fill_rate = total_sales / total_demand if total_demand > 0 else 1.0
    daily_service_level_empirical = (results_df["shortage"] == 0).mean()

    # Cycle service level thực nghiệm:
    # Một cycle là từ 1 kỳ review đến trước kỳ review tiếp theo
    stockout_cycles = 0
    total_cycles = 0

    for start_idx in range(0, len(demand_series), review_period):
        end_idx = min(start_idx + review_period, len(demand_series))
        cycle_df = pd.DataFrame(records[start_idx:end_idx]) if return_details else None

        if return_details:
            total_cycles += 1
            if cycle_df["shortage"].sum() > 0:
                stockout_cycles += 1

    cycle_service_level_empirical = (
        1 - stockout_cycles / total_cycles if total_cycles > 0 else 1.0
    )

    summary = {
        "mu_demand": mu_demand,
        "std_demand": std_demand,
        "review_period": review_period,
        "lead_time": 0,
        "service_level_target": service_level,
        "z_value": z,
        "safety_stock": safety_stock,
        "expected_demand_over_review": expected_demand_over_review,
        "order_up_to_level": order_up_to_level,
        "initial_inventory": initial_inventory,
        "num_periods_simulated": len(demand_series),
        "num_orders": num_orders,
        "total_demand": total_demand,
        "total_sales": total_sales,
        "total_shortage_units": total_shortage_units,
        "fill_rate": fill_rate,
        "cycle_service_level_empirical": cycle_service_level_empirical,
        "daily_service_level_empirical":daily_service_level_empirical,
        "average_ending_inventory": (
            pd.DataFrame(records)["ending_inventory"].mean() if return_details else None
        ),
        "total_order_qty": total_order_qty,
        "total_holding_cost": total_holding_cost,
        "total_shortage_cost": total_shortage_cost,
        "total_order_cost": total_order_cost,
        "total_cost": total_holding_cost + total_shortage_cost + total_order_cost
    }

    
    return results_df, summary


def simulate_periodic_withForecast_L0(
    forecast_series,
    mu_error,
    std_error,
    review_period,
    service_level,
    initial_inventory,
    demand_series,
    holding_cost=0.0,
    shortage_cost=0.0,
    order_cost=0.0,
    return_details=True
):
    """
    Mô phỏng hệ thống tồn kho periodic review với:
    - Lead time L = 0
    - Safety stock:
        SS = z_alpha * sigma_E * sqrt(R)
    - Order-up-to level:
        S = D_hat_R + mu_E * R + SS

    Parameters
    ----------
    forecast_series : list, np.ndarray, pd.Series
        Chuỗi forecast theo từng kỳ.
    mu_error : float
        Trung bình sai số forecast mỗi kỳ.
    std_error : float
        Độ lệch chuẩn sai số forecast mỗi kỳ.
    review_period : int
        Chu kỳ kiểm tra tồn kho R.
    service_level : float
        Mức độ phục vụ mong muốn, ví dụ 0.95.
    initial_inventory : float
        Tồn kho đầu kỳ.
    demand_series : list, np.ndarray, pd.Series
        Chuỗi nhu cầu thực tế dùng để chạy mô phỏng.
    holding_cost : float, default=0.0
        Chi phí lưu kho trên mỗi đơn vị tồn cuối kỳ.
    shortage_cost : float, default=0.0
        Chi phí thiếu hàng trên mỗi đơn vị thiếu.
    order_cost : float, default=0.0
        Chi phí cố định cho mỗi lần đặt hàng.
    return_details : bool, default=True
        Nếu True trả về bảng chi tiết từng kỳ.

    Returns
    -------
    results_df : pd.DataFrame or None
        Bảng mô phỏng chi tiết theo từng kỳ.
    summary : dict
        Thống kê tổng hợp mô phỏng.
    """

    # =========================
    # 0. VALIDATION
    # =========================
    if review_period <= 0:
        raise ValueError("review_period phải > 0")
    if std_error < 0:
        raise ValueError("std_error không được âm")
    if not (0 < service_level < 1):
        raise ValueError("service_level phải nằm trong khoảng (0, 1)")
    if initial_inventory < 0:
        raise ValueError("initial_inventory không được âm")

    demand_series = pd.Series(demand_series, dtype=float).reset_index(drop=True)
    forecast_series = pd.Series(forecast_series, dtype=float).reset_index(drop=True)

    if len(demand_series) == 0:
        raise ValueError("demand_series không được rỗng")
    if len(forecast_series) == 0:
        raise ValueError("forecast_series không được rỗng")
    if len(forecast_series) < len(demand_series):
        raise ValueError("forecast_series phải có độ dài >= demand_series")

    # =========================
    # 1. TÍNH POLICY THEO LÝ THUYẾT
    # =========================
    z = norm.ppf(service_level)

    # Safety stock cố định theo công thức:
    # SS = z_alpha * sigma_E * sqrt(R)
    safety_stock = z * std_error * np.sqrt(review_period)

    # =========================
    # 2. MÔ PHỎNG
    # =========================
    inventory = float(initial_inventory)

    total_holding_cost = 0.0
    total_shortage_cost = 0.0
    total_order_cost = 0.0
    total_order_qty = 0.0
    total_shortage_units = 0.0
    total_sales = 0.0
    num_orders = 0

    records = []

    for t, demand in enumerate(demand_series, start=1):
        beginning_inventory = inventory
        is_review_period = ((t - 1) % review_period == 0)

        order_qty = 0.0
        expected_demand_over_review = np.nan
        bias_correction = np.nan
        order_up_to_level = np.nan

        # Review đầu kỳ; do L=0 nên nhận ngay
        if is_review_period:
            start_idx = t - 1
            end_idx = min(start_idx + review_period, len(forecast_series))

            # Tổng forecast trong kỳ review
            forecast_window = forecast_series.iloc[start_idx:end_idx]
            expected_demand_over_review = forecast_window.sum()

            # Bias correction = mu_E * R
            bias_correction = mu_error * review_period

            # Order-up-to level:
            # S = D_hat_R + mu_E * R + SS
            order_up_to_level = (
                expected_demand_over_review
                + bias_correction
                + safety_stock
            )

            inventory_position = inventory

            if inventory_position < order_up_to_level:
                order_qty = order_up_to_level - inventory_position
                inventory += order_qty
                total_order_qty += order_qty
                num_orders += 1
                if order_qty > 0:
                    total_order_cost += order_cost

        inventory_after_replenishment = inventory

        # Nhu cầu xảy ra trong kỳ
        sales = min(inventory, demand)
        shortage = max(demand - inventory, 0.0)
        ending_inventory = max(inventory - demand, 0.0)

        # Chi phí
        period_holding_cost = ending_inventory * holding_cost
        period_shortage_cost = shortage * shortage_cost

        # Cập nhật tổng
        total_sales += sales
        total_shortage_units += shortage
        total_holding_cost += period_holding_cost
        total_shortage_cost += period_shortage_cost

        # Tồn kho chuyển sang kỳ sau
        inventory = ending_inventory

        if return_details:
            current_forecast = forecast_series.iloc[t - 1]
            forecast_error = demand - current_forecast

            records.append({
                "period": t,
                "forecast": current_forecast,
                "demand": demand,
                "forecast_error": forecast_error,
                "beginning_inventory": beginning_inventory,
                "review_period_flag": is_review_period,
                "z_value": z,
                "mu_error": mu_error,
                "std_error": std_error,
                "safety_stock": safety_stock,
                "forecast_sum_R": expected_demand_over_review,
                "bias_correction": bias_correction,
                "order_up_to_level": order_up_to_level,
                "order_qty": order_qty,
                "inventory_after_replenishment": inventory_after_replenishment,
                "sales": sales,
                "shortage": shortage,
                "ending_inventory": ending_inventory,
                "holding_cost": period_holding_cost,
                "shortage_cost": period_shortage_cost
            })

    # =========================
    # 3. XUẤT DATAFRAME KẾT QUẢ
    # =========================
    results_df = pd.DataFrame(records) if return_details else None

    # =========================
    # 4. TỔNG HỢP KẾT QUẢ
    # =========================
    total_demand = demand_series.sum()
    fill_rate = total_sales / total_demand if total_demand > 0 else 1.0

    if return_details and len(results_df) > 0:
        daily_service_level_empirical = (results_df["shortage"] == 0).mean()
    else:
        daily_service_level_empirical = None

    # Cycle service level thực nghiệm
    stockout_cycles = 0
    total_cycles = 0

    if return_details and len(results_df) > 0:
        for start_idx in range(0, len(demand_series), review_period):
            end_idx = min(start_idx + review_period, len(demand_series))
            cycle_df = results_df.iloc[start_idx:end_idx]
            total_cycles += 1
            if cycle_df["shortage"].sum() > 0:
                stockout_cycles += 1

    cycle_service_level_empirical = (
        1 - stockout_cycles / total_cycles if total_cycles > 0 else 1.0
    )

    summary = {
        "review_period": review_period,
        "lead_time": 0,
        "service_level_target": service_level,
        "z_value": z,
        "mu_error": mu_error,
        "std_error": std_error,
        "safety_stock": safety_stock,
        "initial_inventory": initial_inventory,
        "num_periods_simulated": len(demand_series),
        "num_orders": num_orders,
        "total_demand": total_demand,
        "total_sales": total_sales,
        "total_shortage_units": total_shortage_units,
        "fill_rate": fill_rate,
        "cycle_service_level_empirical": cycle_service_level_empirical,
        "daily_service_level_empirical": daily_service_level_empirical,
        "average_ending_inventory": (
            results_df["ending_inventory"].mean()
            if return_details and len(results_df) > 0 else None
        ),
        "total_order_qty": total_order_qty,
        "total_holding_cost": total_holding_cost,
        "total_shortage_cost": total_shortage_cost,
        "total_order_cost": total_order_cost,
        "total_cost": total_holding_cost + total_shortage_cost + total_order_cost
    }

    return results_df, summary