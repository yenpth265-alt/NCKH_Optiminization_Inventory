import pandas as pd
import numpy as np
import os
import gc

def reduce_mem_usage(df):
    """Hàm tối ưu hóa dung lượng bộ nhớ cho DataFrame"""
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(col_type) or str(col_type) == 'category':
            continue
        if col_type != object:
            try:
                c_min, c_max = df[col].min(), df[col].max()
                # Kiểm tra xem c_min và c_max có phải là numeric không
                if not (np.isscalar(c_min) and np.isscalar(c_max)):
                    df[col] = df[col].astype('category')
                    continue
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            except Exception:
                # Nếu có lỗi, chuyển sang category
                df[col] = df[col].astype('category')
        else:
            df[col] = df[col].astype('category')
    return df

def handle_outliers_and_missing(df):
    """Xử lý ngoại lai (Outliers) bằng phương pháp IQR và điền khuyết"""
    print("\n-> Đang xử lý Outliers (IQR) và Zeroes...")
    
    # Định nghĩa hàm cắt đỉnh (capping) bằng IQR
    def cap_outliers(series):
        active_sales = series[series > 0]
        if len(active_sales) < 10: return series # Bỏ qua nếu dữ liệu quá ít
        Q1, Q3 = active_sales.quantile(0.25), active_sales.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        # Cắt đỉnh những ngày mua sỉ đột biến để làm mịn
        return np.where(series > upper_bound, upper_bound, series)

    # Áp dụng làm mịn cho cột demand
    df['demand'] = df.groupby(['store_id', 'item_id'])['demand'].transform(cap_outliers)
    
    # Nội suy/Điền khuyết: Đảm bảo không có giá trị NaN nào ở cột nhu cầu
    df['demand'] = df['demand'].fillna(0)
    return df

def load_and_preprocess_raw(raw_dir, nrows=None):
    """Hàm tải và hợp nhất dữ liệu Sales với Calendar"""
    print("1. Đang tải và ép cân dữ liệu Calendar...")
    calendar = pd.read_csv(os.path.join(raw_dir, 'calendar.csv'))
    calendar = reduce_mem_usage(calendar)

    print("2. Đang tải dữ liệu Sales...")
    sales = pd.read_csv(os.path.join(raw_dir, 'sales_train_validation.csv'), nrows=nrows) 

    print("3. Đang Melt (kéo giãn) dữ liệu Sales...")
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_melted = pd.melt(sales, id_vars=id_vars, var_name='d', value_name='demand')
    del sales; gc.collect()

    print("4. Đang Merge Sales với Calendar...")
    df_master = pd.merge(sales_melted, calendar, on='d', how='left')
    del sales_melted, calendar; gc.collect()

    df_master['date'] = pd.to_datetime(df_master['date'])
    
    # GỌI HÀM XỬ LÝ OUTLIER Ở ĐÂY
    df_master = handle_outliers_and_missing(df_master)
    
    print("5. Ép cân lần cuối cho DataFrame tổng...")
    df_master = reduce_mem_usage(df_master)
    return df_master


#METHOD TRONG KAGGLE NOTEBOOK
#Xử lý để bảng về dạng long
def create_dt(is_train = True, 
              nrows = None, 
              first_day = 1200, 
              price_dt = None, 
              cal_dt = None, 
              raw_folder = "../dataset/raw/",
              tr_last = 1913,
              max_lags = 57,
              sales_data = 'sales_train_validation.csv'
              ):
    prices = pd.read_csv(f"{raw_folder}/sell_prices.csv", dtype = price_dt)
    for col, col_dtype in price_dt.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv(f"{raw_folder}/calendar.csv", dtype = cal_dt)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in cal_dt.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(f"{raw_folder}/{sales_data}", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return dt

#Thêm các đặc trưng cho dữ liệu
def create_fea(
    dt,
    lags_fe=[7, 28],
    wins_fe=[7, 28],
    std_fe=None,
    price_lags_fe=None,
    price_wins_fe=None,
    price_std_fe=None,
    date_features={
        "wday": "weekday",
        "week": "isocalendar",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
):
    # =========================
    # Sales Features
    # =========================
    lag_cols = []

    if lags_fe is not None:
        lag_cols = [f"lag_{lag}" for lag in lags_fe]
        for lag, lag_col in zip(lags_fe, lag_cols):
            dt[lag_col] = dt.groupby("id")["sales"].shift(lag)

    if wins_fe is not None and lags_fe is not None:
        for win in wins_fe:
            for lag, lag_col in zip(lags_fe, lag_cols):
                dt[f"rmean_{lag}_{win}"] = (
                    dt.groupby("id")[lag_col]
                      .transform(lambda x: x.rolling(win).mean())
                )

    if std_fe is not None and lags_fe is not None:
        for std in std_fe:
            for lag, lag_col in zip(lags_fe, lag_cols):
                dt[f"rstd_{lag}_{std}"] = (
                    dt.groupby("id")[lag_col]
                      .transform(lambda x: x.rolling(std).std())
                )

    # =========================
    # Price Features
    # =========================
    price_lag_cols = []

    if price_lags_fe is not None:
        price_lag_cols = [f"price_lag_{lag}" for lag in price_lags_fe]
        for lag, lag_col in zip(price_lags_fe, price_lag_cols):
            dt[lag_col] = dt.groupby("id")["sell_price"].shift(lag)

    if price_wins_fe is not None and price_lags_fe is not None:
        for win in price_wins_fe:
            for lag, lag_col in zip(price_lags_fe, price_lag_cols):
                dt[f"price_rmean_{lag}_{win}"] = (
                    dt.groupby("id")[lag_col]
                      .transform(lambda x: x.rolling(win).mean())
                )

    if price_std_fe is not None and price_lags_fe is not None:
        for std in price_std_fe:
            for lag, lag_col in zip(price_lags_fe, price_lag_cols):
                dt[f"price_rstd_{lag}_{std}"] = (
                    dt.groupby("id")[lag_col]
                      .transform(lambda x: x.rolling(std).std())
                )

    # =========================
    # Date Features
    # =========================
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_func is None:
            continue

        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            if date_feat_name == "week":
                dt[date_feat_name] = dt["date"].dt.isocalendar().week.astype("int16")
            else:
                dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")

    return dt