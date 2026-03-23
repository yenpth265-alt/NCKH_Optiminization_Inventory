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