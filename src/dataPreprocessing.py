import pandas as pd
import numpy as np
import os
import gc

def reduce_mem_usage(df):
    """Hàm tối ưu hóa dung lượng bộ nhớ cho DataFrame"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'---> Dung lượng ban đầu: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Bỏ qua cột thời gian (datetime) VÀ cột phân loại (category)
        if pd.api.types.is_datetime64_any_dtype(col_type) or str(col_type) == 'category':
            continue
            
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
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
        else:
            df[col] = df[col].astype('category')
            
    print(f'---> Dung lượng sau khi tối ưu: {df.memory_usage().sum() / 1024**2:.2f} MB')
    return df

def load_and_preprocess_raw(raw_dir, nrows=None):
    """Hàm tải và hợp nhất dữ liệu Sales với Calendar"""
    print("1. Đang tải và ép cân dữ liệu Calendar...")
    calendar = pd.read_csv(os.path.join(raw_dir, 'calendar.csv'))
    calendar = reduce_mem_usage(calendar)

    print("\n2. Đang tải dữ liệu Sales...")
    sales = pd.read_csv(os.path.join(raw_dir, 'sales_train_validation.csv'), nrows=nrows) 

    print("\n3. Đang Melt (kéo giãn) dữ liệu Sales...")
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_melted = pd.melt(sales, id_vars=id_vars, var_name='d', value_name='demand')
    
    del sales
    gc.collect()

    print("\n4. Đang Merge Sales với Calendar...")
    df_master = pd.merge(sales_melted, calendar, on='d', how='left')
    
    del sales_melted
    del calendar
    gc.collect()

    df_master['date'] = pd.to_datetime(df_master['date'])
    
    print("\n5. Ép cân lần cuối cho DataFrame tổng...")
    df_master = reduce_mem_usage(df_master)
    
    return df_master