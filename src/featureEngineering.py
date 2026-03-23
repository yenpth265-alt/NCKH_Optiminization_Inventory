import pandas as pd
import numpy as np
import os
import gc

def process_price_features(df, raw_dir):
    """Hợp nhất và tạo đặc trưng Giá cả (Price Features)"""
    print("-> Đang load dữ liệu Giá (sell_prices.csv)...")
    prices = pd.read_csv(os.path.join(raw_dir, 'sell_prices.csv'))
    
    print("-> Đang hợp nhất và tính toán độ co giãn giá...")
    df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    del prices
    gc.collect()

    df['price_max'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
    df['price_discount'] = df['sell_price'] / df['price_max']
    df['price_momentum'] = df['sell_price'] / df.groupby(['store_id', 'item_id'])['sell_price'].shift(1)
    
    # Nới lỏng float32 để tránh lỗi hiển thị RuntimeWarning
    cols_float = ['sell_price', 'price_max', 'price_discount', 'price_momentum']
    for col in cols_float:
        df[col] = df[col].astype(np.float32)
        
    return df

def process_calendar_events(df):
    """Xử lý Missing Value và mã hóa Sự kiện (Label Encoding)"""
    print("-> Đang xử lý Sự kiện lễ tết (Calendar Events)...")
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    
    for col in event_cols:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.add_categories(['No_Event'])
        df[col] = df[col].fillna('No_Event')

    cols_to_encode = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] + event_cols
    for col in cols_to_encode:
        df[col] = df[col].astype('category').cat.codes
        
    return df

def process_time_features(df):
    """Tạo đặc trưng Thời gian"""
    print("-> Đang tạo Đặc trưng Thời gian (Time Features)...")
    df['day'] = df['date'].dt.day
    df['is_weekend'] = df['wday'].isin([1, 2]).astype(np.int8)
    return df

def generate_all_features(df, raw_dir):
    """Hàm Pipeline chạy tổng hợp toàn bộ các bước Feature Engineering"""
    df = process_price_features(df, raw_dir)
    df = process_calendar_events(df)
    df = process_time_features(df)
    return df

def create_lag_features_for_item(df_item):
    """Tạo Lag và Rolling cho một mặt hàng cụ thể trước khi đưa vào ML"""
    df_item = df_item.sort_values('date').reset_index(drop=True)
    
    df_item['lag_7'] = df_item['demand'].shift(7)
    df_item['lag_14'] = df_item['demand'].shift(14)
    df_item['rolling_mean_7'] = df_item['demand'].shift(1).rolling(7).mean()
    
    cols_to_check = ['lag_7', 'lag_14', 'rolling_mean_7']
    df_item = df_item.dropna(subset=cols_to_check)
    
    return df_item


def classify_demand(group):
    """Hàm tính ADI, CV2 và phân loại cho từng sản phẩm"""
    demand = group['demand'].values
    if demand.sum() == 0:
        return pd.Series({'ADI': np.nan, 'CV2': np.nan, 'Category': 'No Sales'})
    
    #Chỉ tính bắt đầu từ ngày đầu tiên có sales
    non_zero_indices = np.where(demand > 0)[0]
    first_sale_idx = non_zero_indices[0]
    active_demand = demand[first_sale_idx:]
    
    days_with_sales = len(active_demand[active_demand > 0])
    adi = len(active_demand) / days_with_sales if days_with_sales > 0 else np.nan
    
    #Lưu ý CV2 chỉ tính trên các ngày có bán hàng (demand >0)
    non_zero_demand = active_demand[active_demand > 0]
    mean_demand = np.mean(non_zero_demand)
    cv2 = (np.std(non_zero_demand) / mean_demand) ** 2 if mean_demand > 0 else np.nan
        
    if adi <= 1.32 and cv2 <= 0.49:
        category = 'Smooth (Đều đặn)'
    elif adi <= 1.32 and cv2 > 0.49:
        category = 'Erratic (Thất thường)'
    elif adi > 1.32 and cv2 <= 0.49:
        category = 'Intermittent (Ngắt quãng)'
    else:
        category = 'Lumpy (Cục bộ)'
        
    return pd.Series({'ADI': round(adi, 2), 'CV2': round(cv2, 2), 'Category': category})