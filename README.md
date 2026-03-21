# 📦 Smart Inventory Optimizer  
### Hệ thống Tối ưu hóa Nhập hàng bằng Machine Learning & Vận trù học

---

## 📌 Giới thiệu

**Smart Inventory Optimizer** là dự án Nghiên cứu Khoa học tập trung vào:

> Ứng dụng **Machine Learning** và **Operations Research (Vận trù học)**  
để **dự báo nhu cầu** và **tối ưu hóa quản trị tồn kho bán lẻ**.

Dự án sử dụng dữ liệu thực tế từ cuộc thi **M5 Forecasting - Accuracy (Walmart)** trên Kaggle để xây dựng một pipeline tự động:

```
Dữ liệu thô 
→ Tiền xử lý 
→ Phân loại nhu cầu 
→ Dự báo AI (LightGBM) 
→ Mô phỏng Monte Carlo 
→ Tối ưu ROP & EOQ
```

---

## 🎯 Mục tiêu

- Dự báo nhu cầu hàng hóa theo thời gian
- Phân loại hành vi tiêu dùng (Demand Pattern)
- Tối ưu:
  - **ROP (Reorder Point)** – Điểm đặt hàng lại
  - **EOQ (Economic Order Quantity)** – Lượng đặt hàng tối ưu
- Hỗ trợ ra quyết định tồn kho theo dữ liệu thực

---

## 📂 Cấu trúc thư mục

```
NCKH_OPTIMINIZATION_INVENTORY/
│
├── dataset/
│   ├── raw/                 
│   │   └── (calendar.csv, sales_train, sell_prices...)
│   └── processed/           
│       └── (file .parquet sau xử lý)
│
├── docs/                    
│   └── Tài liệu, báo cáo, biểu đồ
│
├── src/                     
│   ├── dataPreprocessing.py     
│   └── featureEngineering.py    
│
├── notebooks/               
│   ├── 02_demand_classification.ipynb   
│   └── 03_machine_learning.ipynb        
│
├── app.py                   
├── requirements.txt         
└── README.md                
```

---

## ⚙️ Cài đặt môi trường

### 1. Yêu cầu

- Python **>= 3.8**
- Khuyến nghị: **Anaconda / Miniconda**

---

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 📦 Thư viện chính

- pandas, numpy
- matplotlib, seaborn
- lightgbm
- pyarrow
- statsmodels
- streamlit

---

## 🚀 Hướng dẫn chạy dự án

---

### 🔹 Cách 1: Chạy Web App (Demo trực quan)

```bash
python -m streamlit run app.py
```

👉 Trình duyệt sẽ mở dashboard cho phép:

- Chọn cửa hàng & sản phẩm
- Điều chỉnh:
  - Lead Time
  - Chi phí lưu kho
- Xem:
  - Dự báo AI
  - Biểu đồ tồn kho
  - Kết quả tối ưu ROP & EOQ

---

### 🔹 Cách 2: Chạy môi trường Nghiên cứu

---

### 📥 Bước 0: Chuẩn bị dữ liệu

- Tải dataset từ Kaggle:
  - `m5-forecasting-accuracy.zip`
- Copy các file `.csv` vào:

```
dataset/raw/
```

---

### 🧪 Bước 1: Phân loại nhu cầu

Chạy notebook:

```
notebooks/02_demand_classification.ipynb
```

#### Nội dung:

- Tiền xử lý dữ liệu
- Phân loại bằng phương pháp **Syntetos–Boylan**

| Nhóm | Ý nghĩa |
|------|--------|
| Smooth | Nhu cầu ổn định |
| Erratic | Biến động cao |
| Intermittent | Gián đoạn |
| Lumpy | Không đều + biến động |

#### Phân tích thêm:

- ACF (AutoCorrelation)
- PACF (Partial AutoCorrelation)

---

### 🤖 Bước 2: Dự báo & Tối ưu tồn kho

Chạy notebook:

```
notebooks/03_machine_learning.ipynb
```

#### Pipeline:

1. **ML Routing**
   - Tự động chọn model:
     - LightGBM Regression
     - Tweedie (cho dữ liệu sparse)

2. **Recursive Forecasting**
   - Dự báo cuốn chiếu nhiều bước

3. **Monte Carlo Simulation**
   - Mô phỏng hàng ngàn kịch bản tồn kho

---

### 📊 Kết quả đầu ra

- Tồn kho hiện tại
- EOQ tối ưu
- ROP tối ưu
- Biểu đồ tồn kho theo thời gian

---

## 🧠 Công nghệ sử dụng

### 📊 Data Processing
- Pandas, NumPy
- PyArrow (tối ưu RAM)

### 🤖 Machine Learning
- LightGBM
- Tweedie Loss (xử lý dữ liệu thưa)

### 📈 Time Series
- Statsmodels (ACF/PACF)
- Feature Engineering (Lag, Rolling)

### 📦 Operations Research
- Monte Carlo Simulation
- Grid Search tối ưu ROP & EOQ

### 🌐 Deployment
- Streamlit Dashboard

---
