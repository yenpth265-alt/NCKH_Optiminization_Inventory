# 📦 Smart Inventory Optimizer (Hệ thống Tối ưu hóa Nhập hàng)

Dự án Nghiên cứu Khoa học: **Ứng dụng Machine Learning và Vận trù học trong Dự báo Nhu cầu và Tối ưu hóa Quản trị Tồn kho bán lẻ.**

Dự án này sử dụng tập dữ liệu thực tế từ cuộc thi [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) (Walmart) trên Kaggle để xây dựng một luồng xử lý (pipeline) tự động: Từ dữ liệu thô $\rightarrow$ Phân loại nhu cầu $\rightarrow$ Dự báo AI (LightGBM) $\rightarrow$ Mô phỏng kịch bản đặt hàng để tìm ra ROP (Reorder Point) và EOQ (Economic Order Quantity) tối ưu.

---

## 📂 Cấu trúc Thư mục

Để dự án hoạt động trơn tru, hãy đảm bảo cấu trúc thư mục của bạn như sau:

```
NCKH_OPTIMINIZATION_INVENTORY/
│
├── dataset/
│   ├── raw/                 # Chứa 5 file .csv tải từ Kaggle (calendar, sales_train, sell_prices...)
│   └── processed/           # Chứa các file .parquet sinh ra sau khi chạy code
│
├── docs/                    # Tài liệu đặc tả dự án, báo cáo 
│
├── notebooks/               # Mã nguồn chính (Jupyter Notebooks)
│   ├── 01_data_prep_and_lag.ipynb       # Ép cân, làm sạch dữ liệu và vẽ Lag Chart
│   ├── 02_demand_classification.ipynb   # Phân loại nhóm nhu cầu bằng Toán học (ADI & CV2)
│   ├── 02b_feature_engineering.ipynb    # Trích xuất đặc trưng (Giá cả, Sự kiện, Chu kỳ)
│   └── 03_machine_learning.ipynb        # Train LightGBM & Chạy Mô phỏng Tồn kho
│
├── requirements.txt         # Danh sách thư viện Python cần thiết
└── README.md                # File hướng dẫn này
```

## 🛠️ Cài đặt Môi trường

**1. Yêu cầu hệ thống:**
* Python 3.8 trở lên.
* Khuyên dùng Anaconda hoặc Miniconda.

**2. Cài đặt thư viện:**
Mở Terminal / Command Prompt tại thư mục dự án và chạy lệnh sau:

```
pip install -r requirements.txt
```

(Các thư viện chính bao gồm: pandas, numpy, matplotlib, seaborn, lightgbm, pyarrow, scikit-learn, statsmodels)

## 🚀 Hướng dẫn Chạy Dự án

### Bước 1: Chuẩn bị Dữ liệu
* **Tải dữ liệu:** Tải file `m5-forecasting-accuracy.zip` từ Kaggle, giải nén và copy các file `.csv` (`calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv`) vào thư mục `dataset/raw/`.
* **Chạy File:** Mở và chạy toàn bộ cell trong `01_data_prep_and_lag.ipynb`.
* **Kết quả:** Code sẽ "ép cân" dữ liệu (giảm 70% RAM) và sinh ra file `master_data.parquet` trong thư mục `processed/`.

### Bước 2: Feature Engineering (Trích xuất đặc trưng)
* **Chạy File:** Mở và chạy `02b_feature_engineering.ipynb`.
* **Mô tả:** Hệ thống sẽ hợp nhất dữ liệu giá cả, xử lý các ngày lễ tết, mã hóa biến phân loại và sinh ra các đặc trưng trượt/trễ (Lag/Rolling).
* **Kết quả:** Sinh ra file `featured_data.parquet` sẵn sàng cho AI.

### Bước 3: Phân loại Nhu cầu (Phân tích)
* **Chạy File:** Mở `02_demand_classification.ipynb`.
* **Mô tả:** Sử dụng thuật toán ma trận Syntetos-Boylan (tính ADI và CV²) để chia các mặt hàng thành 4 nhóm: *Smooth*, *Erratic*, *Intermittent*, *Lumpy*.

### Bước 4: Dự báo AI và Mô phỏng Tồn kho (Cốt lõi)
* **Chạy File:** Mở và chạy `03_machine_learning.ipynb`.
* **Mô tả:** * Huấn luyện mô hình LightGBM (Objective: Tweedie) để dự báo nhu cầu 28 ngày tiếp theo.
  * Chạy mô phỏng hàng ngàn kịch bản đặt hàng để tìm mức chi phí thấp nhất.
* **Kết quả:** In ra Báo cáo Hành động (Tồn kho hiện tại, Số lượng cần nhập, Điểm đặt hàng ROP) và các biểu đồ trực quan.

---

## 🧠 Công nghệ Sử dụng
* **Data Processing:** Pandas, Numpy, PyArrow.
* **Machine Learning:** LightGBM (Tweedie distribution for sparse data).
* **Visualization:** Matplotlib, Seaborn.
* **Optimization:** Grid Search Simulation.