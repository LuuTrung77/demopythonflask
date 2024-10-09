import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu
data = pd.read_csv('./UorLaptop.csv')

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X = data.iloc[:, :6]  # Lấy 6 cột thông số của laptop
y = data.iloc[:, 6]    # Cột thứ 7 là giá laptop
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Tạo danh sách các giá trị alpha
alpha_values = np.logspace(-4, 1, 100)  # Thử nghiệm với alpha từ 1e-5 đến 100

# Thiết lập GridSearchCV
ridge = Ridge()
param_grid = {'alpha': alpha_values}
grid_search = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)

# Huấn luyện mô hình
grid_search.fit(X_train, y_train)

# Lấy giá trị alpha tốt nhất
best_alpha = grid_search.best_params_['alpha']

print("Alpha tốt nhất:", best_alpha)
# Dự đoán và đánh giá mô hình với alpha tốt nhất
ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train, y_train)
y_test_pred = ridge_best.predict(X_test)

# In các chỉ số đánh giá cho tập test
print("MSE trên tập kiểm tra:", mean_squared_error(y_test, y_test_pred))
