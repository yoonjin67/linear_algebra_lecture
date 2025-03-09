import numpy as np

A = np.array([[3, 1, -1],
              [2, 1, 5],
              [4, 5, 6]])
print("행렬 A의 고윳값과 고유벡터, 고유공간을 구해보겠습니다.")
print("행렬 A")
print("=" * 30) # 구분선 추가
print(f"{A}")

print("=" * 30)

eigenvalues, eigenvectors = np.linalg.eig(A)

print("=" * 30) # 구분선 추가
print("고윳값과 고유벡터")
print("=" * 30) # 구분선 추가

for index, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors)):
    print(f"{index+1:2d}번째 고윳값: {eigenvalue:8.3f},  고유벡터: {eigenvector}") # 칸 맞춤

print("=" * 30) # 구분선 추가
print("고유 공간 (고유벡터 행렬)")
print("=" * 30) # 구분선 추가
print(eigenvectors)
