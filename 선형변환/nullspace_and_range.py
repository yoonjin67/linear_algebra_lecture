import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def linear_transform_3d_to_2d(matrix, vector):
    """3차원 벡터를 2차원 벡터로 선형 변환합니다."""
    return np.dot(matrix, vector)[:2]  # 처음 2개 요소만 반환

# 변환 행렬 정의 (z축 정보를 소실시켜 xy 평면에 투영)
transform_matrix_3d_to_2d = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 0]])

# 원래 벡터 공간 (3차원 공간의 격자) 생성
u = np.linspace(-2, 2, 3)  # 핵을 더 잘 보이게 격자 간격 줄임
v = np.linspace(-2, 2, 3)
w = np.linspace(-2, 2, 5)  # z축 벡터를 더 많이 생성
xv, yv, zv = np.meshgrid(u, v, w)
vectors_3d = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))

# 각 3차원 벡터에 선형 변환 적용하여 2차원 벡터 얻음
transformed_vectors_2d = np.array([linear_transform_3d_to_2d(transform_matrix_3d_to_2d, vec) for vec in vectors_3d.T]).T

# 핵 시각화: z축 상의 벡터들을 빨간색으로 강조
# z축 상의 벡터는 x=0, y=0 인 벡터들입니다.
null_space_indices = np.logical_and(vectors_3d[0, :] == 0, vectors_3d[1, :] == 0)
null_space_vectors_3d = vectors_3d[:, null_space_indices]

# 치역 시각화: 변환된 2차원 벡터들이 이루는 공간 (xy 평면)
range_x_2d = transformed_vectors_2d[0, :]
range_y_2d = transformed_vectors_2d[1, :]

# 시각화
fig = plt.figure(figsize=(12, 6))

# 1. 원래 3차원 벡터 공간
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# 모든 원래 벡터는 회색으로 표시
ax1.quiver(np.zeros(vectors_3d.shape[1]), np.zeros(vectors_3d.shape[1]), np.zeros(vectors_3d.shape[1]),
           vectors_3d[0, :], vectors_3d[1, :], vectors_3d[2, :],
           length=0.3, normalize=False, alpha=0.2, color='gray')
ax1.scatter(0, 0, 0, color='black', s=50, label='Origin')
# 핵 (z축) 벡터는 빨간색으로 표시
if null_space_vectors_3d.size > 0:
    ax1.quiver(np.zeros(null_space_vectors_3d.shape[1]), np.zeros(null_space_vectors_3d.shape[1]), np.zeros(null_space_vectors_3d.shape[1]),
               null_space_vectors_3d[0, :], null_space_vectors_3d[1, :], null_space_vectors_3d[2, :],
               length=0.5, normalize=False, color='red', linewidth=2, label='Null Space (z-axis)')
ax1.set_xlim([-2, 2])
ax1.set_ylim([-2, 2])
ax1.set_zlim([-2, 2])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Original 3D Vector Space')
ax1.legend()

# 2. 변환된 2차원 벡터 공간 (치역)
ax2 = fig.add_subplot(1, 2, 2)
ax2.quiver(np.zeros(transformed_vectors_2d.shape[1]), np.zeros(transformed_vectors_2d.shape[1]),
           transformed_vectors_2d[0, :], transformed_vectors_2d[1, :],
           angles='xy', scale_units='xy', scale=1, alpha=0.5, color='blue')
ax2.scatter(0, 0, color='black', s=50, label='Origin')
ax2.scatter(range_x_2d, range_y_2d, color='green', s=20, label='Range (xy-plane)')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
ax2.set_xlabel('x\'')
ax2.set_ylabel('y\'')
ax2.set_title('Transformed 2D Vector Space (Range)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print("\nTransformation Matrix (3D to 2D):")
print(transform_matrix_3d_to_2d)
print("\nNull Space Vectors (Original 3D Space - z-axis):")
print(null_space_vectors_3d)
print("\nRange (Transformed 2D Space):")
print("Spanned by the first two columns of the transformation matrix (xy-plane)")
