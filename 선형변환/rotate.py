import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# 원본 이미지 (3x3)
original_image = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

# 회전 각도 (135도를 라디안으로 변환)
angle_degrees = 135
angle_radians = np.deg2rad(angle_degrees)

# scipy.ndimage.rotate 함수를 사용하여 이미지 회전 (각도 단위로 변환된 라디안 값 적용)
rotated_image_radians = rotate(original_image, np.degrees(angle_radians), reshape=False)

# 결과 시각화 (라디안으로 계산된 회전)
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(rotated_image_radians, cmap='gray')
plt.title(f'Rotated Image ({angle_radians:.2f} radians)')

plt.tight_layout()
plt.show()

print(f"\n회전 후 이미지 ({angle_radians:.2f} radians -> {angle_degrees} degrees, 근사):\n{np.round(rotated_image_radians, 2)}")
