import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, affine_transform
from PIL import Image

# 이미지 파일 이름
image_path = "frog.png"

try:
    # 이미지 로드
    original_frog = Image.open(image_path)
    frog_array = np.array(original_frog)

    # 이미지의 rank 확인
    input_rank = frog_array.ndim

    # 1. 회전 변환 (0.5 라디안, 반시계 방향)
    angle_rad = np.degrees(0.5)  # 라디안을 각도로 변환
    rotated_frog = rotate(frog_array, angle_rad, reshape=False)

    # 2. 좌우 확장 변환
    scale_x = 2
    scaled_frog = original_frog.resize((original_frog.width * scale_x, original_frog.height))
    scaled_frog_array = np.array(scaled_frog)

    # 3. 시어링 변환 (x축 시어링 계수 0.5)
    shear_factor_x = 0.5
    # 시어링 매트릭스 생성 (homogeneous coordinates 고려)
    shear_matrix_x = np.array([[1, shear_factor_x, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    # 변환 중심 계산
    center_y, center_x = np.array(frog_array.shape[:2]) / 2.0
    # 오프셋 계산 (컬러 이미지이므로 3차원 오프셋)
    offset = [-center_x * shear_factor_x, 0, 0]

    # 어파인 변환 적용
    sheared_frog = affine_transform(frog_array, shear_matrix_x, offset=offset)

    # 시각화
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 서브플롯 1: 원본 이미지
    axes.flat[0].imshow(original_frog)
    axes.flat[0].set_title("Original Frog")
    axes.flat[0].axis('off')

    # 서브플롯 2: 회전된 이미지
    axes.flat[1].imshow(rotated_frog)
    axes.flat[1].set_title("Rotated Frog (0.5 Radian)")
    axes.flat[1].axis('off')

    # 서브플롯 3: 확장된 이미지
    axes.flat[2].imshow(scaled_frog_array)
    axes.flat[2].set_title("Horizontally Scaled Frog (x2)")
    axes.flat[2].axis('off')

    # 서브플롯 4: 시어링된 이미지
    axes.flat[3].imshow(sheared_frog)
    axes.flat[3].set_title(f"Sheared Frog (x={shear_factor_x})")
    axes.flat[3].axis('off')

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: {image_path} 파일을 찾을 수 없습니다. 현재 작업 디렉토리에 파일이 있는지 확인해주세요.")
except ImportError as e:
    print(f"Error: 필요한 라이브러리를 불러오는 데 실패했습니다: {e}")
    print("numpy, matplotlib, scipy, Pillow가 설치되어 있는지 확인해주세요.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
