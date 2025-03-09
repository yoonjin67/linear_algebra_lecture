import numpy as np

def get_minor_cofactor(matrix, row, col):
    """
    행렬의 (row, col) 위치의 소행렬식, 여인수, 부분행렬을 계산합니다.

    Args:
        matrix: NumPy 정방 행렬
        row: 행을 제거할 행 인덱스 (0-based)
        col: 열을 제거할 열 인덱스 (0-based)

    Returns:
        minor: 소행렬식 값
        cofactor: 여인수 값
        submatrix: 부분행렬 (NumPy 배열)
    """
    submatrix = np.delete(np.delete(matrix, row, axis=0), col, axis=1)
    minor = np.linalg.det(submatrix)
    cofactor = (-1)**(row + col) * minor
    return minor, cofactor, submatrix

def print_minors_cofactors(matrix):
    """
    행렬의 모든 소행렬식, 여인수, 부분행렬을 보기 좋게 출력합니다.

    Args:
        matrix: NumPy 정방 행렬
    """
    n_rows, n_cols = matrix.shape

    print("\n[ 소행렬식, 여인수, 부분행렬 ]")
    print("-" * 40) # 구분선 길이 늘림

    for i in range(n_rows):
        for j in range(n_cols):
            minor, cofactor, submatrix = get_minor_cofactor(matrix, i, j)
            print(f"({i+1}, {j+1}) 위치:")
            print(f"  소행렬식 = {minor:8.3f}, 여인수 = {cofactor:8.3f}") # 칸 맞춤
            print("  부분행렬:")
            print(submatrix) # 부분행렬 출력
            print("-" * 40) # 각 위치별 구분선 추가

def get_determinant_formatted(matrix):
    """
    행렬식을 계산하고 포맷팅하여 문자열로 반환합니다.

    Args:
        matrix: NumPy 정방 행렬

    Returns:
        determinant_str: 포맷팅된 행렬식 문자열
    """
    determinant = np.linalg.det(matrix)
    determinant_str = f"\n[ 행렬식 ]\n" + "-" * 40 + f"\n행렬식 값: {determinant:8.3f}" # 구분선 길이 늘림
    return determinant_str

# 하드코딩된 행렬 (3x3 예시)
matrix_A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("\n[ 입력 행렬 ]")
print("-" * 40) # 구분선 길이 늘림
print(matrix_A)

print_minors_cofactors(matrix_A) # 소행렬식, 여인수, 부분행렬 출력

determinant_output = get_determinant_formatted(matrix_A) # 행렬식 계산 및 포맷팅
print(determinant_output) # 행렬식 출력
