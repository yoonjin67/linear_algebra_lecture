import numpy as np
from fractions import Fraction

def create_unit_matrix(row, col):
    return [[1 if i == j else 0 for j in range(col)] for i in range(row)]

def is_unit_matrix(mat, tolerance=1e-6):
    n = len(mat)
    for i in range(n):
        for j in range(n):
            expected = 1 if i == j else 0
            if abs(mat[i][j] - expected) > tolerance:
                return False
    return True

def forward_phase_with_unit(mat, unit):
    n = len(mat)
    x = [row[:] for row in mat]
    u = [row[:] for row in unit]
    
    for i in range(n):
        # 피봇 선택
        pivot = x[i][i]
        if abs(pivot) < 1e-10:
            max_row = i
            max_val = abs(pivot)
            for k in range(i + 1, n):
                if abs(x[k][i]) > max_val:
                    max_val = abs(x[k][i])
                    max_row = k
            if max_val < 1e-10:
                return None, None
            x[i], x[max_row] = x[max_row][:], x[i][:]
            u[i], u[max_row] = u[max_row][:], u[i][:]
            pivot = x[i][i]

        # 현재 행 정규화
        for j in range(n):
            x[i][j] /= pivot
            u[i][j] /= pivot
        
        # 다른 행들 소거
        for k in range(n):
            if k != i:
                factor = x[k][i]
                for j in range(n):
                    x[k][j] -= factor * x[i][j]
                    u[k][j] -= factor * u[i][j]
    
    return x, u

def matrix_multiply(mat1, mat2, tolerance=1e-10):
    n = len(mat1)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sum_val = 0
            for k in range(n):
                sum_val += mat1[i][k] * mat2[k][j]
            if abs(sum_val) < tolerance:  # 작은 값들은 0으로 처리
                result[i][j] = 0
            else:
                result[i][j] = sum_val
    return result

def create_inverse_matrix(mat):
    if len(mat) != len(mat[0]):
        raise ValueError("정사각행렬이 아닙니다")
    
    n = len(mat)
    unit = create_unit_matrix(n, n)
    result_mat, result_unit = forward_phase_with_unit(mat, unit)
    
    if result_mat is None:
        return None
        
    return result_unit

# 메인 실행 코드
if __name__ == "__main__":
    print("행렬의 크기를 입력하세요: ", end='')
    n = int(input())
    
    print(f"{n}x{n} 행렬의 각 행을 공백으로 구분하여 입력하세요:")
    matrix = []
    for _ in range(n):
        row = list(map(float, input().split()))
        if len(row) != n:
            raise ValueError(f"입력된 열의 개수가 {n}개가 아닙니다")
        matrix.append(row)

    print("\n입력된 행렬:")
    print(np.array(matrix))

    try:
        inv_mat = create_inverse_matrix(matrix)
        if inv_mat is None:
            print("\n역행렬이 존재하지 않습니다.")
        else:
            orig = np.array(matrix)
            inverse = np.array(inv_mat)
            print("\n역행렬:")
            print(inverse)
            # 검증
            result = matrix_multiply(matrix, inv_mat)
            print("\n원래 행렬과 역행렬의 곱:")
            print(np.array(result))
            
            if is_unit_matrix(result):
                print("\n역행렬이 정상적으로 계산되었습니다.")
            else:
                print("\n계산된 역행렬이 정확하지 않습니다.")
                
    except Exception as e:
        print(f"\n오류 발생: {e}")
