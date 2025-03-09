import sys
import numpy as np
import numpy as np

def forward_phase(mat): # a.k.a 가우스 소거법.
    n = len(mat[0])  # 열의 수
    m = len(mat)     # 행의 수
    x = mat.copy()   # 원본 보존을 위한 복사
    
    pivot_row = 0    # 현재 피봇 행
    
    for pivot_col in range(n):  # 각 열에 대해
        # 현재 열에서 피봇 행 아래에서 가장 큰 원소를 찾음
        max_element = abs(x[pivot_row][pivot_col])
        max_row = pivot_row
        
        for i in range(pivot_row + 1, m):
            if abs(x[i][pivot_col]) > max_element:
                max_element = abs(x[i][pivot_col])
                max_row = i
                
        # 피봇이 0이면 다음 열로
        if max_element == 0:
            continue
            
        # 최대 원소를 가진 행을 현재 피봇 행과 교환
        if max_row != pivot_row:
            x[pivot_row], x[max_row] = x[max_row].copy(), x[pivot_row].copy()
        
        # 피봇 행 아래의 모든 행에 대해 소거 수행
        for i in range(pivot_row + 1, m):
            factor = x[i][pivot_col] / x[pivot_row][pivot_col]
            x[i][pivot_col] = 0  # 명시적으로 0으로 설정
            
            for j in range(pivot_col + 1, n):
                x[i][j] -= factor * x[pivot_row][j]
                
        pivot_row += 1
        if pivot_row >= m:  # 모든 행을 처리했으면 종료
            break
            
    return x

def backward_phase(mat):
    n = len(mat[0])  # 열의 수
    m = len(mat)     # 행의 수
    x = mat.copy()   # 원본 보존을 위한 복사

    # 마지막 행부터 위로 올라가며 처리
    for pivot_row in range(m-1, -1, -1):
        # 현재 행의 피봇 열 찾기
        pivot_col = 0
        for j in range(n):
            if abs(x[pivot_row][j]) > 1e-10:  # 수치적 안정성을 위한 임계값
                pivot_col = j
                break
        else:
            continue  # 모든 원소가 0인 행은 건너뜀

        # 피봇 원소를 1로 만들기
        pivot = x[pivot_row][pivot_col]
        if abs(pivot) > 1e-10:  # 0이 아닌 경우에만 처리
            for j in range(pivot_col, n):
                x[pivot_row][j] /= pivot

        # 현재 피봇 열의 위쪽 행들을 0으로 만들기
        for i in range(pivot_row - 1, -1, -1):
            factor = x[i][pivot_col]
            for j in range(pivot_col, n):
                x[i][j] -= factor * x[pivot_row][j]

    return x
def gauss_jordan(matrix):
    matrix = forward_phase(matrix)
    matrix = backward_phase(matrix)
    return matrix
# Test the function
matrix = []
n = int(input("연립일차방정식의 행을 숫자로 넣어 주세요. \n연립일차방정식의 열은, 없는 항을 0으로 하여\n숫자로 넣어 주세요.\n(상수항 포함한 첨가 행렬 꼴일 때)"))
for i in range(n):
    inp = input().split()
    row = []
    for i in inp:
        row.append(int(i))
    matrix.append(row)


matrix = np.array(matrix, dtype=float)
print(matrix)
matrix = gauss_jordan(matrix)
rank = 0
for row in matrix:
    if not np.all(np.abs(row) < 1e-9):
        rank+=1
print("=" * 30)
print("가우스 조던 소거 후:")
print(matrix)
print("=" * 30)
print("=" * 30)
print(f"랭크: {rank}, Nullity: {matrix[0].size-rank}")

