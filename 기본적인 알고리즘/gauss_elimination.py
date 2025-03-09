import sys
import numpy as np

def gauss_elim(matrix):
    step = "1"
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] == 0.0:
            print("가우스 소거법이 불가능합니다. (gauss elimination is impossible)")
            print("행 교환으로 해결 가능한지 측정합니다.")
            for k in range(i+1, n):
                if matrix[k][i] != 0:
                    matrix[i], matrix[k] = matrix[k], matrix[i]
                    break
            else:
                print("행 교환으로도 해결 불가능합니다.(tried all row interchange)")
                return matrix, None  # Moved outside sys.exit()

        for j in range(i+1, n): # j는 다음 줄들의 인덱스
            ratio = matrix[j][i]/matrix[i][i]
            #현재 줄의 대각선 부분을 분모로 두고 matrix[j][i]를 분자로 두고
            for k in range(i,len(matrix[0])):  
                matrix[j][k] -= ratio*matrix[i][k] #이렇게 빼 주면
            #matrix[j][i]는 적어도 0이 된다.
            #이렇게 자기 자신 아래 열의 같은 행 번호들을 0으로 만드는 과정을 그림으로 그리면
            #왼쪽 위 -> 오른쪽 아래로 그어진 대각선 하단은 모두 0이 되는 과정도 알 수 있어진다.
        print("elimination step "+step+":")
        step = chr(ord(step)+1)
        print(matrix)
        print("-----------------")


    print("----gauss eliminated matrix is:")
    print(matrix)

    x = np.zeros(n)  
    for i in range(n-1, -1, -1):
        x[i] = matrix[i][-1]
        # 현재 일차방정식의 상수항은 구하고자 하는 미지수만 포함하지 않음
        for j in range(i+1, n):
            x[i] -= matrix[i][j]*x[j]
            # 상수항에서 이미 구한 미지수의 해*계수만큼을 곱해줌.
        # i*a = j
        # a = j/i
        x[i] /= matrix[i][i]
    
    return matrix, x

# Test the function
matrix = []
n = int(input("연립일차방정식의 행을 숫자로 넣어 주세요. 연립일차방정식의 열은, 없는 항을 0으로 하여 숫자로 넣어 주세요.\n(상수항 포함한 첨가 행렬 꼴일 때)"))
for i in range(n):
    inp = input().split()
    row = []
    for i in inp:
        row.append(int(i))
    matrix.append(row)


matrix = np.array(matrix, dtype=float)
print("original matrix was:")
print(matrix)
a, b = gauss_elim(matrix)

if b is None:
    print("elimination failed!")
else:
    val = 'a'
    for i, value in enumerate(b):
        print(f"{val} = {value}")
        val = chr(ord(val) + 1)
