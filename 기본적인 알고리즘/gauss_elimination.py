import sys
import numpy as np
import pandas as pd

def gauss_elim(matrixToEliminate):
    step = "1"
    rowsLength, columnsLength = matrixToEliminate.shape
    for i in range(rowsLength):
        if matrixToEliminate[i][i] == 0.0:
            print("가우스 소거법이 불가능합니다. (gauss elimination is impossible)")
            print("대각성분이 0으로 보임.")
            print("행 교환으로 해결 가능한지 측정합니다.")
            for k in range(i+1, rowsLength):
                if matrixToEliminate[k][i] != 0:
                    matrixToEliminate.iloc[[i,k]] = matrixToEliminate.iloc[[k,i]].values
                    #iloc에서 대괄호를 2번 써서 [[i,k]] 하면 i행과 k행을 선택
                    break
            else:
                print("행 교환으로도 해결 불가능합니다.(tried all row interchange)")
                return matrixToEliminate, None  # Moved outside sys.exit()
            print("행 교환으로 해결 가능합니다. 가우스 소거법을 다시 진행합니다.")

        for j in range(i+1, rowsLength): # j는 다음 줄들의 인덱스
            ratio = matrixToEliminate.iloc[j, i]/matrixToEliminate.iloc[i, i]

            #주의: df.iloc[i,k]와 같이 쓰면 [i][k] 위치의 요소를 선택.
            #현재 줄의 대각선 부분을 분모로 두고 matrixToEliminate[j][i]를 분자로 두고

            for k in range(i,columnsLength):  
                matrixToEliminate.iloc[j,k] -= ratio*matrixToEliminate.iloc[i, k] #이렇게 빼 주면

            #matrixToEliminate[j][i]는 적어도 0이 된다.
            #이렇게 자기 자신 아래 열의 같은 행 번호들을 0으로 만드는 과정을 그림으로 그리면
            #왼쪽 위 -> 오른쪽 아래로 그어진 대각선 하단은 모두 0이 되는 과정도 알 수 있어진다.

        print("elimination step "+step+":")
        step = chr(ord(step)+1)
        print(matrixToEliminate)
        print("-----------------")


    print("----gauss eliminated matrixToEliminate is:")
    print(matrixToEliminate)

    x = np.zeros(rowsLength)  
    for i in range(rowsLength-1, -1, -1):
        x[i] = matrixToEliminate.iloc[i, -1]
        # 현재 일차방정식의 상수항은 구하고자 하는 미지수만 포함하지 않음
        for j in range(i+1, rowsLength):
            x[i] -= matrixToEliminate.iloc[i, j]*x[j]
            # 상수항에서 이미 구한 미지수의 해*계수만큼을 곱해줌.
        # i*a = j
        # a = j/i
        x[i] /= matrixToEliminate.iloc[i, i]
    
    return matrixToEliminate, x

# Test the function
matrixToEliminate = []
n = int(input("연립일차방정식의 계수행렬을 숫자로 넣어 주세요.\n"))
print("열의 수는 행의 수보다 1만큼 많게 넣어주세요\n")
for i in range(n):
    inp = input().split()
    if len(inp) > n+1:
        print("열의 수가 잘못되었습니니다.")
        sys.exit(1)
    row = []
    for i in inp:
        row.append(int(i))
    matrixToEliminate.append(row)

matrixToEliminate = np.array(matrixToEliminate,dtype=float)
matrixToEliminate =  pd.DataFrame(matrixToEliminate)
print("original matrixToEliminate was:")
print(matrixToEliminate)
a, b = gauss_elim(matrixToEliminate)

if b is None:
    print("elimination failed!")
else:
    val = 'a'
    for i, row in matrixToEliminate.iterrows():
        print(f"{val} = {row.iloc[-1]}")
        val = chr(ord(val) + 1)
