arr = [ [1,2,3],
        [4,5,6],
        [7,8,9]
       ]
print("matrix")
for row in arr:
    for col in row:
        print(col, end = " ")
    print()
print("transpose")
arr = list(map(list,zip(*arr)))
for row in arr:
    for col in row:
        print(col, end = " ")
    print()
