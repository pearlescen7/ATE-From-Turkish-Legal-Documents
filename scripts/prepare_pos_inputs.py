adj = 4
end = 7

for i in range(4):
    for j in range(adj, end):
        if j+i < end:
            print(j, j+i+1)