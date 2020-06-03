import numpy as np
import bitarray

# Function to get no of set bits in binary
# representation of positive integer n */
def countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count

def jaccard_dist(num1, num2, dim):

    numOfDiff = 0
    for p1, p2 in zip(num1, num2):
        n = p1 ^ p2
        #print("Num: {} {} {}".format(p1, p2, n))

        numOfDiff += countSetBits( n )

    return numOfDiff / float(dim)



def read_emb_file(file_path):

    with open(file_path, 'rb') as f:
        '''' 
        arr.fromfile(f)
    
        t = arr.length()
        print(arr)
        print(t)
        '''
        num_of_nodes = int.from_bytes(f.read(4), byteorder='little')
        dim = int.from_bytes(f.read(4), byteorder='little')

        print("{} {}".format(num_of_nodes, dim));

        dimInBytes = int(dim/8)
        embs = []
        for i in range(num_of_nodes):
            #embs.append(int.from_bytes( f.read(dimInBytes), byteorder='little' ))
            embs.append( [int.from_bytes(f.read(1), byteorder='little' ) for _ in range(dimInBytes)])

    return embs


file_path = "./deneme.embedding"

embs = read_emb_file(file_path)

#print(embs)
print(embs[0])
print(embs[-1])


'''
#print(embs)

print(embs[0])
print(embs[1])
#print(embs[2])

j = jaccard_dist(num1=embs[0], num2=embs[1], dim=dim)
print(j)

'''



