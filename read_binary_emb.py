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



def read_binary_emb_file(file_path):

    def _int2boolean(num):

        binary_repr = []
        for _ in range(8):

            binary_repr.append(True if num % 2 else False )
            num = num >> 1

        return binary_repr[::-1]


    with open(file_path, 'rb') as f:

        num_of_nodes = int.from_bytes(f.read(4), byteorder='little')
        dim = int.from_bytes(f.read(4), byteorder='little')

        embs = []

        dimInBytes = int(dim / 8)

        for i in range(num_of_nodes):
            # embs.append(int.from_bytes( f.read(dimInBytes), byteorder='little' ))
            emb = []
            for _ in range(dimInBytes):
                emb.extend(_int2boolean(int.from_bytes(f.read(1), byteorder='little')))

            embs.append(emb)

    return np.asarray(embs, dtype=bool)


file_path = "./karate.embedding"

embs = read_binary_emb_file(file_path)

print("Number of nodes: {} Dimension: {}".format(embs.shape[0], embs.shape[1]))
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



