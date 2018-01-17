import random

def randomLength(x, y):
    len1 = random.randint(x, y)
    len2 = random.randint(x, y)
    return (len1, len2)
    
def randomSequence(size):
    return ''.join(random.choice("ACTG") for _ in range(size))

for i in range(2, 6):
    print(i-1)
    file1 = open("synthData"+str(i-1)+"_str1.txt", "w")
    file2 = open("synthData"+str(i-1)+"_str2.txt", "w")
    len1, len2 = randomLength(10**i, 10**(i+1))
    str1 = randomSequence(len1)
    str2 = randomSequence(len2)
    file1.write(str1)
    file2.write(str2)
    file1.close()
    file2.close()
    
