import random
from scipy.stats import binom

def head_tail(n):
    head = 0 #tura
    tail = 0 #yazı
    for _ in range(n): 
        val = random.randint(0, 1)
        if val == 0:
            head +=1
        else:
            tail += 1
    return head / n, tail / n

#for i in range(1, 100):
#    hr, tr = head_tail(i * 10000)
#    print(f'tura oranı : {hr} , yazı oranı : {tr}')

hr, tr = head_tail(5)
print(f'tura oranı : {hr} , yazı oranı : {tr}')

# 5 kere para atıldığında 3 kere tura gelme olasılığı with binom
res = binom.pmf(3, 5, 0.5)

print(res)
