import numpy
def fun():
    a = 2
    print(a)
    
def fun2(a=3, b=2):
    print([a, " " , b])

def fun3(*a, **kw):
    print(kw)
    print(a)

l = [[1,'c'],[2,'a'],[3,'b']]
print(l)
l = sorted(l, key = lambda x: x[1])
print(l)
t = (1, 2, 3, 'A')
print(t)
print(t[3])