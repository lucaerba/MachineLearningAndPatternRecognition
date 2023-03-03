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

#tuples
t = (1, 2, 3, 'A')
print(t)
print(t[3])

l = list("Hello")
r = list(range(len(l)))

for i in range(len(l)):
    print(i, l[i])

lE = list(enumerate(l))
print(lE)
i, val = lE[0]
print(i,val)

#dict
a = {'A':1, 'B':2, 'C':3}
print(a)
print(a['B'])

a = {}
a['d'] = 5
a['f'] = 6
print(a)

for key in a: #itera le chiavi
    print(key)
for key in a.keys(): #itera le chiavi
    print(key)
for value in a.values(): #itera i valori
    print(value)
