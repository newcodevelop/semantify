a = {"name": "dibyanayan", "id": 101, "marks": {"english": 70, "math": 100, "sciences": 90}}

# lokvs = a.split(',')[0:2]

# lokvs_ = a.split(',')


# nested = lokvs_[2].strip().split(':')

# print(lokvs)

# print(nested)

# for i in lokvs:
#     nested = i.split(',')
#     if len(nested)


import json

json.dump(a, open("./test_file.json", "w"))

k = json.load(open("./test_file.json", "r"))


["I", "am", "raining", "it", "is", "was"]

k = "It was raining"


vec(k) = [0,0,1,1,0,1]

op = 1,2,3,4,5,6
gt = 2,3,6,7,8

2/4 = 0.5



[1,2], [2,3], [3,4], [4,5], [5,6]

gt = [2,3], [3,6], [6,7], [7,8]

bleu-2 = 1/5

print(k)

r_gt (1*768)
r_gen (1*768)

1,2,3,4

exp( logP(2|1) * logP(3|1,2) * log() )


"It is raining"

P ("The boys are playing") = 2-3


1->2->5
1->3->5
1->4->5

1, [2,3,4], [5]


V,E

for i in V:
    if in_degree(i)==0:
        schedule(i) = 1
        out_degree(i) = 0
        delete_edge(i,j) for all (i,j) in E
        in_degree(i,j) = 0 for all (i,j) in E
        
    if schedule(i):
        continue



