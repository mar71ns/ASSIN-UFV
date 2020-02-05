from text_core import *
from extractor_core import *
import pydash
from pydash import intersection, pad, union, min_, find_index
#-------------Teste TextCore-------------------
f = open("vocabulary.txt", encoding="utf8")
frases = f.readlines()
#-------------Teste Extractor-------------------
# sinonimos = get_sinonimos("zzzzzzzzzzz")
antonimos = get_antonimos("aaaaaaaaa")

# i=0
# for f in frases:
#     i+=1
#     if i<2690:
#         continue
#     print("-------------\n")
#     print(pad(i,5,' '),": ",f)
#     sinonimos = get_sinonimos(f)
#     print("\n-------------")
# print("\n\n\n--------------")
