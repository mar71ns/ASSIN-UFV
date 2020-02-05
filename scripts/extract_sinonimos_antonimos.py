from text_core import *
from extractor_core import *
import pydash
from pydash import intersection, pad, union, min_, find_index
#-------------Teste TextCore-------------------

f = open("vocabulary.txt", encoding="utf8")
frases = f.readlines()

i=0
for f in frases:
    #break
    i+=1
    if i<118000:
        continue
    fail = True
    print("===============\n")
    print(pad(i,5,' '),": ",f)
    while fail:
        try:
            sinonimos = get_sinonimos(f)
            print("\n-------------")
            antonimos = get_antonimos(f)
            fail = False
        except Exception as e:
            pass
    print("\n===============")
print("\n\n\n--------------")
