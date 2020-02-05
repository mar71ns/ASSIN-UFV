import nltk
from assin_text_pre_process import *
from text_core import *
from extractor_core import *
import pydash
from pydash import intersection, pad, union, min_, set_, get, find_index, values, flatten_deep, snake_case

nome_arquivos = [
    'assin-ptbr-train',
    'assin-ptpt-train',
    'assin-ptbr-test',
    'assin-ptpt-test',
    'assin-ptpt-dev',
    'assin2-train-only',
    'assin2-dev',
    'assin2-blind-test',
    'assin-ptbr-dev',
]

for nome_arquivo in nome_arquivos:
    f = open( nome_arquivo+".txt", encoding="utf8")
    frases = f.readlines()
    print(nome_arquivo)
    relatorio_data={}
    data_values=[]
    for i in range (0, int((len(frases)/4))):
        ind = i * 4
        f_index = i + 1
        f1 = frases[ind]
        f2 = frases[ind+1]
        print(f_index," (", round(f_index/int((len(frases)/3)),2)," %)")
        res = process_2_sents(f1,f2)
        set_(relatorio_data, f_index, res)
        data_values.append(values(res)[2:])
        #break
    fname = nome_arquivo + "-processed"
    load_json(fname)
    save_json(fname, relatorio_data)
