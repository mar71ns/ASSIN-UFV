#Recupera todos os arquivos XML da pasta e salva os valores em arquivo txt.

import glob

import xml.etree.ElementTree as ET

for filename in glob.glob('*.xml'):
    print(filename)
    f = open(filename, 'r', encoding="utf8")
    xml = ET.parse(f)
    e = ET.ElementTree(xml)
    f_out = open(filename+".txt",'w')
    for elt in e.iter():
#print ("%s" % (elt.text))
        f_out.write(elt.text+"\n")
    f_out.close()
    f.close()
