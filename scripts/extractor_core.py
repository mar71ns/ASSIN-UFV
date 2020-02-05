# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from collections import defaultdict
import requests
import json
import re
import string
import os.path
import pydash
from pydash import set_, get, has, deburr, push, union, snake_case, kebab_case, lower_case, find_index

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------RELATÓRIO---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#######################################################################
##################------RETORNA LISTA DE SINÔNIMOS------###############
#######################################################################
def get_sinonimos(palavra):
    # print("-------------")
    # print(palavra)
    palavra = deburr(kebab_case(palavra))
    base_url="https://www.sinonimos.com.br/"

    list_no_occur = get(nao_consta,"sinonimos")

    if has(sinonimos,palavra):
        #print("Palavra já consta no dicionário de sinônimos")
        return get(sinonimos,palavra)
    if palavra in list_no_occur:
        #print("Não foram encontrados sinônimos para a palavra")
        return []
    else:
        #print("\nNova Palavra: ", palavra)
        request_fail = True
        while request_fail:
            try:
                site = requests.get(base_url + palavra)
                request_fail=False
            except Exception as e:
                print (e)
                pass
        data = BeautifulSoup(site.content,'html.parser')

        try:
            h1 = data.find('h1').getText()
        except Exception as e:
            if palavra is not None:
                list_no_occur = push(list_no_occur,palavra)
            set_(nao_consta,"sinonimos",list_no_occur)
            save_json("nao_consta",nao_consta)
            return []

        if(h1=="Página Não Encontrada"):
            #print(h1)
            if palavra is not None:
                list_no_occur = push(list_no_occur,palavra)
            set_(nao_consta,"sinonimos",list_no_occur)
            save_json("nao_consta",nao_consta)
            return []
        else:
            content = data.find('div', attrs={'id':'content'})
            try:
                div = content.findAll('div', attrs={'class':'s-wrapper'})
            except Exception as e:
                print(e)
                if palavra is not None:
                    list_no_occur = push(list_no_occur,palavra)
                set_(nao_consta,"sinonimos",list_no_occur)
                save_json("nao_consta",nao_consta)
                return []
            aux=0
            for sentido in div:
                aux=aux+1
                lista_sinonimos = []
                try:
                    try:
                        key = lower_case(sentido.find('div', attrs={'class':'sentido'}).getText().strip(":"))
                    except Exception as e:
                        print(e)
                        key = "sinonimos"+str(aux)
                        pass
                    values = sentido.findAll('a', attrs={'class':'sinonimo'}, text=True)
                    values2 = sentido.findAll('span')
                    # print(values2)
                    all_values = union(values, values2)
                    #print(all_values)
                    for value in all_values:
                        lista_sinonimos.append(value.getText().strip(":"))
                    set_(sinonimos,palavra+"."+key,lista_sinonimos)
                    print("Sinônimo Salv@ no Dicionário")
                except Exception as e:
                    print("\nError:\n"+ str(e))
                    return []
            save_json("sinonimos",sinonimos)
            return get(sinonimos,palavra)

#######################################################################
##################------RETORNA LISTA DE ANTÔNIMOS------###############
#######################################################################
def get_antonimos(palavra):
    # print("-------------")
    # print(palavra)
    palavra = deburr(kebab_case(palavra))
    base_url="https://www.antonimos.com.br/"

    list_no_occur = get(nao_consta,"antonimos")

    if has(antonimos,palavra):
        #print("Palavra já consta no dicionário de antônimos")
        return get(antonimos,palavra)
    if palavra in list_no_occur:
        #print("Não foram encontrados antônimos para a palavra")
        return []
    else:
        request_fail = True
        while request_fail:
            try:
                site = requests.get(base_url + palavra)
                request_fail=False
            except Exception as e:
                print (e)
                pass
        data = BeautifulSoup(site.content,'html.parser')

        try:
            h1 = data.find('h1').getText()
        except Exception as e:
            if palavra is not None:
                list_no_occur = push(list_no_occur,palavra)
            set_(nao_consta,"antonimos",list_no_occur)
            save_json("nao_consta",nao_consta)
            return []
        if(h1=="Página Não Encontrada"):
            #print(h1)
            if palavra is not None:
                list_no_occur = push(list_no_occur,palavra)
            set_(nao_consta,"antonimos",list_no_occur)
            save_json("nao_consta",nao_consta)
            return []
        else:
            content = data.find('div', attrs={'id':'content'})
            try:
                div = content.findAll('div', attrs={'class':'s-wrapper'})
            except Exception as e:
                print(e)
                if palavra is not None:
                    list_no_occur = push(list_no_occur,palavra)
                set_(nao_consta,"antonimos",list_no_occur)
                save_json("nao_consta",nao_consta)
            aux=0
            for sentido in div:
                aux=aux+1
                lista_antonimos = []
                try:
                    try:
                        key = lower_case(sentido.find('div', attrs={'class':'sentido'}).getText().strip(":"))
                    except Exception as e:
                        key = lower_case(sentido.find('strong').getText().strip("."))
                        pass
                    #print(sentido.find('p', attrs={'class':'antonimos'}))
                    p = sentido.find('p', attrs={'class':'antonimos'}).getText()[3:]
                    try:
                        p=str(p.encode('raw_unicode_escape').decode('utf-8'))
                    except Exception as e:
                        print(e)
                        pass
                    #print(p)
                    #print(p.encode('utf-8'))
                    all_values = p.split(',')
                    #print(all_values)
                    for value in all_values:
                        lista_antonimos.append(value.strip(":").strip(' '))
                    set_(antonimos,palavra+"."+key,lista_antonimos)
                    print("Antônimo Salv@ no Dicionário")
                except Exception as e:
                    print("\nError:\n"+ str(e))
                    return []
            save_json("antonimos",antonimos)
            return get(antonimos,palavra)

#######################################################################
########################------ OPEN JSON FILE------####################
#######################################################################
def load_json(name):
    filename = name+'.json'
    try:
        if(os.path.isfile(filename)):
            with open(filename, 'r', encoding="utf8") as f:
                return(json.load(f))
                f.close()
        else:
            with open(filename, 'w+', encoding="utf8") as f:
                json.dump({}, f, indent = 4, sort_keys=True, ensure_ascii=False)
                f.close()
                return {}
    except Exception as e:
        print('Erro ao abrir: '+ filename +"\nError:\n"+ str(e))
        with open(filename, 'w+', encoding="utf8") as f:
            json.dump({}, f, indent = 4, sort_keys=True, ensure_ascii=False)
            f.close()
        return {}

#######################################################################
########################------ SAVE JSON FILE------####################
#######################################################################
def save_json(name, data):
    filename = name+'.json'
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(data, f, indent = 4, sort_keys=False, ensure_ascii=False)
        f.close()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#######################################################################
#########################------VARIABLES------#########################
#######################################################################
global base_url, sinonimos, antonimos, nao_consta
antonimos={}

sinonimos = load_json("sinonimos")
antonimos = load_json("antonimos")
nao_consta = load_json("nao_consta")
