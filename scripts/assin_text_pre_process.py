# coding=utf-8
from text_core import *
from extractor_core import *
from pydash import flatten_deep, values, get, concat, set_

####################################################
##############------LE DATASET------################
####################################################
# f = open("assin-ptbr-dev.txt", encoding="utf8")
# frases = f.readlines()
# ind = 0 * 3
# f1 = frases[ind]
# f2 = frases[ind+1]
#
# f1 = "Sebastian Vettel garantiu a pole-position para o Grande Prémio de Singapura de Fórmula 1."
# f2 = "O Grande Prémio de Singapura de Fórmula 1 tem início marcado para as 13h00 de domingo."
#
# i = 0
#######################################################
##############------PROCESSAMENTO------################
#######################################################

#print (f1,"\n",f2)
def process_2_sents(f1,f2):

    ft0_0 = "\tTokens Intersection"
    tk1 ,tk2 = get_tokens_from_2_sents(f1,f2)
    r0_0=len_intersection(tk1,tk2)
    #print_intersection_detail(ft0_0,tk1,tk2)

    ft0_1 = "\tTokens Jaccard"
    r0_1= jaccard_similarity(tk1,tk2)

    ft0_2 = "\tTokens Overlap"
    r0_2= overlap_similarity(tk1,tk2)

    ft0_3 = "\tTokens Cosine"
    r0_3= cos_similarity(tk1,tk2)

    wv1, tks_similar1 = get_wv_most_similar(f1)
    wv2, tks_similar2 = get_wv_most_similar(f2)
    ft0_4= "Tokens Gensim Cosine"
    r0_4= 0
    r0_4= get_gensim_cos_similarity(wv1,wv2)

    ft0_5 = "\tTokens Dice"
    r0_5= dice_similarity(tk1,tk2)

    ft0_6 = "\tTokens Len f1"
    r0_6= len(tk1)

    ft0_7 = "\tTokens Len f2"
    r0_7= len(tk2)

    ft0_8 = "\tTokens Len Proportion"
    r0_8= get_len_proportion_from_2_sents(tk1,tk2)

    #----------------------------------------------------------------

    ft1_0 = "\tWords Intersection"
    wd1 ,wd2 = get_words_from_2_sentS(f1,f2)
    r1_0=len_intersection(wd1,wd2)
    #print_intersection_detail(ft1_0,wd1,wd2)

    ft1_1 = "\tWords Jaccard"
    r1_1= jaccard_similarity(wd1,wd2)

    ft1_2 = "\tWords Overlap"
    r1_2= overlap_similarity(wd1,wd2)

    ft1_3 = "\tWords Cosine"
    r1_3= cos_similarity(wd1,wd2)

    ft1_5 = "\tWords Dice"
    r1_5= dice_similarity(wd1,wd2)

    ft1_6 = "\tWords Len f1"
    r1_6= len(wd1)

    ft1_7 = "\tWords Len f2"
    r1_7= len(wd2)

    ft1_8 = "\tWords Len Proportion"
    r1_8= get_len_proportion_from_2_sents(wd1,wd2)

    #----------------------------------------------------------------

    ft2_0 = "\tNumbers Intersection"
    n1,n2 = get_numbers_from_2_sents(f1,f2)
    r2_0=len_intersection(n1,n2)
    #print_intersection_detail(ft2_0,n1,n2)

    ft2_1 = "\tNumbers Jaccard"
    r2_1= jaccard_similarity(n1,n2)

    ft2_2 = "\tNumbers Overlap"
    r2_2= overlap_similarity(n1,n2)

    ft2_5 = "\tNumbers Dice"
    r2_5= dice_similarity(n1,n2)

    ft2_6 = "\tNumbers Len f1"
    r2_6= len(n1)

    ft2_7 = "\tNumbers Len f2"
    r2_7= len(n2)

    ft2_8 = "\tNumbers Len Proportion"
    r2_8= get_len_proportion_from_2_sents(n1,n2)

    #----------------------------------------------------------------

    ft3_0 = "\tWord Lemas Intersection"
    lm1,lm2 = get_lemmas_from_2_sents(f1,f2)
    r3_0=len_intersection(lm1,lm2)
    #print_intersection_detail(ft3_0,lm1,lm2)

    ft3_1 = "\tWord Lemas Jaccard"
    r3_1=jaccard_similarity(lm1,lm2)

    ft3_2 = "\tWord Lemas Overlap"
    r3_2=overlap_similarity(lm1,lm2)

    ft3_3 = "\tWord Lemas Cosine"
    r3_3= cos_similarity(lm1,lm2)

    lmv1, lm_similar1 = get_wv_most_similar(lm1)
    lmv2, lm_similar2 = get_wv_most_similar(lm2)
    ft3_4 = "\tWord Lemas Gensim Cosine"
    r3_4 = 0
    r3_4= get_gensim_cos_similarity(lmv1,lmv2)

    ft3_5 = "\tWord Lemas Dice"
    r3_5= dice_similarity(lm1,lm2)

    ft3_6 = "\tWord Lemas Len f1"
    r3_6= len(lm1)

    ft3_7 = "\tWord Lemas Len f2"
    r3_7= len(lm2)

    ft3_8 = "\tWord Lemas Len Proportion"
    r3_8= get_len_proportion_from_2_sents(lm1,lm2)

    #----------------------------------------------------------------

    ft5_0 = "\tWord Pos Intersection"
    ptg1,ptg2 = get_word_pos_from_2_sents(f1,f2)
    r5_0=len_intersection(ptg1,ptg2)
    #print_intersection_detail(ft5_0,ptg1,ptg2)

    ft5_1 = "\tWord Pos Jaccard"
    r5_1=jaccard_similarity(ptg1,ptg2)

    ft5_2 = "\tWord Pos Overlap"
    r5_2=overlap_similarity(ptg1,ptg2)

    ft5_3 = "\tWord Pos Cosine"
    r5_3= cos_similarity(ptg1,ptg2)

    ft5_5 = "\tWord Pos Dice"
    r5_5= dice_similarity(ptg1,ptg2)

    ft5_6 = "\tWord Pos Len f1"
    r5_6= len(ptg1)

    ft5_7 = "\tWord Pos Len f2"
    r5_7= len(ptg2)

    ft5_8 = "\tWord Pos Len Proportion"
    r5_8= get_len_proportion_from_2_sents(ptg1,ptg2)

    pos1 = get_obj_pos_from_sent(f1)
    pos2 = get_obj_pos_from_sent(f2)

    #---- ADJ
    l1=get(pos1,'ADJ')
    l2=get(pos2,'ADJ')

    ft5_9 = "\t-ADJ Len f1"
    r5_9 = len(l1)

    ft5_10 = "\t-ADJ Len f2"
    r5_10 = len(l2)

    ft5_11 = "\t-ADJ Abs Difference"
    r5_11 = abs(len(l1) - len(l2))

    ft5_12 = "\t-ADJ Jaccard"
    r5_12=jaccard_similarity(l1,l2)

    ft5_13 = "\t-ADJ Overlap"
    r5_13=overlap_similarity(l1,l2)

    ft5_14 = "\t-ADJ Cosine"
    r5_14= float(cos_similarity(l1,l2))

    #---- ADP
    l1=get(pos1,'ADP')
    l2=get(pos2,'ADP')

    ft5_15 = "\t-ADP Len f1"
    r5_15 = len(l1)

    ft5_16 = "\t-ADP Len f2"
    r5_16 = len(l2)

    ft5_17 = "\t-ADP Abs Difference"
    r5_17 = abs(len(l1) - len(l2))

    ft5_18 = "\t-ADP Jaccard"
    r5_18=jaccard_similarity(l1,l2)

    ft5_19 = "\t-ADP Overlap"
    r5_19=overlap_similarity(l1,l2)

    ft5_20 = "\t-ADP Cosine"
    r5_20= float(cos_similarity(l1,l2))

    #---- ADV
    l1=get(pos1,'ADV')
    l2=get(pos2,'ADV')

    ft5_21 = "\t-ADV Len f1"
    r5_21 = len(l1)

    ft5_22 = "\t-ADV Len f2"
    r5_22 = len(l2)

    ft5_23 = "\t-ADV Abs Difference"
    r5_23 = abs(len(l1) - len(l2))

    ft5_24 = "\t-ADV Jaccard"
    r5_24=jaccard_similarity(l1,l2)

    ft5_25 = "\t-ADV Overlap"
    r5_25=overlap_similarity(l1,l2)

    ft5_26 = "\t-ADV Cosine"
    r5_26= float(cos_similarity(l1,l2))

    #---- AUX
    l1=get(pos1,'AUX')
    l2=get(pos2,'AUX')

    ft5_27 = "\t-AUX Len f1"
    r5_27 = len(l1)

    ft5_28 = "\t-AUX Len f2"
    r5_28 = len(l2)

    ft5_29 = "\t-AUX Abs Difference"
    r5_29 = abs(len(l1) - len(l2))

    ft5_30 = "\t-AUX Jaccard"
    r5_30=jaccard_similarity(l1,l2)

    ft5_31 = "\t-AUX Overlap"
    r5_31=overlap_similarity(l1,l2)

    ft5_32 = "\t-AUX Cosine"
    r5_32= float(cos_similarity(l1,l2))

    #---- CONJ
    l1=get(pos1,'CONJ')
    l2=get(pos2,'CONJ')

    ft5_33 = "\t-CONJ Len f1"
    r5_33 = len(l1)

    ft5_34 = "\t-CONJ Len f2"
    r5_34 = len(l2)

    ft5_35 = "\t-CONJ Abs Difference"
    r5_35 = abs(len(l1) - len(l2))

    ft5_36 = "\t-CONJ Jaccard"
    r5_36=jaccard_similarity(l1,l2)

    ft5_37 = "\t-CONJ Overlap"
    r5_37=overlap_similarity(l1,l2)

    ft5_38 = "\t-CONJ Cosine"
    r5_38= float(cos_similarity(l1,l2))

    #---- CCONJ
    l1=get(pos1,'CCONJ')
    l2=get(pos2,'CCONJ')

    ft5_39 = "\t-CCONJ Len f1"
    r5_39 = len(l1)

    ft5_40 = "\t-CCONJ Len f2"
    r5_40 = len(l2)

    ft5_41 = "\t-CCONJ Abs Difference"
    r5_41 = abs(len(l1) - len(l2))

    ft5_42 = "\t-CCONJ Jaccard"
    r5_42=jaccard_similarity(l1,l2)

    ft5_43 = "\t-CCONJ Overlap"
    r5_43=overlap_similarity(l1,l2)

    ft5_44 = "\t-CCONJ Cosine"
    r5_44= float(cos_similarity(l1,l2))

    #---- DET
    l1=get(pos1,'DET')
    l2=get(pos2,'DET')

    ft5_45 = "\t-DET Len f1"
    r5_45 = len(l1)

    ft5_46 = "\t-DET Len f2"
    r5_46 = len(l2)

    ft5_47 = "\t-DET Abs Difference"
    r5_47 = abs(len(l1) - len(l2))

    ft5_48 = "\t-DET Jaccard"
    r5_48=jaccard_similarity(l1,l2)

    ft5_49 = "\t-DET Overlap"
    r5_49=overlap_similarity(l1,l2)

    ft5_50 = "\t-DET Cosine"
    r5_50= float(cos_similarity(l1,l2))

    #---- INTJ
    l1=get(pos1,'INTJ')
    l2=get(pos2,'INTJ')

    ft5_51 = "\t-INTJ Len f1"
    r5_51 = len(l1)

    ft5_52 = "\t-INTJ Len f2"
    r5_52 = len(l2)

    ft5_53 = "\t-INTJ Abs Difference"
    r5_53 = abs(len(l1) - len(l2))

    ft5_54 = "\t-INTJ Jaccard"
    r5_54=jaccard_similarity(l1,l2)

    ft5_55 = "\t-INTJ Overlap"
    r5_55=overlap_similarity(l1,l2)

    ft5_56 = "\t-INTJ Cosine"
    r5_56= float(cos_similarity(l1,l2))

    #---- NOUN
    l1=get(pos1,'NOUN')
    l2=get(pos2,'NOUN')

    ft5_57 = "\t-NOUN Len f1"
    r5_57 = len(l1)

    ft5_58 = "\t-NOUN Len f2"
    r5_58 = len(l2)

    ft5_59 = "\t-NOUN Abs Difference"
    r5_59 = abs(len(l1) - len(l2))

    ft5_60 = "\t-NOUN Jaccard"
    r5_60=jaccard_similarity(l1,l2)

    ft5_61 = "\t-NOUN Overlap"
    r5_61=overlap_similarity(l1,l2)

    ft5_62 = "\t-NOUN Cosine"
    r5_62= float(cos_similarity(l1,l2))

    #---- NUM
    l1=get(pos1,'NUM')
    l2=get(pos2,'NUM')

    ft5_63 = "\t-NUM Len f1"
    r5_63 = len(l1)

    ft5_64 = "\t-NUM Len f2"
    r5_64 = len(l2)

    ft5_65 = "\t-NUM Abs Difference"
    r5_65 = abs(len(l1) - len(l2))

    ft5_66 = "\t-NUM Jaccard"
    r5_66=jaccard_similarity(l1,l2)

    ft5_67 = "\t-NUM Overlap"
    r5_67=overlap_similarity(l1,l2)

    ft5_68 = "\t-NUM Cosine"
    r5_68= float(cos_similarity(l1,l2))

    #---- PART
    l1=get(pos1,'PART')
    l2=get(pos2,'PART')

    ft5_69 = "\t-PART Len f1"
    r5_69 = len(l1)

    ft5_70 = "\t-PART Len f2"
    r5_70 = len(l2)

    ft5_71 = "\t-PART Abs Difference"
    r5_71 = abs(len(l1) - len(l2))

    ft5_72 = "\t-PART Jaccard"
    r5_72=jaccard_similarity(l1,l2)

    ft5_73 = "\t-PART Overlap"
    r5_73=overlap_similarity(l1,l2)

    ft5_74 = "\t-PART Cosine"
    r5_74= float(cos_similarity(l1,l2))

    #---- PRON
    l1=get(pos1,'PRON')
    l2=get(pos2,'PRON')

    ft5_75 = "\t-PRON Len f1"
    r5_75 = len(l1)

    ft5_76 = "\t-PRON Len f2"
    r5_76 = len(l2)

    ft5_77 = "\t-PRON Abs Difference"
    r5_77 = abs(len(l1) - len(l2))

    ft5_78 = "\t-PRON Jaccard"
    r5_78=jaccard_similarity(l1,l2)

    ft5_79 = "\t-PRON Overlap"
    r5_79=overlap_similarity(l1,l2)

    ft5_80 = "\t-PRON Cosine"
    r5_80= float(cos_similarity(l1,l2))

    #---- PROPN
    l1=get(pos1,'PROPN')
    l2=get(pos2,'PROPN')

    ft5_81 = "\t-PROPN Len f1"
    r5_81 = len(l1)

    ft5_82 = "\t-PROPN Len f2"
    r5_82 = len(l2)

    ft5_83 = "\t-PROPN Abs Difference"
    r5_83 = abs(len(l1) - len(l2))

    ft5_84 = "\t-PROPN Jaccard"
    r5_84=jaccard_similarity(l1,l2)

    ft5_85 = "\t-PROPN Overlap"
    r5_85=overlap_similarity(l1,l2)

    ft5_86 = "\t-PROPN Cosine"
    r5_86= float(cos_similarity(l1,l2))

    #---- PUNCT
    l1=get(pos1,'PUNCT')
    l2=get(pos2,'PUNCT')

    ft5_87 = "\t-PUNCT Len f1"
    r5_87 = len(l1)

    ft5_88 = "\t-PUNCT Len f2"
    r5_88 = len(l2)

    ft5_89 = "\t-PUNCT Abs Difference"
    r5_89 = abs(len(l1) - len(l2))

    ft5_90 = "\t-PUNCT Jaccard"
    r5_90=jaccard_similarity(l1,l2)

    ft5_91 = "\t-PUNCT Overlap"
    r5_91=overlap_similarity(l1,l2)

    ft5_92 = "\t-PUNCT Cosine"
    r5_92= float(cos_similarity(l1,l2))

    #---- SCONJ
    l1=get(pos1,'SCONJ')
    l2=get(pos2,'SCONJ')

    ft5_93 = "\t-SCONJ Len f1"
    r5_93 = len(l1)

    ft5_94 = "\t-SCONJ Len f2"
    r5_94 = len(l2)

    ft5_95 = "\t-SCONJ Abs Difference"
    r5_95 = abs(len(l1) - len(l2))

    ft5_96 = "\t-SCONJ Jaccard"
    r5_96=jaccard_similarity(l1,l2)

    ft5_97 = "\t-SCONJ Overlap"
    r5_97=overlap_similarity(l1,l2)

    ft5_98 = "\t-SCONJ Cosine"
    r5_98= float(cos_similarity(l1,l2))

    #---- SYM
    l1=get(pos1,'SYM')
    l2=get(pos2,'SYM')

    ft5_99 = "\t-SYM Len f1"
    r5_99 = len(l1)

    ft5_100 = "\t-SYM Len f2"
    r5_100 = len(l2)

    ft5_101 = "\t-SYM Abs Difference"
    r5_101 = abs(len(l1) - len(l2))

    ft5_102 = "\t-SYM Jaccard"
    r5_102=jaccard_similarity(l1,l2)

    ft5_103 = "\t-SYM Overlap"
    r5_103=overlap_similarity(l1,l2)

    ft5_104 = "\t-SYM Cosine"
    r5_104= float(cos_similarity(l1,l2))

    #---- VERB
    l1=get(pos1,'VERB')
    l2=get(pos2,'VERB')

    ft5_105 = "\t-VERB Len f1"
    r5_105 = len(l1)

    ft5_106 = "\t-VERB Len f2"
    r5_106 = len(l2)

    ft5_107 = "\t-VERB Abs Difference"
    r5_107 = abs(len(l1) - len(l2))

    ft5_108 = "\t-VERB Jaccard"
    r5_108=jaccard_similarity(l1,l2)

    ft5_109 = "\t-VERB Overlap"
    r5_109=overlap_similarity(l1,l2)

    ft5_110 = "\t-VERB Cosine"
    r5_110= float(cos_similarity(l1,l2))

    #---- X
    l1=get(pos1,'X')
    l2=get(pos2,'X')

    ft5_111 = "\t-X Len f1"
    r5_111 = len(l1)

    ft5_112 = "\t-X Len f2"
    r5_112 = len(l2)

    ft5_113 = "\t-X Abs Difference"
    r5_113 = abs(len(l1) - len(l2))

    ft5_114 = "\t-X Jaccard"
    r5_114=jaccard_similarity(l1,l2)

    ft5_115 = "\t-X Overlap"
    r5_115=overlap_similarity(l1,l2)

    ft5_116 = "\t-X Cosine"
    r5_116= float(cos_similarity(l1,l2))


    #----------------------------------------------------------------

    st1,st2 = get_stems_from_2_sents(f1,f2)

    ft6_0 = "\tWord Stems Intersection"
    r6_0=len_intersection(st1,st2)
    #print_intersection_detail(ft6_0,st1,st2)

    ft6_1 = "\tWord Stems Jaccard"
    r6_1=jaccard_similarity(st1,st2)

    ft6_2 = "\tWord Stems Overlap"
    r6_2=overlap_similarity(st1,st2)

    ft6_3 = "\tWord Stems Cosine"
    r6_3= cos_similarity(st1,st2)

    ft6_5 = "\tWord Stems Dice"
    r6_5= dice_similarity(st1,st2)

    ft6_6 = "\tWord Stems Len f1"
    r6_6= len(st1)

    ft6_7 = "\tWord Stems Len f2"
    r6_7= len(st2)

    ft6_8 = "\tWord Stems Len Proportion"
    r6_8= get_len_proportion_from_2_sents(st1,st2)

    #----------------------------------------------------------------
    we1 = get_word_entitie_from_sent(f1)
    we2 = get_word_entitie_from_sent(f2)

    ft7_0 = "\tWord entities Intersection"
    r7_0=len_intersection(we1,we2)
    #print_intersection_detail(ft7_0,we1,we2)
    #
    ft7_1 = "\tWord entities Jaccard"
    r7_1=jaccard_similarity(we1,we2)
    #
    ft7_2 = "\tWord entities Overlap"
    r7_2=overlap_similarity(we1,we2)
    #
    ft7_3 = "\tWord entities Cosine"
    r7_3= cos_similarity(we1,we2)

    ft7_5 = "\tWord entities Dice"
    r7_5= dice_similarity(we1,we2)

    ft7_6 = "\tWord entities Len f1"
    r7_6= len(we1)

    ft7_7 = "\tWord entities Len f2"
    r7_7= len(we2)

    ft7_8 = "\tWord entities Len Proportion"
    r7_8= get_len_proportion_from_2_sents(we1,we2)

    #----------------------------------------------------------------
    ent1 = get_entities_from_sent(f1)
    ent2 = get_entities_from_sent(f2)

    ft8_0 = "\tEntities Intersection"
    list_ent1 = flatten_deep(values(ent1))
    list_ent2 = flatten_deep(values(ent2))
    r8_0 = len_intersection(list_ent1, list_ent2)
    #print_intersection_detail(ft8_0,ent1,ent2)
    #
    ft8_1 = "\tEntities Jaccard"
    r8_1 = jaccard_similarity(list_ent1, list_ent2)
    #
    ft8_2 = "\tEntities Overlap"
    r8_2 = overlap_similarity(list_ent1, list_ent2)
    #
    ft8_3 = "\tEntities Cosine"
    r8_3 = cos_similarity(list_ent1, list_ent2)

    ft8_5 = "\tEntities Dice"
    r8_5 = dice_similarity(list_ent1, list_ent2)

    ft8_6 = "\tEntities Len f1"
    r8_6 = len(list_ent1)

    ft8_7 = "\tEntities Len f2"
    r8_7 = len(list_ent2)

    ft8_8 = "\tEntities Len Proportion"
    r8_8 = get_len_proportion_from_2_sents(list_ent1, list_ent2)

    #---- PER
    l1=[] if (get(ent1,'PER') is None) else get(ent1,'PER')
    l2=[] if (get(ent2,'PER') is None) else get(ent2,'PER')

    ft8_9 = "\t-PER Len f1"
    r8_9 = len(list_ent1)

    ft8_10 = "\t-PER Len f2"
    r8_10 = len(list_ent2)

    ft8_11 = "\t-PER Abs Difference"
    r8_11 = abs(r8_9 - r8_10)

    #---- PER
    l1=[] if (get(ent1,'PER') is None) else get(ent1,'PER')
    l2=[] if (get(ent2,'PER') is None) else get(ent2,'PER')

    ft8_9 = "\t-PER Len f1"
    r8_9 = len(l1)

    ft8_10 = "\t-PER Len f2"
    r8_10 = len(l2)

    ft8_11 = "\t-PER Abs Difference"
    r8_11 = abs(len(l1) - len(l2))

    #---- LOC
    l1=[] if (get(ent1,'LOC') is None) else get(ent1,'LOC')
    l2=[] if (get(ent2,'LOC') is None) else get(ent2,'LOC')

    ft8_12 = "\t-LOC Len f1"
    r8_12 = len(l1)

    ft8_13 = "\t-LOC Len f2"
    r8_13 = len(l2)

    ft8_14 = "\t-LOC Abs Difference"
    r8_14 = abs(len(l1) - len(l2))

    #---- ORG
    l1=[] if (get(ent1,'ORG') is None) else get(ent1,'ORG')
    l2=[] if (get(ent2,'ORG') is None) else get(ent2,'ORG')

    ft8_15 = "\t-ORG Len f1"
    r8_15 = len(l1)

    ft8_16 = "\t-ORG Len f2"
    r8_16 = len(l2)

    ft8_17 = "\t-ORG Abs Difference"
    r8_17 = abs(len(l1) - len(l2))

    #---- MISC
    l1=[] if (get(ent1,'MISC') is None) else get(ent1,'MISC')
    l2=[] if (get(ent2,'MISC') is None) else get(ent2,'MISC')

    ft8_18 = "\t-MISC Len f1"
    r8_18 = len(l1)

    ft8_19 = "\t-MISC Len f2"
    r8_19 = len(l2)

    ft8_20 = "\t-MISC Abs Difference"
    r8_20 = abs(len(l1) - len(l2))

    #---- TIME
    l1=[] if (get(ent1,'TIME') is None) else get(ent1,'TIME')
    l2=[] if (get(ent2,'TIME') is None) else get(ent2,'TIME')

    ft8_21 = "\t-TIME Len f1"
    r8_21 = len(l1)

    ft8_22 = "\t-TIME Len f2"
    r8_22 = len(l2)

    ft8_23 = "\t-TIME Abs Difference"
    r8_23 = abs(len(l1) - len(l2))

    #---- ORDINAL
    l1=[] if (get(ent1,'ORDINAL') is None) else get(ent1,'ORDINAL')
    l2=[] if (get(ent2,'ORDINAL') is None) else get(ent2,'ORDINAL')

    ft8_24 = "\t-ORDINAL Len f1"
    r8_24 = len(l1)

    ft8_25 = "\t-ORDINAL Len f2"
    r8_25 = len(l2)

    ft8_26 = "\t-ORDINAL Abs Difference"
    r8_26 = abs(len(l1) - len(l2))

    #---- DATE
    l1=[] if (get(ent1,'DATE') is None) else get(ent1,'DATE')
    l2=[] if (get(ent2,'DATE') is None) else get(ent2,'DATE')

    ft8_27 = "\t-DATE Len f1"
    r8_27 = len(l1)

    ft8_28 = "\t-DATE Len f2"
    r8_28 = len(l2)

    ft8_29 = "\t-DATE Abs Difference"
    r8_29 = abs(len(l1) - len(l2))

    #---- CARDINAL
    l1=[] if (get(ent1,'CARDINAL') is None) else get(ent1,'CARDINAL')
    l2=[] if (get(ent2,'CARDINAL') is None) else get(ent2,'CARDINAL')

    ft8_30 = "\t-CARDINAL Len f1"
    r8_30 = len(l1)

    ft8_31 = "\t-CARDINAL Len f2"
    r8_31 = len(l2)

    ft8_32 = "\t-CARDINAL Abs Difference"
    r8_32 = abs(len(l1) - len(l2))

    #---- MONEY
    l1=[] if (get(ent1,'MONEY') is None) else get(ent1,'MONEY')
    l2=[] if (get(ent2,'MONEY') is None) else get(ent2,'MONEY')

    ft8_33 = "\t-MONEY Len f1"
    r8_33 = len(l1)

    ft8_34 = "\t-MONEY Len f2"
    r8_34 = len(l2)

    ft8_35 = "\t-MONEY Abs Difference"
    r8_35 = abs(len(l1) - len(l2))

    #---- QUANTITY
    l1=[] if (get(ent1,'QUANTITY') is None) else get(ent1,'QUANTITY')
    l2=[] if (get(ent2,'QUANTITY') is None) else get(ent2,'QUANTITY')

    ft8_36 = "\t-QUANTITY Len f1"
    r8_36 = len(l1)

    ft8_37 = "\t-QUANTITY Len f2"
    r8_37 = len(l2)

    ft8_38 = "\t-QUANTITY Abs Difference"
    r8_38 = abs(len(l1) - len(l2))

    #----------------------------------------------------------------

    dp1 = get_syntactic_dependency_from_sent(f1)
    dp2 = get_syntactic_dependency_from_sent(f2)

    ft9_0 = "\tDependency Intersection"
    r9_0=len_intersection(dp1,dp2)
    #print_intersection_detail(ft9_0,dp1,dp2)

    ft9_1 = "\tDependency Jaccard"
    r9_1=jaccard_similarity(dp1,dp2)

    ft9_2 = "\tDependency Overlap"
    r9_2=overlap_similarity(dp1,dp2)

    ft9_3 = "\tDependency Cosine"
    r9_3= cos_similarity(dp1,dp2)

    ft9_5 = "\tDependency Dice"
    r9_5= dice_similarity(dp1,dp2)

    ft9_6 = "\tDependency Len f1"
    r9_6= len(dp1)

    ft9_7 = "\tDependency Len f2"
    r9_7= len(dp2)

    ft9_8 = "\tDependency Len Proportion"
    r9_8= get_len_proportion_from_2_sents(dp1,dp2)
    #----------------------------------------------------------------

    wt1 = get_word_tag_from_sent(f1)
    wt2 = get_word_tag_from_sent(f2)

    ft10_0 = "\tWord Tag Intersection"
    r10_0=len_intersection(wt1,wt2)
    #print_intersection_detail(ft10_0,wt1,wt2)

    ft10_1 = "\tWord Tag Jaccard"
    r10_1=jaccard_similarity(wt1,wt2)

    ft10_2 = "\tWord Tag Overlap"
    r10_2=overlap_similarity(wt1,wt2)

    ft10_3 = "\tWord Tag Cosine"
    r10_3= cos_similarity(wt1,wt2)

    ft10_5 = "\tWord Tag Dice"
    r10_5= dice_similarity(wt1,wt2)

    ft10_6 = "\tWord Tag Len f1"
    r10_6= len(wt1)

    ft10_7 = "\tWord Tag Len f2"
    r10_7= len(wt2)

    ft10_8 = "\tWord Tag Len Proportion"
    r10_8= get_len_proportion_from_2_sents(wt1,wt2)
    #----------------------------------------------------------------

    sin1=[]
    sin2=[]
    for token in tk1:
        sin1=concat(sin1,(flatten_deep(values(get_sinonimos(token)))))

    for token in tk2:
        sin2=concat(sin2,(flatten_deep(values(get_sinonimos(token)))))

    ft11_0 = "\tSinonimos Intersection"
    r11_0=len_intersection(sin1,sin2)
    #print_intersection_detail(ft11_0,sin1,sin2)

    ft11_1 = "\tSinonimos Jaccard"
    r11_1=jaccard_similarity(sin1,sin2)

    ft11_2 = "\tSinonimos Overlap"
    r11_2=overlap_similarity(sin1,sin2)

    ft11_3 = "\tSinonimos Cosine"
    r11_3= cos_similarity(sin1,sin2)

    ft11_4 = "\tSinonimos Gensim Cosine"
    r11_4 = 0
    r11_4= get_gensim_cos_similarity(tks_similar1,tks_similar2)

    ft11_5 = "\tSinonimos Dice"
    r11_5= dice_similarity(sin1,sin2)

    ft11_6 = "\tSinonimos Len f1"
    r11_6= len(sin1)

    ft11_7 = "\tSinonimos Len f2"
    r11_7= len(sin2)

    ft11_8 = "\tSinonimos Len Proportion"
    r11_8= get_len_proportion_from_2_sents(sin1,sin2)
    #----------------------------------------------------------------

    ant1=[]
    ant2=[]
    for token in tk1:
        ant1=concat(ant1,(flatten_deep(values(get_antonimos(token)))))

    for token in tk2:
        ant2=concat(ant2,(flatten_deep(values(get_antonimos(token)))))

    ft12_0 = "\tAntonimos Intersection"
    r12_0=len_intersection(ant1,ant2)
    #print_intersection_detail(ft12_0,ant1,ant2)

    ft12_1 = "\tAntonimos Jaccard"
    r12_1=jaccard_similarity(ant1,ant2)

    ft12_2 = "\tAntonimos Overlap"
    r12_2=overlap_similarity(ant1,ant2)

    ft12_3 = "\tAntonimos Cosine"
    r12_3= cos_similarity(ant1,ant2)

    ft12_5 = "\tAntonimos Dice"
    r12_5= dice_similarity(ant1,ant2)

    ft12_6 = "\tAntonimos Len f1"
    r12_6= len(ant1)

    ft12_7 = "\tAntonimos Len f2"
    r12_7= len(ant2)

    ft12_8 = "\tAntonimos Len Proportion"
    r12_8= get_len_proportion_from_2_sents(ant1,ant2)
    #----------------------------------------------------------------

    obj={}
    set_(obj,"f1", f1)
    set_(obj,"f2", f2)

    #print(pad("",55,'-'))   # Tokens
    set_(obj,snake_case(ft0_0), float(r0_0))           # 0 -Intersection
    set_(obj,snake_case(ft0_1), float(r0_1))         # 1 -Jaccard
    set_(obj,snake_case(ft0_2), float(r0_2))         # 2 -Overlap
    set_(obj,snake_case(ft0_3), float(r0_3))  # 3 -Cosine
    set_(obj,snake_case(ft0_4), float(r0_4))  # 4 -Gensim Cosine
    set_(obj,snake_case(ft0_5), float(r0_5))         # 5 -Dice
    set_(obj,snake_case(ft0_6), float(r0_6))         # 6 -Len f1
    set_(obj,snake_case(ft0_7), float(r0_7))         # 7 -Len f2
    set_(obj,snake_case(ft0_8), float(r0_8))         # 8 -f1/f2 or f2/f1

    #print(pad("",55,'-'))   # Words
    set_(obj,snake_case(ft1_0), float(r1_0))
    set_(obj,snake_case(ft1_1), float(r1_1))
    set_(obj,snake_case(ft1_2), float(r1_2))
    set_(obj,snake_case(ft1_3), float(r1_3))
    set_(obj,snake_case(ft1_5), float(r1_5))
    set_(obj,snake_case(ft1_6), float(r1_6))
    set_(obj,snake_case(ft1_7), float(r1_7))
    set_(obj,snake_case(ft1_8), float(r1_8))

    #print(pad("",55,'-'))   # Numbers
    set_(obj,snake_case(ft2_0), float(r2_0))
    set_(obj,snake_case(ft2_1), float(r2_1))
    set_(obj,snake_case(ft2_2), float(r2_2))
    set_(obj,snake_case(ft2_5), float(r2_5))
    set_(obj,snake_case(ft2_6), float(r2_6))
    set_(obj,snake_case(ft2_7), float(r2_7))
    set_(obj,snake_case(ft2_8), float(r2_8))

    #print(pad("",55,'-'))   # Lemas
    set_(obj,snake_case(ft3_0), float(r3_0))
    set_(obj,snake_case(ft3_1), float(r3_1))
    set_(obj,snake_case(ft3_2), float(r3_2))
    set_(obj,snake_case(ft3_3), float(r3_3))
    set_(obj,snake_case(ft3_4), float(r3_4))
    set_(obj,snake_case(ft3_5), float(r3_5))
    set_(obj,snake_case(ft3_6), float(r3_6))
    set_(obj,snake_case(ft3_7), float(r3_7))
    set_(obj,snake_case(ft3_8), float(r3_8))

    #print(pad("",55,'-'))   # Pos Tag
    set_(obj,snake_case(ft5_0), float(r5_0))
    set_(obj,snake_case(ft5_1), float(r5_1))
    set_(obj,snake_case(ft5_2), float(r5_2))
    set_(obj,snake_case(ft5_3), float(r5_3))
    set_(obj,snake_case(ft5_5), float(r5_5))
    set_(obj,snake_case(ft5_6), float(r5_6))
    set_(obj,snake_case(ft5_7), float(r5_7))
    set_(obj,snake_case(ft5_8), float(r5_8))
    set_(obj,snake_case(ft5_9), float(r5_9))     #ADJ
    set_(obj,snake_case(ft5_10), float(r5_10))
    set_(obj,snake_case(ft5_11), float(r5_11))
    set_(obj,snake_case(ft5_12), float(r5_12))
    set_(obj,snake_case(ft5_13), float(r5_13))
    set_(obj,snake_case(ft5_14), float(r5_14))
    set_(obj,snake_case(ft5_15), float(r5_15))     #ADP
    set_(obj,snake_case(ft5_16), float(r5_16))
    set_(obj,snake_case(ft5_17), float(r5_17))
    set_(obj,snake_case(ft5_18), float(r5_18))
    set_(obj,snake_case(ft5_19), float(r5_19))
    set_(obj,snake_case(ft5_20), float(r5_20))
    set_(obj,snake_case(ft5_21), float(r5_21))     #ADV
    set_(obj,snake_case(ft5_22), float(r5_22))
    set_(obj,snake_case(ft5_23), float(r5_23))
    set_(obj,snake_case(ft5_24), float(r5_24))
    set_(obj,snake_case(ft5_25), float(r5_25))
    set_(obj,snake_case(ft5_26), float(r5_26))
    set_(obj,snake_case(ft5_27), float(r5_27))     #AUX
    set_(obj,snake_case(ft5_28), float(r5_28))
    set_(obj,snake_case(ft5_29), float(r5_29))
    set_(obj,snake_case(ft5_30), float(r5_30))
    set_(obj,snake_case(ft5_31), float(r5_31))
    set_(obj,snake_case(ft5_32), float(r5_32))
    set_(obj,snake_case(ft5_33), float(r5_33))     #CONJ
    set_(obj,snake_case(ft5_34), float(r5_34))
    set_(obj,snake_case(ft5_35), float(r5_35))
    set_(obj,snake_case(ft5_36), float(r5_36))
    set_(obj,snake_case(ft5_37), float(r5_37))
    set_(obj,snake_case(ft5_38), float(r5_38))
    set_(obj,snake_case(ft5_39), float(r5_39))     #CCONJ
    set_(obj,snake_case(ft5_40), float(r5_40))
    set_(obj,snake_case(ft5_41), float(r5_41))
    set_(obj,snake_case(ft5_42), float(r5_42))
    set_(obj,snake_case(ft5_43), float(r5_43))
    set_(obj,snake_case(ft5_44), float(r5_44))
    set_(obj,snake_case(ft5_45), float(r5_45))     #DET
    set_(obj,snake_case(ft5_46), float(r5_46))
    set_(obj,snake_case(ft5_47), float(r5_47))
    set_(obj,snake_case(ft5_48), float(r5_48))
    set_(obj,snake_case(ft5_49), float(r5_49))
    set_(obj,snake_case(ft5_50), float(r5_50))
    set_(obj,snake_case(ft5_51), float(r5_51))     #INTJ
    set_(obj,snake_case(ft5_52), float(r5_52))
    set_(obj,snake_case(ft5_53), float(r5_53))
    set_(obj,snake_case(ft5_54), float(r5_54))
    set_(obj,snake_case(ft5_55), float(r5_55))
    set_(obj,snake_case(ft5_56), float(r5_56))
    set_(obj,snake_case(ft5_57), float(r5_57))     #NOUN
    set_(obj,snake_case(ft5_58), float(r5_58))
    set_(obj,snake_case(ft5_59), float(r5_59))
    set_(obj,snake_case(ft5_60), float(r5_60))
    set_(obj,snake_case(ft5_61), float(r5_61))
    set_(obj,snake_case(ft5_62), float(r5_62))
    set_(obj,snake_case(ft5_63), float(r5_63))     #NUM
    set_(obj,snake_case(ft5_64), float(r5_64))
    set_(obj,snake_case(ft5_65), float(r5_65))
    set_(obj,snake_case(ft5_66), float(r5_66))
    set_(obj,snake_case(ft5_67), float(r5_67))
    set_(obj,snake_case(ft5_68), float(r5_68))
    set_(obj,snake_case(ft5_69), float(r5_69))     #PART
    set_(obj,snake_case(ft5_70), float(r5_70))
    set_(obj,snake_case(ft5_71), float(r5_71))
    set_(obj,snake_case(ft5_72), float(r5_72))
    set_(obj,snake_case(ft5_73), float(r5_73))
    set_(obj,snake_case(ft5_74), float(r5_74))
    set_(obj,snake_case(ft5_75), float(r5_75))     #PRON
    set_(obj,snake_case(ft5_76), float(r5_76))
    set_(obj,snake_case(ft5_77), float(r5_77))
    set_(obj,snake_case(ft5_78), float(r5_78))
    set_(obj,snake_case(ft5_79), float(r5_79))
    set_(obj,snake_case(ft5_80), float(r5_80))
    set_(obj,snake_case(ft5_81), float(r5_81))     #PROPN
    set_(obj,snake_case(ft5_82), float(r5_82))
    set_(obj,snake_case(ft5_83), float(r5_83))
    set_(obj,snake_case(ft5_84), float(r5_84))
    set_(obj,snake_case(ft5_85), float(r5_85))
    set_(obj,snake_case(ft5_86), float(r5_86))
    set_(obj,snake_case(ft5_87), float(r5_87))     #PUNCT
    set_(obj,snake_case(ft5_88), float(r5_88))
    set_(obj,snake_case(ft5_89), float(r5_89))
    set_(obj,snake_case(ft5_90), float(r5_90))
    set_(obj,snake_case(ft5_91), float(r5_91))
    set_(obj,snake_case(ft5_92), float(r5_92))
    set_(obj,snake_case(ft5_93), float(r5_93))     #SCONJ
    set_(obj,snake_case(ft5_94), float(r5_94))
    set_(obj,snake_case(ft5_95), float(r5_95))
    set_(obj,snake_case(ft5_96), float(r5_96))
    set_(obj,snake_case(ft5_97), float(r5_97))
    set_(obj,snake_case(ft5_98), float(r5_98))
    set_(obj,snake_case(ft5_99), float(r5_99))     #SYM
    set_(obj,snake_case(ft5_100), float(r5_100))
    set_(obj,snake_case(ft5_101), float(r5_101))
    set_(obj,snake_case(ft5_102), float(r5_102))
    set_(obj,snake_case(ft5_103), float(r5_103))
    set_(obj,snake_case(ft5_104), float(r5_104))
    set_(obj,snake_case(ft5_105), float(r5_105))     #VERB
    set_(obj,snake_case(ft5_106), float(r5_106))
    set_(obj,snake_case(ft5_107), float(r5_107))
    set_(obj,snake_case(ft5_108), float(r5_108))
    set_(obj,snake_case(ft5_109), float(r5_109))
    set_(obj,snake_case(ft5_110), float(r5_110))
    set_(obj,snake_case(ft5_111), float(r5_111))     #X
    set_(obj,snake_case(ft5_112), float(r5_112))
    set_(obj,snake_case(ft5_113), float(r5_113))
    set_(obj,snake_case(ft5_114), float(r5_114))
    set_(obj,snake_case(ft5_115), float(r5_115))
    set_(obj,snake_case(ft5_116), float(r5_116))

    #print(pad("",55,'-'))   # Stems
    set_(obj,snake_case(ft6_0), float(r6_0))
    set_(obj,snake_case(ft6_1), float(r6_1))
    set_(obj,snake_case(ft6_2), float(r6_2))
    set_(obj,snake_case(ft6_3), float(r6_3))
    set_(obj,snake_case(ft6_5), float(r6_5))
    set_(obj,snake_case(ft6_6), float(r6_6))
    set_(obj,snake_case(ft6_7), float(r6_7))
    set_(obj,snake_case(ft6_8), float(r6_8))

    #print(pad("",55,'-'))   # Word Entities
    set_(obj,snake_case(ft7_0), float(r7_0))
    set_(obj,snake_case(ft7_1), float(r7_1))
    set_(obj,snake_case(ft7_2), float(r7_2))
    set_(obj,snake_case(ft7_3), float(r7_3))
    set_(obj,snake_case(ft7_5), float(r7_5))
    set_(obj,snake_case(ft7_6), float(r7_6))
    set_(obj,snake_case(ft7_7), float(r7_7))
    set_(obj,snake_case(ft7_8), float(r7_8))

    #print(pad("",55,'-'))   # Dependency
    set_(obj,snake_case(ft9_0), float(r9_0))
    set_(obj,snake_case(ft9_1), float(r9_1))
    set_(obj,snake_case(ft9_2), float(r9_2))
    set_(obj,snake_case(ft9_3), float(r9_3))
    set_(obj,snake_case(ft9_5), float(r9_5))
    set_(obj,snake_case(ft9_6), float(r9_6))
    set_(obj,snake_case(ft9_7), float(r9_7))
    set_(obj,snake_case(ft9_8), float(r9_8))

    #print(pad("",55,'-'))   # Word Tag
    set_(obj,snake_case(ft10_0), float(r10_0))
    set_(obj,snake_case(ft10_1), float(r10_1))
    set_(obj,snake_case(ft10_2), float(r10_2))
    set_(obj,snake_case(ft10_3), float(r10_3))
    set_(obj,snake_case(ft10_5), float(r10_5))
    set_(obj,snake_case(ft10_6), float(r10_6))
    set_(obj,snake_case(ft10_7), float(r10_7))
    set_(obj,snake_case(ft10_8), float(r10_8))

    #print(pad("",55,'-'))   # Sinônimos
    set_(obj,snake_case(ft11_0), float(r11_0))
    set_(obj,snake_case(ft11_1), float(r11_1))
    set_(obj,snake_case(ft11_2), float(r11_2))
    set_(obj,snake_case(ft11_3), float(r11_3))
    set_(obj,snake_case(ft11_4), float(r11_4))
    set_(obj,snake_case(ft11_5), float(r11_5))
    set_(obj,snake_case(ft11_6), float(r11_6))
    set_(obj,snake_case(ft11_7), float(r11_7))
    set_(obj,snake_case(ft11_8), float(r11_8))

    #print(pad("",55,'-'))   # Antônimos
    set_(obj,snake_case(ft12_0), float(r12_0))
    set_(obj,snake_case(ft12_1), float(r12_1))
    set_(obj,snake_case(ft12_2), float(r12_2))
    set_(obj,snake_case(ft12_3), float(r12_3))
    set_(obj,snake_case(ft12_5), float(r12_5))
    set_(obj,snake_case(ft12_6), float(r12_6))
    set_(obj,snake_case(ft12_7), float(r12_7))
    set_(obj,snake_case(ft12_8), float(r12_8))

    #print(pad("",55,'-'))   # Entities
    set_(obj,snake_case(ft8_0), float(r8_0))
    set_(obj,snake_case(ft8_1), float(r8_1))
    set_(obj,snake_case(ft8_2), float(r8_2))
    set_(obj,snake_case(ft8_3), float(r8_3))
    set_(obj,snake_case(ft8_5), float(r8_5))
    set_(obj,snake_case(ft8_6), float(r8_6))
    set_(obj,snake_case(ft8_7), float(r8_7))
    set_(obj,snake_case(ft8_8), float(r8_8))
    set_(obj,snake_case(ft8_9), float(r8_9))     #PER
    set_(obj,snake_case(ft8_10), float(r8_10))
    set_(obj,snake_case(ft8_11), float(r8_11))
    set_(obj,snake_case(ft8_12), float(r8_12))   #LOC
    set_(obj,snake_case(ft8_13), float(r8_13))
    set_(obj,snake_case(ft8_14), float(r8_14))
    set_(obj,snake_case(ft8_15), float(r8_15))   #ORG
    set_(obj,snake_case(ft8_16), float(r8_16))
    set_(obj,snake_case(ft8_17), float(r8_17))
    set_(obj,snake_case(ft8_18), float(r8_18))   #MISC
    set_(obj,snake_case(ft8_19), float(r8_19))
    set_(obj,snake_case(ft8_20), float(r8_20))
    set_(obj,snake_case(ft8_21), float(r8_21))   #TIME
    set_(obj,snake_case(ft8_22), float(r8_22))
    set_(obj,snake_case(ft8_23), float(r8_23))
    set_(obj,snake_case(ft8_24), float(r8_24))   #ORDINAL
    set_(obj,snake_case(ft8_25), float(r8_25))
    set_(obj,snake_case(ft8_26), float(r8_26))
    set_(obj,snake_case(ft8_27), float(r8_27))   #DATE
    set_(obj,snake_case(ft8_28), float(r8_28))
    set_(obj,snake_case(ft8_29), float(r8_29))
    set_(obj,snake_case(ft8_30), float(r8_30))   #CARDINAL
    set_(obj,snake_case(ft8_31), float(r8_31))
    set_(obj,snake_case(ft8_32), float(r8_32))
    set_(obj,snake_case(ft8_33), float(r8_33))   #MONEY
    set_(obj,snake_case(ft8_34), float(r8_34))
    set_(obj,snake_case(ft8_35), float(r8_35))
    set_(obj,snake_case(ft8_36), float(r8_36))   #QUANTITY
    set_(obj,snake_case(ft8_37), float(r8_37))
    set_(obj,snake_case(ft8_38), float(r8_38))

    #Imprime Adverbios
    adverbios = load_json("adverbios")
    for key, value in adverbios.items():
        feature = "\tADV-" + str(key) + " "
        if isinstance(value,(dict)):
            for k, v in value.items():
                ft = feature +"["+ k +"] "
                res = process_adverb(v,f1,f2)
                set_(obj, snake_case(ft + "Len(f1)"), res[0])
                set_(obj, snake_case(ft + "Len(f2)"), res[1])
                set_(obj, snake_case(ft + "Abs Diff"), res[2])
                set_(obj, snake_case(ft + "Len Inter"), res[3])
        else:
            res = process_adverb(value,f1,f2)
            ft = feature
            set_(obj, snake_case(ft + "Len(f1)"), res[0])
            set_(obj, snake_case(ft + "Len(f2)"), res[1])
            set_(obj, snake_case(ft + "Abs Diff"), res[2])
            set_(obj, snake_case(ft + "Len Inter"), res[3])

    return obj
