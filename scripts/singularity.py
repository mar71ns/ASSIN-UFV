from text_core import *
from extractor_core import *
from pydash import flatten_deep, values, get, concat
print(pad("",80,'.'))
print(pad("",80,'-'))
print(pad("",80,'-'))
print(pad("=== SINGULARITY ===",80,'~'))
print(pad("",80,'-'))
print(pad("",80,'.'))

####################################################
##############------LÊ DATASET------################
####################################################
f = open("assin-ptbr-dev.txt", encoding="utf8")
frases = f.readlines()
ind = 0 * 3
f1 = frases[ind]
f2 = frases[ind+1]
f1 = "Sebastian Vettel garantiu a pole-position para o Grande Prémio de Singapura de Fórmula 1."
f2 = "O Grande Prémio de Singapura de Fórmula 1 tem início marcado para as 13h00 de domingo."

#######################################################
##############------PROCESSAMENTO------################
#######################################################

#print (f1,"\n",f2)

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

# ft0_4= "Tokens Soft Cosine"
# r0_4= soft_cos_similarity(tk1,tk2)

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

# ft1_4 = "\tWords Soft Cosine"
# r1_4= soft_cos_similarity(wd1,wd2)

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

# ft3_4 = "\tWord Lemas Soft Cosine"
# r3_4= soft_cos_similarity(lm1,lm1)

ft3_5 = "\tWord Lemas Dice"
r3_5= dice_similarity(lm1,lm2)

ft3_6 = "\tWord Lemas Len f1"
r3_6= len(lm1)

ft3_7 = "\tWord Lemas Len f2"
r3_7= len(lm2)

ft3_8 = "\tWord Lemas Len Proportion"
r3_8= get_len_proportion_from_2_sents(lm1,lm2)

#----------------------------------------------------------------
#
# ft4_0 = "\tWord Tag Intersection"
# tg1 = get_word_tag_from_sent(f1)
# tg2 = get_word_tag_from_sent(f2)
# r4_0=len_intersection(tg1,tg2)
# #print_intersection_detail(ft4_0,tg1,tg2)
#
# ft4_1 = "\tWord Tag Jaccard"
# r4_1=jaccard_similarity(tg1,tg2)
#
# ft4_2 = "\tWord Tag Overlap"
# r4_2=overlap_similarity(tg1,tg2)
#
# ft4_3 = "\tWord Tag Cosine"
# r4_3= cos_similarity(tg1,tg2)
#
# # ft4_4 = "\tTagger Soft Cosine"
# # r4_4= soft_cos_similarity(tg1,tg2)
#
# ft4_5 = "\tWord Tag Dice"
# r4_5= dice_similarity(tg1,tg2)
#
# ft4_6 = "\tWord Tag Len f1"
# r4_6= len(tg1)
#
# ft4_7 = "\tWord Tag Len f2"
# r4_7= len(tg2)
#
# ft4_8 = "\tWord Tag Len Proportion"
# r4_8= get_len_proportion_from_2_sents(tg1,tg2)

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

# ft5_4 = "\tWord Pos Soft Cosine"
# r5_4= soft_cos_similarity(ptg1,ptg2)

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

# ft6_4 = "\tWord Stems Soft Cosine"
# r6_4= soft_cos_similarity(st1,st2)

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
#
# #ft7_4 = "\tStems Soft Cosine"
# #r7_4= soft_cos_similarity(we1,we2)
#
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

#ft9_4 = "\tDependency Soft Cosine"
#r9_4= soft_cos_similarity(dp1,dp2)

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

#ft10_4 = "\tWord Tag Soft Cosine"
#r10_4= soft_cos_similarity(wt1,wt2)

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

#ft11_4 = "\tSinonimos Soft Cosine"
#r11_4= soft_cos_similarity(sin1,sin2)

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

#ft12_4 = "\tAntonimos Soft Coante"
#r12_4= soft_cos_similarity(ant1,ant2)

ft12_5 = "\tAntonimos Dice"
r12_5= dice_similarity(ant1,ant2)

ft12_6 = "\tAntonimos Len f1"
r12_6= len(ant1)

ft12_7 = "\tAntonimos Len f2"
r12_7= len(ant2)

ft12_8 = "\tAntonimos Len Proportion"
r12_8= get_len_proportion_from_2_sents(ant1,ant2)
#----------------------------------------------------------------

if True:
    #######################################################
    ##############------RELATÓRIO------####################
    #######################################################
    print('\n')
    print(pad("",55,'-'))
    print(pad("RELATÓRIO",55,'-'))
    print(pad("",55,'-'))

    print(f1)
    print(f2)

    print(pad("",55,'-'))   # Tokens
    print('{:40s} {:3d}'.format(ft0_0, r0_0))           # 0 -Intersection
    print('{:40s} {:3.4f}'.format(ft0_1, r0_1))         # 1 -Jaccard
    print('{:40s} {:3.4f}'.format(ft0_2, r0_2))         # 2 -Overlap
    print('{:40s} {:3.4f}'.format(ft0_3, float(r0_3)))  # 3 -Cosine
    #print('{:40s} {:3.4f}'.format(ft0_4, float(r0_4))) # 4 -Soft Cosine
    print('{:40s} {:3.4f}'.format(ft0_5, r0_5))         # 5 -Dice
    print('{:40s} {:3.0f}'.format(ft0_6, r0_6))         # 6 -Len f1
    print('{:40s} {:3.0f}'.format(ft0_7, r0_7))         # 7 -Len f2
    print('{:40s} {:3.4f}'.format(ft0_8, r0_8))         # 8 -f1/f2 or f2/f1

    print(pad("",55,'-'))   # Words
    print('{:40s} {:3d}'.format(ft1_0, r1_0))
    print('{:40s} {:3.4f}'.format(ft1_1, r1_1))
    print('{:40s} {:3.4f}'.format(ft1_2, r1_2))
    print('{:40s} {:3.4f}'.format(ft1_3, float(r1_3)))
    print('{:40s} {:3.4f}'.format(ft1_5, r1_5))
    print('{:40s} {:3.0f}'.format(ft1_6, r1_6))
    print('{:40s} {:3.0f}'.format(ft1_7, r1_7))
    print('{:40s} {:3.4f}'.format(ft1_8, r1_8))

    print(pad("",55,'-'))   # Numbers
    print('{:40s} {:3d}'.format(ft2_0, r2_0))
    print('{:40s} {:3.4f}'.format(ft2_1, r2_1))
    print('{:40s} {:3.4f}'.format(ft2_2, r2_2))
    print('{:40s} {:3.4f}'.format(ft2_5, r2_5))
    print('{:40s} {:3.0f}'.format(ft2_6, r2_6))
    print('{:40s} {:3.0f}'.format(ft2_7, r2_7))
    print('{:40s} {:3.4f}'.format(ft2_8, r2_8))

    print(pad("",55,'-'))   # Lemas
    print('{:40s} {:3d}'.format(ft3_0, r3_0))
    print('{:40s} {:3.4f}'.format(ft3_1, r3_1))
    print('{:40s} {:3.4f}'.format(ft3_2, r3_2))
    print('{:40s} {:3.4f}'.format(ft3_3, float(r3_3)))
    #print('{:40s} {:3.4f}'.format(ft3_4, float(r3_4)))
    print('{:40s} {:3.4f}'.format(ft3_5, r3_5))
    print('{:40s} {:3.0f}'.format(ft3_6, r3_6))
    print('{:40s} {:3.0f}'.format(ft3_7, r3_7))
    print('{:40s} {:3.4f}'.format(ft3_8, r3_8))

    # print(pad("",55,'-'))   # Tagger
    # print('{:40s} {:3d}'.format(ft4_0, r4_0))
    # print('{:40s} {:3.4f}'.format(ft4_1, r4_1))
    # print('{:40s} {:3.4f}'.format(ft4_2, r4_2))
    # print('{:40s} {:3.4f}'.format(ft4_3, float(r4_3)))
    # #print('{:40s} {:3.4f}'.format(ft4_4, float(r4_4)))
    # print('{:40s} {:3.4f}'.format(ft4_5, r4_5))
    # print('{:40s} {:3.0f}'.format(ft4_6, r4_6))
    # print('{:40s} {:3.0f}'.format(ft4_7, r4_7))
    # print('{:40s} {:3.4f}'.format(ft4_8, r4_8))

    print(pad("",55,'-'))   # Pos Tag
    print('{:40s} {:3d}'.format(ft5_0, r5_0))
    print('{:40s} {:3.4f}'.format(ft5_1, r5_1))
    print('{:40s} {:3.4f}'.format(ft5_2, r5_2))
    print('{:40s} {:3.4f}'.format(ft5_3, float(r5_3)))
    #print('{:40s} {:3.4f}'.format(ft5_4, float(r5_4)))
    print('{:40s} {:3.4f}'.format(ft5_5, r5_5))
    print('{:40s} {:3.0f}'.format(ft5_6, r5_6))
    print('{:40s} {:3.0f}'.format(ft5_7, r5_7))
    print('{:40s} {:3.4f}'.format(ft5_8, r5_8))
    print('{:40s} {:3d}'.format(ft5_9, r5_9))     #ADJ
    print('{:40s} {:3d}'.format(ft5_10, r5_10))
    print('{:40s} {:3d}'.format(ft5_11, r5_11))
    print('{:40s} {:3.4f}'.format(ft5_12, r5_12))
    print('{:40s} {:3.4f}'.format(ft5_13, r5_13))
    print('{:40s} {:3.4f}'.format(ft5_14, r5_14))
    print('{:40s} {:3d}'.format(ft5_15, r5_15))     #ADP
    print('{:40s} {:3d}'.format(ft5_16, r5_16))
    print('{:40s} {:3d}'.format(ft5_17, r5_17))
    print('{:40s} {:3.4f}'.format(ft5_18, r5_18))
    print('{:40s} {:3.4f}'.format(ft5_19, r5_19))
    print('{:40s} {:3.4f}'.format(ft5_20, r5_20))
    print('{:40s} {:3d}'.format(ft5_21, r5_21))     #ADV
    print('{:40s} {:3d}'.format(ft5_22, r5_22))
    print('{:40s} {:3d}'.format(ft5_23, r5_23))
    print('{:40s} {:3.4f}'.format(ft5_24, r5_24))
    print('{:40s} {:3.4f}'.format(ft5_25, r5_25))
    print('{:40s} {:3.4f}'.format(ft5_26, r5_26))
    print('{:40s} {:3d}'.format(ft5_27, r5_27))     #AUX
    print('{:40s} {:3d}'.format(ft5_28, r5_28))
    print('{:40s} {:3d}'.format(ft5_29, r5_29))
    print('{:40s} {:3.4f}'.format(ft5_30, r5_30))
    print('{:40s} {:3.4f}'.format(ft5_31, r5_31))
    print('{:40s} {:3.4f}'.format(ft5_32, r5_32))
    print('{:40s} {:3d}'.format(ft5_33, r5_33))     #CONJ
    print('{:40s} {:3d}'.format(ft5_34, r5_34))
    print('{:40s} {:3d}'.format(ft5_35, r5_35))
    print('{:40s} {:3.4f}'.format(ft5_36, r5_36))
    print('{:40s} {:3.4f}'.format(ft5_37, r5_37))
    print('{:40s} {:3.4f}'.format(ft5_38, r5_38))
    print('{:40s} {:3d}'.format(ft5_39, r5_39))     #CCONJ
    print('{:40s} {:3d}'.format(ft5_40, r5_40))
    print('{:40s} {:3d}'.format(ft5_41, r5_41))
    print('{:40s} {:3.4f}'.format(ft5_42, r5_42))
    print('{:40s} {:3.4f}'.format(ft5_43, r5_43))
    print('{:40s} {:3.4f}'.format(ft5_44, r5_44))
    print('{:40s} {:3d}'.format(ft5_45, r5_45))     #DET
    print('{:40s} {:3d}'.format(ft5_46, r5_46))
    print('{:40s} {:3d}'.format(ft5_47, r5_47))
    print('{:40s} {:3.4f}'.format(ft5_48, r5_48))
    print('{:40s} {:3.4f}'.format(ft5_49, r5_49))
    print('{:40s} {:3.4f}'.format(ft5_50, r5_50))
    print('{:40s} {:3d}'.format(ft5_51, r5_51))     #INTJ
    print('{:40s} {:3d}'.format(ft5_52, r5_52))
    print('{:40s} {:3d}'.format(ft5_53, r5_53))
    print('{:40s} {:3.4f}'.format(ft5_54, r5_54))
    print('{:40s} {:3.4f}'.format(ft5_55, r5_55))
    print('{:40s} {:3.4f}'.format(ft5_56, r5_56))
    print('{:40s} {:3d}'.format(ft5_57, r5_57))     #NOUN
    print('{:40s} {:3d}'.format(ft5_58, r5_58))
    print('{:40s} {:3d}'.format(ft5_59, r5_59))
    print('{:40s} {:3.4f}'.format(ft5_60, r5_60))
    print('{:40s} {:3.4f}'.format(ft5_61, r5_61))
    print('{:40s} {:3.4f}'.format(ft5_62, r5_62))
    print('{:40s} {:3d}'.format(ft5_63, r5_63))     #NUM
    print('{:40s} {:3d}'.format(ft5_64, r5_64))
    print('{:40s} {:3d}'.format(ft5_65, r5_65))
    print('{:40s} {:3.4f}'.format(ft5_66, r5_66))
    print('{:40s} {:3.4f}'.format(ft5_67, r5_67))
    print('{:40s} {:3.4f}'.format(ft5_68, r5_68))
    print('{:40s} {:3d}'.format(ft5_69, r5_69))     #PART
    print('{:40s} {:3d}'.format(ft5_70, r5_70))
    print('{:40s} {:3d}'.format(ft5_71, r5_71))
    print('{:40s} {:3.4f}'.format(ft5_72, r5_72))
    print('{:40s} {:3.4f}'.format(ft5_73, r5_73))
    print('{:40s} {:3.4f}'.format(ft5_74, r5_74))
    print('{:40s} {:3d}'.format(ft5_75, r5_75))     #PRON
    print('{:40s} {:3d}'.format(ft5_76, r5_76))
    print('{:40s} {:3d}'.format(ft5_77, r5_77))
    print('{:40s} {:3.4f}'.format(ft5_78, r5_78))
    print('{:40s} {:3.4f}'.format(ft5_79, r5_79))
    print('{:40s} {:3.4f}'.format(ft5_80, r5_80))
    print('{:40s} {:3d}'.format(ft5_81, r5_81))     #PROPN
    print('{:40s} {:3d}'.format(ft5_82, r5_82))
    print('{:40s} {:3d}'.format(ft5_83, r5_83))
    print('{:40s} {:3.4f}'.format(ft5_84, r5_84))
    print('{:40s} {:3.4f}'.format(ft5_85, r5_85))
    print('{:40s} {:3.4f}'.format(ft5_86, r5_86))
    print('{:40s} {:3d}'.format(ft5_87, r5_87))     #PUNCT
    print('{:40s} {:3d}'.format(ft5_88, r5_88))
    print('{:40s} {:3d}'.format(ft5_89, r5_89))
    print('{:40s} {:3.4f}'.format(ft5_90, r5_90))
    print('{:40s} {:3.4f}'.format(ft5_91, r5_91))
    print('{:40s} {:3.4f}'.format(ft5_92, r5_92))
    print('{:40s} {:3d}'.format(ft5_93, r5_93))     #SCONJ
    print('{:40s} {:3d}'.format(ft5_94, r5_94))
    print('{:40s} {:3d}'.format(ft5_95, r5_95))
    print('{:40s} {:3.4f}'.format(ft5_96, r5_96))
    print('{:40s} {:3.4f}'.format(ft5_97, r5_97))
    print('{:40s} {:3.4f}'.format(ft5_98, r5_98))
    print('{:40s} {:3d}'.format(ft5_99, r5_99))     #SYM
    print('{:40s} {:3d}'.format(ft5_100, r5_100))
    print('{:40s} {:3d}'.format(ft5_101, r5_101))
    print('{:40s} {:3.4f}'.format(ft5_102, r5_102))
    print('{:40s} {:3.4f}'.format(ft5_103, r5_103))
    print('{:40s} {:3.4f}'.format(ft5_104, r5_104))
    print('{:40s} {:3d}'.format(ft5_105, r5_105))     #VERB
    print('{:40s} {:3d}'.format(ft5_106, r5_106))
    print('{:40s} {:3d}'.format(ft5_107, r5_107))
    print('{:40s} {:3.4f}'.format(ft5_108, r5_108))
    print('{:40s} {:3.4f}'.format(ft5_109, r5_109))
    print('{:40s} {:3.4f}'.format(ft5_110, r5_110))
    print('{:40s} {:3d}'.format(ft5_111, r5_111))     #X
    print('{:40s} {:3d}'.format(ft5_112, r5_112))
    print('{:40s} {:3d}'.format(ft5_113, r5_113))
    print('{:40s} {:3.4f}'.format(ft5_114, r5_114))
    print('{:40s} {:3.4f}'.format(ft5_115, r5_115))
    print('{:40s} {:3.4f}'.format(ft5_116, r5_116))

    print(pad("",55,'-'))   # Stems
    print('{:40s} {:3d}'.format(ft6_0, r6_0))
    print('{:40s} {:3.4f}'.format(ft6_1, r6_1))
    print('{:40s} {:3.4f}'.format(ft6_2, r6_2))
    print('{:40s} {:3.4f}'.format(ft6_3, float(r6_3)))
    #print('{:40s} {:3.4f}'.format(ft6_4, float(r6_4)))
    print('{:40s} {:3.4f}'.format(ft6_5, r6_5))
    print('{:40s} {:3.0f}'.format(ft6_6, r6_6))
    print('{:40s} {:3.0f}'.format(ft6_7, r6_7))
    print('{:40s} {:3.4f}'.format(ft6_8, r6_8))

    print(pad("",55,'-'))   # Word Entities
    print('{:40s} {:3d}'.format(ft7_0, r7_0))
    print('{:40s} {:3.4f}'.format(ft7_1, r7_1))
    print('{:40s} {:3.4f}'.format(ft7_2, r7_2))
    print('{:40s} {:3.4f}'.format(ft7_3, float(r7_3)))
    #print('{:40s} {:3.4f}'.format(ft7_4, float(r7_4)))
    print('{:40s} {:3.4f}'.format(ft7_5, r7_5))
    print('{:40s} {:3.0f}'.format(ft7_6, r7_6))
    print('{:40s} {:3.0f}'.format(ft7_7, r7_7))
    print('{:40s} {:3.4f}'.format(ft7_8, r7_8))

    print(pad("",55,'-'))   # Dependency
    print('{:40s} {:3d}'.format(ft9_0, r9_0))
    print('{:40s} {:3.4f}'.format(ft9_1, r9_1))
    print('{:40s} {:3.4f}'.format(ft9_2, r9_2))
    print('{:40s} {:3.4f}'.format(ft9_3, float(r9_3)))
    #print('{:40s} {:3.4f}'.format(ft9_4, float(r9_4)))
    print('{:40s} {:3.4f}'.format(ft9_5, r9_5))
    print('{:40s} {:3.0f}'.format(ft9_6, r9_6))
    print('{:40s} {:3.0f}'.format(ft9_7, r9_7))
    print('{:40s} {:3.4f}'.format(ft9_8, r9_8))

    print(pad("",55,'-'))   # Word Tag
    print('{:40s} {:3d}'.format(ft10_0, r10_0))
    print('{:40s} {:3.4f}'.format(ft10_1, r10_1))
    print('{:40s} {:3.4f}'.format(ft10_2, r10_2))
    print('{:40s} {:3.4f}'.format(ft10_3, float(r10_3)))
    #print('{:40s} {:3.4f}'.format(ft10_4, float(r10_4)))
    print('{:40s} {:3.4f}'.format(ft10_5, r10_5))
    print('{:40s} {:3.0f}'.format(ft10_6, r10_6))
    print('{:40s} {:3.0f}'.format(ft10_7, r10_7))
    print('{:40s} {:3.4f}'.format(ft10_8, r10_8))

    print(pad("",55,'-'))   # Sinônimos
    print('{:40s} {:3d}'.format(ft11_0, r11_0))
    print('{:40s} {:3.4f}'.format(ft11_1, r11_1))
    print('{:40s} {:3.4f}'.format(ft11_2, r11_2))
    print('{:40s} {:3.4f}'.format(ft11_3, float(r11_3)))
    #print('{:40s} {:3.4f}'.format(ft11_4, float(r11_4)))
    print('{:40s} {:3.4f}'.format(ft11_5, r11_5))
    print('{:40s} {:3.0f}'.format(ft11_6, r11_6))
    print('{:40s} {:3.0f}'.format(ft11_7, r11_7))
    print('{:40s} {:3.4f}'.format(ft11_8, r11_8))

    print(pad("",55,'-'))   # Antônimos
    print('{:40s} {:3d}'.format(ft12_0, r12_0))
    print('{:40s} {:3.4f}'.format(ft12_1, r12_1))
    print('{:40s} {:3.4f}'.format(ft12_2, r12_2))
    print('{:40s} {:3.4f}'.format(ft12_3, float(r12_3)))
    #print('{:40s} {:3.4f}'.format(ft12_4, float(r12_4)))
    print('{:40s} {:3.4f}'.format(ft12_5, r12_5))
    print('{:40s} {:3.0f}'.format(ft12_6, r12_6))
    print('{:40s} {:3.0f}'.format(ft12_7, r12_7))
    print('{:40s} {:3.4f}'.format(ft12_8, r12_8))

    print(pad("",55,'-'))   # Entities
    print('{:40s} {:3d}'.format(ft8_0, r8_0))
    print('{:40s} {:3.4f}'.format(ft8_1, r8_1))
    print('{:40s} {:3.4f}'.format(ft8_2, r8_2))
    print('{:40s} {:3.4f}'.format(ft8_3, float(r8_3)))
    print('{:40s} {:3.4f}'.format(ft8_5, r8_5))
    print('{:40s} {:3.0f}'.format(ft8_6, r8_6))
    print('{:40s} {:3.0f}'.format(ft8_7, r8_7))
    print('{:40s} {:3.4f}'.format(ft8_8, r8_8))
    print('{:40s} {:3d}'.format(ft8_9, r8_9))     #PER
    print('{:40s} {:3d}'.format(ft8_10, r8_10))
    print('{:40s} {:3d}'.format(ft8_11, r8_11))
    print('{:40s} {:3d}'.format(ft8_12, r8_12))   #LOC
    print('{:40s} {:3d}'.format(ft8_13, r8_13))
    print('{:40s} {:3d}'.format(ft8_14, r8_14))
    print('{:40s} {:3d}'.format(ft8_15, r8_15))   #ORG
    print('{:40s} {:3d}'.format(ft8_16, r8_16))
    print('{:40s} {:3d}'.format(ft8_17, r8_17))
    print('{:40s} {:3d}'.format(ft8_18, r8_18))   #MISC
    print('{:40s} {:3d}'.format(ft8_19, r8_19))
    print('{:40s} {:3d}'.format(ft8_20, r8_20))
    print('{:40s} {:3d}'.format(ft8_21, r8_21))   #TIME
    print('{:40s} {:3d}'.format(ft8_22, r8_22))
    print('{:40s} {:3d}'.format(ft8_23, r8_23))
    print('{:40s} {:3d}'.format(ft8_24, r8_24))   #ORDINAL
    print('{:40s} {:3d}'.format(ft8_25, r8_25))
    print('{:40s} {:3d}'.format(ft8_26, r8_26))
    print('{:40s} {:3d}'.format(ft8_27, r8_27))   #DATE
    print('{:40s} {:3d}'.format(ft8_28, r8_28))
    print('{:40s} {:3d}'.format(ft8_29, r8_29))
    print('{:40s} {:3d}'.format(ft8_30, r8_30))   #CARDINAL
    print('{:40s} {:3d}'.format(ft8_31, r8_31))
    print('{:40s} {:3d}'.format(ft8_32, r8_32))
    print('{:40s} {:3d}'.format(ft8_33, r8_33))   #MONEY
    print('{:40s} {:3d}'.format(ft8_34, r8_34))
    print('{:40s} {:3d}'.format(ft8_35, r8_35))
    print('{:40s} {:3d}'.format(ft8_36, r8_36))   #QUANTITY
    print('{:40s} {:3d}'.format(ft8_37, r8_37))
    print('{:40s} {:3d}'.format(ft8_38, r8_38))

    print(pad("",55,'-'))
    #Imprime Adverbios
    adverbios = load_json("adverbios")
    for key, value in adverbios.items():
        feature = "\tADV-" + str(key) + " "
        if isinstance(value,(dict)):
            for k, v in value.items():
                ft = feature +"["+ k +"] "
                res = process_adverb(v,f1,f2)
                print_adverbs_result(ft, res)
        else:
            res = process_adverb(value,f1,f2)
            print_adverbs_result(feature, res)

    print(pad("",55,'-'))
