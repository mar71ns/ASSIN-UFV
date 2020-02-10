# ASSIN-UFV

## Este repositório são os arquivos do trabalho de dissertação apresentado à Universidade Federal de Viçosa.

Mestrando: Gustavo Soares Martins.
Orientador: Alcione de Paiva Oliveira.


### Arquivos:
O Corpus ASSIN é dividido em 6 arquivos. 3 são em idioma 'ptbr' e 3 em 'ptpt'.
Paraca cada idioma há um arquivo train, dev e test.

Para os outros arquivos ex: nome_arquivo = 'assin-ptbr-train'

Para cada um dos 6 arquivos do corpus original foram gerados:
1- Arquivo com os valores do pré-processamento (nome_arquivo + '-processed.json')
2- Arquivo com os labels para a tarefa de Similaridade Semântica (nome_arquivo + '-labels.json')
3- Arquivo com os labels para a tarefa de Inferência Textual (nome_arquivo + '-labels-classifiers.json')
4- Arquivo com os resultados dos modelos para a tarefa de Similaridade Semântica (nome_arquivo + '-results.txt')
5- Arquivo com os resultados dos modelos para a tarefa de Inferência Textual (nome_arquivo + '-results-classifier.txt')

O arquivo 1 possui os valores de diversas features geradas sobre cada par de frases do corpus.
Os arquivos 2 e 3 são utilizados para o aprendizado supervisionado dos regressores e classificadores, respectivamente.
Os arquivos 4 e 5 contém a configuração dos modelos e seus resultados para cada tarefa.

obs: Para realizar o treinamento em ambas variações do corpus fora criado o arquivo 'assin-train'.

### Modo de uso:

Os arquivos classificadores

