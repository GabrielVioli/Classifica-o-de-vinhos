import pandas as pd
from pandas.core.frame import DataFrame 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

main_file = pd.read_csv(r'C:\Users\gabri\OneDrive\Área de Trabalho\code\format_files\vinho.csv')
dataframe = DataFrame(main_file)

dataframe['style'] = dataframe['style'].replace('red',0)
dataframe['style'] = dataframe['style'].replace('white',1)

y = dataframe['style']
x = dataframe.drop('style', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size = 0.10)

model = ExtraTreesClassifier()
model.fit(x_treino, y_treino)

result = model.score(x_teste, y_teste)

print('acuracia: ', result)
prevision = model.predict(x_teste[400:890])
print(prevision)

lista2 = []
for item2 in prevision:
    lista2.append(item2)

lista  = []
for item in y_teste[400:890]:
    lista.append(item)


print(lista)
print(lista2)

if lista == lista2:
    print(f'100% de acerto')
else:
    print('erros')
