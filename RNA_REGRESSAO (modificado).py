def preProcessamentoDadosCategoricos(previsores, indexesCategoricos):
    from sklearn.preprocessing import LabelEncoder
    LEprevisores = LabelEncoder()
    
    for index in indexesCategoricos:
        previsores[:,index] = LEprevisores.fit_transform(previsores[:,index])

def removerColuna(df, nomeColunas):
    df = df.drop(columns=nomeColunas)
    
    return df
        
def obterConjuntoDeTreinamento(previsores, f):
    conjuntos = {}
    
    from sklearn.model_selection import train_test_split
    previsores_treinamento, previsores_teste, f_treinamento, f_teste = train_test_split(previsores, 
              f, test_size=0.25, random_state=0)
    
    conjuntos["previsores_treinamento"] = previsores_treinamento
    conjuntos["previsores_teste"] = previsores_teste
    conjuntos["f_treinamento"] = f_treinamento
    conjuntos["f_teste"] = f_teste
    
    return conjuntos
    
    
def nnParaSolverEspecifico(numeroNeuronios, solver, previsores, f):
    from sklearn.neural_network import MLPRegressor
    from sklearn import metrics
    
    activationParms = ["identity", "logistic"]#, "tanh", "relu"]
    absErrors = {}
    
    for item in activationParms:
        conjuntos = obterConjuntoDeTreinamento(previsores, f)
        
        nn = MLPRegressor(hidden_layer_sizes=(numeroNeuronios,), 
            solver= solver, activation=item, max_iter=10000,tol=0.001)
    
        treino = nn.fit(conjuntos["previsores_treinamento"], 
                        conjuntos["f_treinamento"])
    
        teste = nn.predict(conjuntos["previsores_teste"])
    
       
        mae = metrics.mean_absolute_error(conjuntos["f_teste"],teste)
        key = "erro: " + str(item) + "-" + str(solver) + "-" + str(numeroNeuronios)
        absErrors[key] = mae
    
    return absErrors
    
    
    
import pandas as pd
base = pd.read_excel('CARROS_USADOS_TREINO.xls')

colunasAremover = ["carID"]
base = removerColuna(base, colunasAremover)

previsores = base.iloc[:, 0:9].values
f = base.iloc[:, 9].values

ColunasCategoricas = [0, 1, 3, 5]


preProcessamentoDadosCategoricos(previsores, ColunasCategoricas)

estatistica = base.describe()

# Separação das variáveis previsoras e f(x)

# NORMALIZAÇÃO DA BASE DE DADOS 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

errors = nnParaSolverEspecifico(10, 'adam', previsores, f)
