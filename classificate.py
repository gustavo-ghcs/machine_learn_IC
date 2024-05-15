import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

# Definir os caminhos
DATANAME = "3DML_urban_point_cloud.xyz"

# Carregar os dados
pcd = pd.read_csv(DATANAME, delimiter=' ')
pcd.dropna(inplace=True)

# Preparar os dados
labels = pcd['Classification']
features = pcd[['X', 'Y', 'Z', 'R', 'G', 'B']]
features_scaled = MinMaxScaler().fit_transform(features)

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.4)

# Treinar o classificador
rf_classifier = RandomForestClassifier(n_estimators=10)
rf_classifier.fit(X_train, y_train)

# Avaliar o modelo
rf_predictions = rf_classifier.predict(X_test)
print(classification_report(y_test, rf_predictions, target_names=['ground', 'vegetation', 'buildings']))

# Salvar o modelo treinado
model_filename = "aprendizado.pkl"
dump(rf_classifier, model_filename)

print(f"Modelo salvo em: {model_filename}")


# _________________________________________________________________________________________________________________________

# Definir o caminho para o modelo salvo e para os novos dados
path = "aprendizado.pkl"
model_path = path
data = "TESTE.xyz"
new_data_path = data

# Carregar o modelo treinado
rf_classifier = load(model_filename)

# Carregar e preparar a nova nuvem de pontos
# Lê o arquivo ignorando a primeira linha (cabeçalho comentado)
new_pcd = pd.read_csv(new_data_path, delimiter=' ', header=None, skiprows=1)

# Define os nomes das colunas conforme o cabeçalho informado
columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'Intensity', 'Return_Number', 'Number_Of_Returns', 'Scan_Direction_Flag', 'Classification', 'Scan_Angle', 'User_Data', 'Point_Source_ID', 'Gps_Time', 'Near_Infrared']
new_pcd.columns = columns

# Selecionar somente as colunas relevantes para a classificação
new_features = new_pcd[['X', 'Y', 'Z', 'R', 'G', 'B']]
new_features_scaled = MinMaxScaler().fit_transform(new_features)

# Classificar a nova nuvem de pontos
new_predictions = rf_classifier.predict(new_features_scaled)

# Adicionar as previsões ao DataFrame e salvar os resultados
new_pcd['Classification'] = new_predictions
output_path = "classificado.xyz"
new_pcd.to_csv(output_path, index=False, sep=' ')

print(f"Nuvem de pontos classificada salva em: {output_path}")