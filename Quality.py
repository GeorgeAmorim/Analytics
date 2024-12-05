import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

# Dados simulados para exemplo
data = pd.DataFrame({
    'Date': np.random.randint(1, 32, 100),
    'Time': np.random.randint(0, 24, 100),
    'CO(GT)': np.random.uniform(0.0, 10.0, 100),
    'PT08.S1(CO)': np.random.uniform(0.0, 5000.0, 100),
    'NMHC(GT)': np.random.uniform(0.0, 500.0, 100),
    'C6H6(GT)': np.random.uniform(0.0, 50.0, 100),
    'PT08.S2(NMHC)': np.random.uniform(0.0, 5000.0, 100),
    'NOx(GT)': np.random.uniform(0.0, 1000.0, 100),
    'PT08.S3(NOx)': np.random.uniform(0.0, 5000.0, 100),
    'NO2(GT)': np.random.uniform(0.0, 500.0, 100),
    'PT08.S4(NO2)': np.random.uniform(0.0, 5000.0, 100),
    'PT08.S5(O3)': np.random.uniform(0.0, 5000.0, 100),
    'T': np.random.uniform(-10.0, 50.0, 100),
    'RH': np.random.uniform(0.0, 100.0, 100),
    'AH': np.random.uniform(0.0, 30.0, 100)
})

# Separar características e alvo
X = data.drop(['Date', 'Time', 'CO(GT)'], axis=1)
y = data['CO(GT)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modelagem
model = xgb.XGBRegressor(n_estimators=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Iniciar a interface Streamlit
st.title('Previsão de Qualidade do Ar')

st.write("Insira as características do ar para exibir a previsão de CO(GT):")

# Campos de entrada para as características do ar com medidas de valor
st.write("**PT08.S1(CO) (0.0 - 5000.0):**")
pt08_s1_co = st.text_input('PT08.S1(CO)', value='2000.0')

st.write("**NMHC(GT) (0.0 - 500.0):**")
nmhc_gt = st.text_input('NMHC(GT)', value='100.0')

st.write("**C6H6(GT) (0.0 - 50.0):**")
c6h6_gt = st.text_input('C6H6(GT)', value='5.0')

st.write("**PT08.S2(NMHC) (0.0 - 5000.0):**")
pt08_s2_nmhc = st.text_input('PT08.S2(NMHC)', value='2000.0')

st.write("**NOx(GT) (0.0 - 1000.0):**")
nox_gt = st.text_input('NOx(GT)', value='300.0')

st.write("**PT08.S3(NOx) (0.0 - 5000.0):**")
pt08_s3_nox = st.text_input('PT08.S3(NOx)', value='2000.0')

st.write("**NO2(GT) (0.0 - 500.0):**")
no2_gt = st.text_input('NO2(GT)', value='50.0')

st.write("**PT08.S4(NO2) (0.0 - 5000.0):**")
pt08_s4_no2 = st.text_input('PT08.S4(NO2)', value='2000.0')

st.write("**PT08.S5(O3) (0.0 - 5000.0):**")
pt08_s5_o3 = st.text_input('PT08.S5(O3)', value='2000.0')

st.write("**Temperatura (-10.0 - 50.0):**")
t = st.text_input('Temperatura', value='20.0')

st.write("**Umidade Relativa (0.0 - 100.0):**")
rh = st.text_input('Umidade Relativa', value='50.0')

st.write("**Umidade Absoluta (0.0 - 30.0):**")
ah = st.text_input('Umidade Absoluta', value='10.0')

# Botão para enviar os dados e gerar a previsão
if st.button('Enviar'):
    try:
        # Preparar dados de entrada
        input_data = np.array([[float(pt08_s1_co), float(nmhc_gt), float(c6h6_gt), float(pt08_s2_nmhc), float(nox_gt), float(pt08_s3_nox),
                                float(no2_gt), float(pt08_s4_no2), float(pt08_s5_o3), float(t), float(rh), float(ah)]])

        # Verificar se o formato de input_data corresponde ao de X
        if input_data.shape[1] == X.shape[1]:
            # Previsão
            predicted_co_gt = model.predict(input_data)[0]
            
            # Mapear o valor previsto para uma categoria de qualidade do ar
            if predicted_co_gt <= 2.0:
                air_quality = 'Boa'
                color = 'green'
                recommendation = 'A qualidade do ar está boa. Continue monitorando para garantir que se mantenha assim.'
            elif predicted_co_gt <= 3.0:
                air_quality = 'Moderada'
                color = 'yellow'
                recommendation = 'A qualidade do ar está moderada. Aguarde uns intantes ate realizar as proximas atividades.'
            elif predicted_co_gt <= 4.0:
                air_quality = 'Ruim'
                color = 'orange'
                recommendation = 'A qualidade do ar está ruim. Pessoas sensíveis podem ser afetadas. Sugestão: Abra janelas e deixe o ar correr.'
            else:
                air_quality = 'Muito Ruim'
                color = 'red'
                recommendation = 'A qualidade do ar está muito ruim. Todos devem sair do local e acionar o orgão de segurança responsavel.'

            # Exibir a previsão de CO(GT) e a qualidade do ar com scatter plot e linhas de referência
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(['Previsão de CO(GT)'], [predicted_co_gt], color=color, s=100)
            ax.axhline(y=1.0, color='green', linestyle='--', label='Boa (<=1.0)')
            ax.axhline(y=2.0, color='yellow', linestyle='--', label='Moderada (<=2.0)')
            ax.axhline(y=3.0, color='orange', linestyle='--', label='Ruim (<=3.0)')
            ax.axhline(y=4.0, color='red', linestyle='--', label='Muito Ruim (>3.0)')
            ax.set_ylim(0, 10)
            ax.set_ylabel('CO(GT)')
            ax.set_title('Previsão de Qualidade do Ar')
            ax.text(0, predicted_co_gt + 0.3, f'{predicted_co_gt:.2f}', color=color, ha='center', va='bottom', fontsize=12)
            ax.legend()

            st.pyplot(fig)
            st.write(f"A qualidade do ar é: **{air_quality}**")
            st.write(f"**Recomendação:** {recommendation}")
        else:
            st.write(f"Erro: O número de características fornecidas não corresponde ao esperado pelo modelo. Esperado: {X.shape[1]}, Recebido: {input_data.shape[1]}")
    except ValueError:
        st.write("Por favor, insira valores numéricos válidos.")
