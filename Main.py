import streamlit as st
import json
import pandas as pd
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly

# --- Carrega modelo ---
def load_model():
    with open('modelo_O3_prophet.json', 'r') as file_in:
        modelo = model_from_json(json.load(file_in))
        return modelo
                
modelo = load_model()

# Adicionando textos ao layout do Streamlit
st.title('Previsão de Níveis de Ozônio (O3) Utilizando a Biblioteca Prophet')

st.caption('''Este projeto utiliza a biblioteca Prophet para prever os níveis de ozônio em ug/m3. O modelo
           criado foi treinado com dados até o dia 05/05/2023 e possui um erro de previsão (RMSE - Erro Quadrático Médio) igual a 17.43 nos dados de teste.
           O usuário pode inserir o número de dias para os quais deseja a previsão, e o modelo gerará um gráfico
           interativo contendo as estimativas baseadas em dados históricos de concentração de O3.
           Além disso, uma tabela será exibida com os valores estimados para cada dia.''')

st.subheader('Insira o número de dias para previsão:')

dias = int(st.number_input("Dias de previsão", min_value=1, value=1, step=1))

# --- Ação ---
if st.button("Prever"):
    futuro = modelo.make_future_dataframe(periods=dias, freq="D")
    previsao = modelo.predict(futuro)

    # Gráfico
    fig = plot_plotly(modelo, previsao)
    st.plotly_chart(fig, use_container_width=True)

    # Tabela (últimos 'dias' = horizonte futuro)
    tabela = previsao[["ds", "yhat"]].tail(dias).copy()
    tabela.columns = ["Data (Dia/Mês/Ano)", "O3 (ug/m3)"]
    tabela["Data (Dia/Mês/Ano)"] = pd.to_datetime(tabela["Data (Dia/Mês/Ano)"]).dt.strftime("%d-%m-%Y")
    tabela["O3 (ug/m3)"] = tabela["O3 (ug/m3)"].round(2)
    tabela.reset_index(drop=True, inplace=True)

    st.write(f"Previsões para os próximos {dias} dia(s):")
    st.dataframe(tabela, height=300)

    # Download CSV
    st.download_button(
        "Baixar CSV",
        data=tabela.to_csv(index=False).encode("utf-8"),
        file_name="previsao_ozonio.csv",
        mime="text/csv",
    )
else:
    st.info("Escolha o número de dias e clique em **Prever**.")
