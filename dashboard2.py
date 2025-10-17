import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
import plotly.express as px
from datetime import datetime
import io

# Configuração da página
st.set_page_config(
    page_title="Dashboard Pesquisa Scooters - Gramado",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">🛵 Dashboard de Pesquisa - Scooters em Gramado</div>', unsafe_allow_html=True)

# Função para carregar dados do CSV
@st.cache_data
def load_data_from_csv(uploaded_file):
    # Lê o arquivo CSV com o separador correto
    df = pd.read_csv(uploaded_file, sep=';', parse_dates=['data_registro'])
    
    # Limpeza e formatação dos dados
    df['data_registro'] = pd.to_datetime(df['data_registro'])
    
    # Mapeamento para labels mais amigáveis
    mapeamento_tempo = {
        'meia_diaria': 'Meia Diária',
        'diaria_completa': 'Diária Completa',
        'estadia_completa': 'Estadia Completa',
        '2_3_dias': '2-3 Dias'
    }
    
    mapeamento_uso = {
        'lazer_passeios': 'Lazer e Passeios',
        'transporte_rapido': 'Transporte Rápido',
        'mobilidade_essencial': 'Mobilidade Essencial',
        'explorar_cidades': 'Explorar Cidades'
    }
    
    mapeamento_fator = {
        'preco': 'Preço',
        'delivery': 'Delivery',
        'conforto': 'Conforto',
        'roteiros': 'Roteiros'
    }
    
    mapeamento_preco = {
        'r121_160': 'R$ 121-160',
        'r80_120': 'R$ 80-120'
    }
    
    mapeamento_idade = {
        '18_25': '18-25 anos',
        '26_37': '26-37 anos',
        '38_49': '38-49 anos',
        '50+': '50+ anos'
    }
    
    mapeamento_viagem = {
        'familia': 'Família',
        'sozinho': 'Sozinho',
        'casal': 'Casal',
        'grupo_amigos': 'Grupo de Amigos'
    }
    
    mapeamento_seguro = {
        'essencial': 'Essencial',
        'se_barato': 'Se for Barato',
        'nao_preciso': 'Não Preciso'
    }
    
    mapeamento_interesse = {
        'sim': 'Sim',
        'nao': 'Não',
        'talvez_preco': 'Talvez (Preço)'
    }
    
    # Aplicar mapeamentos
    df['q2_tempo_aluguel'] = df['q2_tempo_aluguel'].map(mapeamento_tempo)
    df['q3_uso_principal'] = df['q3_uso_principal'].map(mapeamento_uso)
    df['q4_fator_decisivo'] = df['q4_fator_decisivo'].map(mapeamento_fator)
    df['q5_preco_diaria'] = df['q5_preco_diaria'].map(mapeamento_preco)
    df['q6_faixa_etaria'] = df['q6_faixa_etaria'].map(mapeamento_idade)
    df['q7_tipo_viagem'] = df['q7_tipo_viagem'].map(mapeamento_viagem)
    df['q8_valoriza_seguro'] = df['q8_valoriza_seguro'].map(mapeamento_seguro)
    df['q1_interesse'] = df['q1_interesse'].map(mapeamento_interesse)
    
    return df

# Upload do arquivo
uploaded_file = st.sidebar.file_uploader("📁 Carregar arquivo CSV", type=['csv'])

if uploaded_file is not None:
    df = load_data_from_csv(uploaded_file)
    
    # Sidebar com filtros
    st.sidebar.title("🔍 Filtros")
    st.sidebar.markdown("---")
    
    # Filtro por interesse
    interesse_options = ["Todos"] + list(df['q1_interesse'].unique())
    selected_interesse = st.sidebar.selectbox("Interesse", interesse_options)
    
    # Filtro por faixa etária
    faixa_etaria_options = ["Todos"] + list(df['q6_faixa_etaria'].unique())
    selected_faixa_etaria = st.sidebar.selectbox("Faixa Etária", faixa_etaria_options)
    
    # Filtro por tipo de viagem
    tipo_viagem_options = ["Todos"] + list(df['q7_tipo_viagem'].unique())
    selected_tipo_viagem = st.sidebar.selectbox("Tipo de Viagem", tipo_viagem_options)
    
    # Filtro por preço
    preco_options = ["Todos"] + list(df['q5_preco_diaria'].unique())
    selected_preco = st.sidebar.selectbox("Faixa de Preço", preco_options)
    
    # Aplicar filtros
    filtered_df = df.copy()
    if selected_interesse != "Todos":
        filtered_df = filtered_df[filtered_df['q1_interesse'] == selected_interesse]
    if selected_faixa_etaria != "Todos":
        filtered_df = filtered_df[filtered_df['q6_faixa_etaria'] == selected_faixa_etaria]
    if selected_tipo_viagem != "Todos":
        filtered_df = filtered_df[filtered_df['q7_tipo_viagem'] == selected_tipo_viagem]
    if selected_preco != "Todos":
        filtered_df = filtered_df[filtered_df['q5_preco_diaria'] == selected_preco]
    
    # Métricas principais
    st.markdown('<div class="section-header">📊 Métricas Principais</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_respostas = len(filtered_df)
        st.metric("Total de Respostas", total_respostas)
        
    with col2:
        taxa_interesse = (filtered_df['q1_interesse'].isin(['Sim']).sum() / len(filtered_df)) * 100
        st.metric("Taxa de Interesse", f"{taxa_interesse:.1f}%")
        
    with col3:
        uso_lazer = (filtered_df['q3_uso_principal'] == 'Lazer e Passeios').sum()
        st.metric("Uso para Lazer", uso_lazer)
        
    with col4:
        if not filtered_df.empty:
            preco_medio = filtered_df['q5_preco_diaria'].value_counts().index[0]
            st.metric("Faixa de Preço Mais Popular", preco_medio)
        else:
            st.metric("Faixa de Preço Mais Popular", "N/A")
    
    # Gráfico 1: Distribuição de Interesse e Faixa Etária
    st.markdown('<div class="section-header">🎯 Análise de Interesse e Perfil</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not filtered_df.empty:
            interesse_data = filtered_df['q1_interesse'].value_counts().reset_index()
            interesse_data.columns = ['value', 'count']
            
            options = {
                "title": {"text": "Distribuição de Interesse", "left": "center"},
                "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b}: {c} ({d}%)"},
                "legend": {"orient": "vertical", "left": "left"},
                "series": [
                    {
                        "name": "Interesse",
                        "type": "pie",
                        "radius": "50%",
                        "data": [
                            {"value": count, "name": value}
                            for value, count in zip(interesse_data['value'], interesse_data['count'])
                        ],
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowOffsetX": 0,
                                "shadowColor": "rgba(0, 0, 0, 0.5)",
                            }
                        },
                    }
                ],
            }
            st_echarts(options=options, height="400px")
    
    with col2:
        if not filtered_df.empty:
            # Gráfico de Faixa Etária
            faixa_etaria_data = filtered_df['q6_faixa_etaria'].value_counts().reset_index()
            faixa_etaria_data.columns = ['faixa_etaria', 'count']
            
            options = {
                "title": {"text": "Distribuição por Faixa Etária", "left": "center"},
                "tooltip": {"trigger": "axis"},
                "xAxis": {
                    "type": "category",
                    "data": faixa_etaria_data['faixa_etaria'].tolist(),
                },
                "yAxis": {"type": "value"},
                "series": [
                    {
                        "data": faixa_etaria_data['count'].tolist(),
                        "type": "bar",
                        "itemStyle": {
                            "color": {
                                "type": "linear",
                                "x": 0,
                                "y": 0,
                                "x2": 0,
                                "y2": 1,
                                "colorStops": [
                                    {"offset": 0, "color": "#5470c6"},
                                    {"offset": 1, "color": "#91cc75"},
                                ],
                            }
                        },
                    }
                ],
            }
            st_echarts(options=options, height="400px")
    
    # Gráfico 2: Tempo de Aluguel e Uso Principal
    st.markdown('<div class="section-header">⏰ Padrões de Uso</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not filtered_df.empty:
            tempo_aluguel_data = filtered_df['q2_tempo_aluguel'].value_counts().reset_index()
            tempo_aluguel_data.columns = ['tempo', 'count']
            
            options = {
                "title": {"text": "Tempo de Aluguel Preferido", "left": "center"},
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                "series": [
                    {
                        "name": "Tempo de Aluguel",
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "avoidLabelOverlap": False,
                        "itemStyle": {
                            "borderRadius": 10,
                            "borderColor": "#fff",
                            "borderWidth": 2,
                        },
                        "label": {"show": False, "position": "center"},
                        "emphasis": {
                            "label": {"show": True, "fontSize": "18", "fontWeight": "bold"}
                        },
                        "labelLine": {"show": False},
                        "data": [
                            {"value": count, "name": tempo}
                            for tempo, count in zip(tempo_aluguel_data['tempo'], tempo_aluguel_data['count'])
                        ],
                    }
                ],
            }
            st_echarts(options=options, height="400px")
    
    with col2:
        if not filtered_df.empty:
            uso_principal_data = filtered_df['q3_uso_principal'].value_counts().reset_index()
            uso_principal_data.columns = ['uso', 'count']
            
            options = {
                "title": {"text": "Uso Principal dos Scooters", "left": "center"},
                "tooltip": {"trigger": "axis"},
                "xAxis": {
                    "type": "category",
                    "data": uso_principal_data['uso'].tolist(),
                    "axisLabel": {"rotate": 45},
                },
                "yAxis": {"type": "value"},
                "series": [
                    {
                        "data": uso_principal_data['count'].tolist(),
                        "type": "bar",
                        "itemStyle": {"color": "#ee6666"},
                    }
                ],
            }
            st_echarts(options=options, height="400px")
    
    # Gráfico 3: Fatores Decisórios e Preços
    st.markdown('<div class="section-header">💰 Fatores de Decisão e Preços</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not filtered_df.empty:
            fator_data = filtered_df['q4_fator_decisivo'].value_counts().reset_index()
            fator_data.columns = ['fator', 'count']
            
            options = {
                "title": {"text": "Fatores Decisórios na Escolha", "left": "center"},
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "xAxis": {"type": "value"},
                "yAxis": {
                    "type": "category",
                    "data": fator_data['fator'].tolist(),
                    "inverse": True,
                },
                "series": [
                    {
                        "data": fator_data['count'].tolist(),
                        "type": "bar",
                        "itemStyle": {
                            "color": {
                                "type": "linear",
                                "x": 0,
                                "y": 0,
                                "x2": 1,
                                "y2": 0,
                                "colorStops": [
                                    {"offset": 0, "color": "#fac858"},
                                    {"offset": 1, "color": "#73c0de"},
                                ],
                            }
                        },
                    }
                ],
            }
            st_echarts(options=options, height="400px")
    
    with col2:
        if not filtered_df.empty:
            preco_data = filtered_df['q5_preco_diaria'].value_counts().reset_index()
            preco_data.columns = ['preco', 'count']
            
            options = {
                "title": {"text": "Faixas de Preço Preferidas", "left": "center"},
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                "series": [
                    {
                        "name": "Preço Diária",
                        "type": "pie",
                        "radius": "70%",
                        "data": [
                            {"value": count, "name": preco}
                            for preco, count in zip(preco_data['preco'], preco_data['count'])
                        ],
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowOffsetX": 0,
                                "shadowColor": "rgba(0, 0, 0, 0.5)",
                            }
                        },
                    }
                ],
            }
            st_echarts(options=options, height="400px")
    
    # Gráfico 4: Tipo de Viagem e Seguro
    st.markdown('<div class="section-header">👥 Perfil dos Usuários</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not filtered_df.empty:
            viagem_data = filtered_df['q7_tipo_viagem'].value_counts().reset_index()
            viagem_data.columns = ['viagem', 'count']
            
            options = {
                "title": {"text": "Tipo de Viagem", "left": "center"},
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                "series": [
                    {
                        "name": "Tipo de Viagem",
                        "type": "pie",
                        "radius": "60%",
                        "data": [
                            {"value": count, "name": viagem}
                            for viagem, count in zip(viagem_data['viagem'], viagem_data['count'])
                        ],
                        "roseType": "radius",
                        "itemStyle": {
                            "borderRadius": 8
                        },
                    }
                ],
            }
            st_echarts(options=options, height="400px")
    
    with col2:
        if not filtered_df.empty:
            seguro_data = filtered_df['q8_valoriza_seguro'].value_counts().reset_index()
            seguro_data.columns = ['seguro', 'count']
            
            options = {
                "title": {"text": "Valorização do Seguro", "left": "center"},
                "tooltip": {"trigger": "axis"},
                "xAxis": {
                    "type": "category",
                    "data": seguro_data['seguro'].tolist(),
                },
                "yAxis": {"type": "value"},
                "series": [
                    {
                        "data": seguro_data['count'].tolist(),
                        "type": "bar",
                        "itemStyle": {"color": "#91cc75"},
                    }
                ],
            }
            st_echarts(options=options, height="400px")
    
    # Análise Estatística e Insights
    st.markdown('<div class="section-header">📈 Análise Estatística e Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Estatísticas Detalhadas")
        
        if not filtered_df.empty:
            # Taxas calculadas
            total_respostas = len(filtered_df)
            taxa_sim = (filtered_df['q1_interesse'] == 'Sim').sum() / total_respostas * 100
            taxa_talvez = (filtered_df['q1_interesse'] == 'Talvez (Preço)').sum() / total_respostas * 100
            taxa_nao = (filtered_df['q1_interesse'] == 'Não').sum() / total_respostas * 100
            
            st.write(f"**Taxa de Conversão (Sim):** `{taxa_sim:.1f}%`")
            st.write(f"**Interesse Condicional (Talvez):** `{taxa_talvez:.1f}%`")
            st.write(f"**Taxa de Rejeição (Não):** `{taxa_nao:.1f}%`")
            
            # Uso mais popular
            uso_popular = filtered_df['q3_uso_principal'].mode()[0]
            uso_popular_count = (filtered_df['q3_uso_principal'] == uso_popular).sum()
            st.write(f"**Uso Mais Popular:** `{uso_popular}` ({uso_popular_count} respostas)")
            
            # Fator decisivo principal
            fator_principal = filtered_df['q4_fator_decisivo'].mode()[0]
            st.write(f"**Fator Decisivo Principal:** `{fator_principal}`")
            
            # Preço preferido
            preco_preferido = filtered_df['q5_preco_diaria'].mode()[0]
            st.write(f"**Faixa de Preço Preferida:** `{preco_preferido}`")
            
            # Perfil predominante
            perfil_predominante = filtered_df['q7_tipo_viagem'].mode()[0]
            st.write(f"**Perfil Predominante:** `{perfil_predominante}`")
    
    with col2:
        st.subheader("💡 Insights Estratégicos")
        
        if not filtered_df.empty:
            # Cálculos para insights
            total_interessados = (filtered_df['q1_interesse'].isin(['Sim', 'Talvez (Preço)'])).sum()
            perc_interessados = (total_interessados / total_respostas) * 100
            
            perc_lazer = (filtered_df['q3_uso_principal'] == 'Lazer e Passeios').sum() / total_respostas * 100
            perc_essencial_seguro = (filtered_df['q8_valoriza_seguro'] == 'Essencial').sum() / total_respostas * 100
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>🎯 Potencial de Mercado:</strong><br>
            {perc_interessados:.1f}% dos entrevistados demonstraram interesse (Sim ou Talvez)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>🚀 Oportunidade Principal:</strong><br>
            {perc_lazer:.1f}% buscam scooters para lazer e passeios
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-box">
            <strong>🛡️ Diferencial Competitivo:</strong><br>
            {perc_essencial_seguro:.1f}% consideram seguro essencial
            </div>
            """, unsafe_allow_html=True)
            
            # Recomendações baseadas nos dados
            st.markdown("""
            <div class="insight-box">
            <strong>📋 Recomendações Ações:</strong><br>
            • Focar marketing em famílias e casais<br>
            • Manter preços na faixa mais popular<br>
            • Oferecer delivery como diferencial<br>
            • Incluir seguro básico no pacote
            </div>
            """, unsafe_allow_html=True)
    
    # Análise Temporal
    st.markdown('<div class="section-header">📅 Análise Temporal</div>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Agrupar por data
        dados_temporais = filtered_df.groupby(filtered_df['data_registro'].dt.date).size().reset_index()
        dados_temporais.columns = ['data', 'respostas']
        
        options = {
            "title": {"text": "Respostas por Data", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": [str(d) for d in dados_temporais['data'].tolist()],
            },
            "yAxis": {"type": "value"},
            "series": [
                {
                    "data": dados_temporais['respostas'].tolist(),
                    "type": "line",
                    "smooth": True,
                    "itemStyle": {"color": "#ff6b6b"},
                    "lineStyle": {"width": 3},
                }
            ],
        }
        st_echarts(options=options, height="400px")
    
    # Tabela de dados detalhada
    st.markdown('<div class="section-header">📋 Dados da Pesquisa</div>', unsafe_allow_html=True)
    
    # Mostrar dados brutos com opção de download
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(filtered_df, use_container_width=True)
    
    with col2:
        # Botão para download dos dados filtrados
        csv = filtered_df.to_csv(index=False, sep=';')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"dados_filtrados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Estatísticas rápidas
        st.markdown("---")
        st.write("**Resumo Filtrado:**")
        st.write(f"Registros: {len(filtered_df)}")
        if not filtered_df.empty:
            st.write(f"Período: {filtered_df['data_registro'].min().strftime('%d/%m')} a {filtered_df['data_registro'].max().strftime('%d/%m/%Y')}")

else:
    st.info("📁 Por favor, carregue o arquivo CSV da pesquisa para visualizar o dashboard.")
    
    # Exemplo de estrutura esperada
    st.markdown("""
    ### Estrutura esperada do CSV:
    O arquivo deve conter as seguintes colunas (separadas por ponto e vírgula):
    - `id`: Identificador único
    - `data_registro`: Data e hora do registro
    - `q1_interesse`: Interesse (sim/nao/talvez_preco)
    - `q2_tempo_aluguel`: Tempo de aluguel preferido
    - `q3_uso_principal`: Uso principal do scooter
    - `q4_fator_decisivo`: Fator decisivo na escolha
    - `q5_preco_diaria`: Faixa de preço preferida
    - `q6_faixa_etaria`: Faixa etária do respondente
    - `q7_tipo_viagem`: Tipo de viagem
    - `q8_valoriza_seguro`: Valorização do seguro
    - `cupom_gerado`: Código do cupom gerado
    """)

# Rodapé
st.markdown("---")
st.markdown(
    "**Dashboard desenvolvido para análise de pesquisa de mercado sobre scooters em Gramado** • "
    "Dados processados em tempo real a partir do CSV"
)