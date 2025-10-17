import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

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
    .prediction-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">🛵 Dashboard de Pesquisa - Scooters em Gramado</div>', unsafe_allow_html=True)

# Função para converter tipos numpy para Python nativo
def convert_to_native_types(obj):
    """Converte tipos numpy para tipos Python nativos para serialização JSON"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj

# Função para carregar dados do CSV
@st.cache_data
def load_data_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=';', parse_dates=['data_registro'])
    df['data_registro'] = pd.to_datetime(df['data_registro'])
    
    # Mapeamento para labels mais amigáveis
    mapeamento_interesse = {
        'sim': 'Sim',
        'talvez_preco': 'Talvez (Preço)',
        'nao': 'Não'
    }
    
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
    
    # Aplicar mapeamentos
    df['q1_interesse'] = df['q1_interesse'].map(mapeamento_interesse)
    df['q2_tempo_aluguel'] = df['q2_tempo_aluguel'].map(mapeamento_tempo)
    df['q3_uso_principal'] = df['q3_uso_principal'].map(mapeamento_uso)
    df['q4_fator_decisivo'] = df['q4_fator_decisivo'].map(mapeamento_fator)
    df['q5_preco_diaria'] = df['q5_preco_diaria'].map(mapeamento_preco)
    df['q6_faixa_etaria'] = df['q6_faixa_etaria'].map(mapeamento_idade)
    df['q7_tipo_viagem'] = df['q7_tipo_viagem'].map(mapeamento_viagem)
    df['q8_valoriza_seguro'] = df['q8_valoriza_seguro'].map(mapeamento_seguro)
    
    return df

# Função para treinar o modelo de ML
@st.cache_resource
def train_ml_model(df):
    """Treina um modelo de Random Forest para prever interesse"""
    
    # Preparar dados para ML
    ml_df = df.copy()
    
    # Codificar variáveis categóricas
    label_encoders = {}
    features = ['q2_tempo_aluguel', 'q3_uso_principal', 'q4_fator_decisivo', 
                'q5_preco_diaria', 'q6_faixa_etaria', 'q7_tipo_viagem', 'q8_valoriza_seguro']
    
    for feature in features:
        le = LabelEncoder()
        ml_df[feature] = le.fit_transform(ml_df[feature])
        label_encoders[feature] = le
    
    # Codificar target
    le_target = LabelEncoder()
    ml_df['q1_interesse_encoded'] = le_target.fit_transform(ml_df['q1_interesse'])
    
    # Features e target
    X = ml_df[features]
    y = ml_df['q1_interesse_encoded']
    
    # Treinar modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calcular acurácia
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, label_encoders, le_target, accuracy

# Função para gerar dados sintéticos
def generate_synthetic_data(num_samples=400):
    """Gera dados sintéticos baseados nas distribuições observadas"""
    
    synthetic_data = []
    
    for _ in range(num_samples):
        # Gerar dados baseados em distribuições prováveis
        tempo_aluguel = np.random.choice(['Meia Diária', 'Diária Completa', 'Estadia Completa', '2-3 Dias'], 
                                        p=[0.3, 0.4, 0.2, 0.1])
        
        uso_principal = np.random.choice(['Lazer e Passeios', 'Transporte Rápido', 'Mobilidade Essencial', 'Explorar Cidades'],
                                       p=[0.5, 0.2, 0.15, 0.15])
        
        fator_decisivo = np.random.choice(['Preço', 'Delivery', 'Conforto', 'Roteiros'],
                                        p=[0.4, 0.3, 0.2, 0.1])
        
        preco_diaria = np.random.choice(['R$ 80-120', 'R$ 121-160'], p=[0.6, 0.4])
        
        faixa_etaria = np.random.choice(['18-25 anos', '26-37 anos', '38-49 anos', '50+ anos'],
                                      p=[0.2, 0.4, 0.25, 0.15])
        
        tipo_viagem = np.random.choice(['Família', 'Sozinho', 'Casal', 'Grupo de Amigos'],
                                     p=[0.3, 0.2, 0.35, 0.15])
        
        valoriza_seguro = np.random.choice(['Essencial', 'Se for Barato', 'Não Preciso'],
                                         p=[0.5, 0.3, 0.2])
        
        synthetic_data.append({
            'q2_tempo_aluguel': tempo_aluguel,
            'q3_uso_principal': uso_principal,
            'q4_fator_decisivo': fator_decisivo,
            'q5_preco_diaria': preco_diaria,
            'q6_faixa_etaria': faixa_etaria,
            'q7_tipo_viagem': tipo_viagem,
            'q8_valoriza_seguro': valoriza_seguro
        })
    
    return pd.DataFrame(synthetic_data)

# Upload do arquivo
uploaded_file = st.sidebar.file_uploader("📁 Carregar arquivo CSV", type=['csv'])

# Nova seção para predição no sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 Predição com Machine Learning")

if uploaded_file is not None:
    df = load_data_from_csv(uploaded_file)
    
    # Treinar modelo
    with st.sidebar:
        if st.button("🚀 Treinar Modelo de Predição"):
            with st.spinner("Treinando modelo de machine learning..."):
                try:
                    model, label_encoders, le_target, accuracy = train_ml_model(df)
                    st.success(f"Modelo treinado! Acurácia: {accuracy:.2%}")
                    st.session_state['ml_model'] = model
                    st.session_state['label_encoders'] = label_encoders
                    st.session_state['le_target'] = le_target
                    st.session_state['model_trained'] = True
                except Exception as e:
                    st.error(f"Erro ao treinar modelo: {e}")
    
    # Seção de predição em larga escala
    if st.sidebar.checkbox("📊 Simular Predição em Larga Escala"):
        num_samples = st.sidebar.number_input("Número de pessoas para simular:", 
                                            min_value=100, max_value=1000, value=400, step=50)
        
        if st.sidebar.button("🎯 Executar Simulação"):
            if 'model_trained' in st.session_state and st.session_state['model_trained']:
                with st.spinner(f"Gerando predições para {num_samples} pessoas..."):
                    # Gerar dados sintéticos
                    synthetic_df = generate_synthetic_data(num_samples)
                    
                    # Preparar dados para predição
                    prediction_data = synthetic_df.copy()
                    features = ['q2_tempo_aluguel', 'q3_uso_principal', 'q4_fator_decisivo', 
                               'q5_preco_diaria', 'q6_faixa_etaria', 'q7_tipo_viagem', 'q8_valoriza_seguro']
                    
                    # Codificar dados
                    for feature in features:
                        le = st.session_state['label_encoders'][feature]
                        prediction_data[feature] = le.transform(prediction_data[feature])
                    
                    # Fazer predições
                    X_pred = prediction_data[features]
                    predictions_encoded = st.session_state['ml_model'].predict(X_pred)
                    
                    # Decodificar predições
                    predictions = st.session_state['le_target'].inverse_transform(predictions_encoded)
                    
                    # Adicionar predições ao DataFrame
                    synthetic_df['q1_interesse_predito'] = predictions
                    
                    st.session_state['synthetic_predictions'] = synthetic_df
                    st.session_state['prediction_complete'] = True
                    
                st.success(f"Simulação concluída para {num_samples} pessoas!")
            else:
                st.error("Por favor, treine o modelo primeiro!")

    # Sidebar com filtros (código original)
    st.sidebar.title("🔍 Filtros")
    st.sidebar.markdown("---")
    
    # Filtros
    interesse_options = ["Todos"] + list(df['q1_interesse'].unique())
    selected_interesse = st.sidebar.selectbox("Interesse", interesse_options)
    
    faixa_etaria_options = ["Todos"] + list(df['q6_faixa_etaria'].unique())
    selected_faixa_etaria = st.sidebar.selectbox("Faixa Etária", faixa_etaria_options)
    
    tipo_viagem_options = ["Todos"] + list(df['q7_tipo_viagem'].unique())
    selected_tipo_viagem = st.sidebar.selectbox("Tipo de Viagem", tipo_viagem_options)
    
    # Aplicar filtros
    filtered_df = df.copy()
    if selected_interesse != "Todos":
        filtered_df = filtered_df[filtered_df['q1_interesse'] == selected_interesse]
    if selected_faixa_etaria != "Todos":
        filtered_df = filtered_df[filtered_df['q6_faixa_etaria'] == selected_faixa_etaria]
    if selected_tipo_viagem != "Todos":
        filtered_df = filtered_df[filtered_df['q7_tipo_viagem'] == selected_tipo_viagem]

    # MOSTRAR RESULTADOS DA PREDIÇÃO SE EXISTIREM
    if 'prediction_complete' in st.session_state and st.session_state['prediction_complete']:
        st.markdown('<div class="section-header">🤖 Resultados da Simulação de Predição</div>', unsafe_allow_html=True)
        
        synthetic_df = st.session_state['synthetic_predictions']
        
        # Métricas da predição
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pred = len(synthetic_df)
            st.metric("Total Simulado", total_pred)
            
        with col2:
            taxa_sim_pred = (synthetic_df['q1_interesse_predito'] == 'Sim').sum() / len(synthetic_df) * 100
            st.metric("Taxa de Interesse Predita", f"{taxa_sim_pred:.1f}%")
            
        with col3:
            taxa_talvez_pred = (synthetic_df['q1_interesse_predito'] == 'Talvez (Preço)').sum() / len(synthetic_df) * 100
            st.metric("Talvez (Preço) Predito", f"{taxa_talvez_pred:.1f}%")
            
        with col4:
            taxa_nao_pred = (synthetic_df['q1_interesse_predito'] == 'Não').sum() / len(synthetic_df) * 100
            st.metric("Não Predito", f"{taxa_nao_pred:.1f}%")
        
        # Comparação entre dados reais e preditos
        st.markdown("#### 📊 Comparação: Dados Reais vs Predição")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição real
            real_interesse = df['q1_interesse'].value_counts()
            real_option = {
                "title": {"text": "Distribuição Real (Dados Coletados)", "left": "center"},
                "tooltip": {"trigger": "item"},
                "series": [{
                    "name": "Interesse Real",
                    "type": "pie",
                    "radius": "60%",
                    "data": [{"value": int(count), "name": name} for name, count in real_interesse.items()],
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0, 0, 0, 0.5)"}}
                }]
            }
            st_echarts(options=convert_to_native_types(real_option), height="300px")
        
        with col2:
            # Distribuição predita
            pred_interesse = synthetic_df['q1_interesse_predito'].value_counts()
            pred_option = {
                "title": {"text": f"Distribuição Predita ({len(synthetic_df)} pessoas)", "left": "center"},
                "tooltip": {"trigger": "item"},
                "series": [{
                    "name": "Interesse Predito",
                    "type": "pie",
                    "radius": "60%",
                    "data": [{"value": int(count), "name": name} for name, count in pred_interesse.items()],
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0, 0, 0, 0.5)"}}
                }]
            }
            st_echarts(options=convert_to_native_types(pred_option), height="300px")
        
        # Análise detalhada das predições
        st.markdown("#### 📈 Análise Detalhada das Predições")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interesse por faixa etária predito
            interesse_faixa_pred = pd.crosstab(synthetic_df['q6_faixa_etaria'], synthetic_df['q1_interesse_predito'])
            st.subheader("Interesse Predito por Faixa Etária")
            st.dataframe(interesse_faixa_pred.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        with col2:
            # Interesse por tipo de viagem predito
            interesse_viagem_pred = pd.crosstab(synthetic_df['q7_tipo_viagem'], synthetic_df['q1_interesse_predito'])
            st.subheader("Interesse Predito por Tipo de Viagem")
            st.dataframe(interesse_viagem_pred.style.background_gradient(cmap='Greens'), use_container_width=True)
        
        # Tabela com dados da predição
        with st.expander("📋 Visualizar Dados da Simulação Completa"):
            st.dataframe(synthetic_df, use_container_width=True)

    # =========================================================================
    # CÓDIGO ORIGINAL DO DASHBOARD (TODO O RESTANTE PERMANECE IGUAL)
    # =========================================================================

    # Métricas principais
    st.markdown('<div class="section-header">📊 Métricas Principais</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_respostas = len(filtered_df)
        st.metric("Total de Respostas", total_respostas)
        
    with col2:
        taxa_interesse = (filtered_df['q1_interesse'] == 'Sim').sum() / len(filtered_df) * 100
        st.metric("Taxa de Interesse", f"{taxa_interesse:.1f}%")
        
    with col3:
        uso_lazer = (filtered_df['q3_uso_principal'] == 'Lazer e Passeios').sum()
        st.metric("Uso para Lazer", int(uso_lazer))
        
    with col4:
        if not filtered_df.empty:
            preco_counts = filtered_df['q5_preco_diaria'].value_counts()
            if not preco_counts.empty:
                preco_medio = preco_counts.index[0]
                st.metric("Faixa de Preço Mais Popular", preco_medio)

    # GRÁFICO 1: Evolução Temporal do Interesse (Área Stacked)
    st.markdown('<div class="section-header">📈 Evolução Temporal do Interesse</div>', unsafe_allow_html=True)
    
    # Preparar dados para gráfico de área
    daily_interest = filtered_df.groupby([filtered_df['data_registro'].dt.date, 'q1_interesse']).size().unstack(fill_value=0)
    
    # Garantir que todas as categorias existam
    for interesse in ['Sim', 'Talvez (Preço)', 'Não']:
        if interesse not in daily_interest.columns:
            daily_interest[interesse] = 0
    
    daily_interest = daily_interest[['Sim', 'Talvez (Preço)', 'Não']]  # Ordem consistente
    
    area_option = {
        "title": {
            "text": 'Evolução do Interesse ao Longo do Tempo',
            "left": 'center',
            "textStyle": {
                "fontSize": 16
            }
        },
        "tooltip": {
            "trigger": 'axis',
            "axisPointer": {
                "type": 'cross'
            }
        },
        "legend": {
            "data": ['Sim', 'Talvez (Preço)', 'Não'],
            "top": 30
        },
        "grid": {
            "left": '3%',
            "right": '4%',
            "bottom": '3%',
            "containLabel": True
        },
        "xAxis": {
            "type": 'category',
            "boundaryGap": False,
            "data": [str(date) for date in daily_interest.index]
        },
        "yAxis": {
            "type": 'value',
            "name": "Número de Respostas"
        },
        "series": [
            {
                "name": 'Sim',
                "type": 'line',
                "stack": 'Total',
                "areaStyle": {},
                "emphasis": {
                    "focus": 'series'
                },
                "data": [int(x) for x in daily_interest['Sim'].tolist()],
                "itemStyle": {"color": "#28a745"}
            },
            {
                "name": 'Talvez (Preço)',
                "type": 'line',
                "stack": 'Total',
                "areaStyle": {},
                "emphasis": {
                    "focus": 'series'
                },
                "data": [int(x) for x in daily_interest['Talvez (Preço)'].tolist()],
                "itemStyle": {"color": "#ffc107"}
            },
            {
                "name": 'Não',
                "type": 'line',
                "stack": 'Total',
                "areaStyle": {},
                "emphasis": {
                    "focus": 'series'
                },
                "data": [int(x) for x in daily_interest['Não'].tolist()],
                "itemStyle": {"color": "#dc3545"}
            }
        ]
    }
    
    st_echarts(options=convert_to_native_types(area_option), height="400px")

    # GRÁFICO 2: Distribuição de Interesse (Gráfico de Donut)
    st.markdown('<div class="section-header">🎯 Análise de Interesse</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        interesse_data = filtered_df['q1_interesse'].value_counts()
        
        donut_option = {
            "title": {
                "text": 'Distribuição de Interesse',
                "left": 'center',
                "textStyle": {
                    "fontSize": 14
                }
            },
            "tooltip": {
                "trigger": 'item',
                "formatter": '{a} <br/>{b}: {c} ({d}%)'
            },
            "legend": {
                "orient": 'vertical',
                "left": 'left',
                "top": 'center'
            },
            "series": [
                {
                    "name": 'Interesse',
                    "type": 'pie',
                    "radius": ['40%', '70%'],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": '#fff',
                        "borderWidth": 2
                    },
                    "label": {
                        "show": False,
                        "position": 'center'
                    },
                    "emphasis": {
                        "label": {
                            "show": True,
                            "fontSize": '18',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": False
                    },
                    "data": [
                        {"value": int(interesse_data.get('Sim', 0)), "name": 'Sim', "itemStyle": {"color": "#28a745"}},
                        {"value": int(interesse_data.get('Talvez (Preço)', 0)), "name": 'Talvez (Preço)', "itemStyle": {"color": "#ffc107"}},
                        {"value": int(interesse_data.get('Não', 0)), "name": 'Não', "itemStyle": {"color": "#dc3545"}}
                    ]
                }
            ]
        }
        
        st_echarts(options=convert_to_native_types(donut_option), height="400px")
    
    with col2:
        # Gráfico de Faixa Etária
        faixa_etaria_data = filtered_df['q6_faixa_etaria'].value_counts()
        
        bar_option = {
            "title": {
                "text": 'Distribuição por Faixa Etária',
                "left": 'center',
                "textStyle": {
                    "fontSize": 14
                }
            },
            "tooltip": {
                "trigger": 'axis',
                "axisPointer": {
                    "type": 'shadow'
                }
            },
            "xAxis": {
                "type": 'category',
                "data": list(faixa_etaria_data.index)
            },
            "yAxis": {
                "type": 'value',
                "name": "Número de Pessoas"
            },
            "series": [
                {
                    "data": [int(x) for x in faixa_etaria_data.values],
                    "type": 'bar',
                    "itemStyle": {
                        "color": {
                            "type": 'linear',
                            "x": 0,
                            "y": 0,
                            "x2": 0,
                            "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": '#5470c6'},
                                {"offset": 1, "color": '#91cc75'}
                            ]
                        }
                    }
                }
            ]
        }
        
        st_echarts(options=convert_to_native_types(bar_option), height="400px")

    # GRÁFICO 3: Uso Principal e Tempo de Aluguel
    st.markdown('<div class="section-header">⏰ Padrões de Uso</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        uso_data = filtered_df['q3_uso_principal'].value_counts()
        
        uso_option = {
            "title": {
                "text": 'Uso Principal dos Scooters',
                "left": 'center'
            },
            "tooltip": {
                "trigger": 'item',
                "formatter": '{b}: {c} ({d}%)'
            },
            "series": [
                {
                    "name": 'Uso Principal',
                    "type": 'pie',
                    "radius": '60%',
                    "data": [
                        {"value": int(count), "name": uso}
                        for uso, count in uso_data.items()
                    ],
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                }
            ]
        }
        
        st_echarts(options=convert_to_native_types(uso_option), height="400px")
    
    with col2:
        tempo_data = filtered_df['q2_tempo_aluguel'].value_counts()
        
        tempo_option = {
            "title": {
                "text": 'Tempo de Aluguel Preferido',
                "left": 'center'
            },
            "tooltip": {
                "trigger": 'axis',
                "axisPointer": {
                    "type": 'shadow'
                }
            },
            "xAxis": {
                "type": 'category',
                "data": list(tempo_data.index),
                "axisLabel": {
                    "rotate": 45
                }
            },
            "yAxis": {
                "type": 'value'
            },
            "series": [
                {
                    "data": [int(x) for x in tempo_data.values],
                    "type": 'bar',
                    "itemStyle": {
                        "color": '#ee6666'
                    }
                }
            ]
        }
        
        st_echarts(options=convert_to_native_types(tempo_option), height="400px")

    # GRÁFICO 4: Fatores Decisórios e Preços
    st.markdown('<div class="section-header">💰 Fatores de Decisão</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fator_data = filtered_df['q4_fator_decisivo'].value_counts()
        
        fator_option = {
            "title": {
                "text": 'Fatores Decisórios na Escolha',
                "left": 'center'
            },
            "tooltip": {
                "trigger": 'item'
            },
            "series": [
                {
                    "name": 'Fator Decisivo',
                    "type": 'pie',
                    "radius": '60%',
                    "roseType": 'radius',
                    "data": [
                        {"value": int(count), "name": fator}
                        for fator, count in fator_data.items()
                    ],
                    "itemStyle": {
                        "borderRadius": 8
                    }
                }
            ]
        }
        
        st_echarts(options=convert_to_native_types(fator_option), height="400px")
    
    with col2:
        preco_data = filtered_df['q5_preco_diaria'].value_counts()
        
        preco_option = {
            "title": {
                "text": 'Faixas de Preço Preferidas',
                "left": 'center'
            },
            "tooltip": {
                "trigger": 'axis'
            },
            "xAxis": {
                "type": 'category',
                "data": list(preco_data.index)
            },
            "yAxis": {
                "type": 'value'
            },
            "series": [
                {
                    "data": [int(x) for x in preco_data.values],
                    "type": 'bar',
                    "itemStyle": {
                        "color": '#91cc75'
                    }
                }
            ]
        }
        
        st_echarts(options=convert_to_native_types(preco_option), height="400px")

    # Análise Estatística
    st.markdown('<div class="section-header">📈 Análise Estatística</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Estatísticas Detalhadas")
        
        total_respostas = len(filtered_df)
        taxa_sim = (filtered_df['q1_interesse'] == 'Sim').sum() / total_respostas * 100
        taxa_talvez = (filtered_df['q1_interesse'] == 'Talvez (Preço)').sum() / total_respostas * 100
        taxa_nao = (filtered_df['q1_interesse'] == 'Não').sum() / total_respostas * 100
        
        st.write(f"**Taxa de Conversão (Sim):** `{taxa_sim:.1f}%`")
        st.write(f"**Interesse Condicional (Talvez):** `{taxa_talvez:.1f}%`")
        st.write(f"**Taxa de Rejeição (Não):** `{taxa_nao:.1f}%`")
        
        uso_popular = filtered_df['q3_uso_principal'].mode()[0]
        st.write(f"**Uso Mais Popular:** `{uso_popular}`")
        
        fator_principal = filtered_df['q4_fator_decisivo'].mode()[0]
        st.write(f"**Fator Decisivo Principal:** `{fator_principal}`")
    
    with col2:
        st.subheader("💡 Insights Estratégicos")
        
        total_interessados = (filtered_df['q1_interesse'].isin(['Sim', 'Talvez (Preço)'])).sum()
        perc_interessados = (total_interessados / total_respostas) * 100
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🎯 Potencial de Mercado:</strong><br>
        {perc_interessados:.1f}% dos entrevistados demonstraram interesse (Sim ou Talvez)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>📋 Recomendações:</strong><br>
        • Focar em famílias e casais<br>
        • Manter preços competitivos<br>
        • Oferecer delivery como diferencial<br>
        • Incluir seguro básico
        </div>
        """, unsafe_allow_html=True)

    # Tabela de dados
    st.markdown('<div class="section-header">📋 Dados da Pesquisa</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

else:
    st.info("📁 Por favor, carregue o arquivo CSV da pesquisa para visualizar o dashboard.")

# Rodapé
st.markdown("---")
st.markdown(
    "**Dashboard desenvolvido para análise de pesquisa de mercado sobre scooters em Gramado** | "
    "**Recurso de Predição com Machine Learning Adicionado**"
)