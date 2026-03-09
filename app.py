import streamlit as st
import numpy as np
import pandas as pd
import joblib
import keras
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── CONFIGURACIÓN DE PÁGINA ───────────────────────────────────
st.set_page_config(
    page_title="Credit Score AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ESTILOS CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
P
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0f1117; }

.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #8892a4;
    font-weight: 300;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #1e2433 0%, #252d3d 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s;
}

.metric-card:hover { transform: translateY(-2px); }

.result-poor {
    background: linear-gradient(135deg, #2d1515 0%, #3d1a1a 100%);
    border: 2px solid #e53e3e;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

.result-standard {
    background: linear-gradient(135deg, #2d2515 0%, #3d2e1a 100%);
    border: 2px solid #ed8936;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

.result-good {
    background: linear-gradient(135deg, #152d1e 0%, #1a3d2a 100%);
    border: 2px solid #38a169;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e2e8f0;
    border-bottom: 2px solid #2d3748;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.info-box {
    background: #1e2433;
    border-left: 4px solid #4299e1;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #a0aec0;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f2e 0%, #141820 100%);
    border-right: 1px solid #2d3748;
}

.stSlider > div > div > div {
    background: #4299e1 !important;
}

label { color: #a0aec0 !important; font-size: 0.85rem !important; }

.stSelectbox > div > div {
    background: #1e2433 !important;
    border-color: #2d3748 !important;
    color: #e2e8f0 !important;
}

.stNumberInput > div > div > input {
    background: #1e2433 !important;
    border-color: #2d3748 !important;
    color: #e2e8f0 !important;
}

.predict-btn > button {
    background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    transition: all 0.3s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(49,130,206,0.4) !important;
}

div[data-testid="metric-container"] {
    background: #1e2433;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── CARGA DE MODELOS ──────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    try:
        import os
        BASE = r'C:\Users\PC\OneDrive\Documentos\taller_ciencia_de_datos'
        model   = keras.models.load_model(os.path.join(BASE, 'modelo_credit_score.keras'))
        scaler  = joblib.load(os.path.join(BASE, 'scaler.pkl'))
        pca     = joblib.load(os.path.join(BASE, 'pca.pkl'))
        le      = joblib.load(os.path.join(BASE, 'label_encoder.pkl'))
        return model, scaler, pca, le, True
    except Exception as e:
        st.write(e)
        return None, None, None, None, False

model, scaler, pca, le, modelos_ok = cargar_modelos()


# ── HEADER ────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.markdown('<div class="hero-title">💳 Credit Score AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Sistema de evaluación de riesgo crediticio basado en Inteligencia Artificial · ANN Multiclass</div>', unsafe_allow_html=True)

st.markdown("---")

if not modelos_ok:
    st.error("No se encontraron los archivos del modelo. Asegúrate de tener en la misma carpeta: `modelo_credit_score.keras`, `scaler.pkl`, `pca.pkl`, `label_encoder.pkl`")
    st.stop()


# ── SIDEBAR — FORMULARIO ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Datos del Cliente")
    st.markdown('<div class="info-box">Complete todos los campos para obtener la predicción del score crediticio.</div>', unsafe_allow_html=True)

    # Información personal
    st.markdown('<div class="section-header">Información Personal</div>', unsafe_allow_html=True)
    age        = st.slider("Edad", 18, 80, 35)
    occupation = st.selectbox("Ocupación", [
        "Accountant", "Architect", "Developer", "Doctor", "Engineer",
        "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
        "Media_Manager", "Musician", "Scientist", "Teacher", "Writer"
    ])

    # Información financiera
    st.markdown('<div class="section-header">Información Financiera</div>', unsafe_allow_html=True)
    annual_income          = st.number_input("Ingreso Anual ($)", min_value=0.0, value=50000.0, step=1000.0)
    monthly_inhand_salary  = st.number_input("Salario Mensual Neto ($)", min_value=0.0, value=3500.0, step=100.0)
    monthly_balance        = st.number_input("Balance Mensual ($)", min_value=0.0, value=500.0, step=50.0)
    amount_invested        = st.number_input("Inversión Mensual ($)", min_value=0.0, value=200.0, step=50.0)

    # Cuentas y tarjetas
    st.markdown('<div class="section-header">Cuentas y Créditos</div>', unsafe_allow_html=True)
    num_bank_accounts  = st.slider("N° Cuentas Bancarias", 0, 15, 3)
    num_credit_card    = st.slider("N° Tarjetas de Crédito", 0, 15, 2)
    num_of_loan        = st.slider("N° de Préstamos", 0, 10, 1)
    interest_rate      = st.slider("Tasa de Interés (%)", 1, 50, 15)

    # Comportamiento de pago
    st.markdown('<div class="section-header">Comportamiento de Pago</div>', unsafe_allow_html=True)
    delay_from_due_date     = st.slider("Días de Retraso Promedio", 0, 60, 5)
    num_delayed_payment     = st.slider("N° Pagos Atrasados", 0, 30, 2)
    num_credit_inquiries    = st.slider("N° Consultas de Crédito", 0, 20, 3)
    payment_of_min_amount   = st.selectbox("¿Paga el Mínimo?", ["Yes", "No", "NM"])
    payment_behaviour       = st.selectbox("Comportamiento de Pago", [
        "High_spent_Large_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments"
    ])

    # Historial crediticio
    st.markdown('<div class="section-header">Historial Crediticio</div>', unsafe_allow_html=True)
    credit_mix              = st.selectbox("Mezcla de Crédito", ["Good", "Standard", "Bad"])
    outstanding_debt        = st.number_input("Deuda Pendiente ($)", min_value=0.0, value=500.0, step=50.0)
    credit_utilization      = st.slider("Utilización de Crédito (%)", 0.0, 100.0, 30.0)
    credit_history_age      = st.slider("Antigüedad Historial (meses)", 0, 400, 120)
    changed_credit_limit    = st.number_input("Cambio Límite Crédito ($)", value=5.0, step=0.5)
    total_emi               = st.number_input("EMI Mensual Total ($)", min_value=0.0, value=100.0, step=10.0)

    st.markdown("---")
    predict_btn = st.button("🔍 Predecir Credit Score", use_container_width=True, type="primary")


# ── FUNCIÓN DE PREDICCIÓN ─────────────────────────────────────
def predecir(datos):
    # Encoding categóricas (mismo orden que entrenamiento)
    occupation_map       = {"Accountant":0,"Architect":1,"Developer":2,"Doctor":3,
                            "Engineer":4,"Entrepreneur":5,"Journalist":6,"Lawyer":7,
                            "Manager":8,"Mechanic":9,"Media_Manager":10,"Musician":11,
                            "Scientist":12,"Teacher":13,"Writer":14}
    credit_mix_map       = {"Bad":0, "Good":1, "Standard":2}
    payment_min_map      = {"NM":0, "No":1, "Yes":2}
    payment_beh_map      = {
        "High_spent_Large_value_payments":0,
        "High_spent_Medium_value_payments":1,
        "High_spent_Small_value_payments":2,
        "Low_spent_Large_value_payments":3,
        "Low_spent_Medium_value_payments":4,
        "Low_spent_Small_value_payments":5
    }

    X = np.array([[
        datos['age'],
        occupation_map.get(datos['occupation'], 0),
        datos['annual_income'],
        datos['monthly_inhand_salary'],
        datos['num_bank_accounts'],
        datos['num_credit_card'],
        datos['interest_rate'],
        datos['num_of_loan'],
        datos['delay_from_due_date'],
        datos['num_delayed_payment'],
        datos['changed_credit_limit'],
        datos['num_credit_inquiries'],
        credit_mix_map.get(datos['credit_mix'], 1),
        datos['outstanding_debt'],
        datos['credit_utilization'],
        datos['credit_history_age'],
        payment_min_map.get(datos['payment_of_min_amount'], 1),
        datos['total_emi'],
        datos['amount_invested'],
        payment_beh_map.get(datos['payment_behaviour'], 1),
        datos['monthly_balance']
    ]])

    X_scaled = scaler.transform(X)
    X_pca    = pca.transform(X_scaled)
    proba    = model.predict(X_pca, verbose=0)[0]
    clase    = np.argmax(proba)
    return clase, proba


# ── CONTENIDO PRINCIPAL ───────────────────────────────────────
if not predict_btn:
    # Estado inicial — dashboard informativo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clientes Analizados", "12,500", "Dataset de entrenamiento")
    with col2:
        st.metric("Precisión del Modelo", "75.08%", "Accuracy en Test")
    with col3:
        st.metric("Arquitectura", "ANN 4 capas", "128 → 64 → 32 → 3")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Distribución de Clases en Dataset")
        fig_dist = go.Figure(go.Pie(
            labels=['Poor (0)', 'Standard (1)', 'Good (2)'],
            values=[4162, 6111, 2227],
            hole=0.5,
            marker_colors=['#e53e3e', '#ed8936', '#38a169'],
            textinfo='label+percent',
            textfont_size=13
        ))
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            showlegend=False,
            height=300,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.markdown("### Arquitectura del Modelo")
        fig_arch = go.Figure()
        capas    = ['Input\n(15)', 'Dense\n128', 'Dense\n64', 'Dense\n32', 'Output\n3']
        colores  = ['#4299e1', '#3182ce', '#2b6cb0', '#2c5282', '#38a169']
        for i, (capa, color) in enumerate(zip(capas, colores)):
            fig_arch.add_trace(go.Bar(
                x=[capa], y=[128 - i*20],
                marker_color=color,
                text=capa, textposition='outside',
                showlegend=False,
                width=0.5
            ))
        fig_arch.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=300,
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(t=40, b=20),
            barmode='group'
        )
        st.plotly_chart(fig_arch, use_container_width=True)

    st.markdown('<div class="info-box">👈 Completa el formulario en el panel izquierdo y presiona <strong>Predecir Credit Score</strong> para obtener el análisis.</div>', unsafe_allow_html=True)

else:
    # ── PREDICCIÓN ────────────────────────────────────────────
    datos = {
        'age': age, 'occupation': occupation,
        'annual_income': annual_income, 'monthly_inhand_salary': monthly_inhand_salary,
        'num_bank_accounts': num_bank_accounts, 'num_credit_card': num_credit_card,
        'interest_rate': interest_rate, 'num_of_loan': num_of_loan,
        'delay_from_due_date': delay_from_due_date, 'num_delayed_payment': num_delayed_payment,
        'changed_credit_limit': changed_credit_limit, 'num_credit_inquiries': num_credit_inquiries,
        'credit_mix': credit_mix, 'outstanding_debt': outstanding_debt,
        'credit_utilization': credit_utilization, 'credit_history_age': credit_history_age,
        'payment_of_min_amount': payment_of_min_amount, 'total_emi': total_emi,
        'amount_invested': amount_invested, 'payment_behaviour': payment_behaviour,
        'monthly_balance': monthly_balance
    }

    with st.spinner("Analizando perfil crediticio..."):
        clase, proba = predecir(datos)

    etiquetas = ['Poor', 'Standard', 'Good']
    colores_r  = ['#e53e3e', '#ed8936', '#38a169']
    emojis     = ['🔴', '🟡', '🟢']
    clases_css = ['result-poor', 'result-standard', 'result-good']
    descripciones = [
        "Alto riesgo crediticio. Se recomienda revisar el historial de pagos y reducir deudas pendientes.",
        "Riesgo crediticio moderado. El perfil es aceptable con oportunidades de mejora.",
        "Bajo riesgo crediticio. Excelente perfil financiero. Apto para créditos preferenciales."
    ]

    # Resultado principal
    confianza = proba[clase] * 100
    css_class = clases_css[clase]

    st.markdown(f"""
    <div class="{css_class}">
        <div class="result-title">{emojis[clase]} {etiquetas[clase]}</div>
        <div style="font-size:1.1rem; color:#a0aec0; margin-bottom:0.5rem;">
            Confianza: <strong style="color:{colores_r[clase]}">{confianza:.1f}%</strong>
        </div>
        <div style="font-size:0.95rem; color:#cbd5e0;">{descripciones[clase]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Gráficas de resultados
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Probabilidades por Clase")
        fig_proba = go.Figure(go.Bar(
            x=[f"{emojis[i]} {etiquetas[i]}" for i in range(3)],
            y=[p * 100 for p in proba],
            marker_color=colores_r,
            text=[f"{p*100:.1f}%" for p in proba],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        fig_proba.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=320,
            yaxis=dict(range=[0, 110], showgrid=True, gridcolor='#2d3748',
                      title='Probabilidad (%)', tickfont=dict(color='#8892a4')),
            xaxis=dict(tickfont=dict(size=13)),
            margin=dict(t=40, b=20)
        )
        fig_proba.add_hline(y=50, line_dash="dash", line_color="#4a5568",
                            annotation_text="Umbral 50%", annotation_font_color="#8892a4")
        st.plotly_chart(fig_proba, use_container_width=True)

    with col2:
        st.markdown("### Indicador de Riesgo")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confianza,
            title={'text': f"Confianza — {etiquetas[clase]}", 'font': {'color': '#e2e8f0', 'size': 14}},
            number={'suffix': "%", 'font': {'color': colores_r[clase], 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8892a4', 'tickfont': {'color': '#8892a4'}},
                'bar': {'color': colores_r[clase]},
                'bgcolor': '#1e2433',
                'bordercolor': '#2d3748',
                'steps': [
                    {'range': [0, 50],  'color': '#2d1515'},
                    {'range': [50, 75], 'color': '#2d2515'},
                    {'range': [75, 100],'color': '#152d1e'}
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 3},
                    'thickness': 0.8,
                    'value': confianza
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=320,
            margin=dict(t=60, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Resumen del perfil del cliente
    st.markdown("### Resumen del Perfil Analizado")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Ingreso Anual", f"${annual_income:,.0f}")
    col_b.metric("Deuda Pendiente", f"${outstanding_debt:,.0f}")
    col_c.metric("Utilización Crédito", f"{credit_utilization:.1f}%")
    col_d.metric("Pagos Atrasados", f"{num_delayed_payment}")

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Edad", f"{age} años")
    col_f.metric("N° Préstamos", f"{num_of_loan}")
    col_g.metric("Historial", f"{credit_history_age} meses")
    col_h.metric("Mezcla Crédito", credit_mix)

    # Radar chart del perfil
    st.markdown("### Perfil de Riesgo — Radar")
    categorias = ['Ingresos', 'Historial', 'Puntualidad', 'Endeudamiento', 'Utilización', 'Estabilidad']

    # Normalizar valores para el radar (0-10)
    v_ingresos     = min(annual_income / 100000 * 10, 10)
    v_historial    = min(credit_history_age / 400 * 10, 10)
    v_puntualidad  = max(10 - (num_delayed_payment / 30 * 10), 0)
    v_endeudamiento= max(10 - (outstanding_debt / 5000 * 10), 0)
    v_utilizacion  = max(10 - (credit_utilization / 100 * 10), 0)
    v_estabilidad  = min(credit_history_age / 400 * 10 + (1 if payment_of_min_amount == 'Yes' else 0), 10)

    valores = [v_ingresos, v_historial, v_puntualidad, v_endeudamiento, v_utilizacion, v_estabilidad]
    valores_cerrado = valores + [valores[0]]
    categorias_cerrado = categorias + [categorias[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=valores_cerrado,
        theta=categorias_cerrado,
        fill='toself',
        fillcolor=f"rgba({','.join(['227,83,83' if clase==0 else '237,137,54' if clase==1 else '56,161,105'])},0.2)",
        line=dict(color=colores_r[clase], width=2),
        name='Perfil del Cliente'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10],
                           gridcolor='#2d3748', tickfont=dict(color='#8892a4')),
            angularaxis=dict(gridcolor='#2d3748', tickfont=dict(color='#e2e8f0', size=13)),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        height=400,
        showlegend=False,
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Recomendaciones
    st.markdown("### Recomendaciones")
    if clase == 0:
        st.error("**Alto Riesgo** — Acciones sugeridas:")
        st.markdown("""
        - Reducir la deuda pendiente por debajo del 30% del límite disponible
        - Establecer pagos automáticos para evitar retrasos
        - Evitar nuevas consultas de crédito por al menos 6 meses
        - Considerar un plan de consolidación de deudas
        """)
    elif clase == 1:
        st.warning("**Riesgo Moderado** — Acciones sugeridas:")
        st.markdown("""
        - Mantener la utilización de crédito por debajo del 30%
        - Incrementar el historial de pagos puntuales
        - Diversificar los tipos de crédito gradualmente
        - Aumentar el ahorro mensual para mejorar el balance
        """)
    else:
        st.success("**Bajo Riesgo** — Perfil destacado:")
        st.markdown("""
        - Mantener los buenos hábitos financieros actuales
        - Apto para productos crediticios con tasas preferenciales
        - Considerar inversiones para optimizar el patrimonio
        - Perfil elegible para límites de crédito más altos
        """)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding:1rem 0;">
    Credit Score AI · Modelo ANN Multiclass · Taller Ciencia de Datos<br>
    Accuracy: 75.08% · 12,500 clientes · TensorFlow 2.19
</div>
""", unsafe_allow_html=True)
