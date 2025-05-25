import streamlit as st

def model_dropdown(models, key="model_select", label="Selecione um modelo salvo"):
    options = [
        f"{m['created_at']} | {m['framework'].upper()}-{m['model_type'].capitalize()} | {m['run_uuid']}"
        for m in models
    ]
    if not options:
        st.warning("Nenhum modelo treinado foi encontrado.")
        return None
    idx = st.selectbox(label, list(range(len(options))), format_func=lambda i: options[i], key=key)
    return models[idx] if models else None
