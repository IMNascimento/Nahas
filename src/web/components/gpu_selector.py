import streamlit as st

def gpu_selector(framework: str, key_prefix="gpu"):
    """
    Componente Streamlit para seleção de GPU.
    Retorna: (usar_gpu: bool, gpu_index: int|None, gpu_label: str|None)
    """
    use_gpu = st.checkbox("Usar GPU para treino?", key=f"{key_prefix}_use_gpu")
    gpu_index = None
    gpu_label = None

    if use_gpu:
        framework = framework.lower()
        gpu_options = []
        # Se for keras, usa o TensorFlow por baixo!
        if framework in ("tensorflow", "keras"):
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                gpu_options = [f"{i} - {gpu.name}" for i, gpu in enumerate(gpus)]
            except Exception as e:
                st.warning(f"Erro ao listar GPUs TensorFlow/Keras: {e}")
                gpus = []
        elif framework == "pytorch":
            try:
                import torch
                gpu_options = [f"{i} - {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
            except Exception as e:
                st.warning(f"Erro ao listar GPUs PyTorch: {e}")
                gpu_options = []
        else:
            st.info("Selecione um framework antes.")

        if gpu_options:
            gpu_label = st.selectbox("Selecione a GPU", gpu_options, key=f"{key_prefix}_gpu_select")
            gpu_index = int(gpu_label.split(" ")[0])
        else:
            st.warning("Nenhuma GPU encontrada para o framework selecionado.")

    return use_gpu, gpu_index, gpu_label

