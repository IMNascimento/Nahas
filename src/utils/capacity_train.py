from __future__ import annotations

import subprocess
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# -------------------------------------------------------------------
# GPU: memória livre/total (preferência NVML; fallback nvidia-smi)
# -------------------------------------------------------------------
def gpu_mem_info() -> List[Dict[str, int]]:
    """
    Retorna lista de dicts: [{'index':0, 'total': bytes, 'free': bytes}, ...]
    Se não houver GPU ou falhar, retorna [].
    """
    info: List[Dict[str, int]] = []
    # Tenta NVML
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            info.append({"index": i, "total": int(mem.total), "free": int(mem.free)})
        pynvml.nvmlShutdown()
        return info
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        for i, line in enumerate(out.strip().splitlines()):
            total, free = line.split(",")
            info.append({
                "index": i,
                "total": int(total.strip()) * 1024 * 1024,  # MiB -> bytes
                "free": int(free.strip()) * 1024 * 1024
            })
        return info
    except Exception:
        return []  # sem GPU / sem nvidia-smi


# -------------------------------------------------------------------
# Estimativa de consumo de memória por job (heurística)
# -------------------------------------------------------------------
def estimate_job_mem_bytes(cfg: Dict[str, Any], model_type: str) -> int:
    """
    Palpite conservador (bytes). Ajuste os coeficientes conforme sua realidade.
    Garante um clamp entre 0.8GB e 10GB para evitar absurdos.
    """
    B  = int(cfg.get("batch_size", 32))
    L  = int(cfg.get("window_size", 96))
    SA = int(cfg.get("steps_ahead", 1))
    D  = max(1, int(len(cfg.get("relevant_columns", []))))

    if model_type.lower() == "lstm":
        layers = cfg.get("layers_config", [128, 64])
        try:
            H = sum(int(h) for h in layers) / max(1, len(layers))  # hidden médio
        except Exception:
            H = 128
        base = 1_200_000_000  # 1.2 GB base (framework/lib)
        act  = B * L * (H + D) * 16 * 4   # ativações rough
    else:  # transformer
        H = int(cfg.get("embed_dim", 32))
        heads = int(cfg.get("num_heads", 2))
        Ls = int(cfg.get("num_layers", 2))
        base = 1_500_000_000  # 1.5 GB
        act  = B * L * H * Ls * heads * 8

    out = base + act + SA * 16_000_000  # troco por saída
    out = int(min(max(out, 800_000_000), 10_000_000_000))  # clamp 0.8–10 GB
    return out


# -------------------------------------------------------------------
# “Assinatura” do job + memória empírica observada
# -------------------------------------------------------------------
EMPIRICAL_PEAK: defaultdict = defaultdict(int)  # (signature) -> bytes

def job_signature(cfg: Dict[str, Any], model_type: str) -> Tuple:
    """
    Cria uma assinatura estável para agrupar jobs “parecidos”
    e aprender o pico empírico de memória.
    """
    if model_type.lower() == "lstm":
        arch = tuple(cfg.get("layers_config", []))
    else:
        arch = (int(cfg.get("num_layers", 2)),
                int(cfg.get("embed_dim", 32)),
                int(cfg.get("num_heads", 2)))

    return (
        model_type.lower(),
        arch,
        int(cfg.get("batch_size", 32)),
        int(cfg.get("window_size", 96)),
        int(cfg.get("steps_ahead", 1)),
        tuple(sorted(cfg.get("relevant_columns", [])))
    )

def record_empirical_peak(cfg: Dict[str, Any], model_type: str, peak_bytes: int | None) -> None:
    """
    Salva o pico observado (em bytes) para a assinatura do job.
    Use isso no retorno do seu worker (se você conseguir medir).
    """
    if not peak_bytes:
        return
    sig = job_signature(cfg, model_type)
    EMPIRICAL_PEAK[sig] = max(EMPIRICAL_PEAK[sig], int(peak_bytes))


# -------------------------------------------------------------------
# Escolha de GPU levando memoria em conta
# -------------------------------------------------------------------
def pick_gpu_for_job(
    cfg: Dict[str, Any],
    model_type: str,
    max_workers_per_gpu: int,
    gpu_slots_in_use: List[int],
    safety_ratio: float = 0.20
) -> int | None:
    """
    Retorna o índice de uma GPU adequada OU None se não houver capacidade.
    - Respeita limite de 'max_workers_per_gpu' por GPU (via gpu_slots_in_use).
    - Usa estimativa (ou pico empírico, se já aprendido) + margem de segurança.
    """
    mems = gpu_mem_info()
    if not mems:
        return None

    # estimativa vs. empirico
    sig = job_signature(cfg, model_type)
    empirical = EMPIRICAL_PEAK.get(sig, 0)
    need = max(empirical, estimate_job_mem_bytes(cfg, model_type))
    margin = int(need * safety_ratio)
    need_total = need + margin

    for m in mems:
        idx = m["index"]
        if idx < len(gpu_slots_in_use) and gpu_slots_in_use[idx] >= max_workers_per_gpu:
            continue
        if m["free"] >= need_total:
            return idx

    return None
