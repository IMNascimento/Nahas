"""
Helper para comparação de modelos (original vs fine-tuned).
Facilita análise de performance e decisão de rollback.
"""
import json
import os
from typing import Dict, Any, Optional
import pandas as pd


class ModelComparison:
    """Classe para comparar métricas entre modelos."""
    
    @staticmethod
    def load_metrics(metrics_path: str) -> Dict[str, Any]:
        """Carrega métricas de um arquivo JSON."""
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Arquivo de métricas não encontrado: {metrics_path}")
        
        with open(metrics_path, "r") as f:
            return json.load(f)
    
    @staticmethod
    def compare_models(
        original_metrics: Dict[str, Any],
        finetuned_metrics: Dict[str, Any],
        metric_names: list = None
    ) -> Dict[str, Any]:
        """
        Compara métricas entre modelo original e fine-tuned.
        
        Args:
            original_metrics: Métricas do modelo original
            finetuned_metrics: Métricas do modelo fine-tuned
            metric_names: Lista de métricas para comparar (padrão: ['rmse', 'mae', 'r2'])
        
        Returns:
            Dict com comparações detalhadas
        """
        if metric_names is None:
            metric_names = ['rmse', 'mae', 'mse', 'r2']
        
        comparison = {
            'original': {},
            'finetuned': {},
            'deltas': {},
            'improvement_pct': {},
            'verdict': {}
        }
        
        # Pega métricas de teste
        orig_test = original_metrics.get('test', {})
        ft_test = finetuned_metrics.get('test', {})
        
        for metric in metric_names:
            orig_val = orig_test.get(metric)
            ft_val = ft_test.get(metric)
            
            if orig_val is None or ft_val is None:
                continue
            
            comparison['original'][metric] = orig_val
            comparison['finetuned'][metric] = ft_val
            
            delta = ft_val - orig_val
            comparison['deltas'][metric] = delta
            
            # Cálculo de melhoria (%)
            if abs(orig_val) > 1e-10:
                if metric == 'r2':  # Maior é melhor
                    improvement = ((ft_val - orig_val) / abs(orig_val)) * 100
                else:  # RMSE, MAE, MSE: menor é melhor
                    improvement = ((orig_val - ft_val) / abs(orig_val)) * 100
                
                comparison['improvement_pct'][metric] = improvement
            else:
                comparison['improvement_pct'][metric] = None
            
            # Veredito por métrica
            if metric == 'r2':
                comparison['verdict'][metric] = 'improved' if delta > 0 else 'degraded' if delta < 0 else 'unchanged'
            else:
                comparison['verdict'][metric] = 'improved' if delta < 0 else 'degraded' if delta > 0 else 'unchanged'
        
        # Veredito geral
        verdicts = list(comparison['verdict'].values())
        if verdicts.count('improved') > verdicts.count('degraded'):
            comparison['overall_verdict'] = 'IMPROVED'
        elif verdicts.count('degraded') > verdicts.count('improved'):
            comparison['overall_verdict'] = 'DEGRADED'
        else:
            comparison['overall_verdict'] = 'MIXED'
        
        return comparison
    
    @staticmethod
    def should_rollback(
        comparison: Dict[str, Any],
        critical_metrics: list = None,
        degradation_threshold: float = 5.0
    ) -> bool:
        """
        Decide se deve fazer rollback baseado na comparação.
        
        Args:
            comparison: Resultado de compare_models()
            critical_metrics: Métricas críticas (default: ['rmse', 'mae'])
            degradation_threshold: % de degradação tolerável
        
        Returns:
            True se recomenda rollback
        """
        if critical_metrics is None:
            critical_metrics = ['rmse', 'mae']
        
        for metric in critical_metrics:
            improvement = comparison['improvement_pct'].get(metric)
            if improvement is not None and improvement < -degradation_threshold:
                return True
        
        return False
    
    @staticmethod
    def generate_report(comparison: Dict[str, Any]) -> str:
        """Gera relatório textual da comparação."""
        lines = ["=" * 60]
        lines.append("RELATÓRIO DE COMPARAÇÃO: ORIGINAL vs FINE-TUNED")
        lines.append("=" * 60)
        lines.append("")
        
        lines.append(f"Veredito Geral: {comparison['overall_verdict']}")
        lines.append("")
        
        lines.append("Detalhes por Métrica:")
        lines.append("-" * 60)
        
        for metric in comparison['original'].keys():
            orig = comparison['original'][metric]
            ft = comparison['finetuned'][metric]
            delta = comparison['deltas'][metric]
            improvement = comparison['improvement_pct'].get(metric, 0)
            verdict = comparison['verdict'][metric]
            
            lines.append(f"\n{metric.upper()}:")
            lines.append(f"  Original:    {orig:.6f}")
            lines.append(f"  Fine-tuned:  {ft:.6f}")
            lines.append(f"  Delta:       {delta:+.6f}")
            
            if improvement is not None:
                lines.append(f"  Melhoria:    {improvement:+.2f}%")
            
            emoji = "✓" if verdict == "improved" else "✗" if verdict == "degraded" else "="
            lines.append(f"  Veredito:    {emoji} {verdict.upper()}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    @staticmethod
    def to_dataframe(comparison: Dict[str, Any]) -> pd.DataFrame:
        """Converte comparação para DataFrame pandas."""
        data = []
        
        for metric in comparison['original'].keys():
            row = {
                'Metric': metric.upper(),
                'Original': comparison['original'][metric],
                'Fine-tuned': comparison['finetuned'][metric],
                'Delta': comparison['deltas'][metric],
                'Improvement (%)': comparison['improvement_pct'].get(metric),
                'Verdict': comparison['verdict'][metric]
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df


class ModelVersionManager:
    """Gerencia versionamento e rollback de modelos."""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def list_versions(self, base_name: str) -> list:
        """Lista todas as versões de um modelo."""
        versions = []
        for fname in os.listdir(self.models_dir):
            if fname.startswith(base_name):
                fpath = os.path.join(self.models_dir, fname)
                stat = os.stat(fpath)
                versions.append({
                    'filename': fname,
                    'path': fpath,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': stat.st_mtime
                })
        
        # Ordena por data de modificação
        versions.sort(key=lambda x: x['modified'], reverse=True)
        return versions
    
    def rollback_to_backup(self, backup_path: str, target_path: str) -> bool:
        """
        Faz rollback de um modelo para uma versão de backup.
        
        Args:
            backup_path: Caminho do backup
            target_path: Caminho onde restaurar
        
        Returns:
            True se sucesso
        """
        import shutil
        
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup não encontrado: {backup_path}")
        
        try:
            # Cria backup do estado atual antes de sobrescrever
            if os.path.exists(target_path):
                temp_backup = target_path + ".before_rollback"
                shutil.copy2(target_path, temp_backup)
            
            # Restaura o backup
            shutil.copy2(backup_path, target_path)
            return True
        
        except Exception as e:
            print(f"Erro durante rollback: {e}")
            return False


# Funções auxiliares de conveniência
def quick_compare(original_metrics_path: str, finetuned_metrics_path: str) -> Dict[str, Any]:
    """Comparação rápida entre dois arquivos de métricas."""
    comp = ModelComparison()
    orig = comp.load_metrics(original_metrics_path)
    ft = comp.load_metrics(finetuned_metrics_path)
    return comp.compare_models(orig, ft)


def print_comparison_report(original_metrics_path: str, finetuned_metrics_path: str):
    """Imprime relatório de comparação no console."""
    comparison = quick_compare(original_metrics_path, finetuned_metrics_path)
    report = ModelComparison.generate_report(comparison)
    print(report)
    
    # Recomendação de rollback
    if ModelComparison.should_rollback(comparison):
        print("\n⚠️  RECOMENDAÇÃO: Considere fazer rollback para o modelo original.")
        print("   A degradação de performance excede o limite aceitável.")
    else:
        print("\n✓ OK: Performance aceitável ou melhorada.")