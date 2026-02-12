"""Operaciones pesadas no-hot-path: backup, benchmark e import/export."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def run_model_benchmark(prompts: Optional[List[str]] = None, models: Optional[List[str]] = None, limit_models: int = 5) -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.run_model_benchmark(prompts=prompts, models=models, limit_models=limit_models)


def get_benchmark_results(limit: int = 20) -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.get_benchmark_results(limit=limit)


def run_backup(reason: str = "manual") -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.run_backup(reason=reason)


def list_backups(limit: int = 50) -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.list_backups(limit=limit)


def restore_backup(backup_id: str, overwrite_profiles: bool = False) -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.restore_backup(backup_id=backup_id, overwrite_profiles=overwrite_profiles)


def start_backup_scheduler(interval_seconds: int | None = None) -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.start_backup_scheduler(interval_seconds=interval_seconds)


def export_configuration() -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.export_configuration()


def import_configuration(payload: Dict[str, Any], overwrite_profiles: bool = False) -> Dict[str, Any]:
    import rag_mejorado as rag

    return rag.import_configuration(payload, overwrite_profiles=overwrite_profiles)
