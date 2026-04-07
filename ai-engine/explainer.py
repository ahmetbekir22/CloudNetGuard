"""
CloudNetGuard — XAI açıklama modülü.
SHAP ile her anomali tespiti için feature önem skorları üretir.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch

sys.path.insert(0, "/app/shared")
from schema import FEATURE_NAMES, FeatureExplanation

log = logging.getLogger(__name__)

# Anomali tipini feature pattern'lerine göre tahmin et
_TYPE_RULES: list[tuple[str, dict[str, float]]] = [
    # (tip, {feature: minimum_normalize_değer})
    ("tunnel", {"entropy": 0.7, "query_length": 0.6, "record_type_TXT": 0.5}),
    ("ddos",   {"query_rate": 0.7}),
    ("flux",   {"unique_domains": 0.6, "is_nxdomain": 0.5}),
]


def _guess_type(feature_vector: list[float]) -> str:
    fdict = dict(zip(FEATURE_NAMES, feature_vector))
    best, best_score = "tunnel", 0.0
    for atype, rules in _TYPE_RULES:
        score = sum(
            max(0.0, fdict.get(f, 0.0) - threshold)
            for f, threshold in rules.items()
        )
        if score > best_score:
            best_score = score
            best = atype
    return best if best_score > 0 else "tunnel"


class SHAPExplainer:
    """
    Autoencoder için SHAP KernelExplainer sarıcısı.
    İlk çağrıda background verisine göre başlatılır.
    """

    def __init__(self, model: torch.nn.Module, background: np.ndarray, threshold: float) -> None:
        try:
            import shap  # type: ignore
            self._shap = shap
        except ImportError:
            log.warning("shap kütüphanesi bulunamadı, gradient tabanlı açıklama kullanılacak.")
            self._shap = None

        self._model = model
        self._threshold = threshold
        self._background = background[:50]   # KernelExplainer için küçük temsili set

        if self._shap is not None:
            def _predict(x: np.ndarray) -> np.ndarray:
                t = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    err = self._model.reconstruction_error(t).numpy()
                return err.reshape(-1, 1)

            self._explainer = self._shap.KernelExplainer(_predict, self._background)
        else:
            self._explainer = None

    def explain(
        self,
        feature_vector: list[float],
        anomaly_score: float,
        reconstruction_error: float,
    ) -> tuple[list[FeatureExplanation], str]:
        """
        Döndürür: (top_features listesi, doğal dil özeti)
        """
        farray = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        predicted_type = _guess_type(feature_vector)

        if self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(farray, nsamples=50)
                importances = np.abs(shap_values[0]).flatten()
            except Exception as exc:
                log.warning("SHAP hesaplanamadı: %s — gradient yedek kullanılıyor", exc)
                importances = self._gradient_importance(farray)
        else:
            importances = self._gradient_importance(farray)

        # Normalize et
        total = importances.sum() + 1e-9
        norm_imp = importances / total

        # Top-5 feature
        top_idx = np.argsort(norm_imp)[::-1][:5]
        top_features: list[FeatureExplanation] = []
        fdict = dict(zip(FEATURE_NAMES, feature_vector))

        for idx in top_idx:
            name = FEATURE_NAMES[idx]
            val  = fdict[name]
            direction = "high" if val > 0.5 else "low"
            if name.startswith("record_type_") or name == "is_nxdomain":
                direction = "present" if val > 0.5 else "absent"
            top_features.append(
                FeatureExplanation(
                    feature=name,
                    importance=round(float(norm_imp[idx]), 4),
                    value=round(float(val), 4),
                    direction=direction,
                )
            )

        summary = self._build_summary(predicted_type, top_features)
        return top_features, summary

    def _gradient_importance(self, farray: np.ndarray) -> np.ndarray:
        """SHAP yoksa basit gradient tabanlı skor."""
        t = torch.tensor(farray, requires_grad=True)
        err = self._model.reconstruction_error(t)
        err.backward()
        return t.grad.abs().numpy().flatten()

    @staticmethod
    def _build_summary(predicted_type: str, features: list[FeatureExplanation]) -> str:
        if not features:
            return "Anomali tespit edildi."
        top_name = features[0].feature
        descriptions: dict[str, str] = {
            "tunnel":  f"Yüksek {top_name} değeri DNS tünelleme göstergesi",
            "ddos":    f"Yüksek sorgu oranı DDoS saldırısı göstergesi",
            "flux":    f"Rastgele domain üretimi (domain flux) tespit edildi",
        }
        return descriptions.get(predicted_type, f"Anomali tespit edildi ({predicted_type})")
