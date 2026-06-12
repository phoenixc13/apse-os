"""
APSE OS - AI Provider Registry
Manages Google AI Studio, GitHub Models, HuggingFace and local providers.
Declare your provider in configs/providers/ and register here.
"""
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ProviderType(Enum):
    LOCAL = 'local'
    REMOTE_API = 'remote_api'
    HUB_AND_INFERENCE = 'hub_and_inference'


class LicenseMode(Enum):
    APACHE2 = 'apache-2.0'
    MIT = 'mit'
    PROVIDER_TERMS = 'provider-terms'
    PER_MODEL_CARD = 'per-model-card'
    COMMERCIAL = 'commercial'
    RESEARCH_ONLY = 'research-only'


@dataclass
class ProviderDescriptor:
    id: str
    name: str
    type: ProviderType
    portal_url: str
    auth_env_var: str
    license_mode: LicenseMode
    models: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    enabled: bool = False

    def is_authenticated(self) -> bool:
        token = os.environ.get(self.auth_env_var, '')
        return bool(token)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'portal_url': self.portal_url,
            'auth_env_var': self.auth_env_var,
            'authenticated': self.is_authenticated(),
            'license_mode': self.license_mode.value,
            'models': self.models,
            'enabled': self.enabled,
            'notes': self.notes
        }


class ProviderRegistry:
    """
    APSE AI Provider Registry.
    Connects to Google AI Studio, GitHub Models, HuggingFace and local runtimes.
    """

    BUILTIN_PROVIDERS = [
        ProviderDescriptor(
            id='google',
            name='Google AI Studio (Gemini)',
            type=ProviderType.REMOTE_API,
            portal_url='https://ai.google.dev/aistudio',
            auth_env_var='GOOGLE_API_KEY',
            license_mode=LicenseMode.PROVIDER_TERMS,
            models=['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash'],
            notes=['requires_google_api_key', 'remote_inference_only', 'rate_limits_apply']
        ),
        ProviderDescriptor(
            id='github-models',
            name='GitHub Models',
            type=ProviderType.REMOTE_API,
            portal_url='https://github.com/marketplace/models',
            auth_env_var='GITHUB_TOKEN',
            license_mode=LicenseMode.PER_MODEL_CARD,
            models=['multiple_catalog'],
            notes=['github_credentials_required', 'catalog_and_inference_api', 'check_per_model_license']
        ),
        ProviderDescriptor(
            id='huggingface',
            name='HuggingFace Hub & Inference',
            type=ProviderType.HUB_AND_INFERENCE,
            portal_url='https://huggingface.co/models',
            auth_env_var='HF_TOKEN',
            license_mode=LicenseMode.PER_MODEL_CARD,
            models=['openvla-small', 'openvla-7b', 'llava-next', 'phi-3-vision'],
            notes=['check_model_card_license', 'inference_endpoints_available', 'some_models_gated']
        ),
        ProviderDescriptor(
            id='local',
            name='Local ONNX Runtime',
            type=ProviderType.LOCAL,
            portal_url='',
            auth_env_var='',
            license_mode=LicenseMode.APACHE2,
            models=['custom'],
            notes=['no_internet_required', 'lowest_latency', 'hardware_dependent'],
            enabled=True
        ),
    ]

    def __init__(self):
        self._providers: Dict[str, ProviderDescriptor] = {}
        for p in self.BUILTIN_PROVIDERS:
            self._providers[p.id] = p
            if p.is_authenticated():
                p.enabled = True
                print(f'[REGISTRY] Provider enabled: {p.id} (token found)')
            else:
                print(f'[REGISTRY] Provider registered (no token): {p.id}')

    def get_provider(self, provider_id: str) -> Optional[ProviderDescriptor]:
        return self._providers.get(provider_id)

    def list_providers(self, only_enabled: bool = False) -> List[ProviderDescriptor]:
        providers = list(self._providers.values())
        if only_enabled:
            return [p for p in providers if p.enabled]
        return providers

    def register_custom(self, provider: ProviderDescriptor):
        self._providers[provider.id] = provider
        print(f'[REGISTRY] Custom provider registered: {provider.id}')

    def best_available(self) -> Optional[ProviderDescriptor]:
        priority = ['local', 'google', 'github-models', 'huggingface']
        for pid in priority:
            p = self._providers.get(pid)
            if p and p.enabled:
                return p
        return None

    def summary(self) -> dict:
        return {
            'total': len(self._providers),
            'enabled': sum(1 for p in self._providers.values() if p.enabled),
            'providers': [p.to_dict() for p in self._providers.values()]
        }


# --- Singleton ---
_registry = ProviderRegistry()


def get_registry() -> ProviderRegistry:
    return _registry
