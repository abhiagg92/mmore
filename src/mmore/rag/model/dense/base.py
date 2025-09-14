from dataclasses import dataclass
from typing import Optional

from langchain_aws import BedrockEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from .multimodal import MultimodalEmbeddings

_OPENAI_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

_GOOGLE_MODELS = ["textembedding-gecko@001"]

_COHERE_MODELS = [
    "embed-english-light-v2.0",
    "embed-english-v2.0",
    "embed-multilingual-v2.0",
]

_MISTRAL_MODELS = ["mistral-textembedding-7B-v1", "mistral-textembedding-13B-v1"]

_NVIDIA_MODELS = ["nvidia-clarity-text-embedding-v1", "nvidia-megatron-embedding-530B"]

_AWS_MODELS = ["amazon-titan-embedding-xlarge", "amazon-titan-embedding-light"]

_OLLAMA_MODELS = ["llama2", "llama3", "vicuna", "alpaca", "wizardlm"]


loaders = {
    "OPENAI": OpenAIEmbeddings,
    # 'GOOGLE': VertexAIEmbeddings,
    "COHERE": CohereEmbeddings,
    "MISTRAL": MistralAIEmbeddings,
    "NVIDIA": NVIDIAEmbeddings,
    "AWS": BedrockEmbeddings,
    "HF": lambda model, **kwargs: HuggingFaceEmbeddings(
        model_name=model, model_kwargs={"trust_remote_code": True}, **kwargs
    ),
    "OLLAMA": OllamaEmbeddings,
    "FAKE": lambda **kwargs: FakeEmbeddings(
        size=2048
    ),  # For testing purposes, don't use in production
}


@dataclass
class DenseModelConfig:
    model_name: str
    is_multimodal: bool = False
    organization: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama server URL

    def __post_init__(self) -> None:
        if self.organization:
            self.organization = self.organization.upper()
        elif self.model_name in _OPENAI_MODELS:
            self.organization = "OPENAI"
        elif self.model_name in _GOOGLE_MODELS:
            self.organization = "GOOGLE"
        elif self.model_name in _COHERE_MODELS:
            self.organization = "COHERE"
        elif self.model_name in _MISTRAL_MODELS:
            self.organization = "MISTRAL"
        elif self.model_name in _NVIDIA_MODELS:
            self.organization = "NVIDIA"
        elif self.model_name in _AWS_MODELS:
            self.organization = "AWS"
        elif self.model_name in _OLLAMA_MODELS:
            self.organization = "OLLAMA"
        elif self.model_name == "debug":
            self.organization = "FAKE"  # For testing purposes
        else:
            self.organization = "HF"


class DenseModel(Embeddings):
    @classmethod
    def from_config(cls, config: DenseModelConfig) -> Embeddings:
        if config.organization == "HF" and config.is_multimodal:
            return MultimodalEmbeddings(model_name=config.model_name)
        elif config.organization == "OLLAMA":
            return OllamaEmbeddings(
                model=config.model_name,
                base_url=config.base_url,
            )
        else:
            return loaders[config.organization](model=config.model_name)
