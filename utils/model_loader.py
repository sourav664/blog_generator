import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from google import genai
from google.genai import types
from utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import BlogGeneratorException


class ApiKeyManager:
    """
    Loads and manages all environment-based API keys.
    """

    def __init__(self):
        load_dotenv()

        self.api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        }

        log.info("Initializing ApiKeyManager")

        # Log loaded key statuses without exposing secrets
        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded successfully from environment")
            else:
                log.warning(f"{key} is missing in environment variables")

    def get(self, key: str):
        """
        Retrieve a specific API key.

        Args:
            key (str): Name of the API key.

        Returns:
            str | None: API key value if found.
        """
        return self.api_keys.get(key)


class ModelLoader:
    """
    Loads embedding models and LLMs dynamically based on YAML configuration and environment settings.
    """

    def __init__(self):
        """
        Initialize the ModelLoader and load configuration.
        """
        try:
            self.api_key_mgr = ApiKeyManager()
            self.config = load_config()
            log.info("YAML configuration loaded successfully", config_keys=list(self.config.keys()))
        except Exception as e:
            log.error("Error initializing ModelLoader", error=str(e))
            raise BlogGeneratorException("Failed to initialize ModelLoader", sys)

    

    # ----------------------------------------------------------------------
    # LLM Loader
    # ----------------------------------------------------------------------
    def load_llm(self):
        """
        Load and return a chat-based LLM according to the configured provider.

        Supported providers:
            - OpenAI
            - Google (Gemini)
            - Groq

        Returns:
            ChatOpenAI | ChatGoogleGenerativeAI | ChatGroq: LLM instance
        """
        try:
            llm_block = self.config["llm"]
            provider_key = os.getenv("LLM_PROVIDER", "openai")

            if provider_key not in llm_block:
                log.error("LLM provider not found in configuration", provider=provider_key)
                raise ValueError(f"LLM provider '{provider_key}' not found in configuration")

            llm_config = llm_block[provider_key]
            provider = llm_config.get("provider")
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0.2)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "google":
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            elif provider == "groq":
                llm = ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                    temperature=temperature,
                )

            elif provider == "openai":
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                    temperature=temperature,
                )

            else:
                log.error("Unsupported LLM provider encountered", provider=provider)
                raise ValueError(f"Unsupported LLM provider: {provider}")

            log.info("LLM loaded successfully", provider=provider, model=model_name)
            return llm

        except Exception as e:
            log.error("Error loading LLM", error=str(e))
            raise BlogGeneratorException("Failed to load LLM", sys)
        
        
    def image_model(self, prompt):
        """
        Load and return a Image-based LLM according to the configured provider.

        Supported provider:
            - OpenAI
            - Google (Gemini)

        Returns:
            OpenAI | GoogleGenAI | ChatGroq: LLM instance
        """
        
        try:
            llm_block = self.config["llm"]
            provider_key = os.getenv("LLM_PROVIDER", "openai")

            if provider_key not in llm_block:
                log.error("LLM provider not found in configuration", provider=provider_key)
                raise ValueError(f"LLM provider '{provider_key}' not found in configuration")

            llm_config = llm_block[provider_key]
            provider = llm_config.get("provider")
            model_name = llm_config.get("model_name")

            log.info("Loading LLM", provider=provider, model=model_name)
            
            
            if provider == "openai":
                client = OpenAI(
                      api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                )
                image_generator = client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    size="1024x1024"
                )
                
            elif provider == "google":
                client = genai.Client(api_key=self.api_key_mgr.get("GOOGLE_API_KEY"))
                
                image_generator = client.models.generate_content(
                                        model=model_name,
                                        contents=prompt,
                                        config=types.GenerateContentConfig(
                                            response_modalities=["IMAGE"],
                                            safety_settings=[
                                                types.SafetySetting(
                                                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                                    threshold="BLOCK_ONLY_HIGH",
                                                )
                                            ],
                                        ),
                                    )
                
            else:
                log.error("Unsupported LLM provider encountered", provider=provider)
                raise ValueError(f"Unsupported LLM provider: {provider}")

            log.info("LLM loaded successfully", provider=provider, model=model_name)
            return image_generator

        except Exception as e:
            log.error("Error loading Image LLM", error=str(e))
            raise BlogGeneratorException("Failed to load LLM", sys)


# ----------------------------------------------------------------------
# Standalone Testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        loader = ModelLoader()

        # # Test embedding model
        # embeddings = loader.load_embeddings()
        # print(f"Embedding Model Loaded: {embeddings}")
        # result = embeddings.embed_query("Hello, how are you?")
        # print(f"Embedding Result: {result[:5]} ...")

        # Test LLM
        llm = loader.load_llm()
        print(f"LLM Loaded: {llm}")
        result = llm.invoke("Hello, how are you?")
        print(f"LLM Result: {result.content[:200]}")

        log.info("ModelLoader test completed successfully")

    except BlogGeneratorException as e:
        log.error("Critical failure in ModelLoader test", error=str(e))
        
        
# Write a clean, enterprise-grade Python module for dynamic model loading in a structured AI backend system.
# The system must follow clean architecture principles and separate API key management, configuration loading, 
# and model initialization logic. Use environment variables and YAML configuration to determine which LLM provider to load. 
# Support OpenAI, Google Gemini, and Groq chat models. Include structured logging at every stage, avoid exposing secrets, 
# ensure async loop safety for gRPC-based embedding APIs, and wrap all failures using a custom domain exception class. 
# Provide complete documentation, comments, error handling, and a standalone test block for local validation.