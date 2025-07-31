#!/usr/bin/env python
# coding: utf-8

# # DEPENDENCIES:

# In[1]:


get_ipython().system('pip install openai')


# In[2]:


get_ipython().system('pip install rapidfuzz numpy')

get_ipython().system('pip install torch sentence-transformers faiss-cpu transformers')

get_ipython().system('pip install deep-translator')


# In[3]:


get_ipython().system('pip install langdetect')


# # Chatbot:

# In[5]:


"""
RADEETA Water Distribution and Sanitation Chatbot System - Simplified FAQ + GPT Version 2.3.0

Simplified architecture: FAQ + GPT (no RAG, GPT-based translation)
"""

import os
import json
import re
import time
import warnings
import gc
import logging
import traceback
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rapidfuzz import fuzz

# OpenAI and GitHub Models integration
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("OpenAI library loaded successfully")
except ImportError as e:
    OPENAI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"OpenAI library not available: {e}")

# Google Colab integration for secrets
try:
    from google.colab import userdata
    COLAB_AVAILABLE = True
    logger.info("Google Colab userdata available")
except ImportError:
    COLAB_AVAILABLE = False
    logger.warning("Google Colab userdata not available")

# Production logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('radeeta_chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Language detection with graceful fallbacks
try:
    from langdetect import detect
    try:
        from langdetect import LangDetectError
    except ImportError:
        try:
            from langdetect.lang_detect_exception import LangDetectError
        except ImportError:
            class LangDetectError(Exception):
                pass

    LANGDETECT_AVAILABLE = True
    logger.info("LangDetect library loaded successfully")
except ImportError as e:
    logger.warning(f"LangDetect not available: {e}")
    LANGDETECT_AVAILABLE = False
    class LangDetectError(Exception):
        pass

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# SIMPLIFIED CONFIGURATION
# ============================================================================

class Config:
    """Simplified configuration for FAQ + GPT architecture."""

    # Core paths
    FAQ_FILE_PATH = "radeeta_faq_perfect_105.json"

    # Language detection and translation
    SUPPORTED_LANGUAGES = ['fr', 'en', 'ar']
    DEFAULT_LANGUAGE = 'fr'
    DETECTION_CONFIDENCE_THRESHOLD = 0.8

    # FAQ system optimization
    FAQ_SIMILARITY_THRESHOLD = 35
    FAQ_HIGH_CONFIDENCE = 0.85
    FAQ_MEDIUM_CONFIDENCE = 0.65
    FAQ_LOW_CONFIDENCE = 0.45

    # GPT routing threshold - if FAQ confidence < this, use GPT
    GPT_ROUTING_THRESHOLD = 0.70

    # Performance optimization
    CACHE_SIZE = 1000
    THREAD_POOL_SIZE = 4

    # Organization metadata
    ORGANIZATION = "RADEETA"
    ORGANIZATION_FULL = "Régie Autonome de Distribution d'Eau et d'Assainissement de Taza"
    DOMAIN = "Water Distribution and Sanitation"

    # Contact information
    CONTACT_INFO = {
        'general': {
            'phone': "0535-55-00-00",
            'emergency': "0535-55-00-00/15/16/17",
            'website': "www.radeeta.ma",
            'address': "Avenue Mohammed V, Taza"
        },
        'billing': {
            'phone': "0535-55-00-00/15/16/17",
            'email': "contact@radeeta.ma",
            'hours': "Lundi-Vendredi 8:30h-16:30h"
        },
        'technical': {
            'phone': "0535-67-BB-BB",
            'emergency': "0535-55-00-00/15/16/17, Veuillez appelez/vous presentez à l'agence en cas d'urgence"
        }
    }

    # GitHub Models configuration
    try:
        if COLAB_AVAILABLE:
            GITHUB_TOKEN = userdata.get('RADEETAchatbot')
            if GITHUB_TOKEN:
                os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN
        else:
            GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
    except:
        GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

    GITHUB_ENDPOINT = "https://models.github.ai/inference"
    GITHUB_MODEL = "openai/gpt-4.1"
    GITHUB_MAX_TOKENS = 400
    GITHUB_TEMPERATURE = 0.3

# ============================================================================
# UTILITIES
# ============================================================================

class Utils:
    """Utility functions for text processing."""

    @staticmethod
    @lru_cache(maxsize=500)
    def preprocess_french_text(text: str) -> str:
        """Enhanced French text preprocessing with caching."""
        if not text:
            return ""

        # Normalize and clean
        text = re.sub(r'\s+', ' ', text.lower().strip())
        text = re.sub(r'[^\w\s\-\'àâäçéèêëïîôöùûüÿñ]', '', text)

        # French-specific optimizations
        replacements = {
            "qu'est-ce que": "quest ce que",
            "d'eau": "eau",
            "où": "ou",
            "être": "etre",
            "c'est": "cest",
            "numéro": "numero",
            "téléphone": "telephone",
            "adresse": "adresse"
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    @staticmethod
    def validate_json_file(file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate JSON file structure efficiently."""
        if not Path(file_path).exists():
            return False, f"File {file_path} not found"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list) or len(data) == 0:
                return False, "Invalid data structure"

            # Validation for FAQ items
            for item in data:
                if not isinstance(item, dict) or 'question' not in item or 'answer' not in item:
                    return False, "Invalid FAQ item structure"

            return True, None

        except json.JSONDecodeError as e:
            return False, f"JSON error: {e}"
        except Exception as e:
            return False, f"File error: {e}"

    @staticmethod
    def safe_load_json(file_path: str, fallback: Any = None) -> Any:
        """Safely load JSON with error handling and fallback."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return fallback or []

# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

class LanguageDetector:
    """Language detection with pattern matching fallback."""

    def __init__(self):
        self.stats = defaultdict(int)

    def detect_language(self, text: str) -> Dict[str, Any]:
        """Enhanced language detection with confidence scoring."""
        if not text or len(text.strip()) < 3:
            return self._create_result('unknown', 0.0, 'empty_input')

        # Try LangDetect first
        if LANGDETECT_AVAILABLE:
            try:
                detected = detect(text)
                confidence = self._calculate_langdetect_confidence(text, detected)

                if confidence >= Config.DETECTION_CONFIDENCE_THRESHOLD:
                    self.stats['langdetect_success'] += 1
                    return self._create_result(detected, confidence, 'langdetect')

            except (LangDetectError, Exception) as e:
                logger.debug(f"LangDetect failed for text '{text[:50]}...': {e}")
                pass

        # Fallback to pattern matching
        lang, conf = self._pattern_detect(text)
        self.stats['pattern_fallback'] += 1
        return self._create_result(lang, conf, 'pattern')

    def _calculate_langdetect_confidence(self, text: str, detected_lang: str) -> float:
        """Calculate confidence based on text characteristics."""
        base_confidence = min(0.9, 0.6 + (len(text) / 100) * 0.3)

        # Boost for supported languages
        if detected_lang in Config.SUPPORTED_LANGUAGES:
            base_confidence += 0.1

        # Domain-specific terms
        domain_terms_fr = ['facture', 'eau', 'assainissement', 'radeeta']
        domain_terms_en = ['bill', 'water', 'sanitation', 'radeeta']

        if detected_lang == 'fr' and any(term in text.lower() for term in domain_terms_fr):
            base_confidence += 0.1
        elif detected_lang == 'en' and any(term in text.lower() for term in domain_terms_en):
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    @lru_cache(maxsize=200)
    def _pattern_detect(self, text: str) -> Tuple[str, float]:
        """Pattern-based detection with caching."""
        text_lower = text.lower()

        # Arabic script detection
        if re.search(r'[\u0600-\u06FF]', text):
            return 'ar', 0.95

        # Language indicators with weights
        french_indicators = [
            ('le', 0.2), ('la', 0.2), ('les', 0.2), ('de', 0.1), ('du', 0.1),
            ('eau', 0.3), ('facture', 0.4), ('comment', 0.2), ('où', 0.2),
            ('radeeta', 0.3), ('assainissement', 0.4)
        ]

        english_indicators = [
            ('the', 0.2), ('and', 0.1), ('water', 0.3), ('bill', 0.4),
            ('how', 0.2), ('what', 0.2), ('service', 0.3), ('radeeta', 0.3)
        ]

        french_score = sum(weight for word, weight in french_indicators if word in text_lower)
        english_score = sum(weight for word, weight in english_indicators if word in text_lower)

        total_words = len(text_lower.split())

        if french_score > english_score and french_score > 0:
            confidence = min(0.8, 0.5 + (french_score / max(total_words, 1)))
            return 'fr', confidence
        elif english_score > 0:
            confidence = min(0.8, 0.5 + (english_score / max(total_words, 1)))
            return 'en', confidence

        # Default to French for RADEETA context
        return 'fr', 0.5

    def _create_result(self, language: str, confidence: float, method: str) -> Dict[str, Any]:
        """Create standardized detection result."""
        self.stats['total_detections'] += 1
        return {
            'language': language,
            'confidence': confidence,
            'method': method,
            'supported': language in Config.SUPPORTED_LANGUAGES
        }

# ============================================================================
# GPT HANDLER FOR BOTH RESPONSES AND TRANSLATION
# ============================================================================

class GPTHandler:
    """Unified GPT handler for responses and translation using GitHub Models."""

    def __init__(self):
        self.github_token = None
        self.client = None
        self.endpoint = Config.GITHUB_ENDPOINT
        self.model = Config.GITHUB_MODEL
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'translation_requests': 0,
            'response_requests': 0
        }

        # Initialize token and client
        self._initialize_token()

        if self.github_token and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(
                    base_url=self.endpoint,
                    api_key=self.github_token,
                )
                logger.info(f"✅ GPT handler initialized: {self.model}")
            except Exception as e:
                logger.error(f"❌ GPT initialization failed: {e}")
                self.client = None
        else:
            logger.warning("⚠️ GPT not available - missing token or OpenAI library")

    def _initialize_token(self):
        """Initialize GitHub token from Colab secret or environment."""
        try:
            if COLAB_AVAILABLE:
                self.github_token = userdata.get('RADEETAchatbot')
                if self.github_token:
                    os.environ["GITHUB_TOKEN"] = self.github_token
                    logger.info("✅ GitHub token loaded from Colab secret")
                else:
                    logger.warning("⚠️ No token found in Colab secret 'RADEETAchatbot'")
            else:
                self.github_token = os.environ.get("GITHUB_TOKEN")
                if self.github_token:
                    logger.info("✅ GitHub token loaded from environment")
                else:
                    logger.warning("⚠️ No GitHub token found in environment")
        except Exception as e:
            logger.error(f"❌ Token initialization failed: {e}")
            self.github_token = None

    def translate_to_french(self, text: str, source_language: str) -> Dict[str, Any]:
        """Translate text to French using GPT."""
        if not self.client or source_language == 'fr':
            return {
                'translated_text': text,
                'source_language': source_language,
                'translation_needed': False,
                'success': True
            }

        try:
            self.stats['total_requests'] += 1
            self.stats['translation_requests'] += 1

            # Create translation prompt
            system_prompt = f"""You are a professional translator specializing in water utility terminology.
Translate the following {source_language} text to French, maintaining technical accuracy for RADEETA water utility context.

IMPORTANT:
- Only provide the translation, no explanations
- Maintain technical water/sanitation terminology
- Keep RADEETA as "RADEETA"
- Preserve question structure if it's a question"""

            user_message = f"Translate this {source_language} text to French: {text}"

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=0.1,  # Low temperature for consistent translation
                max_tokens=200,  # Shorter for translations
                top_p=0.9
            )

            translated_text = response.choices[0].message.content.strip()

            # Track usage
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                self.stats['total_tokens_used'] += tokens_used

            self.stats['successful_requests'] += 1

            return {
                'translated_text': translated_text,
                'source_language': source_language,
                'translation_needed': True,
                'success': True,
                'tokens_used': tokens_used
            }

        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Translation failed: {e}")

            return {
                'translated_text': text,
                'source_language': source_language,
                'translation_needed': False,
                'success': False,
                'error': str(e)
            }

    def translate_from_french(self, text: str, target_language: str) -> Dict[str, Any]:
        """Translate text from French to target language using GPT."""
        if not self.client or target_language == 'fr':
            return {'translated_text': text, 'success': True}

        try:
            self.stats['total_requests'] += 1
            self.stats['translation_requests'] += 1

            system_prompt = f"""You are a professional translator specializing in water utility terminology.
Translate the following French text to {target_language}, maintaining technical accuracy.

IMPORTANT:
- Only provide the translation, no explanations
- Maintain technical water/sanitation terminology
- Keep RADEETA as "RADEETA"
- Preserve response structure and contact information format"""

            user_message = f"Translate this French text to {target_language}: {text}"

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=300,
                top_p=0.9
            )

            translated_text = response.choices[0].message.content.strip()

            # Track usage
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                self.stats['total_tokens_used'] += tokens_used

            self.stats['successful_requests'] += 1

            return {
                'translated_text': translated_text,
                'success': True,
                'tokens_used': tokens_used
            }

        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Back-translation failed: {e}")

            return {
                'translated_text': text,
                'success': False,
                'error': str(e)
            }

    def generate_response(self, query: str, language: str = "fr") -> Dict[str, Any]:
        """Generate RADEETA response using GPT."""
        if not self.client:
            return {
                'text': f"Service GPT non disponible. Contactez le support: {Config.CONTACT_INFO['general']['phone']}",
                'success': False,
                'error': 'no_client'
            }

        try:
            self.stats['total_requests'] += 1
            self.stats['response_requests'] += 1

            # RADEETA-specific system prompt
            system_prompt = f"""Vous êtes un assistant expert de la RADEETA ({Config.ORGANIZATION_FULL}),
spécialisée dans les services de distribution d'eau potable et d'assainissement à Taza, Maroc.

INFORMATIONS RADEETA:
- Téléphone général: {Config.CONTACT_INFO['general']['phone']}
- Urgences 24h/24: {Config.CONTACT_INFO['general']['emergency']}
- Service facturation: {Config.CONTACT_INFO['billing']['phone']}
- Email: {Config.CONTACT_INFO['billing']['email']}
- Site web: {Config.CONTACT_INFO['general']['website']}
- Adresse: {Config.CONTACT_INFO['general']['address']}

MISSION:
- Aider les clients avec facturation, qualité eau, coupures, raccordements
- Fournir informations précises sur services RADEETA
- Diriger vers bons services selon question

CONSIGNES:
- Répondez TOUJOURS en français
- Ton professionnel mais accessible
- Maximum 4 phrases par réponse
- Incluez contact pertinent si nécessaire
- Si incertain, dirigez vers service client
- Basez-vous sur vos connaissances générales des services d'eau et d'assainissement"""

            user_message = f"QUESTION: {query}\n\nFournissez une réponse claire et utile:"

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=Config.GITHUB_TEMPERATURE,
                top_p=0.9,
                max_tokens=Config.GITHUB_MAX_TOKENS
            )

            response_text = response.choices[0].message.content.strip()

            # Clean response
            response_text = self._clean_response(response_text)

            # Track usage
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                self.stats['total_tokens_used'] += tokens_used

            self.stats['successful_requests'] += 1

            return {
                'text': response_text,
                'success': True,
                'tokens_used': tokens_used,
                'model': self.model,
                'confidence': 0.92,  # High confidence for GPT-4.1
                'service': 'GPT',
                'method': 'gpt'
            }

        except Exception as e:
            self.stats['failed_requests'] += 1

            # User-friendly error in French
            error_msg = str(e).lower()
            if 'unauthorized' in error_msg:
                user_error = f"Problème d'authentification système. Contactez le support: {Config.CONTACT_INFO['general']['phone']}"
            elif 'rate limit' in error_msg or 'quota' in error_msg:
                user_error = "Service temporairement surchargé. Réessayez dans quelques minutes."
            elif 'timeout' in error_msg:
                user_error = "Délai de réponse dépassé. Veuillez réessayer."
            else:
                user_error = f"Service temporairement indisponible. Contactez notre service client: {Config.CONTACT_INFO['general']['phone']}"

            logger.error(f"GPT error: {e}")

            return {
                'text': user_error,
                'success': False,
                'error': str(e),
                'service': 'GPT (Error)'
            }

    def _clean_response(self, response: str) -> str:
        """Clean and format response."""
        # Remove AI prefixes
        response = re.sub(r'^(Réponse|Response|Answer):\s*', '', response, flags=re.IGNORECASE)

        # Ensure proper punctuation
        if response and not response.endswith(('.', '!', '?')):
            response += '.'

        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:]

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        success_rate = 0
        if self.stats['total_requests'] > 0:
            success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100

        return {
            'service': 'GPT Handler',
            'model': self.model,
            'endpoint': self.endpoint,
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': success_rate,
            'total_tokens_used': self.stats['total_tokens_used'],
            'translation_requests': self.stats['translation_requests'],
            'response_requests': self.stats['response_requests'],
            'avg_tokens_per_request': self.stats['total_tokens_used'] / max(1, self.stats['successful_requests']),
            'available': self.client is not None
        }

# ============================================================================
# FAQ SYSTEM
# ============================================================================

class FAQSystem:
    """FAQ system with improved matching."""

    def __init__(self, faq_file: str = Config.FAQ_FILE_PATH):
        self.faq_file = faq_file
        self.faq_data = []
        self.processed_questions = []
        self.category_index = defaultdict(list)
        self.keyword_index = defaultdict(set)
        self.cache = {}
        self.stats = defaultdict(int)

        logger.info("Initializing FAQ system")
        self._load_and_index_data()

    def _load_and_index_data(self) -> None:
        """Load and index FAQ data."""
        is_valid, error = Utils.validate_json_file(self.faq_file)
        if not is_valid:
            logger.error(f"FAQ validation failed: {error}")
            return

        self.faq_data = Utils.safe_load_json(self.faq_file, [])
        if not self.faq_data:
            logger.error("No FAQ data loaded")
            return

        # Add default contact questions if not present
        self._add_default_questions()
        self._build_indexes()
        logger.info(f"FAQ system loaded: {len(self.faq_data)} questions")

    def _add_default_questions(self) -> None:
        """Add default contact questions if not already present."""
        contact_questions = [
            {
                "question": "Quel est le numéro de téléphone de la RADEETA ?",
                "answer": f"Le numéro de téléphone principal est {Config.CONTACT_INFO['general']['phone']}. Pour les urgences, composez le {Config.CONTACT_INFO['general']['emergency']}.",
                "category": "Service Client",
                "id": "contact_phone"
            },
            {
                "question": "Où se trouve le siège de la RADEETA ?",
                "answer": f"Le siège de la RADEETA est situé à l'adresse suivante : {Config.CONTACT_INFO['general']['address']}.",
                "category": "Service Client",
                "id": "contact_address"
            }
        ]

        existing_ids = {q.get('id') for q in self.faq_data if 'id' in q}

        for question in contact_questions:
            if question['id'] not in existing_ids:
                self.faq_data.append(question)
                logger.info(f"Added default question: {question['id']}")

    def _build_indexes(self) -> None:
        """Build indexes for faster searching."""
        for i, item in enumerate(self.faq_data):
            if not isinstance(item, dict) or 'question' not in item:
                continue

            # Process question text
            processed = Utils.preprocess_french_text(item['question'])
            keywords = set(processed.split())

            # Add synonyms and related terms
            if 'téléphone' in processed or 'numero' in processed:
                keywords.update(['contact', 'appeler', 'joindre'])
            if 'adresse' in processed:
                keywords.update(['localisation', 'siège', 'bureau'])

            self.processed_questions.append({
                'index': i,
                'processed': processed,
                'keywords': keywords,
                'category': item.get('category', 'General'),
                'priority': self._calculate_priority(item)
            })

            # Build category index
            category = item.get('category', 'General')
            self.category_index[category].append(i)

            # Build keyword index
            for word in keywords:
                if len(word) > 2:
                    self.keyword_index[word].add(i)

    def _calculate_priority(self, item: Dict) -> float:
        """Calculate priority for FAQ items."""
        score = 1.0

        # High priority categories
        high_priority = ['Urgence', 'Facturation', 'Coupure', 'Qualité', 'Service Client']
        if any(cat in item.get('category', '') for cat in high_priority):
            score += 0.5

        # Priority keywords
        priority_words = {
            'urgence': 0.2, 'facture': 0.3, 'paiement': 0.3,
            'coupure': 0.4, 'qualité': 0.3, 'téléphone': 0.2,
            'contact': 0.2, 'adresse': 0.2, 'radeeta': 0.1
        }

        question_lower = item.get('question', '').lower()
        answer_lower = item.get('answer', '').lower()

        for word, weight in priority_words.items():
            if word in question_lower or word in answer_lower:
                score += weight

        return min(score, 2.0)

    @lru_cache(maxsize=Config.CACHE_SIZE)
    def find_best_answer(self, query: str, min_confidence: float = 0.4) -> Optional[Dict]:
        """Find best FAQ answer with improved scoring."""
        self.stats['total_searches'] += 1

        if not query:
            return None

        processed_query = Utils.preprocess_french_text(query)
        query_words = set(processed_query.split())

        # Keyword-based filtering
        candidate_indices = set()
        for word in query_words:
            if word in self.keyword_index:
                candidate_indices.update(self.keyword_index[word])

            # Partial matches for longer words
            if len(word) > 4:
                for keyword in self.keyword_index.keys():
                    if word in keyword or keyword in word:
                        candidate_indices.update(self.keyword_index[keyword])

        if not candidate_indices:
            candidate_indices = set(range(len(self.processed_questions)))

        # Score candidates
        best_score = 0
        best_match = None

        for i in candidate_indices:
            if i >= len(self.processed_questions):
                continue

            item = self.processed_questions[i]
            original_item = self.faq_data[item['index']]

            # Calculate similarity
            fuzzy_score = fuzz.token_set_ratio(processed_query, item['processed'])
            partial_score = fuzz.partial_ratio(processed_query, item['processed'])

            # Keyword overlap
            keyword_overlap = len(query_words & item['keywords'])
            keyword_bonus = (keyword_overlap / max(len(query_words), 1)) * 25

            # Priority bonus
            priority_bonus = item['priority'] * 10

            # Combined score
            total_score = (fuzzy_score * 0.6 + partial_score * 0.4) + keyword_bonus + priority_bonus

            if total_score > best_score and (fuzzy_score >= Config.FAQ_SIMILARITY_THRESHOLD or partial_score >= 75):
                best_score = total_score
                best_match = {
                    'answer': original_item['answer'],
                    'matched_question': original_item['question'],
                    'category': item['category'],
                    'confidence': min(total_score / 130, 1.0),
                    'id': original_item.get('id', item['index']),
                    'method': 'FAQ'
                }

        if best_match and best_match['confidence'] >= min_confidence:
            self.stats['successful_matches'] += 1
            return best_match

        return None

    def get_category_suggestions(self, query: str) -> List[str]:
        """Get category suggestions based on query."""
        query_lower = query.lower()

        category_patterns = {
            'Facturation et Paiement': ['facture', 'paiement', 'tarif', 'cout', 'montant', 'payer'],
            'Qualité de l\'Eau': ['qualité', 'eau', 'potable', 'analyse', 'goût', 'odeur', 'couleur'],
            'Coupures d\'Eau': ['coupure', 'panne', 'manque', 'pression', 'réseau'],
            'Service Client': ['contact', 'téléphone', 'agence', 'adresse', 'heure', 'horaires']
        }

        suggestions = []
        for category, keywords in category_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                suggestions.append(category)

        return suggestions

# ============================================================================
# MAIN CHATBOT SYSTEM - SIMPLIFIED FAQ + GPT
# ============================================================================

class RADEETAChatbotSystem:
    """Simplified RADEETA chatbot with FAQ + GPT architecture."""

    def __init__(self,
                 enable_translation: bool = True,
                 enable_gpt: bool = True):

        logger.info("Initializing simplified RADEETA System (FAQ + GPT)")

        # Initialize components
        self.language_detector = LanguageDetector()
        self.faq_system = FAQSystem()
        self.gpt_handler = GPTHandler() if enable_gpt else None

        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'response_times': [],
            'methods_used': defaultdict(int),
            'languages_detected': defaultdict(int),
            'failed_queries': 0,
            'gpt_used': 0
        }

        # Capabilities detection
        self.capabilities = {
            'faq': len(self.faq_system.faq_data) > 0,
            'gpt': self.gpt_handler and self.gpt_handler.client is not None,
            'translation': enable_translation and self.gpt_handler and self.gpt_handler.client is not None
        }

        logger.info(f"System initialized - Capabilities: {self.capabilities}")

    def process_query(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """Process query with simplified FAQ + GPT routing."""
        start_time = time.time()
        self.stats['total_queries'] += 1

        if not query or not query.strip():
            return self._error_response("Empty query", start_time)

        try:
            original_query = query.strip()
            source_lang = 'fr'
            translation_result = {}

            # Language detection and translation
            if self.capabilities['translation']:
                detection = self.language_detector.detect_language(original_query)
                source_lang = detection['language']
                self.stats['languages_detected'][source_lang] += 1

                if source_lang != 'fr' and detection['confidence'] > Config.DETECTION_CONFIDENCE_THRESHOLD:
                    translation_result = self.gpt_handler.translate_to_french(original_query, source_lang)

                    if translation_result['success']:
                        query = translation_result['translated_text']
                        if debug:
                            logger.info(f"Translated {source_lang} -> fr: {original_query} -> {query}")
                    else:
                        if debug:
                            logger.warning(f"Translation failed, using original query")

            # Routing decision: FAQ or GPT
            method = self._select_method(query, debug)
            self.stats['methods_used'][method] += 1

            # Route to appropriate handler
            if method == 'faq':
                result = self._handle_faq(query, debug)
            else:  # method == 'gpt'
                result = self._handle_gpt(query, debug)

            # Back-translation if needed
            if (self.capabilities['translation'] and
                translation_result.get('translation_needed') and
                source_lang != 'fr' and
                result.get('success', True)):

                back_translation = self.gpt_handler.translate_from_french(
                    result['answer'], source_lang
                )
                if back_translation['success']:
                    result['original_answer_french'] = result['answer']
                    result['answer'] = back_translation['translated_text']
                    if debug:
                        logger.info(f"Back-translated to {source_lang}")

            # Finalize response
            response_time = time.time() - start_time
            self.stats['response_times'].append(response_time)

            result.update({
                'response_time': response_time,
                'method': method,
                'original_query': original_query,
                'timestamp': datetime.now().isoformat(),
                'source_language': source_lang
            })

            if debug:
                logger.info(f"Query processed: {method} in {response_time:.3f}s")
                if 'confidence' in result:
                    logger.info(f"Confidence: {result['confidence']:.2f}")

            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            self.stats['failed_queries'] += 1
            return self._error_response(str(e), start_time)

    def _select_method(self, query: str, debug: bool = False) -> str:
        """Select processing method: FAQ or GPT."""
        if not self.capabilities['faq']:
            return 'gpt' if self.capabilities['gpt'] else 'fallback'

        # Try FAQ first
        faq_result = self.faq_system.find_best_answer(query, min_confidence=0.3)
        faq_confidence = faq_result['confidence'] if faq_result else 0.0

        if debug:
            logger.info(f"FAQ confidence: {faq_confidence:.2f}")

        # Decision logic
        if faq_confidence >= Config.GPT_ROUTING_THRESHOLD:
            return 'faq'
        elif self.capabilities['gpt']:
            return 'gpt'
        elif faq_result:  # Use FAQ even with low confidence if GPT not available
            return 'faq'
        else:
            return 'fallback'

    def _handle_faq(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """Handle FAQ queries."""
        result = self.faq_system.find_best_answer(query)

        if result:
            if debug:
                logger.info(f"FAQ match: ID {result['id']}, confidence {result['confidence']:.2f}")

            # Enhance answer with contact info if relevant
            answer = result['answer']
            if 'contact' in result['category'].lower() or 'téléphone' in query.lower():
                answer = self._enhance_with_contact_info(answer, query)

            return {
                'answer': answer,
                'confidence': result['confidence'],
                'category': result['category'],
                'source': 'FAQ',
                'success': True
            }
        else:
            return self._handle_fallback(query, debug)

    def _handle_gpt(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """Handle GPT queries."""
        if not self.capabilities['gpt']:
            return self._handle_fallback(query, debug)

        if debug:
            logger.info("Using GPT for response generation")

        self.stats['gpt_used'] += 1

        gpt_result = self.gpt_handler.generate_response(query)

        if gpt_result['success']:
            return {
                'answer': gpt_result['text'],
                'confidence': gpt_result['confidence'],
                'source': 'GPT',
                'tokens_used': gpt_result.get('tokens_used', 0),
                'model': gpt_result.get('model', 'unknown'),
                'success': True
            }
        else:
            if debug:
                logger.warning(f"GPT failed: {gpt_result.get('error', 'unknown')}")
            return self._handle_fallback(query, debug)

    def _handle_fallback(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """Handle fallback when other methods fail."""
        if debug:
            logger.info("Using fallback response")

        # Try to find related FAQ questions
        related_questions = []
        if self.capabilities['faq']:
            for item in self.faq_system.faq_data:
                if fuzz.partial_ratio(query.lower(), item['question'].lower()) > 65:
                    related_questions.append(item['question'])
                    if len(related_questions) >= 3:
                        break

        response_parts = []

        if related_questions:
            response_parts.append("Je n'ai pas trouvé de réponse exacte, mais voici des questions similaires:")
            response_parts.extend(f"- {q}" for q in related_questions[:3])
        else:
            response_parts.append("Je n'ai pas trouvé de réponse exacte à votre question.")

        # Add contact information
        contact_info = self._get_contact_info_based_on_query(query)
        response_parts.extend(contact_info)

        # Add category suggestions
        if self.capabilities['faq']:
            suggestions = self.faq_system.get_category_suggestions(query)
            if suggestions:
                response_parts.append(f"\nVotre question pourrait concerner ces catégories: {', '.join(suggestions)}")

        return {
            'answer': '\n'.join(response_parts),
            'confidence': 0.0,
            'source': 'Fallback',
            'suggestions': related_questions,
            'success': True
        }

    def _enhance_with_contact_info(self, answer: str, query: str) -> str:
        """Enhance answer with relevant contact information."""
        query_lower = query.lower()

        if 'facture' in query_lower or 'paiement' in query_lower:
            contact_info = Config.CONTACT_INFO['billing']
            if 'phone' not in answer.lower():
                answer += f"\n\nPour toute question de facturation: {contact_info['phone']} ({contact_info['hours']})"
        elif 'urgence' in query_lower or 'coupure' in query_lower:
            contact_info = Config.CONTACT_INFO['technical']
            if 'urgence' not in answer.lower():
                answer += f"\n\nUrgences 24h/24: {contact_info['emergency']}"
        else:
            contact_info = Config.CONTACT_INFO['general']
            if 'téléphone' not in answer.lower():
                answer += f"\n\nContact général: {contact_info['phone']}"

        return answer

    def _get_contact_info_based_on_query(self, query: str) -> List[str]:
        """Return context-appropriate contact information."""
        query_lower = query.lower()
        contact_info = []

        if 'facture' in query_lower or 'paiement' in query_lower:
            contact_info.extend([
                "\nPour des questions de facturation:",
                f"• Service client facturation: {Config.CONTACT_INFO['billing']['phone']}",
                f"• Email: {Config.CONTACT_INFO['billing']['email']}",
                f"• Heures d'ouverture: {Config.CONTACT_INFO['billing']['hours']}"
            ])
        elif 'urgence' in query_lower or 'coupure' in query_lower:
            contact_info.extend([
                "\nPour les urgences:",
                f"• Urgences 24h/24: {Config.CONTACT_INFO['technical']['emergency']}",
                f"• Service technique: {Config.CONTACT_INFO['technical']['phone']}"
            ])
        else:
            contact_info.extend([
                "\nPour une assistance personnalisée:",
                f"• Téléphone: {Config.CONTACT_INFO['general']['phone']}",
                f"• Site web: {Config.CONTACT_INFO['general']['website']}",
                f"• Adresse: {Config.CONTACT_INFO['general']['address']}"
            ])

        return contact_info

    def _error_response(self, error: str, start_time: float) -> Dict[str, Any]:
        """Generate error response."""
        response_time = time.time() - start_time
        self.stats['response_times'].append(response_time)

        return {
            'answer': "Je rencontre des difficultés techniques. Veuillez réessayer ou contacter notre service client.",
            'confidence': 0.0,
            'source': 'Error',
            'error': error,
            'response_time': response_time,
            'success': False,
            'contact_info': Config.CONTACT_INFO['general']
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            'performance': {
                'total_queries': self.stats['total_queries'],
                'failed_queries': self.stats['failed_queries'],
                'success_rate': ((self.stats['total_queries'] - self.stats['failed_queries']) /
                                max(1, self.stats['total_queries'])) * 100,
                'avg_response_time': np.mean(self.stats['response_times']) if self.stats['response_times'] else 0,
                'p95_response_time': np.percentile(self.stats['response_times'], 95) if self.stats['response_times'] else 0,
                'methods_used': dict(self.stats['methods_used']),
                'languages_detected': dict(self.stats['languages_detected']),
                'gpt_used': self.stats['gpt_used']
            },
            'capabilities': self.capabilities,
            'faq_stats': {
                'total_questions': len(self.faq_system.faq_data),
                'categories': len(self.faq_system.category_index),
                'searches': self.faq_system.stats['total_searches'],
                'matches': self.faq_system.stats['successful_matches'],
                'match_rate': (self.faq_system.stats['successful_matches'] /
                              max(1, self.faq_system.stats['total_searches'])) * 100
            } if self.capabilities['faq'] else {},
            'gpt_stats': self.gpt_handler.get_stats() if self.capabilities['gpt'] else {
                'available': False,
                'reason': 'No token or OpenAI library not available'
            }
        }

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'details': {}
        }

        # Check core components
        health['components']['faq'] = {
            'status': 'healthy' if self.capabilities['faq'] else 'failed',
            'questions': len(self.faq_system.faq_data),
            'categories': len(self.faq_system.category_index)
        }

        health['components']['gpt'] = {
            'status': 'healthy' if self.capabilities['gpt'] else 'disabled',
            'client_available': self.gpt_handler.client is not None if self.gpt_handler else False,
            'model': Config.GITHUB_MODEL if self.capabilities['gpt'] else None,
            'endpoint': Config.GITHUB_ENDPOINT
        }

        health['components']['translation'] = {
            'status': 'healthy' if self.capabilities['translation'] else 'disabled',
            'gpt_available': self.capabilities['gpt']
        }

        # Performance metrics
        health['performance'] = {
            'avg_response_time': np.mean(self.stats['response_times']) if self.stats['response_times'] else 0,
            'recent_errors': self.stats['failed_queries'],
            'gpt_usage': self.stats['gpt_used']
        }

        # Determine overall status
        if not self.capabilities['faq']:
            health['status'] = 'unhealthy'
        elif self.stats['failed_queries'] > 0 and (self.stats['failed_queries'] / max(1, self.stats['total_queries'])) > 0.2:
            health['status'] = 'degraded'
        elif self.stats['response_times'] and np.mean(self.stats['response_times']) > 3.0:
            health['status'] = 'degraded'

        return health

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def validate_system_files() -> bool:
    """Validate system files."""
    logger.info("Validating system files...")

    is_valid, error = Utils.validate_json_file(Config.FAQ_FILE_PATH)
    if is_valid:
        logger.info("✓ FAQ file valid")
        data = Utils.safe_load_json(Config.FAQ_FILE_PATH, [])
        if len(data) < 10:
            logger.warning(f"⚠ FAQ file has only {len(data)} items (recommended minimum: 50)")
            return False
    else:
        logger.error(f"✗ FAQ file error: {error}")
        return False

    return True

def test_gpt_integration():
    """Test GPT integration specifically for RADEETA."""
    print("🧪 Testing GPT integration for RADEETA...")

    try:
        handler = GPTHandler()

        if not handler.client:
            print("❌ GPT handler not available")
            print("   Check your Colab secret 'RADEETAchatbot' or GITHUB_TOKEN environment variable")
            return False

        # Test RADEETA-specific queries
        test_queries = [
            "Comment consulter ma facture d'eau?",
            "Que faire en cas de coupure d'eau?",
            "Quel est le numéro d'urgence RADEETA?",
            "Comment signaler une fuite d'eau?"
        ]

        print(f"Testing {len(test_queries)} RADEETA queries...")

        total_tokens = 0
        successful_tests = 0

        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Test {i}: {query}")

            result = handler.generate_response(query)

            if result['success']:
                successful_tests += 1
                tokens = result.get('tokens_used', 0)
                total_tokens += tokens

                print(f"✅ Success | Tokens: {tokens}")
                print(f"   Answer: {result['text'][:100]}...")
            else:
                print(f"❌ Failed: {result.get('error', 'unknown')}")

        # Test translation
        print(f"\n📋 Testing translation:")
        translation_test = handler.translate_to_french("How to pay water bill?", "en")
        if translation_test['success']:
            print(f"✅ Translation Success | Tokens: {translation_test.get('tokens_used', 0)}")
            print(f"   Result: {translation_test['translated_text']}")
        else:
            print(f"❌ Translation Failed")

        # Show statistics
        stats = handler.get_stats()
        print(f"\n📊 GPT Test Results:")
        print(f"   Success rate: {successful_tests}/{len(test_queries)} ({(successful_tests/len(test_queries)*100):.1f}%)")
        print(f"   Total tokens used: {total_tokens}")
        print(f"   Average tokens per request: {total_tokens/max(1,successful_tests):.1f}")
        print(f"   Service: {stats['service']}")
        print(f"   Model: {stats['model']}")

        if successful_tests >= len(test_queries) * 0.75:
            print("🎉 GPT integration working well!")
            return True
        else:
            print("⚠️ GPT integration has issues")
            return False

    except Exception as e:
        print(f"❌ GPT integration test failed: {e}")
        return False

def run_system_tests() -> Tuple[RADEETAChatbotSystem, Dict[str, Any]]:
    """Run comprehensive system tests."""
    logger.info("Running simplified RADEETA system tests (FAQ + GPT)")

    # Initialize system
    chatbot = RADEETAChatbotSystem()

    if not chatbot.capabilities['faq']:
        return chatbot, {'error': 'System initialization failed'}

    # Test scenarios
    test_cases = [
        # FAQ tests (high confidence, should use FAQ)
        ("Comment consulter ma facture d'eau?", "faq"),
        ("Signaler une coupure d'eau", "faq"),
        ("Contact service client", "faq"),
        ("Quel est le numéro de téléphone?", "faq"),
        ("Adresse du siège", "faq"),

        # GPT tests (complex queries that should trigger GPT)
        ("Expliquez le processus de traitement de l'eau potable", "gpt"),
        ("Technologies innovantes pour économiser l'eau", "gpt"),
        ("Impact environnemental des systèmes d'assainissement", "gpt"),
        ("Optimisation de la consommation d'eau industrielle", "gpt"),

        # Multilingual tests
        ("How to pay water bill?", "translation"),
        ("Water quality control process", "translation"),
        ("Emergency phone number", "translation"),

        # Fallback tests
        ("Question hors sujet", "fallback"),
        ("aaaa bbbb cccc", "fallback"),
        ("Quand est la prochaine éclipse?", "fallback")
    ]

    results = {
        'total_tests': len(test_cases),
        'passed': 0,
        'failed': 0,
        'gpt_used': 0,
        'results': []
    }

    for query, expected_type in test_cases:
        try:
            result = chatbot.process_query(query, debug=True)

            success = result.get('success', True) and result.get('confidence', 0) > 0.3
            if success:
                results['passed'] += 1
            else:
                results['failed'] += 1

            # Check if GPT was used
            if result.get('source') == 'GPT':
                results['gpt_used'] += 1

            results['results'].append({
                'query': query,
                'expected': expected_type,
                'method': result.get('method', 'unknown'),
                'source': result.get('source', 'unknown'),
                'confidence': result.get('confidence', 0),
                'success': success,
                'response_time': result.get('response_time', 0),
                'answer_length': len(result.get('answer', '')),
                'gpt_used': result.get('source') == 'GPT',
                'tokens_used': result.get('tokens_used', 0)
            })

            logger.info(f"✓ {query} -> {result.get('source', 'unknown')} ({result.get('confidence', 0):.2f})")

        except Exception as e:
            results['failed'] += 1
            logger.error(f"✗ {query} -> Error: {e}")
            results['results'].append({
                'query': query,
                'error': str(e),
                'success': False
            })

    # Final statistics
    results['success_rate'] = (results['passed'] / results['total_tests']) * 100
    results['gpt_usage_rate'] = (results['gpt_used'] / results['total_tests']) * 100
    results['system_stats'] = chatbot.get_statistics()

    # Calculate average metrics
    if results['results']:
        results['avg_confidence'] = np.mean([r['confidence'] for r in results['results'] if 'confidence' in r])
        results['avg_response_time'] = np.mean([r['response_time'] for r in results['results'] if 'response_time' in r])
        results['total_tokens_used'] = sum([r.get('tokens_used', 0) for r in results['results']])

    logger.info(f"Tests completed: {results['passed']}/{results['total_tests']} passed ({results['success_rate']:.1f}%)")
    logger.info(f"GPT used: {results['gpt_used']} times ({results['gpt_usage_rate']:.1f}%)")

    return chatbot, results

def interactive_demo():
    """Interactive demonstration."""
    print("="*70)
    print("RADEETA SIMPLIFIED SYSTEM v2.3.0 - INTERACTIVE DEMO")
    print("FAQ + GPT Architecture")
    print("="*70)

    # Validate files
    if not validate_system_files():
        print("❌ File validation failed. Please check your JSON files.")
        return

    print("Initializing simplified system (FAQ + GPT)...")
    try:
        chatbot = RADEETAChatbotSystem()

        if not chatbot.capabilities['faq']:
            print("❌ System initialization failed!")
            return

        print("✅ System ready!")
        print(f"Capabilities: {chatbot.capabilities}")

        if chatbot.capabilities['gpt']:
            print("🚀 GPT integration: ACTIVE")
            print(f"   Routing threshold: {Config.GPT_ROUTING_THRESHOLD}")
            print("   Low FAQ confidence queries will use GPT!")
        else:
            print("⚠️ GPT not available")

        print("\nCommands:")
        print("  - Ask questions naturally")
        print("  - 'stats' for detailed statistics")
        print("  - 'health' for system health check")
        print("  - 'test' to run quick tests")
        print("  - 'gpt' to test GPT specifically")
        print("  - 'quit' to exit")

        # Demo queries
        demo_queries = [
            "Comment consulter ma facture?",  # Should use FAQ
            "Numéro urgence",  # Should use FAQ
            "Technologies innovantes pour économiser l'eau",  # Should use GPT
            "Processus de traitement de l'eau potable détaillé",  # Should use GPT
        ]

        print("\nQuick demo (testing routing logic):")
        for query in demo_queries:
            result = chatbot.process_query(query, debug=True)
            method = result.get('source', 'unknown')
            confidence = result.get('confidence', 0)
            tokens = result.get('tokens_used', 0)

            print(f"🔍 {query}")
            print(f"   -> {method} | Confidence: {confidence:.2f} | Time: {result.get('response_time', 0):.3f}s")
            if tokens > 0:
                print(f"   -> GPT tokens used: {tokens}")
            print(f"   💬 {result.get('answer', '')[:150]}...\n")

        print("\n" + "="*50)
        print("Interactive session started!")

        while True:
            try:
                user_input = input("\n🤖 RADEETA> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break

                elif user_input.lower() == 'stats':
                    stats = chatbot.get_statistics()
                    print("\n📊 System Statistics:")
                    print(f"  Total queries: {stats['performance']['total_queries']}")
                    print(f"  Success rate: {stats['performance']['success_rate']:.1f}%")
                    print(f"  GPT used: {stats['performance']['gpt_used']} times")
                    print(f"  Avg response time: {stats['performance']['avg_response_time']:.3f}s")
                    print(f"  Methods used: {stats['performance']['methods_used']}")

                    if 'gpt_stats' in stats:
                        gpt_stats = stats['gpt_stats']
                        print(f"\n🤖 GPT Statistics:")
                        print(f"  Total requests: {gpt_stats.get('total_requests', 0)}")
                        print(f"  Success rate: {gpt_stats.get('success_rate', 0):.1f}%")
                        print(f"  Total tokens used: {gpt_stats.get('total_tokens_used', 0)}")

                elif user_input.lower() == 'health':
                    health = chatbot.health_check()
                    print(f"\n🏥 System health: {health['status'].upper()}")
                    for component, details in health['components'].items():
                        emoji = "✅" if details['status'] == "healthy" else "⚠️" if details['status'] == "disabled" else "❌"
                        print(f"  {component.upper()}: {emoji} {details['status']}")

                elif user_input.lower() == 'test':
                    print("\nRunning tests...")
                    _, results = run_system_tests()
                    print(f"\nTest results: {results['passed']}/{results['total_tests']} passed")
                    print(f"GPT usage: {results['gpt_used']} times")
                    print(f"Total tokens used: {results.get('total_tokens_used', 0)}")

                elif user_input.lower() == 'gpt':
                    print("\nTesting GPT specifically...")
                    test_gpt_integration()

                else:
                    result = chatbot.process_query(user_input, debug=True)

                    method = result.get('source', 'unknown')
                    confidence = result.get('confidence', 0)
                    time_taken = result.get('response_time', 0)
                    tokens = result.get('tokens_used', 0)

                    print(f"\n📋 Method: {method} | Confidence: {confidence:.2f} | Time: {time_taken:.3f}s")
                    if tokens > 0:
                        print(f"🤖 GPT tokens used: {tokens}")
                        print(f"💰 Cost estimate: ~${tokens * 0.00003:.6f}")

                    print(f"\n💬 Response:")
                    print(result.get('answer', 'No response'))
                    print("\n" + "-"*50)

            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ System initialization failed: {e}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for simplified RADEETA chatbot."""
    print("="*70)
    print(f"RADEETA SIMPLIFIED CHATBOT SYSTEM v2.3.0 (FAQ + GPT)")
    print("="*70)
    print(f"{Config.ORGANIZATION_FULL}")
    print(f"Domain: {Config.DOMAIN}")
    print("="*70)

    # Display GPT capability
    if OPENAI_AVAILABLE and Config.GITHUB_TOKEN:
        print("🚀 GPT Integration: ENABLED")
        print(f"   Model: {Config.GITHUB_MODEL}")
        print(f"   Routing Threshold: {Config.GPT_ROUTING_THRESHOLD}")
        print(f"   Endpoint: {Config.GITHUB_ENDPOINT}")
        print("   FAQ queries < 70% confidence will use GPT")
    elif OPENAI_AVAILABLE:
        print("⚠️ GPT Integration: Available but no token found")
        print("   Add token to Colab secret 'RADEETAchatbot' or set GITHUB_TOKEN")
    else:
        print("❌ GPT Integration: OpenAI library not available")
        print("   Install with: pip install openai")

    # Display system capabilities
    print("\nSIMPLIFIED SYSTEM ARCHITECTURE:")
    print("✅ FAQ: Advanced fuzzy matching with enhanced recall")

    if Config.GITHUB_TOKEN:
        print("✅ GPT: Response generation + translation for complex queries")
        print("✅ Translation: GPT-based multi-language support")
    else:
        print("❌ GPT: Requires GitHub token")
        print("❌ Translation: Requires GPT integration")

    if LANGDETECT_AVAILABLE:
        print("✅ Language Detection: LangDetect + pattern matching")
    else:
        print("⚠️ Language Detection: Pattern-based only")

    # System status
    if Config.GITHUB_TOKEN and LANGDETECT_AVAILABLE:
        status = "🟢 Fully Operational"
    elif Config.GITHUB_TOKEN:
        status = "🟡 Operational (Limited Language Detection)"
    else:
        status = "🟡 FAQ Only Mode"

    print(f"\nSTATUS: {status}")

    print("\nSIMPLIFIED FUNCTIONS:")
    print("1. validate_system_files() - FAQ file integrity checks")
    print("2. chatbot, results = run_system_tests() - Comprehensive testing")
    print("3. test_gpt_integration() - Test GPT specifically")
    print("4. interactive_demo() - Interactive interface")
    print("5. Manual usage:")
    print("   chatbot = RADEETAChatbotSystem()")
    print("   result = chatbot.process_query('question', debug=True)")
    print("   # Will automatically route FAQ vs GPT based on confidence")

    print(f"\nREQUIRED FILES:")
    print(f"  - {Config.FAQ_FILE_PATH} (min 50 recommended questions)")

    print("\nREQUIRED SETUP:")
    print("  - GitHub token in Colab secret 'RADEETAchatbot' or GITHUB_TOKEN env var")
    print("  - Token should have 'models:read' permission")

    print("\nSUPPORTED LANGUAGES:")
    print("  - French (primary)")
    if Config.GITHUB_TOKEN:
        print("  - English/Arabic (auto-translated via GPT)")

    print("\nSIMPLIFIED ROUTING LOGIC:")
    print("  1. User asks a question")
    print("  2. System tries FAQ matching")
    print("  3. If FAQ confidence ≥ 70%:")
    print("     → Returns FAQ answer")
    print("  4. If FAQ confidence < 70%:")
    print("     → Routes to GPT for response generation")
    print("  5. Translation handled by GPT for non-French queries")

    print("\nCONTACT INFORMATION:")
    for category, info in Config.CONTACT_INFO.items():
        print(f"  {category.capitalize()}:")
        for key, value in info.items():
            print(f"    {key}: {value}")

    print("="*70)
    print("SIMPLIFIED SYSTEM READY (FAQ + GPT)")
    print("="*70)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_monthly_cost(total_tokens: int, queries_per_day: int = 100) -> Dict[str, float]:
    """Estimate monthly cost for GPT usage."""
    cost_per_1k_tokens = 0.03  # Approximate for GPT-4

    cost_current = (total_tokens / 1000) * cost_per_1k_tokens

    # Estimate monthly cost
    if total_tokens > 0:
        avg_tokens_per_query = total_tokens / max(1, queries_per_day)
        monthly_tokens = avg_tokens_per_query * queries_per_day * 30
        monthly_cost = (monthly_tokens / 1000) * cost_per_1k_tokens
    else:
        monthly_cost = 0

    return {
        'current_cost': cost_current,
        'estimated_monthly_cost': monthly_cost,
        'cost_per_1k_tokens': cost_per_1k_tokens,
        'total_tokens': total_tokens
    }

def optimize_gpt_usage():
    """Provide tips for optimizing GPT usage and costs."""
    tips = """
🎯 GPT OPTIMIZATION TIPS:

💰 Cost Optimization:
  • Adjust GPT_ROUTING_THRESHOLD (currently 0.70)
  • Higher threshold = more GPT usage = higher costs but better answers
  • Lower threshold = more FAQ usage = lower costs but may miss complex queries

⚡ Performance Optimization:
  • GPT responses are typically high quality
  • But require internet connection and use tokens
  • Monitor token usage to stay within limits

📊 Usage Monitoring:
  • Check stats regularly: chatbot.get_statistics()
  • Monitor 'gpt_used' and 'total_tokens_used'
  • Use estimate_monthly_cost() for cost projections

🎛️ Configuration Options:
  • GITHUB_MAX_TOKENS: Controls response length (default: 400)
  • GITHUB_TEMPERATURE: Controls response creativity (default: 0.3)
  • GPT_ROUTING_THRESHOLD: Controls FAQ vs GPT routing (default: 0.70)

🔧 Troubleshooting:
  • Rate limits: Wait and retry
  • Token limits: Reduce max_tokens
  • Authentication errors: Check token permissions
"""
    return tips

def quick_setup_guide():
    """Provide quick setup guide for new users."""
    guide = """
🚀 QUICK SETUP GUIDE FOR SIMPLIFIED RADEETA CHATBOT:

1️⃣ GET GITHUB TOKEN:
   • Go to: https://github.com/settings/tokens
   • Create new token (classic)
   • Select 'models:read' permission
   • Copy the token (starts with 'ghp_')

2️⃣ ADD TO COLAB:
   • Click 🔑 key icon in Colab sidebar
   • Add secret named: RADEETAchatbot
   • Paste your GitHub token as value
   • Toggle ON to enable

3️⃣ INSTALL DEPENDENCIES:
   !pip install openai rapidfuzz langdetect

4️⃣ TEST SETUP:
   • Run: test_gpt_integration()
   • Should show "✅ GPT integration working well!"

5️⃣ START USING:
   chatbot = RADEETAChatbotSystem()
   result = chatbot.process_query("Complex water question", debug=True)

6️⃣ MONITOR USAGE:
   stats = chatbot.get_statistics()
   print(f"GPT used: {stats['performance']['gpt_used']} times")

⚠️ TROUBLESHOOTING:
   • "No token found" → Check Colab secret name exactly: 'RADEETAchatbot'
   • "Unauthorized" → Token needs 'models:read' permission
   • "Rate limit" → Wait a few minutes, then retry

🎯 ROUTING LOGIC:
   • High FAQ confidence (≥70%) → FAQ response
   • Low FAQ confidence (<70%) → GPT response
   • Adjust threshold in Config.GPT_ROUTING_THRESHOLD
"""
    return guide

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run main
    main()

    # System validation
    print("\n🔍 Running system validation...")

    try:
        if validate_system_files():
            print("✅ Files validated successfully")

            # Test GPT integration
            print("\n🤖 Testing GPT integration...")
            gpt_working = test_gpt_integration()

            # Test initialization
            test_system = RADEETAChatbotSystem()

            if test_system.capabilities['faq']:
                print("✅ System initialization successful")
                print(f"   Capabilities: {test_system.capabilities}")

                # Test complex query (should trigger GPT)
                complex_query = "Expliquez les technologies innovantes pour optimiser la distribution d'eau"
                test_result = test_system.process_query(complex_query, debug=True)

                if test_result.get('success', True):
                    print("✅ Complex query test passed")
                    print(f"   Method used: {test_result.get('source', 'unknown')}")
                    print(f"   Confidence: {test_result.get('confidence', 0):.2f}")

                    if test_result.get('source') == 'GPT':
                        print("🚀 GPT routing working!")
                        print(f"   Tokens used: {test_result.get('tokens_used', 0)}")
                        print(f"   Model: {test_result.get('model', 'unknown')}")

                # Simple test (should use FAQ)
                simple_result = test_system.process_query("Comment consulter ma facture?", debug=True)
                if simple_result.get('success', True) and simple_result.get('confidence', 0) > 0.5:
                    print("✅ Simple FAQ test passed")
                    print(f"   Method: {simple_result.get('source', 'unknown')} (should be FAQ)")

                print("🎉 SIMPLIFIED SYSTEM READY!")
                print("   Try: interactive_demo()")

            else:
                print("❌ System initialization failed")
        else:
            print("❌ File validation failed")

    except Exception as e:
        print(f"❌ Validation error: {e}")
        print("Please check your JSON files and dependencies.")

    print("\n" + "="*70)
    print("🎯 SIMPLIFIED QUICK START:")
    print("="*70)
    print("1. Make sure your GitHub token is in Colab secret 'RADEETAchatbot'")
    print("2. Run: chatbot = RADEETAChatbotSystem()")
    print("3. Ask simple questions (routes to FAQ): 'Quel est le numéro de téléphone?'")
    print("4. Ask complex questions (routes to GPT): 'Technologies de traitement d'eau'")
    print("5. Check stats: chatbot.get_statistics()")
    print("6. Interactive demo: interactive_demo()")
    print("\n💡 System automatically routes FAQ vs GPT based on confidence!")
    print("💰 Monitor token usage for cost tracking")
    print("🎛️ Adjust GPT_ROUTING_THRESHOLD to control routing behavior")
    print("="*70)

# Print setup guide when module is imported
if __name__ == "__main__":
    print("\n" + quick_setup_guide())
    print("\n" + optimize_gpt_usage())


# Fixing Data Structure:

# In[4]:


import json

# Load existing file
with open('radeeta_faq_perfect_105.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract just the FAQ data array
faq_data = data['data']

# Save in the correct format for the chatbot
with open('radeeta_faq_perfect_105.json', 'w', encoding='utf-8') as f:
    json.dump(faq_data, f, ensure_ascii=False, indent=2)

print(f"✅ Fixed FAQ file structure! {len(faq_data)} questions ready.")


# In[ ]:


interactive_demo()


# # DEPLOYEMENT:

# In[ ]:


# ============================================================================
# RADEETA CHATBOT - COMPLETE DEPLOYMENT CELL
# Run this AFTER your chatbot creation cell
# ============================================================================

print("🚀 Starting RADEETA Chatbot Deployment...")

# Install deployment dependencies
get_ipython().system('pip install flask==2.3.3 flask-cors==4.0.0 pyngrok==7.0.0')

# Import deployment libraries
from pyngrok import ngrok
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import traceback

# ============================================================================
# STEP 1: CONNECT TO CHATBOT
# ============================================================================

print("🔍 Connecting to your chatbot...")

deployment_chatbot = None

# Look for your chatbot instance
if 'chatbot' in globals():
    deployment_chatbot = globals()['chatbot']
    print(f"✅ Found 'chatbot': {type(deployment_chatbot)}")
elif 'RADEETAChatbotSystem' in globals():
    try:
        deployment_chatbot = RADEETAChatbotSystem()
        print("✅ Created new RADEETAChatbotSystem instance")
    except:
        deployment_chatbot = RADEETAChatbotSystem
        print("✅ Found RADEETAChatbotSystem class")
elif 'radeeta_chatbot' in globals():
    deployment_chatbot = globals()['radeeta_chatbot']
    print(f"✅ Found 'radeeta_chatbot': {type(deployment_chatbot)}")

if deployment_chatbot:
    print("✅ Chatbot connected successfully!")
    # Test the chatbot
    try:
        if hasattr(deployment_chatbot, 'process_query'):
            test_response = deployment_chatbot.process_query("Test")
            print(f"✅ Chatbot test successful: {str(test_response)[:50]}...")
        else:
            print("⚠️ Chatbot found but no 'process_query' method detected")
    except Exception as e:
        print(f"⚠️ Chatbot test failed: {e}")
else:
    print("❌ Chatbot not found! Make sure to run your chatbot cell first.")

# ============================================================================
# STEP 2: SETUP NGROK
# ============================================================================

print("🔧 Setting up ngrok...")

try:
    from google.colab import userdata
    ngrok_token = userdata.get('NGROK_TOKEN')
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        print("✅ Ngrok token configured!")
    else:
        print("⚠️ Add your ngrok token to Colab secrets as 'NGROK_TOKEN'")
        print("🔗 Get token: https://dashboard.ngrok.com/get-started/your-authtoken")
except Exception as e:
    print(f"❌ Ngrok setup error: {e}")

# ============================================================================
# STEP 3: FLASK APPLICATION
# ============================================================================

print("🌐 Creating Flask application...")

app = Flask(__name__)
CORS(app)

# Global variables
conversation_history = {}
active_sessions = set()
start_time = time.time()

# Complete HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RADEETA - Assistant Virtuel</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>💧</text></svg>">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .chatbot-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 25px 80px rgba(0, 0, 0, 0.15);
            overflow: hidden;
            width: 100%;
            max-width: 1000px;
            height: 750px;
            display: flex;
            flex-direction: column;
            backdrop-filter: blur(20px);
        }
        .header {
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            padding: 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1), transparent 50%);
        }
        .header h1 {
            font-size: 30px;
            margin-bottom: 10px;
            font-weight: 700;
            position: relative;
            z-index: 1;
        }
        .header p {
            opacity: 0.9;
            font-size: 16px;
            position: relative;
            z-index: 1;
        }
        .status-indicator {
            position: absolute;
            top: 30px;
            right: 30px;
            width: 16px;
            height: 16px;
            background: #4CAF50;
            border-radius: 50%;
            animation: pulse 2s infinite;
            box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.3);
            z-index: 2;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            background: #f8fafe;
            scroll-behavior: smooth;
        }
        .messages-container::-webkit-scrollbar {
            width: 8px;
        }
        .messages-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        .messages-container::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        .message {
            margin-bottom: 25px;
            display: flex;
            align-items: flex-start;
            animation: slideIn 0.5s ease-out;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { justify-content: flex-end; }
        .message.bot { justify-content: flex-start; }
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            margin: 0 12px;
            flex-shrink: 0;
        }
        .message.user .message-avatar {
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            order: 2;
        }
        .message.bot .message-avatar {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            color: #1976D2;
        }
        .message-content {
            max-width: 70%;
            padding: 18px 24px;
            border-radius: 20px;
            font-size: 15px;
            line-height: 1.6;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            word-wrap: break-word;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            border-bottom-right-radius: 6px;
        }
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e3f2fd;
            border-bottom-left-radius: 6px;
        }
        .message-info {
            font-size: 11px;
            opacity: 0.8;
            margin-top: 12px;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }
        .method-badge {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            color: #1976D2;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .confidence-bar {
            width: 50px;
            height: 4px;
            background: #e0e0e0;
            border-radius: 2px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff5722, #ff9800, #4caf50);
            transition: width 0.8s ease;
            border-radius: 2px;
        }
        .input-area {
            padding: 30px;
            background: white;
            border-top: 1px solid #e3f2fd;
        }
        .input-container {
            display: flex;
            gap: 16px;
            align-items: flex-end;
            margin-bottom: 20px;
        }
        .input-wrapper { flex: 1; position: relative; }
        #messageInput {
            width: 100%;
            border: 2px solid #e3f2fd;
            border-radius: 25px;
            padding: 16px 24px;
            font-size: 15px;
            outline: none;
            transition: all 0.3s ease;
            resize: none;
            min-height: 56px;
            max-height: 120px;
            font-family: inherit;
        }
        #messageInput:focus {
            border-color: #2196F3;
            box-shadow: 0 0 0 4px rgba(33, 150, 243, 0.1);
            transform: translateY(-1px);
        }
        #sendButton {
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            border: none;
            border-radius: 50%;
            width: 56px;
            height: 56px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        }
        #sendButton:hover:not(:disabled) {
            transform: scale(1.05) translateY(-2px);
            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.4);
        }
        #sendButton:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .typing-indicator {
            display: none;
            padding: 15px 20px;
            background: white;
            border-radius: 20px;
            border-bottom-left-radius: 6px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            max-width: 70%;
        }
        .typing-dots {
            display: flex;
            gap: 6px;
            align-items: center;
        }
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #666;
            animation: typingBounce 1.4s infinite ease-in-out both;
        }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        .typing-dot:nth-child(3) { animation-delay: 0s; }
        @keyframes typingBounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 25px;
            padding: 20px;
            background: rgba(255,255,255,0.8);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .suggestion-chip {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            color: #1976D2;
            border: none;
            padding: 10px 16px;
            border-radius: 20px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        .suggestion-chip:hover {
            background: linear-gradient(135deg, #bbdefb, #90caf9);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(33, 150, 243, 0.3);
        }
        .controls {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .control-btn {
            background: #f8fafe;
            border: 1px solid #e3f2fd;
            border-radius: 20px;
            padding: 10px 16px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        .control-btn:hover {
            background: #e3f2fd;
            transform: translateY(-1px);
        }
        @media (max-width: 768px) {
            body { padding: 10px; }
            .chatbot-container { height: 100vh; border-radius: 0; }
            .header { padding: 20px; }
            .header h1 { font-size: 24px; }
            .messages-container { padding: 20px; }
            .message-content { max-width: 85%; }
            .input-area { padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="chatbot-container">
        <div class="header">
            <div class="status-indicator"></div>
            <h1>💧 RADEETA - Assistant Virtuel</h1>
            <p>Régie Autonome de Distribution d'Eau et d'Assainissement de Taza</p>
        </div>

        <div class="chat-area">
            <div class="messages-container" id="messagesContainer">
                <div class="message bot">
                    <div class="message-avatar">🤖</div>
                    <div class="message-content">
                        👋 <strong>Bienvenue chez RADEETA !</strong><br><br>
                        Je suis votre assistant virtuel et je peux vous aider avec :<br><br>
                        💰 <strong>Facturation et paiements</strong><br>
                        🔧 <strong>Urgences et coupures</strong><br>
                        💧 <strong>Qualité de l'eau</strong><br>
                        📍 <strong>Informations pratiques</strong><br>
                        📞 <strong>Contact et horaires</strong>
                        <div class="message-info">
                            <span class="method-badge">SYSTÈME</span>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: 100%"></div>
                            </div>
                            <span>100%</span>
                        </div>
                    </div>
                </div>

                <div class="suggestions" id="suggestions">
                    <button class="suggestion-chip" onclick="sendSuggestion('Où se situe l\\'agence RADEETA ?')">📍 Adresse de l'agence</button>
                    <button class="suggestion-chip" onclick="sendSuggestion('Comment consulter ma facture ?')">💰 Consulter facture</button>
                    <button class="suggestion-chip" onclick="sendSuggestion('Numéro d\\'urgence RADEETA')">🚨 Numéro d'urgence</button>
                    <button class="suggestion-chip" onclick="sendSuggestion('Qualité de l\\'eau potable')">💧 Qualité de l'eau</button>
                    <button class="suggestion-chip" onclick="sendSuggestion('Horaires d\\'ouverture')">🕒 Horaires</button>
                    <button class="suggestion-chip" onclick="sendSuggestion('Comment payer ma facture ?')">💳 Paiement</button>
                </div>

                <div class="typing-indicator" id="typingIndicator">
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea id="messageInput" placeholder="Tapez votre question ici..." rows="1"></textarea>
                    </div>
                    <button id="sendButton">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M2,21L23,12L2,3V10L17,12L2,14V21Z"/>
                        </svg>
                    </button>
                </div>

                <div class="controls">
                    <button class="control-btn" onclick="clearChat()">🗑️ Effacer</button>
                    <button class="control-btn" onclick="showStats()">📊 Statistiques</button>
                    <button class="control-btn" onclick="window.open('/api/health', '_blank')">🔧 État</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        const messagesContainer = document.getElementById('messagesContainer');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');

        let sessionId = 'session_' + Date.now();
        let messageCount = 0;

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            setupEventListeners();
            messageInput.focus();
        });

        function setupEventListeners() {
            // Input handling
            messageInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 120) + 'px';
                sendButton.disabled = !this.value.trim();
            });

            // Enter key handling
            messageInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // Send button
            sendButton.addEventListener('click', sendMessage);
        }

        // Send message function
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, 'user');
            messageInput.value = '';
            messageInput.style.height = 'auto';
            sendButton.disabled = true;

            // Show typing indicator
            showTypingIndicator();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId
                    })
                });

                hideTypingIndicator();

                if (!response.ok) {
                    throw new Error('Network error');
                }

                const data = await response.json();

                if (data.status === 'success') {
                    addMessage(data.data.answer, 'bot', {
                        method: data.data.method,
                        confidence: data.data.confidence,
                        responseTime: data.data.response_time
                    });
                } else {
                    addMessage('❌ ' + data.message, 'bot');
                }
            } catch (error) {
                hideTypingIndicator();
                addMessage('❌ Erreur de connexion. Veuillez réessayer.', 'bot');
            }
        }

        function showTypingIndicator() {
            typingIndicator.style.display = 'block';
            scrollToBottom();
        }

        function hideTypingIndicator() {
            typingIndicator.style.display = 'none';
        }

        function sendSuggestion(suggestion) {
            messageInput.value = suggestion;
            messageInput.focus();
            messageInput.dispatchEvent(new Event('input'));
            sendMessage();
            document.getElementById('suggestions').style.display = 'none';
        }

        // Add message to chat
        function addMessage(content, sender, metadata = {}) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = sender === 'user' ? '👤' : '🤖';

            let messageInfo = '';
            if (sender === 'bot' && metadata.method) {
                const confidencePercent = Math.round((metadata.confidence || 0) * 100);
                const responseTime = metadata.responseTime ? `${Math.round(metadata.responseTime * 1000)}ms` : '';

                messageInfo = `
                    <div class="message-info">
                        <span class="method-badge">${metadata.method}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                        <span>${confidencePercent}%</span>
                        ${responseTime ? `<span>⏱️ ${responseTime}</span>` : ''}
                    </div>
                `;
            }

            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = content.replace(/\\n/g, '<br>') + messageInfo;

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);

            scrollToBottom();
            messageCount++;

            if (sender === 'user') {
                document.getElementById('suggestions').style.display = 'none';
            }
        }

        function scrollToBottom() {
            setTimeout(() => {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }, 100);
        }

        function clearChat() {
            if (confirm('Êtes-vous sûr de vouloir effacer la conversation ?')) {
                const messages = messagesContainer.querySelectorAll('.message:not(:first-child)');
                messages.forEach(msg => msg.remove());
                document.getElementById('suggestions').style.display = 'flex';
                messageCount = 0;
            }
        }

        async function showStats() {
            try {
                const response = await fetch('/api/statistics');
                const data = await response.json();

                if (data.status === 'success') {
                    const stats = data.data;
                    const statsMessage = `
📊 <strong>Statistiques RADEETA</strong><br><br>
📈 <strong>Performance:</strong><br>
• Total requêtes: ${stats.performance?.total_queries || 'N/A'}<br>
• Taux de succès: ${stats.performance?.success_rate || 'N/A'}%<br>
• Uptime: ${stats.performance?.uptime_hours || 'N/A'}h<br><br>
💬 <strong>Session actuelle:</strong><br>
• Messages échangés: ${messageCount}<br>
• Session ID: ${sessionId.substring(0, 12)}...
                    `;
                    addMessage(statsMessage, 'bot', { method: 'statistiques', confidence: 1.0 });
                } else {
                    addMessage('❌ Impossible de récupérer les statistiques', 'bot');
                }
            } catch (error) {
                addMessage('❌ Erreur lors de la récupération des statistiques', 'bot');
            }
        }
    </script>
</body>
</html>
'''

# ============================================================================
# STEP 4: API ROUTES
# ============================================================================

@app.route('/')
def home():
    return HTML_TEMPLATE

@app.route('/api/health')
def health_check():
    try:
        health_data = {
            'server_time': datetime.now().isoformat(),
            'chatbot_connected': deployment_chatbot is not None,
            'chatbot_type': type(deployment_chatbot).__name__ if deployment_chatbot else None,
            'uptime_seconds': time.time() - start_time,
            'active_sessions': len(active_sessions),
            'total_conversations': len(conversation_history)
        }

        if deployment_chatbot and hasattr(deployment_chatbot, 'health_check'):
            try:
                chatbot_health = deployment_chatbot.health_check()
                health_data['chatbot_health'] = chatbot_health
            except:
                health_data['chatbot_health'] = 'Health check not available'

        return jsonify({'status': 'success', 'data': health_data})

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    start_request_time = time.time()

    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not message:
            return jsonify({
                'status': 'error',
                'message': 'Message vide'
            }), 400

        if not deployment_chatbot:
            return jsonify({
                'status': 'error',
                'message': 'Chatbot non disponible. Vérifiez que votre chatbot est bien initialisé.'
            }), 503

        # Track session
        active_sessions.add(session_id)
        if session_id not in conversation_history:
            conversation_history[session_id] = []

        # Process with chatbot
        try:
            if hasattr(deployment_chatbot, 'process_query'):
                response = deployment_chatbot.process_query(message)

                # Handle different response formats
                if isinstance(response, dict):
                    answer = response.get('answer', str(response))
                    method = response.get('method', 'process_query')
                    confidence = response.get('confidence', 0.8)
                    source = response.get('source', 'chatbot')
                    tokens_used = response.get('tokens_used', 0)
                else:
                    answer = str(response)
                    method = 'process_query'
                    confidence = 0.8
                    source = 'chatbot'
                    tokens_used = 0

            elif hasattr(deployment_chatbot, 'get_response'):
                answer = deployment_chatbot.get_response(message)
                method = 'get_response'
                confidence = 0.8
                source = 'chatbot'
                tokens_used = 0

            elif callable(deployment_chatbot):
                answer = deployment_chatbot(message)
                method = 'callable'
                confidence = 0.8
                source = 'chatbot'
                tokens_used = 0

            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Méthode de traitement non trouvée sur le chatbot'
                }), 500

        except Exception as e:
            print(f"Chatbot processing error: {e}")
            traceback.print_exc()
            answer = f"Désolé, j'ai rencontré un problème technique : {str(e)}"
            method = 'error'
            confidence = 0
            source = 'system'
            tokens_used = 0

        # Calculate timing
        response_time = time.time() - start_request_time

        # Store conversation
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'bot_response': answer,
            'method': method,
            'confidence': confidence,
            'response_time': response_time
        }
        conversation_history[session_id].append(conversation_entry)

        # Keep only last 100 messages per session
        if len(conversation_history[session_id]) > 100:
            conversation_history[session_id] = conversation_history[session_id][-100:]

        return jsonify({
            'status': 'success',
            'data': {
                'answer': answer,
                'method': method,
                'confidence': confidence,
                'source': source,
                'response_time': response_time,
                'tokens_used': tokens_used,
                'message_id': len(conversation_history[session_id])
            }
        })

    except Exception as e:
        print(f"API Error: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': 'Erreur interne du serveur'
        }), 500

@app.route('/api/statistics')
def statistics():
    try:
        total_messages = sum(len(conv) for conv in conversation_history.values())
        total_sessions = len(conversation_history)
        uptime_hours = (time.time() - start_time) / 3600

        base_stats = {
            'performance': {
                'total_queries': total_messages,
                'success_rate': 95,  # Default
                'uptime_hours': round(uptime_hours, 2),
                'active_sessions': len(active_sessions),
                'total_sessions': total_sessions
            }
        }

        # Try to get chatbot-specific stats
        if deployment_chatbot and hasattr(deployment_chatbot, 'get_statistics'):
            try:
                chatbot_stats = deployment_chatbot.get_statistics()
                base_stats.update(chatbot_stats)
            except Exception as e:
                print(f"Error getting chatbot stats: {e}")

        return jsonify({'status': 'success', 'data': base_stats})

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint non trouvé'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Erreur interne du serveur'
    }), 500

# ============================================================================
# STEP 5: START SERVER
# ============================================================================

def run_flask():
    try:
        print("🌐 Flask server starting...")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"❌ Flask server error: {e}")

if deployment_chatbot:
    print("🚀 Starting deployment server...")

    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Wait for server to initialize
    print("⏳ Waiting for server to start...")
    time.sleep(3)

    # Create public URL with ngrok
    try:
        public_url = ngrok.connect(5000, "http")

        print("\n" + "🎉" + "="*70)
        print("✅ RADEETA CHATBOT IS NOW LIVE!")
        print("="*72)
        print(f"🌐 PUBLIC URL: {public_url}")
        print(f"🏠 LOCAL URL:  http://localhost:5000")
        print("="*72)
        print("🤖 CHATBOT STATUS:")
        print(f"   • Type: {type(deployment_chatbot).__name__}")
        print(f"   • Methods: {[m for m in dir(deployment_chatbot) if not m.startswith('_') and callable(getattr(deployment_chatbot, m))][:3]}...")
        print("="*72)
        print("✨ FEATURES:")
        print("   • Professional web interface")
        print("   • Real-time chat interaction")
        print("   • Responsive design (mobile-friendly)")
        print("   • Typing indicators and animations")
        print("   • Message history and statistics")
        print("   • Quick suggestion buttons")
        print("="*72)
        print("📱 INSTRUCTIONS:")
        print("   1. Click the PUBLIC URL above")
        print("   2. Start chatting with RADEETA")
        print("   3. Try the suggestion buttons")
        print("   4. Share the link with others!")
        print("="*72)
        print("⚡ The chatbot will stay online as long as this cell runs")
        print("🛑 To stop: Interrupt this cell (Runtime > Interrupt execution)")
        print("="*72)

        # Keep the service alive with status updates
        try:
            print(f"⏰ Server started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            while True:
                time.sleep(60)  # Status update every minute
                uptime = time.time() - start_time
                hours = int(uptime // 3600)
                minutes = int((uptime % 3600) // 60)
                total_sessions = len(conversation_history)
                total_messages = sum(len(conv) for conv in conversation_history.values())

                status_msg = f"⏰ {datetime.now().strftime('%H:%M:%S')} | Uptime: {hours}h {minutes}m | Sessions: {total_sessions} | Messages: {total_messages}"
                print(f"🟢 ONLINE: {public_url}")
                print(f"📊 {status_msg}")
                print("-" * 50)

        except KeyboardInterrupt:
            print("\n🛑 Stopping RADEETA chatbot...")
            print("✅ Server stopped gracefully")
            try:
                ngrok.disconnect(public_url)
                print("✅ Ngrok tunnel closed")
            except:
                pass
            print("👋 Thank you for using RADEETA chatbot!")

    except Exception as e:
        print(f"\n❌ Ngrok connection failed: {e}")
        print("💡 Possible solutions:")
        print("   1. Check your ngrok token in Colab secrets")
        print("   2. Verify ngrok account limits")
        print("   3. Try restarting the cell")
        print(f"\n🏠 Server running locally: http://localhost:5000")
        print("📝 You can still test locally, but others won't be able to access it")

        # Keep local server alive
        try:
            while True:
                time.sleep(60)
                uptime = time.time() - start_time
                hours = int(uptime // 3600)
                minutes = int((uptime % 3600) // 60)
                print(f"🏠 Local server running | Uptime: {hours}h {minutes}m")
        except KeyboardInterrupt:
            print("\n🛑 Local server stopped")

else:
    print("\n" + "❌" + "="*70)
    print("❌ DEPLOYMENT FAILED - CHATBOT NOT FOUND!")
    print("="*72)
    print("🔍 TROUBLESHOOTING:")
    print("   1. Make sure you've run your chatbot creation cell first")
    print("   2. Check that your chatbot variable is named 'chatbot' or similar")
    print("   3. Verify your chatbot has a 'process_query' method")
    print("   4. Try running: print(globals().keys()) to see available variables")
    print("="*72)
    print("💡 EXPECTED CHATBOT STRUCTURE:")
    print("   • Variable name: 'chatbot' or 'RADEETAChatbotSystem'")
    print("   • Required method: process_query(message)")
    print("   • Optional methods: get_statistics(), health_check()")
    print("="*72)
    print("🔄 After fixing your chatbot, re-run this deployment cell")
    print("="*72)

print("\n✅ Deployment cell execution completed!")

