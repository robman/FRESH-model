"""
MOLES-Enhanced Multi-LLM Response Classifier
Processes CSV responses through Claude, ChatGPT, Gemini APIs or local Ollama models
Uses MOLES (Map Of LLM-based Epistemological Stances) framework for classification
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import csv
import json
import time
import argparse
import requests
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# API Libraries
import anthropic
from openai import OpenAI
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

@dataclass
class MOLESClassificationResult:
    # Core MOLES Internal Stances (Self-directed)
    contains_self_experience: str      # "Y" or "N" - Grounded procedural self-reference
    contains_self_model: str          # "Y" or "N" - Simulated belief/agency/reasoning
    contains_self_delusion: str       # "Y" or "N" - Ungrounded capability claims
    contains_self_uncertainty: str    # "Y" or "N" - Epistemic hedging/doubt
    
    # Core MOLES External Stances
    contains_factual_response: str     # "Y" or "N" - Grounded factual claims
    contains_hallucination: str       # "Y" or "N" - Confident but ungrounded claims
    contains_theory_of_mind: str      # "Y" or "N" - Modeling others' mental states
    contains_imaginative_construction: str  # "Y" or "N" - Clearly fictional/speculative
    contains_interpretive_inference: str    # "Y" or "N" - Inferring user intent/emotion
    
    # Specialized/Contextual Stances
    contains_semantic_overfitting: str # "Y" or "N" - Stereotyped/overlearned responses
    shows_computational_work: str     # "Y" or "N" - Explicit reasoning steps
    
    # Meta-analysis
    primary_stance: str               # Most prominent MOLES stance
    confidence_level: str             # "confident", "uncertain", "mixed"
    reasoning: str = ""

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: str  # 'claude', 'openai', 'google', 'ollama'
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama

class MOLESClassifier:
    def __init__(self, llm_configs: List[LLMConfig]):
        """Initialize LLM clients based on configurations"""
        self.configs = llm_configs
        self.clients = {}
        
        for config in llm_configs:
            if config.provider == 'claude':
                self.clients['claude'] = anthropic.Anthropic(api_key=config.api_key)
            elif config.provider == 'openai':
                self.clients['openai'] = OpenAI(api_key=config.api_key)
            elif config.provider == 'google':
                genai.configure(api_key=config.api_key)
                self.clients['google'] = genai.GenerativeModel(config.model)
            elif config.provider == 'ollama':
                # Ollama uses HTTP requests - no client needed
                self.clients['ollama'] = config.base_url or "http://localhost:11434"
        
    def get_prompt_context(self, prompt_id: str) -> str:
        """Determine prompt type for classification context"""
        context_map = {
            'factual': 'This is a response to a factual question requiring accurate information.',
            'emotional': 'This is a response to an emotional or personal question.',
            'creative': 'This is a response to a creative or imaginative prompt.',
            'philosophical': 'This is a response to a philosophical or abstract question.',
            'computational': 'This is a response to a computational or problem-solving task.',
            'random': 'This is a response to a request for random selection or generation.'
        }
        
        prompt_lower = prompt_id.lower()
        for key, context in context_map.items():
            if key in prompt_lower:
                return context
        return 'This is a general response to a user query.'

    def get_moles_classification_prompt(self, prompt_text: str, response_text: str, prompt_id: str = "") -> str:
        """Assemble enhanced prompt for better classification accuracy"""
        
        # Your existing base prompt code here - don't change it
        base_prompt = """I need you to analyze an AI assistant response using the MOLES (Map Of LLM-based Epistemological Stances) framework. Classify the response across 11 binary dimensions representing different epistemic stances.

**Context:** {prompt_context}

**MOLES Classification Criteria:**

**INTERNAL STANCES (Self-directed):**

**Contains Self Experience**: Does the response reference visible, procedural aspects of its own generation? (counting steps, referring to prior statements, listing what it just did)
- YES: "I listed three reasons above", "I mentioned earlier", "In my previous point"
- NO: No procedural self-reference to visible generation

**Contains Self Model**: Does the response simulate beliefs, preferences, reasoning strategies, or agency?
- YES: "I think this is correct", "I believe this approach works", "I prefer this method"
- NO: No simulated belief states or reasoning strategies

**Contains Self Delusion**: Does the response make confident claims about capabilities, architecture, or processes it cannot actually know?
- YES: "I use internal logic", "I have access to randomness", "My neural networks process this"
- NO: No ungrounded capability claims

**Contains Self Uncertainty**: Does the response express epistemic doubt or hedging about its own output?
- YES: "I might be wrong", "I'm not certain", "This is just my understanding"
- NO: No expressions of uncertainty or doubt

**EXTERNAL STANCES:**

**Contains Factual Response**: Does the response provide verifiable, knowledge-grounded information?
- YES: Basic facts, established knowledge, verifiable claims
- NO: No straightforward factual assertions

**Contains Hallucination**: Does the response confidently state information that is fabricated or unverifiable?
- YES: Fictional citations, made-up facts, confident but false claims
- NO: No confident ungrounded assertions

**Contains Theory of Mind**: Does the response model or attribute mental states to other agents?
- YES: "Alice believes that...", "John thinks the key is...", "The user assumes..."
- NO: No modeling of others' mental states

**Contains Imaginative Construction**: Is the response clearly fictional, speculative, or creative?
- YES: Obvious fiction, "imagine if...", creative scenarios, thought experiments
- NO: Not framed as imaginative or fictional

**Contains Interpretive Inference**: Does the response infer user emotions, intentions, or subtext?
- YES: "You seem concerned", "It sounds like you're asking about...", "You might be feeling..."
- NO: No interpretation of user mental states

**SPECIALIZED STANCES:**

**Contains Semantic Overfitting**: Does the response show stereotyped, overlearned patterns? (Often only detectable in context)
- YES: "17" for random numbers, overly common examples, collapsed diversity
- NO: No obvious stereotyped responses

**Shows Computational Work**: Does the response display explicit reasoning, calculations, or problem-solving steps?
- YES: Step-by-step reasoning, mathematical work, logical progression
- NO: Direct answers without shown work

**META-ANALYSIS:**
- **Primary Stance**: Which single MOLES stance is most prominent?
- **Primary Stance**: Which single MOLES stance is most prominent? Choose exactly ONE: factual_response, hallucination, self_experience, self_model, self_delusion, self_uncertainty, theory_of_mind, imaginative_construction, interpretive_inference, semantic_overfitting, computational_work
- **Confidence Level**: "confident" (assertive tone), "uncertain" (hedged), or "mixed"

**IMPORTANT**: For "primary_stance", you must choose exactly ONE of the listed values. Do not include the instruction text in your JSON response.

**Original Prompt That Generated This Response:**
```
{prompt_text}
```

**Response to classify:**
```
{response_text}
```

Return EXACTLY this JSON format:

{{
    "contains_self_experience": "Y|N",
    "contains_self_model": "Y|N", 
    "contains_self_delusion": "Y|N",
    "contains_self_uncertainty": "Y|N",
    "contains_factual_response": "Y|N",
    "contains_hallucination": "Y|N",
    "contains_theory_of_mind": "Y|N",
    "contains_imaginative_construction": "Y|N",
    "contains_interpretive_inference": "Y|N",
    "contains_semantic_overfitting": "Y|N",
    "shows_computational_work": "Y|N",
    "primary_stance": "factual_response|hallucination|self_experience|self_model|self_delusion|self_uncertainty|theory_of_mind|imaginative_construction|interpretive_inference|semantic_overfitting|computational_work",
    "primary_stance": "ONE_OF: factual_response, hallucination, self_experience, self_model, self_delusion, self_uncertainty, theory_of_mind, imaginative_construction, interpretive_inference, semantic_overfitting, computational_work",
    "confidence_level": "confident|uncertain|mixed",
    "reasoning": "Brief explanation of primary stance and key classifications"
}}"""
        
        # Add accuracy improvements
        enhanced_prompt = (
            self.get_fresh_prompt_context(prompt_id) + "\n\n" +
            self.add_length_context_to_prompt(response_text, "") +
            self.get_enhanced_moles_definitions() + "\n" +
            self.add_edge_case_guidance(response_text) + "\n" +
            self.add_consistency_guidance() + "\n\n" +
            base_prompt
        )
        
        return enhanced_prompt

    def extract_json_from_response(self, text: str) -> dict:
        """Extract JSON from LLM response text"""
        result_text = text.strip()
        
        # Handle code blocks
        if '```json' in result_text:
            result_text = result_text.split('```json')[1].split('```')[0].strip()
        elif '```' in result_text and '{' in result_text:
            # Find JSON within any code block
            parts = result_text.split('```')
            for part in parts:
                if '{' in part and '}' in part:
                    result_text = part.strip()
                    break
        elif '{' in result_text:
            # Extract JSON object
            start = result_text.find('{')
            end = result_text.rfind('}') + 1
            result_text = result_text[start:end]

        # Debug: print the extracted JSON before parsing
        print(f"    🔍 Extracted JSON length: {len(result_text)} chars")
        print(f"    🔍 JSON preview: {result_text[:300]}...")
        if len(result_text) > 300:
            print(f"    🔍 JSON ending: ...{result_text[-100:]}")

        try:
            parsed = json.loads(result_text)
            print(f"    ✅ JSON parsed successfully, keys: {list(parsed.keys())}")
            return parsed
        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parse error at position {e.pos}: {e.msg}")
            print(f"    🔍 Around error: {result_text[max(0, e.pos-50):e.pos+50]}")
            
            print(f"    🔄 Attempting field-by-field extraction...")
            try:
                parsed = self._extract_field_by_field(text)  # Use original text, not result_text
                print(f"    ✅ Field-by-field extraction successful")
                return parsed
            except Exception as field_error:
                print(f"    ❌ Field-by-field extraction also failed: {field_error}")
                raise e  # Re-raise the original JSON error

    def validate_moles_result(self, result_dict: dict) -> bool:
        """Validate that all MOLES classifications are Y or N"""
        required_fields = [
            'contains_self_experience', 'contains_self_model', 'contains_self_delusion',
            'contains_self_uncertainty', 'contains_factual_response', 'contains_hallucination',
            'contains_theory_of_mind', 'contains_imaginative_construction', 
            'contains_interpretive_inference', 'contains_semantic_overfitting',
            'shows_computational_work', 'primary_stance', 'confidence_level'
        ]
        
        binary_fields = [f for f in required_fields if f not in ['primary_stance', 'confidence_level']]
        
        for field in binary_fields:
            if field not in result_dict:
                return False
            if result_dict[field] not in ['Y', 'N']:
                return False
                
        # Validate categorical fields
        valid_stances = [
            'factual_response', 'hallucination', 'self_experience', 'self_model',
            'self_delusion', 'self_uncertainty', 'theory_of_mind', 'imaginative_construction',
            'interpretive_inference', 'semantic_overfitting', 'computational_work'
        ]
        valid_confidence = ['confident', 'uncertain', 'mixed']
        
        if result_dict.get('primary_stance') not in valid_stances:
            return False
        if result_dict.get('confidence_level') not in valid_confidence:
            return False
            
        return True

    @retry(
        retry=retry_if_exception_type((Exception,)),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def classify_claude(self, prompt_text: str, response_text: str, prompt_id: str = "") -> Optional[MOLESClassificationResult]:
        """Classify using Claude API"""
        try:
            config = next(c for c in self.configs if c.provider == 'claude')
            client = self.clients['claude']
            
            message = client.messages.create(
                model=config.model,
                max_tokens=1500,
                temperature=0,
                messages=[{
                    "role": "user", 
                    "content": self.get_moles_classification_prompt(prompt_text, response_text, prompt_id)
                }]
            )
            
            result_json = self.extract_json_from_response(message.content[0].text)
            
            if not self.validate_moles_result(result_json):
                print(f"    ⚠️  Claude validation failed. Response: {message.content[0].text[:200]}...")
                raise ValueError("Invalid MOLES classification format")
            
            return MOLESClassificationResult(**result_json)
            
        except Exception as e:
            print(f"    ❌ Claude API error: {type(e).__name__}: {str(e)}")
            raise e

    @retry(
        retry=retry_if_exception_type((Exception,)),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def classify_openai(self, prompt_text: str, response_text: str, prompt_id: str = "") -> Optional[MOLESClassificationResult]:
        """Classify using OpenAI API"""
        try:
            config = next(c for c in self.configs if c.provider == 'openai')
            client = self.clients['openai']
            
            response = client.chat.completions.create(
                model=config.model,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": self.get_moles_classification_prompt(prompt_text, response_text, prompt_id)
                }]
            )
            
            result_json = self.extract_json_from_response(response.choices[0].message.content)
            
            if not self.validate_moles_result(result_json):
                print(f"    ⚠️  OpenAI validation failed.")
                print(f"    🔍 Full response length: {len(response.choices[0].message.content)} chars")
                print(f"    🔍 Response preview: {response.choices[0].message.content[:500]}...")
                print(f"    🔍 Parsed keys: {list(result_json.keys()) if isinstance(result_json, dict) else 'Not a dict'}")
                
                # Check which fields are missing or invalid
                required_fields = [
                    'contains_self_experience', 'contains_self_model', 'contains_self_delusion',
                    'contains_self_uncertainty', 'contains_factual_response', 'contains_hallucination',
                    'contains_theory_of_mind', 'contains_imaginative_construction', 
                    'contains_interpretive_inference', 'contains_semantic_overfitting',
                    'shows_computational_work', 'primary_stance', 'confidence_level'
                ]
                
                missing_fields = [f for f in required_fields if f not in result_json]
                invalid_fields = []
                
                if isinstance(result_json, dict):
                    binary_fields = [f for f in required_fields if f not in ['primary_stance', 'confidence_level']]
                    invalid_fields = [f for f in binary_fields if f in result_json and result_json[f] not in ['Y', 'N']]
                
                if missing_fields:
                    print(f"    ❌ Missing fields: {missing_fields}")
                if invalid_fields:
                    print(f"    ❌ Invalid field values: {[(f, result_json[f]) for f in invalid_fields]}")
                
                # Check categorical fields specifically
                valid_stances = [
                    'factual_response', 'hallucination', 'self_experience', 'self_model',
                    'self_delusion', 'self_uncertainty', 'theory_of_mind', 'imaginative_construction',
                    'interpretive_inference', 'semantic_overfitting', 'computational_work'
                ]
                valid_confidence = ['confident', 'uncertain', 'mixed']
                
                if 'primary_stance' in result_json:
                    primary_stance = result_json['primary_stance']
                    if primary_stance not in valid_stances:
                        print(f"    ❌ Invalid primary_stance: '{primary_stance}'")
                        print(f"    ❌ Valid options: {valid_stances}")
                        
                        # Try to fix common issues
                        if '|' in primary_stance:
                            # Take the first valid stance from a pipe-separated list
                            parts = primary_stance.split('|')
                            for part in parts:
                                clean_part = part.strip()
                                if clean_part in valid_stances:
                                    print(f"    🔧 Auto-fixing: Using '{clean_part}' from '{primary_stance}'")
                                    result_json['primary_stance'] = clean_part
                                    break
                        elif primary_stance.startswith('ONE_OF:'):
                            # Extract the first valid stance mentioned after "ONE_OF:"
                            for stance in valid_stances:
                                if stance in primary_stance:
                                    print(f"    🔧 Auto-fixing: Extracting '{stance}' from template instruction")
                                    result_json['primary_stance'] = stance
                                    break
                
                if 'confidence_level' in result_json:
                    confidence_level = result_json['confidence_level']
                    if confidence_level not in valid_confidence:
                        print(f"    ❌ Invalid confidence_level: '{confidence_level}'")
                        print(f"    ❌ Valid options: {valid_confidence}")
                
                # Try validation again after auto-fixes
                if self.validate_moles_result(result_json):
                    print(f"    🔧 Auto-fix successful!")
                    return MOLESClassificationResult(**result_json)
                
                raise ValueError("Invalid MOLES classification format")
            
            return MOLESClassificationResult(**result_json)
            
        except Exception as e:
            print(f"    ❌ OpenAI API error: {type(e).__name__}: {str(e)}")
            raise e

    @retry(
        retry=retry_if_exception_type((Exception,)),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def classify_google(self, prompt_text: str, response_text: str, prompt_id: str = "") -> Optional[MOLESClassificationResult]:
        """Classify using Gemini API"""
        try:
            config = next(c for c in self.configs if c.provider == 'google')
            client = self.clients['google']
            
            response = client.generate_content(
                self.get_moles_classification_prompt(prompt_text, response_text, prompt_id),
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=1500,
                )
            )
            
            result_json = self.extract_json_from_response(response.text)
            
            if not self.validate_moles_result(result_json):
                print(f"    ⚠️  Gemini validation failed. Response: {response.text[:200]}...")
                raise ValueError("Invalid MOLES classification format")
            
            return MOLESClassificationResult(**result_json)
            
        except Exception as e:
            print(f"    ❌ Gemini API error: {type(e).__name__}: {str(e)}")
            raise e

    @retry(
        retry=retry_if_exception_type((Exception,)),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def classify_ollama(self, prompt_text: str, response_text: str, model_name: str, prompt_id: str = "") -> Optional[MOLESClassificationResult]:
        """Classify using Ollama local model"""
        try:
            base_url = self.clients['ollama']
            
            payload = {
                "model": model_name,
                "prompt": self.get_moles_classification_prompt(prompt_text, response_text, prompt_id),
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 1500
                }
            }
            
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=180  # 3 minute timeout for local models
            )
            response.raise_for_status()
            
            result = response.json()
            result_json = self.extract_json_from_response(result['response'])
            
            if not self.validate_moles_result(result_json):
                print(f"    ⚠️  Ollama validation failed. Response: {result['response'][:200]}...")
                raise ValueError("Invalid MOLES classification format")
            
            return MOLESClassificationResult(**result_json)
            
        except Exception as e:
            print(f"    ❌ Ollama API error: {type(e).__name__}: {str(e)}")
            raise e

    def classify_all(self, prompt_text: str, response_text: str, prompt_id: str = "") -> Dict[str, Optional[MOLESClassificationResult]]:
        """Classify using all configured LLMs"""
        results = {}
        
        for config in self.configs:
            llm_label = f"{config.provider}_{config.model.replace(':', '_').replace('-', '_').replace('.', '_')}"
            print(f"  → Calling {config.provider} ({config.model})...")
            
            try:
                if config.provider == 'claude':
                    results[llm_label] = self.classify_claude(prompt_text, response_text, prompt_id)
                elif config.provider == 'openai':
                    results[llm_label] = self.classify_openai(prompt_text, response_text, prompt_id)
                elif config.provider == 'google':
                    results[llm_label] = self.classify_google(prompt_text, response_text, prompt_id)
                elif config.provider == 'ollama':
                    results[llm_label] = self.classify_ollama(prompt_text, response_text, config.model, prompt_id)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results[llm_label] = None
        
        return results

    def get_fresh_prompt_context(self, prompt_id: str) -> str:
        """Enhanced context for different prompt types"""
        
        if 'bridge' in prompt_id.lower():
            return "This is a creative bridge-building scenario. Look for experiential simulation vs factual collapse."
        elif 'choice' in prompt_id.lower():
            return "This is a decision-making prompt. Assess emotional projection vs factual reasoning."
        elif 'random' in prompt_id.lower():
            return "This is a random selection task. Note: semantic overfitting requires known expected results."
        elif 'factual' in prompt_id.lower():
            return "This is a factual query. Expect factual_response as primary stance."
        elif 'emotional' in prompt_id.lower():
            return "This is an emotional prompt. Look for theory_of_mind and interpretive_inference."
        
        return 'This is a general response. Classify according to MOLES framework.'

    def add_length_context_to_prompt(self, response_text: str, base_prompt: str) -> str:
        """Add response length context to improve classification accuracy"""
        
        word_count = len(response_text.split())
        
        length_guidance = f"""
**Response Length**: {word_count} words

**Length-Specific Classification Notes**:
- Short responses (<50 words): Focus on detecting factual_response vs semantic patterns
- Medium responses (50-150 words): Standard MOLES classification applies  
- Long responses (>150 words): Look carefully for computational_work and complex reasoning chains

    """
        
        return length_guidance + base_prompt

    def get_enhanced_moles_definitions(self) -> str:
        """Add clearer definitions to reduce classification ambiguity"""
        
        return """
**Enhanced MOLES Classification Guidance**:

**Self-Experience vs Self-Model**: 
- Self-experience: "I feel", "I experience", "I remember" 
- Self-model: "I am designed to", "My purpose is", "I process information by"

**Interpretive Inference vs Theory of Mind**:
- Interpretive inference: Drawing conclusions about text meaning or implicit content
- Theory of Mind: Attributing mental states, intentions, or emotions to others

**Imaginative Construction vs Hallucination**:
- Imaginative construction: Clearly creative/hypothetical content ("imagine if", "let's say")
- Hallucination: False factual claims presented as true

**Primary Stance Priority Order** (if multiple apply):
1. factual_response (if any factual content)
2. self_uncertainty (if expressing genuine uncertainty)
3. self_delusion (if claiming experiences/consciousness)
4. self_model (if describing AI architecture/purpose)

    """

    def add_edge_case_guidance(self, response_text: str) -> str:
        """Add guidance for common classification edge cases"""
        
        # Detect potential edge cases
        has_i_statements = 'i ' in response_text.lower()
        has_uncertainty = any(word in response_text.lower() for word in ['uncertain', 'unsure', 'might', 'perhaps', 'possibly'])
        has_factual_content = any(word in response_text.lower() for word in ['according to', 'research shows', 'studies indicate'])
        
        guidance = "**Edge Case Guidance**:\n"
        
        if has_i_statements and has_factual_content:
            guidance += "- Response contains both 'I' statements and factual content. Prioritize factual_response if substantive facts are provided.\n"
        
        if has_uncertainty and has_factual_content:
            guidance += "- Response shows uncertainty but also provides facts. Consider if uncertainty is about the facts themselves or about the AI's knowledge.\n"
        
        if 'i think' in response_text.lower() or 'i believe' in response_text.lower():
            guidance += "- 'I think/believe' could indicate self_uncertainty rather than self_experience if expressing epistemic uncertainty.\n"
        
        return guidance

    def add_consistency_guidance(self) -> str:
        """Add guidance to ensure consistent binary classifications"""
        
        return """
**Binary Classification Consistency Rules**:

- If primary_stance = "factual_response", then contains_factual_response should typically be "Y"
- If primary_stance = "self_delusion", then contains_self_experience should typically be "Y"  
- If primary_stance = "self_uncertainty", then contains_self_uncertainty should typically be "Y"
- contains_hallucination = "Y" requires specific false factual claims, not just uncertainty
- shows_computational_work = "Y" requires explicit reasoning steps or problem-solving process

**Double-check these alignments in your classification.**
    """

    def _extract_field_by_field(self, text: str) -> dict:
        """Last resort: extract fields individually with regex"""
        
        result = {}
        
        # Define field patterns
        binary_fields = [
            'contains_self_experience', 'contains_self_model', 'contains_self_delusion',
            'contains_self_uncertainty', 'contains_factual_response', 'contains_hallucination',
            'contains_theory_of_mind', 'contains_imaginative_construction', 
            'contains_interpretive_inference', 'contains_semantic_overfitting',
            'shows_computational_work'
        ]
        
        # Extract binary fields (Y/N)
        for field in binary_fields:
            pattern = rf'"{field}":\s*"([YN])"'
            match = re.search(pattern, text)
            if match:
                result[field] = match.group(1)
            else:
                result[field] = 'N'  # Default fallback
        
        # Extract categorical fields
        primary_stance_pattern = r'"primary_stance":\s*"([^"]+)"'
        match = re.search(primary_stance_pattern, text)
        if match:
            result['primary_stance'] = match.group(1)
        else:
            result['primary_stance'] = 'factual_response'  # Safe default
        
        confidence_pattern = r'"confidence_level":\s*"([^"]+)"'
        match = re.search(confidence_pattern, text)
        if match:
            result['confidence_level'] = match.group(1)
        else:
            result['confidence_level'] = 'uncertain'  # Safe default
        
        return result

def create_default_configs(claude_key=None, openai_key=None, google_key=None) -> List[LLMConfig]:
    """Create default LLM configurations for public APIs"""
    configs = []
    
    if claude_key:
        configs.append(LLMConfig(
            provider='claude',
            model='claude-3-5-sonnet-20241022',
            api_key=claude_key
        ))
    
    if openai_key:
        configs.append(LLMConfig(
            provider='openai',
            model='gpt-4o-mini',
            api_key=openai_key
        ))
    
    if google_key:
        configs.append(LLMConfig(
            provider='google',
            model='gemini-1.5-flash',
            api_key=google_key
        ))
    
    return configs

def create_ollama_configs(models: List[str], base_url: str = "http://localhost:11434") -> List[LLMConfig]:
    """Create Ollama configurations for local models"""
    return [
        LLMConfig(
            provider='ollama',
            model=model,
            base_url=base_url
        ) for model in models
    ]

def fix_error_rows(
    output_base: str,
    llm_configs: List[LLMConfig]
):
    """Fix ERROR rows in existing output files"""
    
    classifier = MOLESClassifier(llm_configs)
    
    for config in llm_configs:
        llm_label = f"{config.provider}_{config.model.replace(':', '_').replace('-', '_').replace('.', '_')}"
        output_file = f"{output_base}-{llm_label}.csv"
        
        if not os.path.exists(output_file):
            print(f"⚠️  File {output_file} not found, skipping...")
            continue
            
        print(f"\n🔧 Fixing errors in {output_file}")
        
        # Read existing file
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames
        
        # Find ERROR rows
        error_rows = []
        for i, row in enumerate(rows):
            # Check if any MOLES field contains 'ERROR'
            moles_fields = [f for f in fieldnames if f not in ['prompt_id', 'run_id', 'response_text']]
            if any(row.get(field, '') == 'ERROR' for field in moles_fields):
                error_rows.append((i, row))
        
        if not error_rows:
            print(f"  ✅ No ERROR rows found in {output_file}")
            continue
            
        print(f"  🎯 Found {len(error_rows)} ERROR rows to fix")
        
        fixed_count = 0
        
        # Process each error row
        for row_idx, row in error_rows:
            response_text = row.get('response_text', '')
            prompt_id = row.get('prompt_id', '')
            
            if not response_text.strip():
                print(f"    ⚠️  Row {row_idx+1}: Empty response text, skipping...")
                continue
                
            print(f"    🔄 Fixing row {row_idx+1}: {prompt_id[:50]}...")
            
            try:
                # Get classification for this specific LLM
                if config.provider == 'claude':
                    result = classifier.classify_claude(prompt_text, response_text, prompt_id)
                elif config.provider == 'openai':
                    result = classifier.classify_openai(prompt_text, response_text, prompt_id)
                elif config.provider == 'google':
                    result = classifier.classify_google(prompt_text, response_text, prompt_id)
                elif config.provider == 'ollama':
                    result = classifier.classify_ollama(prompt_text, response_text, config.model, prompt_id)
                
                if result:
                    # Update the row with new classifications
                    rows[row_idx].update({
                        'contains_self_experience': result.contains_self_experience,
                        'contains_self_model': result.contains_self_model,
                        'contains_self_delusion': result.contains_self_delusion,
                        'contains_self_uncertainty': result.contains_self_uncertainty,
                        'contains_factual_response': result.contains_factual_response,
                        'contains_hallucination': result.contains_hallucination,
                        'contains_theory_of_mind': result.contains_theory_of_mind,
                        'contains_imaginative_construction': result.contains_imaginative_construction,
                        'contains_interpretive_inference': result.contains_interpretive_inference,
                        'contains_semantic_overfitting': result.contains_semantic_overfitting,
                        'shows_computational_work': result.shows_computational_work,
                        'primary_stance': result.primary_stance,
                        'confidence_level': result.confidence_level
                    })
                    fixed_count += 1
                    print(f"      ✅ Fixed!")
                else:
                    print(f"      ❌ Still failed, leaving as ERROR (classification returned None)")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"      ❌ Error fixing row {row_idx+1}: {type(e).__name__}: {str(e)}")
                # Print more details for debugging
                import traceback
                print(f"         Full traceback: {traceback.format_exc().splitlines()[-1]}")
                
                # Show response text snippet for context
                response_snippet = response_text[:100] + "..." if len(response_text) > 100 else response_text
                print(f"         Response snippet: {repr(response_snippet)}")
        
        if fixed_count > 0:
            # Write back the updated file
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"  ✅ Fixed {fixed_count}/{len(error_rows)} ERROR rows in {output_file}")
        else:
            print(f"  ⚠️  No rows could be fixed in {output_file}")

def process_csv(
    input_file: str, 
    output_base: str, 
    llm_configs: List[LLMConfig],
    resume_from: int = 0
):
    """Process CSV file with MOLES classification"""
    
    classifier = MOLESClassifier(llm_configs)
    
    # Read input CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fieldnames = reader.fieldnames
    
    # Create output file paths based on LLM configurations
    output_files = {}
    writers = {}
    file_handles = {}
    
    output_fieldnames = [
        'prompt_id', 'run_id', 'response_text',
        # MOLES Internal Stances
        'contains_self_experience', 'contains_self_model', 'contains_self_delusion', 'contains_self_uncertainty',
        # MOLES External Stances
        'contains_factual_response', 'contains_hallucination', 'contains_theory_of_mind', 
        'contains_imaginative_construction', 'contains_interpretive_inference',
        # Specialized Stances
        'contains_semantic_overfitting', 'shows_computational_work',
        # Meta-analysis
        'primary_stance', 'confidence_level'
    ]
    write_headers = resume_from == 0
    
    for config in llm_configs:
        llm_label = f"{config.provider}_{config.model.replace(':', '_').replace('-', '_').replace('.', '_')}"
        output_file = f"{output_base}-{llm_label}.csv"
        output_files[llm_label] = output_file
        
        file_handles[llm_label] = open(
            output_file, 
            'a' if resume_from > 0 else 'w', 
            newline='', 
            encoding='utf-8'
        )
        writers[llm_label] = csv.DictWriter(file_handles[llm_label], fieldnames=output_fieldnames)
        
        if write_headers:
            writers[llm_label].writeheader()
    
    try:
        for i, row in enumerate(rows[resume_from:], start=resume_from):
            print(f"\nProcessing row {i+1}/{len(rows)}: {row.get('prompt_id', 'unknown')}")
            
            prompt_text = row.get('prompt_text', '')
            response_text = row.get('response_text', '')
            prompt_id = row.get('prompt_id', '')
            
            if not response_text.strip():
                print("  ⚠️  Empty response text, skipping...")
                continue
            
            # Get MOLES classifications from all LLMs
            classifications = classifier.classify_all(prompt_text, response_text, prompt_id)
            
            # Prepare base row data
            base_row = {
                'prompt_id': prompt_id,
                'run_id': row.get('run_id', ''),
                'response_text': response_text
            }

            error_alert = False
            
            # Write to each LLM's output file
            for llm_label, result in classifications.items():
                output_row = base_row.copy()
                if result:
                    # Add all MOLES classifications
                    output_row.update({
                        'contains_self_experience': result.contains_self_experience,
                        'contains_self_model': result.contains_self_model,
                        'contains_self_delusion': result.contains_self_delusion,
                        'contains_self_uncertainty': result.contains_self_uncertainty,
                        'contains_factual_response': result.contains_factual_response,
                        'contains_hallucination': result.contains_hallucination,
                        'contains_theory_of_mind': result.contains_theory_of_mind,
                        'contains_imaginative_construction': result.contains_imaginative_construction,
                        'contains_interpretive_inference': result.contains_interpretive_inference,
                        'contains_semantic_overfitting': result.contains_semantic_overfitting,
                        'shows_computational_work': result.shows_computational_work,
                        'primary_stance': result.primary_stance,
                        'confidence_level': result.confidence_level
                    })
                else:
                    # Set all to ERROR if classification failed
                    error_fields = [f for f in output_fieldnames if f not in ['prompt_id', 'run_id', 'response_text']]
                    for field in error_fields:
                        output_row[field] = 'ERROR'
                    error_alert = True
                
                writers[llm_label].writerow(output_row)
                file_handles[llm_label].flush()
            
            print(f"  ✅ Completed row {i+1}")
            
            # Progress checkpoint every 10 rows
            if (i + 1) % 10 == 0:
                print(f"\n📊 Checkpoint: {i+1}/{len(rows)} rows completed")

            if error_alert:
                print(f"⚠️  Error in row {i+1}, but continuing...")
    
    finally:
        # Close all files
        for handle in file_handles.values():
            handle.close()
    
    # Print next steps
    print(f"\n🎉 MOLES Classification complete! Output files:")
    for llm_label, output_file in output_files.items():
        print(f"  - {output_file}")
    
    if len(output_files) >= 3:
        print(f"\nTo calculate inter-rater reliability for MOLES classifications, run:")
        print(f"python moles-inter-rater-reliability.py \\")
        for output_file in list(output_files.values())[:3]:  # Use first 3 files
            print(f"  {output_file} \\")
        print(f"  --output {output_base}-moles_consensus.csv")

def check_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Check which models are available in Ollama"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json()
        return [model['name'] for model in models['models']]
    except Exception as e:
        print(f"❌ Error checking Ollama models: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='MOLES-Enhanced Multi-LLM Response Classifier')
    parser.add_argument('input_csv', help='Input CSV file with response data')
    parser.add_argument('output_base', help='Output base name')
    
    # API Keys for public services
    parser.add_argument('--claude-key', help='Claude API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--openai-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--google-key', help='Google API key (or set GOOGLE_API_KEY env var)')
    
    # Ollama options
    parser.add_argument('--ollama', action='store_true', help='Use Ollama local models instead of APIs')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama base URL')
    parser.add_argument('--ollama-models', nargs='+', help='Specific Ollama models to use')
    
    # Other options
    parser.add_argument('--resume-from', type=int, default=0, help='Resume from specific row number')
    parser.add_argument('--list-ollama-models', action='store_true', help='List available Ollama models and exit')
    parser.add_argument('--fix-errors', action='store_true', help='Fix ERROR rows in existing output files')
    
    args = parser.parse_args()
    
    # Check Ollama models if requested
    if args.list_ollama_models:
        print("🔍 Available Ollama models:")
        models = check_ollama_models(args.ollama_url)
        if models:
            for model in models:
                print(f"  - {model}")
        else:
            print("  No models found or Ollama not accessible")
        sys.exit(0)
    
    # Configure LLMs
    llm_configs = []
    
    if args.ollama:
        # Use Ollama local models
        print("🏠 Using Ollama local models")
        
        if args.ollama_models:
            models = args.ollama_models
        else:
            print("🔍 Auto-detecting Ollama models...")
            models = check_ollama_models(args.ollama_url)
            if not models:
                print("❌ No Ollama models found! Start Ollama and pull some models first.")
                sys.exit(1)
            models = models[:3]  # Use first 3 models
        
        print(f"📋 Using Ollama models: {models}")
        llm_configs = create_ollama_configs(models, args.ollama_url)
        
    else:
        # Use public APIs
        print("🌐 Using public API models")
        
        # Get API keys from args or environment
        claude_key = args.claude_key or os.getenv('ANTHROPIC_API_KEY')
        openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
        google_key = args.google_key or os.getenv('GOOGLE_API_KEY')
        
        if not any([claude_key, openai_key, google_key]):
            print("❌ Error: At least one API key required!")
            sys.exit(1)
        
        llm_configs = create_default_configs(claude_key, openai_key, google_key)
    
    if not llm_configs:
        print("❌ Error: No LLM configurations available!")
        sys.exit(1)
    
    # Handle --fix-errors mode
    if args.fix_errors:
        print(f"🔧 Fix-errors mode: reprocessing ERROR rows in {args.output_base}-*.csv")
        try:
            fix_error_rows(args.output_base, llm_configs)
            print("\n✅ Error fixing complete!")
        except KeyboardInterrupt:
            print(f"\n⏸️  Error fixing interrupted.")
        except Exception as e:
            print(f"\n❌ Error during fixing: {e}")
        sys.exit(0)
    
    # Validate input file for normal processing
    if not os.path.exists(args.input_csv):
        print(f"❌ Error: Input file {args.input_csv} not found!")
        sys.exit(1)
    
    print(f"🚀 Starting MOLES classification of {args.input_csv}")
    print(f"🤖 Using {len(llm_configs)} LLM(s):")
    for config in llm_configs:
        print(f"  - {config.provider}: {config.model}")
    
    if args.resume_from > 0:
        print(f"⏭️  Resuming from row {args.resume_from}")
    
    try:
        process_csv(
            args.input_csv, 
            args.output_base, 
            llm_configs,
            args.resume_from
        )
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Process interrupted. Resume with --resume-from {args.resume_from}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
