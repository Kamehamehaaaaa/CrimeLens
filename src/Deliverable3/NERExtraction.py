"""
Advanced NER-Based Entity Extraction for CrimeLens
Replaces heuristic pattern matching with proper Named Entity Recognition

Requirements:
pip install spacy
python -m spacy download en_core_web_sm
"""

import spacy
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class NEREntityExtractor:
    """
    Advanced entity extractor using spaCy NER + rule-based patterns.
    
    Extracts:
    - Persons (PERSON entities)
    - Objects (via dependency parsing and custom patterns)
    - Locations (GPE, LOC entities)
    - Actions (verb phrases with direct objects)
    """
    
    def __init__(self, known_actions=None, known_objects=None, known_locations=None):
        """
        Initialize with spaCy and domain knowledge.
        
        Args:
            known_actions: Set of valid action verbs from training data
            known_objects: Set of valid objects from training data
            known_locations: Set of valid locations from training data
        """
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        self.known_actions = known_actions or set()
        self.known_objects = known_objects or set()
        self.known_locations = known_locations or set()
        
        # Define weapon/object keywords for better extraction
        self.weapon_keywords = {
            'knife', 'gun', 'pistol', 'rifle', 'blade', 'dagger', 'machete',
            'bat', 'club', 'hammer', 'crowbar', 'pipe', 'sword', 'axe',
            'rope', 'wire', 'chain', 'handcuffs', 'tape'
        }
        
        self.valuable_keywords = {
            'wallet', 'purse', 'jewelry', 'watch', 'ring', 'necklace',
            'phone', 'laptop', 'tablet', 'cash', 'money', 'card'
        }
        
        self.all_object_keywords = self.weapon_keywords | self.valuable_keywords
        
        # Crime-related action verbs
        self.crime_verbs = {
            'stabbed', 'shot', 'killed', 'murdered', 'attacked', 'assaulted',
            'robbed', 'stole', 'threatened', 'grabbed', 'took', 'snatched',
            'hit', 'punched', 'kicked', 'beat', 'struck', 'slapped',
            'broke', 'forced', 'entered', 'fled', 'escaped', 'ran'
        }
    
    def extract_persons(self, doc: spacy.tokens.Doc) -> List[str]:
        """Extract person names using NER."""
        persons = []
        
        # Get PERSON entities from spaCy
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Clean name
                name = ent.text.strip()
                # Filter out pronouns and common non-names
                if name.lower() not in ['he', 'she', 'they', 'him', 'her', 'them', 'i', 'you']:
                    persons.append(name)
        
        # Fallback: Look for capitalized words that might be names
        if not persons:
            for token in doc:
                if token.pos_ == "PROPN" and token.text[0].isupper():
                    skip_words = {'The', 'A', 'An', 'This', 'That', 'When', 'Where'}
                    if token.text not in skip_words:
                        persons.append(token.text)
        
        return list(set(persons))  # Remove duplicates
    
    def extract_locations(self, doc: spacy.tokens.Doc) -> List[str]:
        """Extract locations using NER and known locations."""
        locations = []
        
        # Get location entities from spaCy
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:  # Geo-political, location, facility
                locations.append(ent.text.lower())
        
        # Check against known locations
        text_lower = doc.text.lower()
        for known_loc in self.known_locations:
            if known_loc.lower() in text_lower:
                locations.append(known_loc)
        
        # Common location patterns
        location_patterns = [
            r'\bat (?:the )?(\w+)',  # "at the park"
            r'\bin (?:the )?(\w+)',  # "in the apartment"
            r'\bon (?:the )?(\w+)',  # "on the street"
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text_lower)
            locations.extend(matches)
        
        return list(set(locations)) if locations else ['unknown']
    
    def extract_objects(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Extract objects using multiple strategies:
        1. Keyword matching
        2. Dependency parsing (direct objects)
        3. Known objects from training
        """
        objects = []
        text_lower = doc.text.lower()
        
        # Strategy 1: Keyword matching
        for keyword in self.all_object_keywords:
            if keyword in text_lower:
                objects.append(keyword)
        
        # Strategy 2: Check known objects
        for known_obj in self.known_objects:
            if known_obj.lower() in text_lower:
                objects.append(known_obj)
        
        # Strategy 3: Dependency parsing - find direct objects of verbs
        for token in doc:
            if token.dep_ in ["dobj", "pobj"] and token.pos_ == "NOUN":
                obj = token.text.lower()
                # Filter out abstract nouns
                if obj not in ['thing', 'way', 'time', 'place', 'person']:
                    objects.append(obj)
        
        # Strategy 4: Look for "with a/an X" patterns
        with_pattern = r'with (?:a |an |the )?(\w+)'
        matches = re.findall(with_pattern, text_lower)
        objects.extend(matches)
        
        return list(set(objects)) if objects else ['unknown']
    
    def extract_actions(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Extract action verbs using:
        1. Known crime verbs
        2. Past tense verbs
        3. Main verbs in sentence
        """
        actions = []
        text_lower = doc.text.lower()
        
        # Strategy 1: Check known actions
        for known_action in self.known_actions:
            if known_action.lower() in text_lower:
                actions.append(known_action)
        
        # Strategy 2: Check crime verb keywords
        for verb in self.crime_verbs:
            if verb in text_lower:
                actions.append(verb)
        
        # Strategy 3: Extract past tense verbs
        for token in doc:
            if token.pos_ == "VERB" and token.tag_ in ["VBD", "VBN"]:  # Past tense
                lemma = token.lemma_.lower()
                if lemma not in ['be', 'have', 'do', 'say', 'go', 'get']:  # Filter auxiliaries
                    actions.append(token.text.lower())
        
        return list(set(actions)) if actions else ['unknown']
    
    def extract_suspect_victim_roles(self, persons: List[str], doc: spacy.tokens.Doc, 
                                     action: str) -> Tuple[str, str]:
        """
        Determine suspect and victim based on sentence structure.
        
        Heuristic: The person before the action verb is usually the suspect,
        the person after is usually the victim.
        """
        if len(persons) >= 2:
            # Find action position
            action_pos = -1
            for i, token in enumerate(doc):
                if action.lower() in token.text.lower():
                    action_pos = i
                    break
            
            if action_pos > 0:
                # Find persons before and after action
                persons_with_pos = []
                for person in persons:
                    for i, token in enumerate(doc):
                        if person.lower() in token.text.lower():
                            persons_with_pos.append((person, i))
                            break
                
                if len(persons_with_pos) >= 2:
                    # Sort by position
                    persons_with_pos.sort(key=lambda x: x[1])
                    
                    # First person before action = suspect
                    # First person after action = victim
                    suspect = persons_with_pos[0][0]
                    victim = persons_with_pos[1][0]
                    
                    return suspect, victim
            
            # Fallback: first is suspect, second is victim
            return persons[0], persons[1]
        
        elif len(persons) == 1:
            return persons[0], "Unknown Victim"
        else:
            return "Unknown Suspect", "Unknown Victim"
    
    def parse_sentence(self, sentence: str, default_crime_type: str = None) -> Optional[pd.DataFrame]:
        """
        Parse a single sentence into structured crime data.
        
        Args:
            sentence: Natural language crime description
            default_crime_type: Placeholder crime type (will be predicted)
            
        Returns:
            DataFrame with one row of extracted entities
        """
        # Process with spaCy
        doc = self.nlp(sentence)
        
        # Extract all entities
        persons = self.extract_persons(doc)
        locations = self.extract_locations(doc)
        objects = self.extract_objects(doc)
        actions = self.extract_actions(doc)
        
        # Need at least one action to proceed
        if not actions:
            return None
        
        # Take first of each (or most relevant)
        action = actions[0]
        obj = objects[0] if objects else 'unknown'
        location = locations[0] if locations else 'unknown'
        
        # Determine suspect and victim
        suspect, victim = self.extract_suspect_victim_roles(persons, doc, action)
        
        # Use default crime type as placeholder
        crime_type = default_crime_type if default_crime_type else 'unknown'
        
        # Create DataFrame
        data = {
            'scene_id': [1],
            'event_id': [1],
            'crime_type': [crime_type],
            'suspect': [suspect],
            'victim': [victim],
            'object': [obj],
            'location': [location],
            'action': [action]
        }
        
        return pd.DataFrame(data)
    
    def parse_multiple_sentences(self, text: str, default_crime_type: str = None) -> Optional[pd.DataFrame]:
        """
        Parse multiple sentences as separate events in one scene.
        
        Args:
            text: Multi-line text with one sentence per line
            default_crime_type: Placeholder crime type
            
        Returns:
            DataFrame with multiple events
        """
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        
        events = []
        for i, sentence in enumerate(sentences):
            parsed = self.parse_sentence(sentence, default_crime_type)
            if parsed is not None:
                parsed['event_id'] = i + 1
                events.append(parsed)
        
        if not events:
            return None
        
        result = pd.concat(events, ignore_index=True)
        result['scene_id'] = 1
        return result


# ============================================================
# INTEGRATION WITH EXISTING SYSTEM
# ============================================================

class ImprovedCrimeSentenceParser:
    """
    Drop-in replacement for heuristic parser in Streamlit app.
    Uses NER instead of pattern matching.
    """
    
    def __init__(self, graph_builder):
        """Initialize with graph builder to get known entities."""
        self.graph_builder = graph_builder
        
        # Get known entities from trained model
        known_actions = set(graph_builder.action_encoder.classes_) if hasattr(graph_builder, 'action_encoder') else set()
        known_objects = set(graph_builder.object_encoder.classes_) if hasattr(graph_builder, 'object_encoder') else set()
        known_locations = set(graph_builder.location_encoder.classes_) if hasattr(graph_builder, 'location_encoder') else set()
        
        # Initialize NER extractor
        self.extractor = NEREntityExtractor(
            known_actions=known_actions,
            known_objects=known_objects,
            known_locations=known_locations
        )
        
        # Get default crime type (first in encoder)
        self.default_crime = graph_builder.crime_encoder.classes_[0] if hasattr(graph_builder, 'crime_encoder') else 'unknown'
    
    def parse(self, sentence: str) -> Optional[pd.DataFrame]:
        """Parse single sentence."""
        return self.extractor.parse_sentence(sentence, self.default_crime)
    
    def parse_multiple(self, sentences: List[str]) -> Optional[pd.DataFrame]:
        """Parse multiple sentences."""
        text = '\n'.join(sentences)
        return self.extractor.parse_multiple_sentences(text, self.default_crime)


# ============================================================
# EVALUATION: Compare NER vs Heuristic
# ============================================================

def compare_parsers(test_sentences: List[str]):
    """
    Compare NER-based parser vs heuristic parser.
    
    Usage:
        test_sentences = [
            "John stabbed Maria with a knife at home",
            "The suspect robbed the victim at the parking lot",
            "Mike attacked Sarah with a bat and fled"
        ]
        compare_parsers(test_sentences)
    """
    from FinalGenerator import CrimeGraphBuilder
    
    # Initialize parsers
    graph_builder = CrimeGraphBuilder()
    
    # Load some data to fit encoders
    # from advanced_data_generator import generate_advanced_data
    # df = generate_advanced_data(n_scenes=100, seed=42)
    from Architecture_2_generator import DataManager
    dm = DataManager("/Users/rohitbogulla/Desktop/Sem 3/Applied ML 2/CrimeLens/data/realistic_crime_data_2.csv")
    df = dm.getData()
    graph_builder.fit(df)
    
    # NER parser
    ner_parser = ImprovedCrimeSentenceParser(graph_builder)
    
    print("\n" + "="*70)
    print("NER PARSER vs HEURISTIC PARSER COMPARISON")
    print("="*70)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n[Test {i}] {sentence}")
        print("-"*70)
        
        # Parse with NER
        ner_result = ner_parser.parse(sentence)
        
        if ner_result is not None:
            print("✓ NER Parser Result:")
            print(f"  Suspect: {ner_result['suspect'].iloc[0]}")
            print(f"  Victim: {ner_result['victim'].iloc[0]}")
            print(f"  Action: {ner_result['action'].iloc[0]}")
            print(f"  Object: {ner_result['object'].iloc[0]}")
            print(f"  Location: {ner_result['location'].iloc[0]}")
        else:
            print("✗ NER Parser: Failed to parse")


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Example 1: Standalone usage
    print("Example 1: Standalone NER Extraction")
    print("="*70)
    
    extractor = NEREntityExtractor()
    
    test_sentence = "John Smith stabbed Maria Garcia with a knife at the apartment and fled"
    result = extractor.parse_sentence(test_sentence)
    
    if result is not None:
        print(f"Input: {test_sentence}\n")
        print("Extracted:")
        print(result.to_string(index=False))
    
    # Example 2: Multiple sentences
    print("\n\nExample 2: Multiple Events")
    print("="*70)
    
    multi_text = """John attacked Sarah with a bat at the park
John fled the scene
A witness called the police"""
    
    result = extractor.parse_multiple_sentences(multi_text)
    if result is not None:
        print(result[['suspect', 'victim', 'action', 'object', 'location']].to_string(index=False))
    
    # Example 3: Comparison
    print("\n\nExample 3: Parser Comparison")
    test_sentences = [
        "John stabbed Maria with a knife at home",
        "The suspect robbed the victim with a gun at the parking lot",
        "Mike attacked Sarah with a baseball bat at the bar and fled the scene"
    ]
    
    compare_parsers(test_sentences)