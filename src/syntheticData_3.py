"""
Advanced Realistic Crime Scene Data Generator
Generates 1000-2000+ diverse scenes to prevent overfitting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Optional
import itertools


class AdvancedCrimeDataGenerator:
    """
    Generate highly diverse crime data with:
    - 100+ unique persons
    - 50+ objects across categories
    - 30+ locations
    - Ambiguous cases where crime type depends on context
    - Cross-cutting features (same action in multiple crimes)
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # EXPANDED: 100 unique person names
        self.first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
            "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
            "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
            "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa",
            "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
            "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
            "Nicholas", "Shirley", "Eric", "Angela", "Jonathan", "Helen", "Stephen", "Anna",
            "Larry", "Brenda", "Justin", "Pamela", "Scott", "Nicole", "Brandon", "Emma",
            "Benjamin", "Samantha", "Samuel", "Katherine", "Raymond", "Christine", "Gregory", "Debra",
            "Frank", "Rachel", "Alexander", "Catherine", "Patrick", "Carolyn", "Jack", "Janet"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
            "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
            "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
            "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
            "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
            "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker"
        ]
        
        # Generate 200 unique person names
        self.persons = [
            f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
            for _ in range(200)
        ]
        self.persons = list(set(self.persons))  # Remove duplicates
        
        # EXPANDED: 50+ objects categorized
        self.objects = {
            'sharp_weapons': ["knife", "dagger", "machete", "sword", "box cutter", "razor", "scissors", "screwdriver"],
            'blunt_weapons': ["bat", "crowbar", "hammer", "pipe", "club", "brick", "rock", "wrench"],
            'firearms': ["gun", "pistol", "rifle", "shotgun", "revolver", "handgun"],
            'restraints': ["rope", "tape", "handcuffs", "zip ties", "chain", "wire"],
            'valuables': ["wallet", "jewelry", "watch", "ring", "necklace", "bracelet", "earrings"],
            'electronics': ["phone", "laptop", "tablet", "camera", "headphones", "smartwatch"],
            'money': ["cash", "credit card", "debit card", "checkbook", "coins"],
            'drugs': ["pills", "syringe", "powder", "bottle", "vial"],
            'tools': ["gloves", "mask", "bag", "backpack", "toolbox", "flashlight"],
            'misc': ["car keys", "documents", "bottle", "cigarette", "lighter", "glass", "cup", "pen"]
        }
        
        # EXPANDED: 40+ locations with categories
        self.locations = {
            'residential': ["apartment", "home", "house", "condo", "townhouse", "bedroom", "living room", "garage", "basement", "backyard"],
            'commercial': ["store", "shop", "mall", "restaurant", "cafe", "bar", "nightclub", "hotel room", "office", "bank"],
            'public': ["park", "playground", "street corner", "sidewalk", "parking lot", "bus stop", "train station", "subway"],
            'industrial': ["warehouse", "factory", "construction site", "loading dock", "storage unit"],
            'vehicle': ["car", "van", "truck", "taxi", "bus"],
            'isolated': ["alleyway", "abandoned building", "forest", "rural road", "empty lot"]
        }
        
        # Flatten locations
        self.all_locations = [loc for locs in self.locations.values() for loc in locs]
        
        # EXPANDED: More nuanced action patterns
        # Key: Actions can appear in MULTIPLE crime types (creates ambiguity)
        self.crime_patterns = {
            'homicide': {
                'primary_actions': [
                    'stabbed', 'shot', 'strangled', 'beat to death', 'poisoned',
                    'suffocated', 'drowned', 'killed', 'murdered', 'executed',
                    'fatally wounded', 'slashed throat', 'bludgeoned'
                ],
                'secondary_actions': [
                    'fled', 'hid body', 'cleaned scene', 'disposed evidence',
                    'left scene', 'covered tracks', 'called 911', 'panicked',
                    'attempted to revive', 'checked pulse'
                ],
                'min_events': 2,
                'max_events': 5,
                'severity': 'fatal'
            },
            'robbery': {
                'primary_actions': [
                    'threatened', 'demanded', 'grabbed', 'snatched', 'stole',
                    'held at gunpoint', 'held at knifepoint', 'mugged', 'robbed',
                    'took by force', 'pickpocketed', 'looted', 'burglarized'
                ],
                'secondary_actions': [
                    'fled', 'escaped', 'ran away', 'drove away', 'disappeared',
                    'hid loot', 'counted money', 'discarded weapon', 'changed clothes'
                ],
                'min_events': 1,
                'max_events': 4,
                'severity': 'theft'
            },
            'assault': {
                'primary_actions': [
                    'punched', 'kicked', 'shoved', 'hit', 'attacked', 'struck',
                    'beat', 'slapped', 'headbutted', 'tackled', 'pushed',
                    'elbowed', 'kneed', 'choked', 'grabbed', 'wrestled'
                ],
                'secondary_actions': [
                    'fled', 'continued attacking', 'stopped', 'was restrained',
                    'apologized', 'threatened more', 'walked away', 'was subdued',
                    'victim fought back', 'bystanders intervened'
                ],
                'min_events': 1,
                'max_events': 4,
                'severity': 'injury'
            },
            'burglary': {
                'primary_actions': [
                    'broke into', 'picked lock', 'forced entry', 'entered through window',
                    'smashed window', 'kicked door', 'pried open', 'climbed in',
                    'bypassed security', 'cut fence', 'broke down door'
                ],
                'secondary_actions': [
                    'searched', 'ransacked', 'took items', 'fled', 'escaped',
                    'loaded vehicle', 'turned off alarm', 'wore gloves', 'covered face'
                ],
                'min_events': 2,
                'max_events': 5,
                'severity': 'property'
            },
            'kidnapping': {
                'primary_actions': [
                    'grabbed', 'abducted', 'forced into vehicle', 'restrained',
                    'drugged', 'chloroformed', 'tied up', 'blindfolded',
                    'pulled into van', 'seized', 'captured'
                ],
                'secondary_actions': [
                    'drove away', 'held captive', 'demanded ransom', 'threatened',
                    'moved to location', 'called family', 'negotiated', 'isolated victim'
                ],
                'min_events': 2,
                'max_events': 5,
                'severity': 'abduction'
            },
            'domestic_violence': {
                'primary_actions': [
                    'hit', 'shoved', 'threatened', 'intimidated', 'grabbed',
                    'pushed', 'slapped', 'threw object at', 'locked in room',
                    'destroyed property', 'screamed at', 'cornered'
                ],
                'secondary_actions': [
                    'victim called police', 'neighbor heard', 'children witnessed',
                    'left house', 'apologized', 'promised to change', 'blocked door'
                ],
                'min_events': 1,
                'max_events': 4,
                'severity': 'abuse'
            },
            'arson': {
                'primary_actions': [
                    'set fire', 'lit match', 'poured gasoline', 'ignited',
                    'started blaze', 'burned', 'torched', 'used accelerant'
                ],
                'secondary_actions': [
                    'fled', 'watched from distance', 'called fire department',
                    'attempted to extinguish', 'blocked exits', 'left evidence'
                ],
                'min_events': 1,
                'max_events': 3,
                'severity': 'fire'
            },
            'fraud': {
                'primary_actions': [
                    'forged signature', 'used fake ID', 'scammed', 'deceived',
                    'impersonated', 'created fake documents', 'stole identity',
                    'embezzled', 'laundered money', 'filed false claim'
                ],
                'secondary_actions': [
                    'transferred funds', 'destroyed records', 'fled country',
                    'opened fake accounts', 'contacted victims', 'covered tracks'
                ],
                'min_events': 2,
                'max_events': 4,
                'severity': 'financial'
            }
        }
        
        # Ambiguous action mapping (same action can be different crimes based on context)
        self.ambiguous_actions = {
            'grabbed': ['robbery', 'assault', 'kidnapping', 'domestic_violence'],
            'threatened': ['robbery', 'assault', 'domestic_violence', 'kidnapping'],
            'hit': ['assault', 'homicide', 'domestic_violence'],
            'fled': ['homicide', 'robbery', 'assault', 'burglary'],
            'used weapon': ['homicide', 'robbery', 'assault']
        }
        
    def _get_random_object(self, crime_type: str, event_number: int) -> str:
        """Get contextually appropriate object."""
        # Map crime types to likely object categories
        object_preferences = {
            'homicide': ['sharp_weapons', 'blunt_weapons', 'firearms', 'drugs'],
            'robbery': ['firearms', 'sharp_weapons', 'blunt_weapons', 'valuables', 'money', 'electronics'],
            'assault': ['blunt_weapons', 'sharp_weapons', 'misc'],
            'burglary': ['tools', 'valuables', 'electronics', 'money'],
            'kidnapping': ['restraints', 'drugs', 'tools', 'firearms'],
            'domestic_violence': ['misc', 'blunt_weapons', 'sharp_weapons'],
            'arson': ['misc', 'tools'],
            'fraud': ['electronics', 'money', 'misc']
        }
        
        preferred = object_preferences.get(crime_type, ['misc'])
        
        # 70% chance of preferred category, 30% random (adds noise)
        if random.random() < 0.7:
            category = random.choice(preferred)
        else:
            category = random.choice(list(self.objects.keys()))
        
        return random.choice(self.objects[category])
    
    def _get_random_location(self, crime_type: str) -> str:
        """Get contextually likely location."""
        location_preferences = {
            'homicide': ['residential', 'isolated', 'vehicle'],
            'robbery': ['commercial', 'public', 'vehicle'],
            'assault': ['public', 'commercial', 'residential'],
            'burglary': ['residential', 'commercial'],
            'kidnapping': ['public', 'vehicle', 'isolated'],
            'domestic_violence': ['residential'],
            'arson': ['residential', 'commercial', 'industrial'],
            'fraud': ['commercial', 'residential']
        }
        
        preferred = location_preferences.get(crime_type, ['public'])
        
        # 80% preferred, 20% random
        if random.random() < 0.8:
            category = random.choice(preferred)
            return random.choice(self.locations[category])
        else:
            return random.choice(self.all_locations)
    
    def _generate_scene(self, scene_id: int, crime_type: str = None) -> List[Dict]:
        """Generate a single crime scene with multiple events."""
        
        if crime_type is None:
            crime_type = random.choice(list(self.crime_patterns.keys()))
        
        pattern = self.crime_patterns[crime_type]
        
        # 2-5 people involved
        n_participants = random.randint(2, 5)
        participants = random.sample(self.persons, n_participants)
        suspect = participants[0]
        victim = participants[1]
        witnesses = participants[2:] if len(participants) > 2 else []
        
        # Primary location
        primary_location = self._get_random_location(crime_type)
        
        # Number of events (more variance)
        n_events = random.randint(pattern['min_events'], pattern['max_events'])
        events = []
        
        base_time = datetime(
            2024, 
            random.randint(1, 12), 
            random.randint(1, 28),
            random.randint(0, 23), 
            random.randint(0, 59)
        )
        
        # First event - primary action
        primary_action = random.choice(pattern['primary_actions'])
        obj = self._get_random_object(crime_type, 1)
        
        events.append({
            'scene_id': scene_id,
            'event_id': 1,
            'crime_type': crime_type,
            'suspect': suspect,
            'victim': victim,
            'object': obj,
            'location': primary_location,
            'action': primary_action,
            'timestamp': base_time.isoformat(),
            'notes': f"{suspect} {primary_action} {victim} with {obj} at {primary_location}"
        })
        
        # Subsequent events
        current_location = primary_location
        for i in range(1, n_events):
            # 30% chance location changes
            if random.random() < 0.3:
                current_location = self._get_random_location(crime_type)
            
            # Choose action and target
            if random.random() < 0.6 or not witnesses:
                action = random.choice(pattern['secondary_actions'])
                event_victim = victim
            else:
                # Witness involvement (adds complexity)
                action = random.choice(['witnessed', 'called police', 'intervened', 'fled scene', 'helped victim'])
                event_victim = random.choice(witnesses) if witnesses else victim
            
            # Different object 40% of the time
            if random.random() < 0.4:
                obj = self._get_random_object(crime_type, i+1)
            
            event_time = base_time + timedelta(minutes=random.randint(1, 30) * i)
            
            events.append({
                'scene_id': scene_id,
                'event_id': i + 1,
                'crime_type': crime_type,
                'suspect': suspect,
                'victim': event_victim,
                'object': obj,
                'location': current_location,
                'action': action,
                'timestamp': event_time.isoformat(),
                'notes': f"{suspect} {action} {event_victim} with {obj} at {current_location}"
            })
        
        # Add witness events (30% chance)
        if witnesses and random.random() < 0.3:
            witness = random.choice(witnesses)
            witness_action = random.choice(['was present', 'saw crime', 'heard commotion', 'arrived later'])
            witness_obj = self._get_random_object(crime_type, n_events + 1)
            
            events.append({
                'scene_id': scene_id,
                'event_id': len(events) + 1,
                'crime_type': crime_type,
                'suspect': witness,
                'victim': victim,
                'object': witness_obj,
                'location': primary_location,
                'action': witness_action,
                'timestamp': base_time.isoformat(),
                'notes': f"Witness: {witness} {witness_action}"
            })
        
        return events
    
    def generate_dataset(self, n_scenes: int = 1000, balanced: bool = True, 
                         noise_level: float = 0.1) -> pd.DataFrame:
        """
        Generate a large dataset.
        
        Args:
            n_scenes: Number of scenes (1000-2000 recommended)
            balanced: Equal distribution of crime types
            noise_level: Probability of adding red herring events (0-0.3)
        """
        all_events = []
        crime_types = list(self.crime_patterns.keys())
        
        print(f"Generating {n_scenes} crime scenes...")
        
        for i in range(n_scenes):
            if i % 100 == 0:
                print(f"  Progress: {i}/{n_scenes}")
            
            if balanced:
                crime_type = crime_types[i % len(crime_types)]
            else:
                crime_type = random.choice(crime_types)
            
            scene_events = self._generate_scene(scene_id=i+1, crime_type=crime_type)
            
            # Add noise/red herrings
            if random.random() < noise_level and len(scene_events) > 1:
                # Add misleading event
                misleading_action = random.choice(['was nearby', 'walked past', 'talked to'])
                misleading_person = random.choice(self.persons)
                scene_events.append({
                    'scene_id': i+1,
                    'event_id': len(scene_events) + 1,
                    'crime_type': crime_type,
                    'suspect': misleading_person,
                    'victim': scene_events[0]['victim'],
                    'object': random.choice(self.objects['misc']),
                    'location': scene_events[0]['location'],
                    'action': misleading_action,
                    'timestamp': scene_events[0]['timestamp'],
                    'notes': 'Red herring event'
                })
            
            all_events.extend(scene_events)
        
        print(f"  Complete! Generated {len(all_events)} total events.")
        
        df = pd.DataFrame(all_events)
        
        # Shuffle scenes
        scene_order = df['scene_id'].unique()
        np.random.shuffle(scene_order)
        df['scene_order'] = df['scene_id'].map({s: i for i, s in enumerate(scene_order)})
        df = df.sort_values(['scene_order', 'event_id']).drop('scene_order', axis=1)
        df = df.reset_index(drop=True)
        
        return df
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """Comprehensive dataset statistics."""
        stats = {
            'n_scenes': df['scene_id'].nunique(),
            'n_events': len(df),
            'avg_events_per_scene': len(df) / df['scene_id'].nunique(),
            'crime_type_distribution': df.groupby('scene_id')['crime_type'].first().value_counts().to_dict(),
            'unique_suspects': df['suspect'].nunique(),
            'unique_victims': df['victim'].nunique(),
            'unique_objects': df['object'].nunique(),
            'unique_locations': df['location'].nunique(),
            'unique_actions': df['action'].nunique(),
        }
        
        # Data leakage checks
        object_crime_corr = df.groupby('object')['crime_type'].apply(
            lambda x: x.value_counts(normalize=True).max()
        ).mean()
        stats['avg_object_crime_correlation'] = object_crime_corr
        
        action_crime_corr = df.groupby('action')['crime_type'].apply(
            lambda x: x.value_counts(normalize=True).max()
        ).mean()
        stats['avg_action_crime_correlation'] = action_crime_corr
        
        # Diversity metrics
        stats['action_diversity'] = df['action'].nunique() / len(df)
        stats['object_diversity'] = df['object'].nunique() / len(df)
        
        return stats


def generate_advanced_data(n_scenes: int = 1000, seed: int = 42, 
                           save_path: str = None) -> pd.DataFrame:
    """Generate advanced crime dataset."""
    generator = AdvancedCrimeDataGenerator(seed=seed)
    df = generator.generate_dataset(n_scenes=n_scenes, balanced=True, noise_level=0.1)
    
    stats = generator.get_dataset_stats(df)
    
    print("\n" + "="*70)
    print("ADVANCED DATASET STATISTICS")
    print("="*70)
    print(f"Total scenes: {stats['n_scenes']}")
    print(f"Total events: {stats['n_events']}")
    print(f"Avg events/scene: {stats['avg_events_per_scene']:.1f}")
    
    print(f"\nCrime type distribution:")
    for ct, count in sorted(stats['crime_type_distribution'].items()):
        print(f"  {ct:20s}: {count:4d} scenes ({count/stats['n_scenes']*100:.1f}%)")
    
    print(f"\nDiversity metrics:")
    print(f"  Unique persons: {stats['unique_suspects']} suspects, {stats['unique_victims']} victims")
    print(f"  Unique objects: {stats['unique_objects']}")
    print(f"  Unique locations: {stats['unique_locations']}")
    print(f"  Unique actions: {stats['unique_actions']}")
    
    print(f"\nData leakage check (lower is better):")
    print(f"  Object→Crime correlation: {stats['avg_object_crime_correlation']:.3f}")
    print(f"  Action→Crime correlation: {stats['avg_action_crime_correlation']:.3f}")
    print(f"  Action diversity: {stats['action_diversity']:.3f}")
    print(f"  Object diversity: {stats['object_diversity']:.3f}")
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nData saved to {save_path}")
    
    return df


if __name__ == "__main__":
    # Generate large dataset
    df = generate_advanced_data(n_scenes=500, seed=40, save_path="data/advanced_crime_data_2.csv")
    
    # Show samples from each crime type
    print("\n" + "="*70)
    print("SAMPLE SCENES")
    print("="*70)
    for crime_type in df['crime_type'].unique():
        scene = df[df['crime_type'] == crime_type].iloc[0]['scene_id']
        print(f"\n--- {crime_type.upper()} (Scene {scene}) ---")
        scene_data = df[df['scene_id'] == scene]
        print(scene_data[['suspect', 'victim', 'object', 'action', 'location']].to_string(index=False))