"""
Realistic Crime Scene Data Generator
Creates data where graph structure matters for prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class RealisticCrimeDataGenerator:
    """
    Generates crime scene data where:
    - Same objects appear in different crime types
    - Multiple people involved with different roles
    - Graph structure (relationships) determines crime type
    - Not trivially solvable by simple feature lookup
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Shared entities across crime types (key for making it non-trivial)
        self.persons = [
            "John Smith", "Maria Garcia", "James Wilson", "Sarah Johnson",
            "Michael Brown", "Emily Davis", "Robert Miller", "Jennifer Martinez",
            "William Anderson", "Lisa Thompson", "David Taylor", "Amanda White",
            "Christopher Lee", "Jessica Harris", "Daniel Clark", "Ashley Robinson"
        ]
        
        # Objects that can appear in MULTIPLE crime types
        self.objects = {
            'weapons': ["knife", "gun", "bat", "crowbar", "hammer"],
            'valuables': ["wallet", "jewelry", "phone", "laptop", "cash"],
            'tools': ["rope", "tape", "gloves", "mask", "bag"],
            'misc': ["car keys", "documents", "bottle", "cigarette"]
        }
        
        self.locations = [
            "apartment", "parking lot", "alleyway", "home", "office",
            "warehouse", "park", "bar", "restaurant", "street corner",
            "shopping mall", "gas station", "hotel room", "basement"
        ]
        
        # Crime type is determined by ACTION PATTERNS, not just objects
        self.crime_patterns = {
            'homicide': {
                'primary_actions': ['stabbed', 'shot', 'strangled', 'beat to death', 'poisoned'],
                'secondary_actions': ['fled', 'hid body', 'cleaned scene', 'disposed evidence'],
                'min_events': 2,
                'max_events': 4,
                'weapon_required': True
            },
            'robbery': {
                'primary_actions': ['threatened', 'demanded', 'grabbed', 'snatched', 'held at gunpoint'],
                'secondary_actions': ['fled', 'escaped in vehicle', 'ran away', 'disappeared'],
                'min_events': 2,
                'max_events': 3,
                'weapon_required': False  # Can be with or without weapon
            },
            'assault': {
                'primary_actions': ['punched', 'kicked', 'shoved', 'hit', 'attacked'],
                'secondary_actions': ['fled', 'continued attacking', 'stopped', 'was restrained'],
                'min_events': 1,
                'max_events': 3,
                'weapon_required': False
            },
            'burglary': {
                'primary_actions': ['broke into', 'picked lock', 'entered through window', 'forced entry'],
                'secondary_actions': ['searched', 'took items', 'fled', 'escaped'],
                'min_events': 2,
                'max_events': 4,
                'weapon_required': False
            },
            'kidnapping': {
                'primary_actions': ['grabbed', 'forced into vehicle', 'restrained', 'abducted'],
                'secondary_actions': ['drove away', 'held captive', 'demanded ransom', 'threatened'],
                'min_events': 2,
                'max_events': 4,
                'weapon_required': False
            }
        }
    
    def _get_random_object(self, crime_type: str, use_weapon: bool = None) -> str:
        """Get an object - weapons can appear in any crime type."""
        if use_weapon is None:
            use_weapon = random.random() < 0.5
        
        if use_weapon:
            return random.choice(self.objects['weapons'])
        else:
            category = random.choice(['valuables', 'tools', 'misc'])
            return random.choice(self.objects[category])
    
    def _generate_scene(self, scene_id: int, crime_type: str = None) -> list:
        """Generate a single crime scene with multiple events."""
        
        if crime_type is None:
            crime_type = random.choice(list(self.crime_patterns.keys()))
        
        pattern = self.crime_patterns[crime_type]
        
        # Select participants (2-4 people involved)
        n_participants = random.randint(2, 4)
        participants = random.sample(self.persons, n_participants)
        suspect = participants[0]
        victim = participants[1]
        witnesses = participants[2:] if len(participants) > 2 else []
        
        # Select location(s)
        primary_location = random.choice(self.locations)
        
        # Generate events
        n_events = random.randint(pattern['min_events'], pattern['max_events'])
        events = []
        
        base_time = datetime(2024, random.randint(1, 12), random.randint(1, 28),
                            random.randint(0, 23), random.randint(0, 59))
        
        # First event - primary action
        primary_action = random.choice(pattern['primary_actions'])
        use_weapon = pattern['weapon_required'] or random.random() < 0.4
        obj = self._get_random_object(crime_type, use_weapon)
        
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
            # Sometimes location changes
            if random.random() < 0.3:
                current_location = random.choice(self.locations)
            
            # Mix of secondary actions and sometimes witnesses
            if random.random() < 0.7 or not witnesses:
                action = random.choice(pattern['secondary_actions'])
                event_victim = victim
            else:
                # Witness interaction
                action = random.choice(['saw', 'heard', 'called police', 'intervened'])
                event_victim = random.choice(witnesses)
            
            # Sometimes different object
            if random.random() < 0.4:
                obj = self._get_random_object(crime_type)
            
            event_time = base_time + timedelta(minutes=random.randint(1, 15) * i)
            
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
        
        # Add some noise: additional people/objects present but not directly involved
        if random.random() < 0.3 and witnesses:
            noise_event = {
                'scene_id': scene_id,
                'event_id': len(events) + 1,
                'crime_type': crime_type,
                'suspect': witnesses[0],  # Not the real suspect
                'victim': victim,
                'object': random.choice(self.objects['misc']),
                'location': primary_location,
                'action': random.choice(['was present', 'was nearby', 'saw']),
                'timestamp': base_time.isoformat(),
                'notes': "Witness/bystander presence"
            }
            events.append(noise_event)
        
        return events
    
    def generate_dataset(self, n_scenes: int = 100, 
                         balanced: bool = True) -> pd.DataFrame:
        """
        Generate a complete dataset.
        
        Args:
            n_scenes: Number of crime scenes to generate
            balanced: If True, equal distribution of crime types
        """
        all_events = []
        crime_types = list(self.crime_patterns.keys())
        
        for i in range(n_scenes):
            if balanced:
                crime_type = crime_types[i % len(crime_types)]
            else:
                crime_type = random.choice(crime_types)
            
            scene_events = self._generate_scene(scene_id=i+1, crime_type=crime_type)
            all_events.extend(scene_events)
        
        df = pd.DataFrame(all_events)
        
        # Shuffle to avoid ordering bias
        scene_order = df['scene_id'].unique()
        np.random.shuffle(scene_order)
        df['scene_order'] = df['scene_id'].map({s: i for i, s in enumerate(scene_order)})
        df = df.sort_values(['scene_order', 'event_id']).drop('scene_order', axis=1)
        df = df.reset_index(drop=True)
        
        return df
    
    def get_dataset_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics about the generated dataset."""
        stats = {
            'n_scenes': df['scene_id'].nunique(),
            'n_events': len(df),
            'avg_events_per_scene': len(df) / df['scene_id'].nunique(),
            'crime_type_distribution': df.groupby('scene_id')['crime_type'].first().value_counts().to_dict(),
            'unique_suspects': df['suspect'].nunique(),
            'unique_victims': df['victim'].nunique(),
            'unique_objects': df['object'].nunique(),
            'unique_locations': df['location'].nunique(),
            'unique_actions': df['action'].nunique()
        }
        
        # Check for data leakage (object -> crime_type correlation)
        object_crime_corr = df.groupby('object')['crime_type'].apply(
            lambda x: x.value_counts(normalize=True).max()
        ).mean()
        stats['avg_object_crime_correlation'] = object_crime_corr
        
        # Check action -> crime_type correlation
        action_crime_corr = df.groupby('action')['crime_type'].apply(
            lambda x: x.value_counts(normalize=True).max()
        ).mean()
        stats['avg_action_crime_correlation'] = action_crime_corr
        
        return stats


def generate_realistic_data(n_scenes: int = 200, seed: int = 42) -> pd.DataFrame:
    """Quick function to generate realistic crime data."""
    generator = RealisticCrimeDataGenerator(seed=seed)
    df = generator.generate_dataset(n_scenes=n_scenes, balanced=True)
    
    stats = generator.get_dataset_stats(df)
    
    print("="*60)
    print("GENERATED DATASET STATISTICS")
    print("="*60)
    print(f"Total scenes: {stats['n_scenes']}")
    print(f"Total events: {stats['n_events']}")
    print(f"Avg events/scene: {stats['avg_events_per_scene']:.1f}")
    print(f"\nCrime type distribution:")
    for ct, count in stats['crime_type_distribution'].items():
        print(f"  {ct}: {count}")
    print(f"\nUnique entities:")
    print(f"  Suspects: {stats['unique_suspects']}")
    print(f"  Objects: {stats['unique_objects']}")
    print(f"  Locations: {stats['unique_locations']}")
    print(f"  Actions: {stats['unique_actions']}")
    print(f"\nData leakage check:")
    print(f"  Object→Crime correlation: {stats['avg_object_crime_correlation']:.2f}")
    print(f"  Action→Crime correlation: {stats['avg_action_crime_correlation']:.2f}")
    print("  (Lower is better, 0.2-0.5 means graph structure matters)")
    
    return df


# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    # Generate dataset
    df = generate_realistic_data(n_scenes=10000, seed=42)
    
    # Save to CSV
    df.to_csv("data/realistic_crime_data_3.csv", index=False)
    print(f"\nData saved to realistic_crime_data_3.csv")
    
    # Show sample
    print("\nSample scenes:")
    for scene_id in df['scene_id'].unique()[:3]:
        print(f"\n--- Scene {scene_id} ---")
        scene = df[df['scene_id'] == scene_id]
        print(scene[['suspect', 'victim', 'object', 'action', 'location', 'crime_type']].to_string())