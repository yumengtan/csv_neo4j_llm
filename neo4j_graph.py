import os
import traceback
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
import logging
from neo4j import GraphDatabase
import re
import argparse
from dotenv import load_dotenv

load_dotenv()

# Neo4j Connection Settings
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
TRAINED_MODEL_PATH = os.environ.get("MODEL_PATH")

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

class CSVSchemaExtraction(BaseModel):
    """Schema extraction result from LLM analysis"""
    entities: List[str] = Field(description="List of main entity types (2-6 entities)")
    relationships: List[Dict[str, str]] = Field(description="Relationships in format [{'source': 'Entity1', 'target': 'Entity2', 'type': 'RELATIONSHIP_TYPE'}]")
    column_mappings: Dict[str, List[str]] = Field(description="Mapping of entity types to their relevant columns")
    key_columns: Dict[str, str] = Field(description="Best key column for each entity", default_factory=dict)
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the schema decisions")

class GraphDocument(BaseModel):
    """Represents extracted graph structure"""
    nodes: List[Dict[str, Any]] = Field(description="List of nodes with properties")
    relationships: List[Dict[str, Any]] = Field(description="List of relationships")
    metadata: Dict[str, Any] = Field(default_factory=dict)

def sanitize_label(label):
    """Ensure label is valid for Neo4j"""
    if not label:
        return "Unknown"
    if label and label[0].isdigit():
        label = 'n' + label
    return ''.join(c if c.isalnum() else '_' for c in label)

def sanitize_property_key(key):
    """Ensure property key is valid for Neo4j"""
    if not key:
        return "unknown"
    return re.sub(r'[^a-zA-Z0-9_]', '_', key.lower())

def filter_high_null_columns(df, null_threshold=0.5):
    """Filter out columns with more than half empty values"""
    columns_to_keep = []
    columns_filtered = []
    
    for col in df.columns:
        null_ratio = df[col].isnull().sum() / len(df) if len(df) > 0 else 0
        if null_ratio <= null_threshold:
            columns_to_keep.append(col)
        else:
            columns_filtered.append(col)
    
    if columns_filtered:
        LOGGER.info(f"Filtered {len(columns_filtered)} columns with more than half empty values: {columns_filtered}")
        LOGGER.info(f"Keeping {len(columns_to_keep)} columns for processing")
    
    return df[columns_to_keep], columns_filtered

def get_llm(trained_model_path=TRAINED_MODEL_PATH):
    """Load and configure LLM model"""
    
    model = AutoModelForCausalLM.from_pretrained(
        trained_model_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.05,
        top_p=0.85,
        top_k=40,   
        do_sample=True,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=pipe)

def connect_to_neo4j():
    """Create and return a Neo4j connection"""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def clear_database(driver):
    """Clear all data from Neo4j database"""
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")
        LOGGER.info("Database cleared.")

def analyze_key_candidates(columns, df):
    """Analyze columns to find best key candidates. Used to identify what makes each entity unique in this dataset"""
    candidates = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        total_rows = len(df)
        if total_rows == 0:
            continue
            
        unique_values = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        uniqueness_ratio = unique_values / total_rows
        completeness_ratio = 1 - (null_count / total_rows)
        base_score = (uniqueness_ratio * 0.7) + (completeness_ratio * 0.3)
        
        col_lower = col.lower()
        pattern_bonus = 0
        
        if any(pattern in col_lower for pattern in ['_id', 'id_', 'key', 'code', 'number']):
            pattern_bonus += 0.3
        elif col_lower.endswith('id') or col_lower.startswith('id'):
            pattern_bonus += 0.25
        elif 'name' in col_lower:
            pattern_bonus += 0.15
            
        final_score = min(base_score + pattern_bonus, 1.0)
        candidates.append((col, final_score))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [col for col, score in candidates]


def determine_entity_key(entity_name, row_data, columns, df, schema_hints=None):
    """Determine entity key using schema hints and data analysis"""
    if not columns:
        return f"{entity_name}_unknown"
    
    if schema_hints and 'key_columns' in schema_hints:
        suggested_key = schema_hints['key_columns'].get(entity_name)
        if suggested_key and suggested_key in row_data and pd.notna(row_data[suggested_key]):
            return str(row_data[suggested_key])
    
    key_candidates = analyze_key_candidates(columns, df)
    
    for candidate in key_candidates:
        if candidate in row_data and pd.notna(row_data[candidate]):
            return str(row_data[candidate])
    
    # Use first non-null value
    for col in columns:
        if col in row_data and pd.notna(row_data[col]):
            return str(row_data[col])
    
    # Create composite key
    values = [str(row_data[col]) for col in columns if col in row_data and pd.notna(row_data[col])]
    if values:
        return '_'.join(values) if values else f"{entity_name}_unknown"
    
    return f"{entity_name}_unknown"


class EntityStore:
    """Handles entity storage and deduplication"""
    
    def __init__(self):
        self.entities = {}
        self.entity_counts = {}
        
    def add_or_merge_entity(self, entity_type, entity_key, entity_data):
        """Add new entity or merge with existing one"""
        full_key = f"{entity_type}:{entity_key}"
        
        if full_key not in self.entities:
            entity_id = entity_key
            self.entities[full_key] = {
                'id': entity_id,
                'type': entity_type,
                'properties': {**entity_data, 'id': entity_id, 'entity_key': entity_key},
                'key': entity_key
            }
            self.entity_counts[entity_type] = self.entity_counts.get(entity_type, 0) + 1
        else:
            # Merge additional properties
            existing_entity = self.entities[full_key]
            for prop_key, prop_value in entity_data.items():
                if prop_key not in existing_entity['properties'] or not existing_entity['properties'][prop_key]:
                    existing_entity['properties'][prop_key] = prop_value
            
        return self.entities[full_key]['id']
    
    def get_all_entities(self):
        return list(self.entities.values())
    
    def get_entity_counts(self):
        return self.entity_counts.copy()

def deduplicate_relationships(relationships):
    """Remove duplicate relationships"""
    unique_rels = {}
    
    for rel in relationships:
        rel_key = f"{rel['source_id']}:{rel['type']}:{rel['target_id']}"
        if rel_key not in unique_rels:
            unique_rels[rel_key] = rel
    
    return list(unique_rels.values())

def create_csv_text_representation(df, sample_size=5):
    """Convert CSV to text representation for LLM analysis"""
    text_parts = []
    
    text_parts.append("Dataset Overview:")
    text_parts.append(f"- Total records: {len(df)}")
    text_parts.append(f"- Number of columns: {len(df.columns)}")
    text_parts.append(f"- Column names: {', '.join(df.columns)}")
    text_parts.append("")
    
    text_parts.append("Column Analysis:")
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df) if len(df) > 0 else 0
        null_count = df[col].isnull().sum()
        sample_values = df[col].dropna().unique()[:3].tolist()
        
        col_info = f"- {col}: {unique_count} unique values ({unique_ratio:.1%} unique), "
        col_info += f"{null_count} nulls, examples: {sample_values}"
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info += " [NUMERIC]"
        else:
            col_info += " [TEXT]"
            
        if unique_ratio > 0.8:
            col_info += " [HIGH_UNIQUENESS - potential key]"
        elif any(pattern in col.lower() for pattern in ['id', 'key', 'code']):
            col_info += " [ID_PATTERN - potential key]"
            
        text_parts.append(col_info)
    
    text_parts.append("")
    text_parts.append(f"Sample Data (first {min(sample_size, len(df))} records):")
    for i, (idx, row) in enumerate(df.head(sample_size).iterrows()):
        record_parts = [f"{col}='{value}'" for col, value in row.items() if pd.notna(value)]
        text_parts.append(f"Record {i+1}: " + ", ".join(record_parts))

    return "\n".join(text_parts)


def create_schema_prompt(data_text_representation):
    """Create optimized prompt for schema extraction. Confidence is now an arbitrary number thrown out by the LLM. Not sure if confidence score is needed"""
   
    column_names = []
    if 'Column names: ' in data_text_representation:
        col_line = data_text_representation.split('Column names: ')[1].split('\n')[0]
        column_names = [col.strip() for col in col_line.split(',')]
    
    return f"""You are an expert data analyst and knowledge graph designer. Analyze this data and extract a logical schema for creating a Neo4j graph database.

{data_text_representation}

## ANALYSIS GUIDELINES:

### 1. Entity Identification (2-6 entities max)
- Focus on meaningful concepts (nouns like Person, Company, Product)
- Group related attributes together logically
- Use singular, clear entity names
- Each entity should represent a distinct real-world concept
- Avoid creating entities for simple categorical values

### 2. Key Column Selection (Critical for Deduplication)
- Identify the BEST unique identifier column for each entity
- Prioritize columns with high uniqueness (>80%) and low null values
- Prefer ID-like columns (ending in 'id', 'key', 'code', 'number')
- Use name/title columns if no clear ID exists
- This choice is crucial for preventing duplicate entities

### 3. Relationship Discovery
- Create meaningful relationships: WORKS_FOR, BELONGS_TO, MANAGES, LIVES_AT,
- Focus on real-world connections between entities
- Avoid generic relationships like "RELATED_TO" or "CONNECTED_TO"
- Use verb-based relationship names in UPPERCASE

### 4. Column Mapping Strategy
- Assign columns to the entity they best describe
- Categorical/lookup data should be properties, not separate entities
- Ensure each entity has multiple meaningful properties

## REQUIREMENTS:
- Use ONLY these exact column names: {', '.join(column_names)}
- Choose 2-6 meaningful entity types maximum
- Pick the most unique/identifying column as key for each entity
- Create logical, meaningful relationships
- Column names in mappings must match CSV headers exactly

## OUTPUT FORMAT:

Return your analysis as JSON in this EXACT format:

{{
    "entities": ["Entity1", "Entity2", "Entity3"],
    "relationships": [{{"source": "Entity1", "target": "Entity2", "type": "RELATIONSHIP_TYPE"}}, {{"source": "Entity2", "target": "Entity3", "type": "ANOTHER_RELATIONSHIP"}}],
    "column_mappings": {{"Entity1": ["actual_column_name1", "actual_column_name2"], "Entity2": ["actual_column_name3", "actual_column_name4"], "Entity3": ["actual_column_name5"]}},
    "key_columns": {{"Entity1": "best_unique_column_for_entity1", "Entity2": "best_unique_column_for_entity2", "Entity3": "best_unique_column_for_entity3"}},
    "confidence": 0-1,
    "reasoning": "Explain your entity choices, key column selections, and relationship decisions"
}}

## CRITICAL NOTES:
- key_columns must specify the column with highest uniqueness for each entity
- Choose ID-pattern columns first, then name columns, then best available
- All column names must exactly match the CSV headers provided above
- Confidence should reflect how well the data fits a graph structure (0.0-1.0)
- Reasoning should explain WHY you chose specific key columns

JSON:"""

def is_valid_schema_json(parsed_json):
    """Check if JSON is a valid schema with better scoring"""
    if not isinstance(parsed_json, dict):
        print("Not a dictionary")
        return False
    
    LOGGER.info(f"Validating JSON with keys: {list(parsed_json.keys())}")
    
    
    # Core schema fields (required)
    core_fields = ['entities', 'relationships', 'column_mappings']
    missing_core = [field for field in core_fields if field not in parsed_json]
    
    if missing_core:
        print(f"Missing core fields: {missing_core}")
        return False
    
    # Check entities quality
    entities = parsed_json.get('entities', [])
    if not entities:
        print("No entities found")
        return False
        
    if any(entity in ['Entity1', 'Entity2'] for entity in entities):
        print(f"Template entities found: {entities}")
        return False
    
    print(f"Valid entities: {entities}")
    
    # Check column mappings quality
    mappings = parsed_json.get('column_mappings', {})
    if not mappings:
        print("No column mappings found")
        return False
    
    mapping_score = 0
    for entity, columns in mappings.items():
        if not columns:
            print(f"Empty columns for entity {entity}")
            return False
        if any(col in ['col1', 'col2', 'column1', 'column2'] for col in columns):
            print(f"Template columns found in {entity}: {columns}")
            return False
        mapping_score += len(columns)
    
    print(f"Valid column mappings with {mapping_score} total columns")
    
    if 'key_columns' in parsed_json and parsed_json['key_columns']:
        print(f"Has key_columns: {parsed_json['key_columns']}")
    
    return True

def extract_natural_language_analysis(response):
    """Improved natural language extraction with better column mapping detection"""
    
    result = {
        "entities": [],
        "relationships": [],
        "column_mappings": {},
        "key_columns": {},
        "confidence": 0.5,
        "reasoning": "Extracted from LLM natural language analysis"
    }
    
    try:        
        entities = []
        
        entities_json_match = re.search(r'"entities":\s*\[(.*?)\]', response, re.DOTALL)
        if entities_json_match:
            entities_text = entities_json_match.group(1)
            entities = re.findall(r'"([^"]+)"', entities_text)

        if not entities:
            entities_section = re.search(r'Entities:\s*\n((?:\s*-\s*[^\n]+\n?)+)', response, re.IGNORECASE | re.MULTILINE)
            if entities_section:
                entities_text = entities_section.group(1)
                bullet_entities = re.findall(r'-\s*([^\n]+)', entities_text)
                entities = [entity.strip() for entity in bullet_entities 
                           if entity.strip() and not any(skip in entity.lower() for skip in ['entity1', 'entity2'])]
        
        result["entities"] = entities

        relationships = []
        
        rel_json_match = re.search(r'"relationships":\s*\[(.*?)\]', response, re.DOTALL)
        if rel_json_match:
            rel_text = rel_json_match.group(1)
            # Find relationship objects in JSON
            rel_objects = re.findall(r'\{[^}]*\}', rel_text)
            for rel_obj in rel_objects:
                source_match = re.search(r'"source":\s*"([^"]+)"', rel_obj)
                target_match = re.search(r'"target":\s*"([^"]+)"', rel_obj)
                type_match = re.search(r'"type":\s*"([^"]+)"', rel_obj)
                
                if source_match and target_match and type_match:
                    relationships.append({
                        "source": source_match.group(1),
                        "target": target_match.group(1),
                        "type": type_match.group(1)
                    })
                    LOGGER.info(f"Found relationship from JSON: {source_match.group(1)} -[{type_match.group(1)}]-> {target_match.group(1)}")
        
        # If no JSON relationships, create default
        if not relationships and len(entities) >= 2:
            relationships.append({
                "source": entities[0],
                "target": entities[1],
                "type": "ASSOCIATED_WITH"
            })
        
        result["relationships"] = relationships
        
        # Extract column mappings from JSON format
        column_mappings = {}
        key_columns = {}
        
        # Try JSON format first
        mappings_json_match = re.search(r'"column_mappings":\s*\{(.*?)\}', response, re.DOTALL)
        if mappings_json_match:
            mappings_text = mappings_json_match.group(1)
            LOGGER.info(f"Found column mappings JSON text: {mappings_text}")
            
            # Extract entity: [columns] patterns
            entity_patterns = re.findall(r'"([^"]+)":\s*\[(.*?)\]', mappings_text)
            for entity, columns_str in entity_patterns:
                columns = re.findall(r'"([^"]+)"', columns_str)
                if columns:
                    column_mappings[entity] = columns
                    LOGGER.info(f"Found mapping for {entity}: {columns}")
        
        # Try key_columns from JSON
        key_json_match = re.search(r'"key_columns":\s*\{(.*?)\}', response, re.DOTALL)
        if key_json_match:
            key_text = key_json_match.group(1)
            key_patterns = re.findall(r'"([^"]+)":\s*"([^"]+)"', key_text)
            for entity, key_col in key_patterns:
                key_columns[entity] = key_col
                LOGGER.info(f"Found key column for {entity}: {key_col}")
        
        # If no key columns found, use first column from mappings
        if not key_columns and column_mappings:
            for entity, columns in column_mappings.items():
                if columns:
                    key_columns[entity] = columns[0]
        
        result["column_mappings"] = column_mappings
        result["key_columns"] = key_columns
        
        # Calculate confidence
        confidence_score = 0.3
        if len(result["entities"]) > 0:
            confidence_score += 0.2
        if len(result["relationships"]) > 0:
            confidence_score += 0.2
        if len(result["column_mappings"]) > 0:
            confidence_score += 0.3
        
        result["confidence"] = min(confidence_score, 1.0)
        
        print("Final extracted schema: ")
        print(f"  Entities: {result['entities']}")
        print(f"  Relationships: {len(result['relationships'])}")
        print(f"  Column mappings: {len(result['column_mappings'])}")
        print(f"  Key columns: {len(result['key_columns'])}")
        print(f"  Confidence: {result['confidence']}")
        
        return result
        
    except Exception as e:
        print(f"Natural language extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_llm_schema_response(response):
    """Parse LLM response with better debugging"""
    print(f"Parsing LLM response ({len(response)} chars)")
    
    # Try to extract JSON first
    json_result = extract_largest_complete_json(response)
    
    if json_result:
        print(f"Found JSON: {json_result}")
        
        if is_valid_schema_json(json_result):
            return json_result
        else:
            print("JSON validation failed")
    else:
        print("No JSON found")
    
    # Fallback
    return extract_natural_language_analysis(response)

def extract_largest_complete_json(response):
    """LLM Models spits out the whole template + json response.
    Retrieve the largest json response which will be the complete schema.
    """
    
    # Find all opening braces
    brace_positions = [i for i, char in enumerate(response) if char == '{']
    print(f"Found {len(brace_positions)} opening braces at positions: {brace_positions}")
    
    valid_json_objects = []
    
    # Try parsing from each position
    for start_pos in brace_positions:
        try:
            brace_count = 0
            end_pos = start_pos
            
            # Find matching closing brace
            for i in range(start_pos, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            
            if brace_count == 0:
                json_str = response[start_pos:end_pos + 1]
                try:
                    parsed = json.loads(json_str)
                    json_size = len(json_str)
                    
                    print(f"  Keys: {list(parsed.keys())}")
                    
                    valid_json_objects.append({
                        'position': start_pos,
                        'size': json_size,
                        'json_str': json_str,
                        'parsed': parsed
                    })
                    
                except json.JSONDecodeError:
                    pass
                
        except Exception:
            continue
    
    if not valid_json_objects:
        print("No valid JSON found")
        return None
    
    # Sort by size (largest first)
    valid_json_objects.sort(key=lambda x: (x['size']), reverse=True)
    
    for i, obj in enumerate(valid_json_objects):
        print(f"  {i+1}. Position {obj['position']}, Size {obj['size']}, Keys: {list(obj['parsed'].keys())}")
    
    # Return the largest JSON object
    chosen = valid_json_objects[0]
    print(f"Choosing largest JSON at position {chosen['position']} (size: {chosen['size']})")
    
    return chosen['parsed']


def validate_extracted_schema(schema_data, df):
    """Validate and clean extracted schema"""
    schema_data.setdefault('entities', [])
    schema_data.setdefault('relationships', [])
    schema_data.setdefault('column_mappings', {})
    schema_data.setdefault('key_columns', {})
    schema_data.setdefault('confidence', 0.5)
    schema_data.setdefault('reasoning', 'Automated extraction')
    
    # Validate entities
    if not schema_data['entities']:
        schema_data['entities'] = ['Entity']
        schema_data['confidence'] *= 0.5
    
    # Validate column mappings
    all_columns = set(df.columns)
    mapped_columns = set()
    
    for entity, columns in list(schema_data['column_mappings'].items()):
        valid_columns = [col for col in columns if col in all_columns]
        
        if not valid_columns and columns:
            invalid_columns = [col for col in columns if col not in all_columns]
            LOGGER.info(f"Entity '{entity}' has invalid columns: {invalid_columns}")
            
            # Try to find matching columns by entity name or column similarity
            for invalid_col in invalid_columns:
                # Look for columns that contain the entity name (case-insensitive)
                entity_lower = entity.lower()
                for df_col in df.columns:
                    if entity_lower in df_col.lower() or df_col.lower() in entity_lower:
                        valid_columns.append(df_col)
                        LOGGER.info(f"Mapped invalid column '{invalid_col}' to existing column '{df_col}' for entity '{entity}'")
                        break
        
        schema_data['column_mappings'][entity] = valid_columns
        mapped_columns.update(valid_columns)
        
        # Validate key column
        suggested_key = schema_data['key_columns'].get(entity)
        if suggested_key and suggested_key not in valid_columns:
            if valid_columns:
                key_candidates = analyze_key_candidates(valid_columns, df)
                new_key = key_candidates[0] if key_candidates else valid_columns[0]
                schema_data['key_columns'][entity] = new_key
                LOGGER.info(f"Updated key column for '{entity}': '{suggested_key}' → '{new_key}'")
            else:
                schema_data['key_columns'].pop(entity, None)
                LOGGER.warning(f"Removed key column for '{entity}' - no valid columns found")
        elif not suggested_key and valid_columns:
            key_candidates = analyze_key_candidates(valid_columns, df)
            schema_data['key_columns'][entity] = key_candidates[0] if key_candidates else valid_columns[0]
    
    # Remove entities with no valid columns
    entities_to_remove = []
    for entity in schema_data['entities']:
        if entity not in schema_data['column_mappings'] or not schema_data['column_mappings'][entity]:
            entities_to_remove.append(entity)
            LOGGER.warning(f"Removing entity '{entity}' - no valid columns")
    
    for entity in entities_to_remove:
        schema_data['entities'].remove(entity)
        schema_data['column_mappings'].pop(entity, None)
        schema_data['key_columns'].pop(entity, None)
    
    # Assign unmapped columns
    unmapped_columns = all_columns - mapped_columns
    if unmapped_columns and schema_data['entities']:
        first_entity = schema_data['entities'][0]
        schema_data['column_mappings'].setdefault(first_entity, []).extend(list(unmapped_columns))
        LOGGER.info(f"Assigned unmapped columns {list(unmapped_columns)} to entity '{first_entity}'")
        
        if first_entity not in schema_data['key_columns']:
            all_entity_columns = schema_data['column_mappings'][first_entity]
            key_candidates = analyze_key_candidates(all_entity_columns, df)
            if key_candidates:
                schema_data['key_columns'][first_entity] = key_candidates[0]
    
    # Validate relationships
    valid_entities = set(schema_data['entities'])
    valid_relationships = []
    
    for rel in schema_data['relationships']:
        if (isinstance(rel, dict) and 'source' in rel and 'target' in rel and 'type' in rel and
            rel['source'] in valid_entities and rel['target'] in valid_entities):
            valid_relationships.append(rel)
        else:
            LOGGER.warning(f"Removing invalid relationship: {rel}")
    
    schema_data['relationships'] = valid_relationships
    
    # Final validation: ensure all entities have at least one column
    final_entities = []
    for entity in schema_data['entities']:
        if entity in schema_data['column_mappings'] and schema_data['column_mappings'][entity]:
            final_entities.append(entity)
        else:
            LOGGER.warning(f"Final check: Removing entity '{entity}' - no columns")
    
    schema_data['entities'] = final_entities
    
    return schema_data

def extract_schema_from_text(df, llm):
    """Extract schema using LLM with improved parsing"""
    csv_text = create_csv_text_representation(df)
    prompt = create_schema_prompt(csv_text)
    
    try:
        response = llm(prompt)
        print(f"llm response: {response}")
        schema_data = parse_llm_schema_response(response)
        
        if schema_data:
            validated_schema = validate_extracted_schema(schema_data, df)
            
            if validated_schema['confidence'] > 0.3:
                print(f"  Entities: {validated_schema['entities']}")
                print(f"  Relationships: {len(validated_schema['relationships'])}")
                print(f"  Confidence: {validated_schema['confidence']}")
                return CSVSchemaExtraction(**validated_schema)
        
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        traceback.print_exc()
    
    print("Falling back method for schema creation...")
    key_candidates = analyze_key_candidates(list(df.columns), df)
    best_key = key_candidates[0] if key_candidates else df.columns[0]
    
    return CSVSchemaExtraction(
        entities=["Record"],
        relationships=[],
        column_mappings={"Record": list(df.columns)},
        key_columns={"Record": best_key},
        confidence=0.3,
        reasoning="Generic fallback - LLM extraction failed"
    )

def extract_entity_data(row, columns, available_columns):
    """Extract entity data from a row for specific columns"""
    entity_data = {}
    has_data = False
    
    for col in columns:
        if col in available_columns and pd.notna(row[col]):
            entity_data[sanitize_property_key(col)] = str(row[col])
            has_data = True
    
    return entity_data if has_data else None

def identify_reference_columns(df, schema):
    """
    Automatically detect columns that contain references to other entities.
    This is the key function that finds columns like 'Knows' that should create relationships.
    """
    reference_columns = {}
    
    for entity_name, columns in schema.column_mappings.items():
        if entity_name not in schema.key_columns:
            continue
            
        key_column = schema.key_columns[entity_name]
        
        # Get all possible values for this entity's key column (e.g., all person names)
        if key_column in df.columns:
            entity_values = set(df[key_column].dropna().astype(str))
         
            # Check ALL columns for references
            for col in df.columns:
                if col == key_column:  # Skip the key column itself
                    continue
                    
                col_values = set(df[col].dropna().astype(str))
                if not col_values:
                    continue
                
                # Calculate overlap - how many values in this column match entity names
                overlap = len(col_values.intersection(entity_values))
                overlap_ratio = overlap / len(col_values) if col_values else 0
                
             
                if overlap_ratio > 0.4:
                    # Determine relationship name from column name
                    rel_type = col.upper().replace(' ', '_').replace('-', '_')
                    
                    reference_columns[col] = {
                        'source_entity': entity_name,
                        'target_entity': entity_name, 
                        'relationship_type': rel_type,
                        'overlap_ratio': overlap_ratio,
                        'total_matches': overlap
                    }
    
    return reference_columns


def clean_schema_for_references(schema, reference_columns):
    """
    Remove reference columns from entity property mappings since they'll become relationships.
    """
    for col_name in reference_columns.keys():
        for entity_name, columns in schema.column_mappings.items():
            if col_name in columns:
                schema.column_mappings[entity_name] = [c for c in columns if c != col_name]
    
    return schema


def create_all_entities_first(df, schema, entity_store):
    """
    First pass: Create ALL entities across ALL rows.
    This ensures every person/company/location exists before we try to link them.
    """
    
    entities_created = {}
    
    for idx, row in df.iterrows():
        for entity_name, columns in schema.column_mappings.items():
            if not columns:
                continue
                
            # Extract data for this entity from this row
            entity_data = extract_entity_data(row, columns, df.columns)
            if entity_data:
                # Determine the unique key for this entity
                entity_key = determine_entity_key(
                    entity_name, row, columns, df, 
                    schema_hints={'key_columns': schema.key_columns}
                )
                
                # Add to entity store (handles deduplication automatically)
                entity_store.add_or_merge_entity(entity_name, entity_key, entity_data)
                
                # Track what we created
                if entity_name not in entities_created:
                    entities_created[entity_name] = set()
                entities_created[entity_name].add(entity_key)
    
    return entities_created


def create_schema_relationships(df, schema, entity_store):
    """
    Create relationships defined in the schema (like Person-WORKS_FOR->Company).
    These come from the same row.
    """
    
    relationships = []
    relationship_counts = {}
    
    for idx, row in df.iterrows():
        # Get entity IDs for this row
        row_entities = {}
        for entity_name, columns in schema.column_mappings.items():
            if not columns:
                continue
                
            entity_data = extract_entity_data(row, columns, df.columns)
            if entity_data:
                entity_key = determine_entity_key(
                    entity_name, row, columns, df, 
                    schema_hints={'key_columns': schema.key_columns}
                )
                full_key = f"{entity_name}:{entity_key}"
                if full_key in entity_store.entities:
                    row_entities[entity_name] = entity_store.entities[full_key]['id']
        
        # Create relationships between entities in this row
        for rel_def in schema.relationships:
            source_entity = rel_def['source']
            target_entity = rel_def['target']
            rel_type = rel_def['type']
            
            # Skip self-referential relationships (we'll handle these separately)
            if source_entity == target_entity:
                continue
                
            if source_entity in row_entities and target_entity in row_entities:
                source_id = row_entities[source_entity]
                target_id = row_entities[target_entity]
                
                relationships.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'type': rel_type,
                    'properties': {}
                })
                
                relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
    
    return relationships


def create_reference_relationships(df, reference_columns, entity_store, schema):
    """
    Create relationships from reference columns (like 'Knows' pointing to other people).
    This is where the cross-row lookup magic happens!
    """
    
    relationships = []
    relationship_counts = {}
    failed_lookups = {}
    
    for idx, row in df.iterrows():
        # For each reference column in this row
        for col_name, ref_info in reference_columns.items():
            if col_name not in row or pd.isna(row[col_name]):
                continue
                
            target_name = str(row[col_name]).strip()
            if not target_name:
                continue
            
            source_entity = ref_info['source_entity']
            target_entity = ref_info['target_entity']
            rel_type = ref_info['relationship_type']
            
            # Find the SOURCE entity (from this row)
            source_columns = schema.column_mappings.get(source_entity, [])
            if source_columns:
                source_key = determine_entity_key(
                    source_entity, row, source_columns, df,
                    schema_hints={'key_columns': schema.key_columns}
                )
                source_full_key = f"{source_entity}:{source_key}"
                
                if source_full_key in entity_store.entities:
                    source_id = entity_store.entities[source_full_key]['id']
                    
                    # Find the TARGET entity (lookup across all entities)
                    target_full_key = f"{target_entity}:{target_name}"
                    
                    if target_full_key in entity_store.entities:
                        target_id = entity_store.entities[target_full_key]['id']
                        
                        # Avoid self-loops
                        if source_id != target_id:
                            relationships.append({
                                'source_id': source_id,
                                'target_id': target_id,
                                'type': rel_type,
                                'properties': {
                                    'source_column': col_name,
                                    'original_value': target_name
                                }
                            })
                            
                            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
                    else:
                        # Track failed lookups for debugging
                        failed_lookups[target_name] = failed_lookups.get(target_name, 0) + 1
                        

    return relationships

def create_graph_documents_from_schema(df, schema: CSVSchemaExtraction):
    """Create graph documents with proper entity deduplication"""

     # Step 1: Identify reference columns (like 'Knows')
    reference_columns = identify_reference_columns(df, schema)
    
    # Step 2: Clean schema to remove reference columns from properties
    schema_cleaned = clean_schema_for_references(schema, reference_columns)
    
    # Step 3: Create entity store and create ALL entities first
    entity_store = EntityStore()
    create_all_entities_first(df, schema_cleaned, entity_store)
    
    # Step 4: Create schema relationships (same-row relationships)
    schema_relationships = create_schema_relationships(df, schema_cleaned, entity_store)
    
    # Step 5: Create reference relationships (cross-row relationships) 
    reference_relationships = create_reference_relationships(df, reference_columns, entity_store, schema_cleaned)
    
    # Step 6: Combine and deduplicate all relationships
    all_relationships = schema_relationships + reference_relationships
    unique_relationships = deduplicate_relationships(all_relationships)
    
    print(f"   👥 Total unique entities: {len(entity_store.get_all_entities())}")
    print(f"   🔗 Total unique relationships: {len(unique_relationships)}")
    
    # Create final counts
    all_entities = entity_store.get_all_entities()
    entity_counts = entity_store.get_entity_counts()
    
    relationship_counts = {}
    for rel in unique_relationships:
        rel_type = rel['type']
        relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
    
    print("Entity breakdown:")
    for entity_type, count in entity_counts.items():
        print(f"      {entity_type}: {count}")
    
    print("Relationship breakdown:")
    for rel_type, count in relationship_counts.items():
        print(f"      {rel_type}: {count}")
    
    # Create graph document
    graph_document = GraphDocument(
        nodes=all_entities,
        relationships=unique_relationships,
        metadata={
            'total_rows_processed': len(df),
            'unique_entities': len(all_entities),
            'total_relationships': len(unique_relationships),
            'schema_confidence': schema.confidence,
            'entity_counts': entity_counts,
            'relationship_counts': relationship_counts,
            'reference_columns_detected': len(reference_columns),
            'reference_columns': list(reference_columns.keys())
        }
    )
    
    return [graph_document]

def create_constraints_from_schema(driver, schema: CSVSchemaExtraction):
    """Create uniqueness constraints for entities"""
    with driver.session(database=NEO4J_DATABASE) as session:
        for entity in schema.entities:
            entity_label = sanitize_label(entity)
            query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{entity_label}) REQUIRE n.id IS UNIQUE"
            session.run(query)

def save_graph_documents_to_neo4j_batch(driver, graph_documents, batch_size=1000):
    """Save graph documents in batches"""
    if len(graph_documents) == 1:
        graph_doc = graph_documents[0]
        LOGGER.info(f"Processing graph document with {len(graph_doc.nodes)} unique nodes and {len(graph_doc.relationships)} relationships")
        
        if 'entity_counts' in graph_doc.metadata:
            LOGGER.info("Entity breakdown:")
            for entity_type, count in graph_doc.metadata['entity_counts'].items():
                LOGGER.info(f"  {entity_type}: {count} entities")
        
        if 'relationship_counts' in graph_doc.metadata:
            LOGGER.info("Relationship breakdown:")
            for rel_type, count in graph_doc.metadata['relationship_counts'].items():
                LOGGER.info(f"  {rel_type}: {count} relationships")
    
    # Collect all unique nodes
    unique_nodes = {}
    for doc in graph_documents:
        for node in doc.nodes:
            unique_nodes[node['id']] = node
    
    nodes_list = list(unique_nodes.values())
    nodes_created_by_type = {}
    
    # Create nodes in batches
    with driver.session(database=NEO4J_DATABASE) as session:
        for i in range(0, len(nodes_list), batch_size):
            batch_nodes = nodes_list[i:i + batch_size]
            
            for node in batch_nodes:
                entity_label = sanitize_label(node['type'])
                properties = node['properties']
                
                query = f"""
                MERGE (n:{entity_label} {{id: $id}})
                SET n += $properties
                """
                session.run(query, {"id": properties['id'], "properties": properties})
                
                nodes_created_by_type[entity_label] = nodes_created_by_type.get(entity_label, 0) + 1
    
    # Collect unique relationships
    unique_relationships = {}
    for doc in graph_documents:
        for rel in doc.relationships:
            rel_key = f"{rel['source_id']}:{rel['type']}:{rel['target_id']}"
            if rel_key not in unique_relationships:
                unique_relationships[rel_key] = rel
    
    all_relationships = list(unique_relationships.values())
    relationships_created_by_type = {}
    
    # Create relationships
    with driver.session(database=NEO4J_DATABASE) as session:
        for i in range(0, len(all_relationships), batch_size):
            batch_rels = all_relationships[i:i + batch_size]
            
            for rel in batch_rels:
                rel_type = sanitize_label(rel['type'])
                
                query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                SET r += $properties
                """
                session.run(query, {
                    "source_id": rel['source_id'],
                    "target_id": rel['target_id'],
                    "properties": rel.get('properties', {})
                })
                
                relationships_created_by_type[rel_type] = relationships_created_by_type.get(rel_type, 0) + 1
    
    LOGGER.info("Relationships created by type:")
    for rel_type, count in relationships_created_by_type.items():
        LOGGER.info(f"  {rel_type}: {count}")
    
    return nodes_created_by_type, relationships_created_by_type


def load_data_file(file_path):
    """Load CSV or Excel file into DataFrame"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    
def process_file(file, batch_size=2000):
    """Main function to process file"""
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        
        df = load_data_file(file)
        LOGGER.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Filter columns with >50% null values
        df_filtered, filtered_columns = filter_high_null_columns(df, null_threshold=0.5)
        
        if len(df_filtered.columns) == 0:
            raise ValueError("No columns remaining after filtering high-null columns")
        
        if len(df_filtered) > 0:
            LOGGER.info("Sample data (first row):")
            for col, val in df_filtered.iloc[0].items():
                LOGGER.info(f"  {col}: {val}")

        try:
            llm = get_llm()
        except Exception as e:
            LOGGER.error(f"Failed to load LLM model: {str(e)}")
            raise
        
        try:
            schema = extract_schema_from_text(df_filtered, llm)
        except Exception as e:
            LOGGER.error(f"Schema extraction failed: {str(e)}")
            schema = None
        
        try:
            graph_documents = create_graph_documents_from_schema(df_filtered, schema)
        except Exception as e:
            LOGGER.error(f"Graph document creation failed: {str(e)}")
            raise
        
        try:
            driver = connect_to_neo4j()
        except Exception as e:
            LOGGER.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
        
        try:
            # clear database before run
            clear_database(driver)  
            create_constraints_from_schema(driver, schema)
            
            nodes_by_type, rels_by_type = save_graph_documents_to_neo4j_batch(driver, graph_documents, batch_size)
            
            LOGGER.info(f"Total entities created: {sum(nodes_by_type.values())}")
            LOGGER.info(f"Total relationships created: {sum(rels_by_type.values())}")
            
        finally:
            if 'driver' in locals():
                driver.close()
                LOGGER.info("Neo4j connection closed")
            
    except Exception as e:
        LOGGER.error(f"Processing failed: {str(e)}")
        LOGGER.error(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description="Process structured data in csv/excel files into neo4j knowledge graph")
    parser.add_argument("file", help="Path to CSV or Excel file")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for processing")
    
    args = parser.parse_args()
    
    process_file(args.file, args.batch_size)
    
if __name__ == "__main__":
    main()