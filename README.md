# CSV to Neo4j Knowledge Graph Extraction

A Python tool that automatically converts structured data (CSV/Excel files) into Neo4j knowledge graphs using large language models for intelligent schema extraction. Currently it connects to local neo4j db. If you want to change to neo4j Aura DB, change the NEO4J URI in .env file. 

## Overview

This tool analyzes tabular data and automatically:
- Identifies entities and relationships within the data
- Creates a logical graph schema using LLM analysis
- Detects cross-references between entities (e.g., "Person A knows Person B")
- Builds a Neo4j knowledge graph with proper deduplication
- Handles complex relationship mappings across rows

## Key Features

- **Intelligent Schema Extraction**: Uses LLM to analyze data structure and suggest optimal entity-relationship models
- **Cross-Reference Detection**: Automatically finds columns that reference other entities
- **Entity Deduplication**: Ensures unique entities across the entire dataset
- **Relationship Discovery**: Creates both same-row and cross-row relationships
- **Constraint Management**: Automatically creates Neo4j constraints for data integrity
- **Batch Processing**: Handles large datasets efficiently

## Requirements

### Dependencies
```python
pandas
torch
transformers
langchain_huggingface
pydantic
neo4j
python-dotenv
```

### Environment Variables
Create a `.env` file with:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j
MODEL_PATH=<model path>
```

## Usage

### Command Line
```bash
python script.py data.csv --batch-size 2000
```

### Programmatic Usage
```python
from script import process_file

# Process a CSV file
process_file("data.csv", batch_size=2000)
```

## How It Works: Complete Flow

### Overall Architecture

The system follows a pipeline approach with distinct phases:

```
CSV/Excel → Data Loading → Schema Extraction → Graph Construction → Neo4j Storage
```

### Main Entry Point: `process_file()`

**Purpose**: Orchestrates the entire pipeline from file input to Neo4j storage

**Flow**:
1. **File Validation**: Checks if file exists and is readable
2. **Data Loading**: Loads CSV/Excel using appropriate pandas method
3. **Data Filtering**: Removes columns with >50% null values
4. **LLM Initialization**: Loads the language model for schema extraction
5. **Schema Extraction**: Analyzes data and creates entity-relationship schema
6. **Graph Construction**: Converts data into graph documents
7. **Neo4j Connection**: Establishes database connection
8. **Database Operations**: Clears database, creates constraints, saves data
9. **Cleanup**: Closes connections and logs statistics

```python
def process_file(file, batch_size=2000):
    # 1. File validation and loading
    df = load_data_file(file)
    
    # 2. Data preprocessing
    df_filtered, filtered_columns = filter_high_null_columns(df, null_threshold=0.5)
    
    # 3. LLM setup
    llm = get_llm()
    
    # 4. Schema extraction
    schema = extract_schema_from_text(df_filtered, llm)
    
    # 5. Graph construction
    graph_documents = create_graph_documents_from_schema(df_filtered, schema)
    
    # 6. Neo4j operations
    driver = connect_to_neo4j()
    clear_database(driver)
    create_constraints_from_schema(driver, schema)
    save_graph_documents_to_neo4j_batch(driver, graph_documents, batch_size)
```

## Core Data Models

### CSVSchemaExtraction
**Purpose**: Structured representation of the extracted schema
```python
class CSVSchemaExtraction(BaseModel):
    entities: List[str]                    # ["Person", "Company"]
    relationships: List[Dict[str, str]]    # [{"source": "Person", "target": "Company", "type": "WORKS_FOR"}]
    column_mappings: Dict[str, List[str]]  # {"Person": ["name", "age"], "Company": ["company_name"]}
    key_columns: Dict[str, str]            # {"Person": "name", "Company": "company_name"}
    confidence: float                      # 0.0-1.0 confidence score
    reasoning: str                         # LLM's explanation
```

### GraphDocument
**Purpose**: Final graph structure ready for Neo4j
```python
class GraphDocument(BaseModel):
    nodes: List[Dict[str, Any]]           # [{"id": "alice", "type": "Person", "properties": {...}}]
    relationships: List[Dict[str, Any]]   # [{"source_id": "alice", "target_id": "acme", "type": "WORKS_FOR"}]
    metadata: Dict[str, Any]              # Statistics and processing info
```

## Phase 1: Data Loading and Preprocessing

### `load_data_file(file_path)`

**Purpose**: Loads CSV or Excel files into pandas DataFrame

**Flow**:
1. **File Extension Detection**: Checks `.csv`, `.xlsx`, `.xls`
2. **Format-Specific Loading**: Uses `pd.read_csv()` or `pd.read_excel()`
3. **Error Handling**: Raises `ValueError` for unsupported formats

```python
def load_data_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
```

### `filter_high_null_columns(df, null_threshold=0.5)`

**Purpose**: Removes columns with excessive null values to improve schema quality

**Flow**:
1. **Null Ratio Calculation**: For each column, calculates `null_count / total_rows`
2. **Threshold Filtering**: Keeps columns with null ratio ≤ 0.5
3. **Logging**: Reports filtered and kept columns
4. **Return**: Filtered DataFrame and list of removed columns

```python
def filter_high_null_columns(df, null_threshold=0.5):
    columns_to_keep = []
    columns_filtered = []
    
    for col in df.columns:
        null_ratio = df[col].isnull().sum() / len(df) if len(df) > 0 else 0
        if null_ratio <= null_threshold:
            columns_to_keep.append(col)
        else:
            columns_filtered.append(col)
    
    return df[columns_to_keep], columns_filtered
```

## Phase 2: Schema Extraction

### `create_csv_text_representation(df, sample_size=5)`

**Purpose**: Converts DataFrame to text format optimized for LLM analysis

**Flow**:
1. **Dataset Overview**: Creates summary with row count, column count, column names
2. **Column Analysis**: For each column:
   - Counts unique values and calculates uniqueness ratio
   - Counts null values
   - Extracts sample values
   - Identifies data types (numeric/text)
   - Flags potential key columns (high uniqueness, ID patterns)
3. **Sample Data**: Shows first N records in key=value format
4. **Text Assembly**: Combines all sections into structured text

```python
def create_csv_text_representation(df, sample_size=5):
    text_parts = []
    
    # Dataset overview
    text_parts.append(f"Total records: {len(df)}")
    text_parts.append(f"Columns: {', '.join(df.columns)}")
    
    # Column analysis
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)
        null_count = df[col].isnull().sum()
        sample_values = df[col].dropna().unique()[:3].tolist()
        
        col_info = f"{col}: {unique_count} unique ({unique_ratio:.1%}), {null_count} nulls"
        if unique_ratio > 0.8:
            col_info += " [POTENTIAL_KEY]"
        text_parts.append(col_info)
    
    # Sample records
    for i, (idx, row) in enumerate(df.head(sample_size).iterrows()):
        record = [f"{col}='{value}'" for col, value in row.items() if pd.notna(value)]
        text_parts.append(f"Record {i+1}: " + ", ".join(record))
    
    return "\n".join(text_parts)
```

### `create_schema_prompt(data_text_representation)`

**Purpose**: Creates optimized prompt for LLM schema extraction

**Flow**:
1. **Column Name Extraction**: Parses column names from data representation
2. **Instruction Assembly**: Combines analysis guidelines with data
3. **Format Specification**: Defines exact JSON output format required
4. **Constraint Definition**: Lists rules for entity identification and relationships

**Key Prompt Sections**:
- **Entity Guidelines**: Focus on 2-6 meaningful concepts
- **Key Column Selection**: Prioritize unique identifiers
- **Relationship Discovery**: Create meaningful verb-based relationships
- **Output Format**: Exact JSON schema specification

### `extract_schema_from_text(df, llm)`

**Purpose**: Main schema extraction orchestrator

**Flow**:
1. **Text Preparation**: Converts DataFrame to LLM-readable format
2. **Prompt Generation**: Creates structured prompt with guidelines
3. **LLM Invocation**: Sends prompt to language model
4. **Response Parsing**: Extracts and validates JSON schema
5. **Schema Validation**: Ensures schema matches actual data
6. **Fallback Handling**: Creates generic schema if extraction fails

```python
def extract_schema_from_text(df, llm):
    # 1. Create text representation
    csv_text = create_csv_text_representation(df)
    
    # 2. Generate prompt
    prompt = create_schema_prompt(csv_text)
    
    # 3. Get LLM response
    response = llm(prompt)
    
    # 4. Parse response
    schema_data = parse_llm_schema_response(response)
    
    # 5. Validate and clean
    if schema_data:
        validated_schema = validate_extracted_schema(schema_data, df)
        if validated_schema['confidence'] > 0.3:
            return CSVSchemaExtraction(**validated_schema)
    
    # 6. Fallback
    return create_fallback_schema(df)
```

### `parse_llm_schema_response(response)`

**Purpose**: Extracts structured schema from LLM's text response

**Flow**:
1. **JSON Extraction**: Uses `extract_largest_complete_json()` to find JSON blocks
2. **JSON Validation**: Validates structure with `is_valid_schema_json()`
3. **Fallback Parsing**: If JSON fails, uses `extract_natural_language_analysis()`
4. **Quality Assessment**: Ensures extracted schema meets minimum requirements

### `extract_largest_complete_json(response)`

**Purpose**: Finds and extracts the largest valid JSON object from LLM response

**Flow**:
1. **Brace Detection**: Locates all opening braces `{` in response
2. **Bracket Matching**: For each opening brace, finds matching closing brace
3. **JSON Parsing**: Attempts to parse each potential JSON block
4. **Size Ranking**: Sorts valid JSON objects by size
5. **Best Selection**: Returns the largest valid JSON object

```python
def extract_largest_complete_json(response):
    brace_positions = [i for i, char in enumerate(response) if char == '{']
    valid_json_objects = []
    
    for start_pos in brace_positions:
        brace_count = 0
        # Find matching closing brace
        for i in range(start_pos, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = response[start_pos:i + 1]
                    try:
                        parsed = json.loads(json_str)
                        valid_json_objects.append({
                            'size': len(json_str),
                            'parsed': parsed
                        })
                    except json.JSONDecodeError:
                        pass
                    break
    
    # Return largest valid JSON
    if valid_json_objects:
        return max(valid_json_objects, key=lambda x: x['size'])['parsed']
    return None
```

### `validate_extracted_schema(schema_data, df)`

**Purpose**: Validates and repairs extracted schema against actual data

**Flow**:
1. **Structure Validation**: Ensures all required fields exist
2. **Column Validation**: Checks that mapped columns exist in DataFrame
3. **Entity Cleanup**: Removes entities with no valid columns
4. **Key Column Validation**: Ensures key columns are valid and optimal
5. **Relationship Validation**: Verifies relationships reference valid entities
6. **Unmapped Column Handling**: Assigns orphaned columns to entities

```python
def validate_extracted_schema(schema_data, df):
    # 1. Ensure required fields
    schema_data.setdefault('entities', [])
    schema_data.setdefault('column_mappings', {})
    
    # 2. Validate column mappings
    all_columns = set(df.columns)
    for entity, columns in list(schema_data['column_mappings'].items()):
        valid_columns = [col for col in columns if col in all_columns]
        schema_data['column_mappings'][entity] = valid_columns
    
    # 3. Remove empty entities
    entities_to_remove = [entity for entity in schema_data['entities'] 
                         if not schema_data['column_mappings'].get(entity)]
    for entity in entities_to_remove:
        schema_data['entities'].remove(entity)
        schema_data['column_mappings'].pop(entity, None)
    
    # 4. Validate key columns
    for entity, columns in schema_data['column_mappings'].items():
        if entity not in schema_data.get('key_columns', {}):
            key_candidates = analyze_key_candidates(columns, df)
            if key_candidates:
                schema_data.setdefault('key_columns', {})[entity] = key_candidates[0]
    
    return schema_data
```

## Phase 3: Graph Construction

### `create_graph_documents_from_schema(df, schema)`

**Purpose**: Main orchestrator for converting schema and data into graph structure

**Flow**:
1. **Reference Detection**: Identifies columns that reference other entities
2. **Schema Cleaning**: Removes reference columns from property mappings
3. **Entity Creation**: Creates all entities across all rows first
4. **Schema Relationships**: Creates same-row relationships
5. **Reference Relationships**: Creates cross-row relationships
6. **Deduplication**: Removes duplicate relationships
7. **Graph Document Assembly**: Combines everything into final structure

```python
def create_graph_documents_from_schema(df, schema):
    # 1. Detect reference columns
    reference_columns = identify_reference_columns(df, schema)
    
    # 2. Clean schema
    schema_cleaned = clean_schema_for_references(schema, reference_columns)
    
    # 3. Create entity store and all entities
    entity_store = EntityStore()
    create_all_entities_first(df, schema_cleaned, entity_store)
    
    # 4. Create relationships
    schema_relationships = create_schema_relationships(df, schema_cleaned, entity_store)
    reference_relationships = create_reference_relationships(df, reference_columns, entity_store, schema_cleaned)
    
    # 5. Combine and deduplicate
    all_relationships = schema_relationships + reference_relationships
    unique_relationships = deduplicate_relationships(all_relationships)
    
    # 6. Create graph document
    return GraphDocument(
        nodes=entity_store.get_all_entities(),
        relationships=unique_relationships,
        metadata={...}
    )
```

### `identify_reference_columns(df, schema)`

**Purpose**: Automatically detects columns containing references to other entities

**Flow**:
1. **Entity Value Collection**: For each entity, collects all possible key values
2. **Column Analysis**: For each column in DataFrame:
   - Extracts unique values from column
   - Calculates overlap with entity key values
   - Computes overlap ratio
3. **Reference Detection**: Columns with >40% overlap are marked as references
4. **Relationship Naming**: Converts column names to relationship types

```python
def identify_reference_columns(df, schema):
    reference_columns = {}
    
    for entity_name, columns in schema.column_mappings.items():
        key_column = schema.key_columns.get(entity_name)
        if not key_column:
            continue
            
        # Get all entity values
        entity_values = set(df[key_column].dropna().astype(str))
        
        # Check all columns for references
        for col in df.columns:
            if col == key_column:
                continue
                
            col_values = set(df[col].dropna().astype(str))
            overlap = len(col_values.intersection(entity_values))
            overlap_ratio = overlap / len(col_values) if col_values else 0
            
            if overlap_ratio > 0.4:  # 40% threshold
                reference_columns[col] = {
                    'source_entity': entity_name,
                    'target_entity': entity_name,
                    'relationship_type': col.upper().replace(' ', '_'),
                    'overlap_ratio': overlap_ratio
                }
    
    return reference_columns
```

### `EntityStore` Class

**Purpose**: Manages entity creation and deduplication

**Key Methods**:

#### `add_or_merge_entity(entity_type, entity_key, entity_data)`
**Flow**:
1. **Key Generation**: Creates full key as `"EntityType:entity_key"`
2. **Existence Check**: Checks if entity already exists
3. **New Entity**: If not exists, creates new entity with properties
4. **Merge Existing**: If exists, merges new properties with existing
5. **Return ID**: Returns entity ID for relationship creation

```python
def add_or_merge_entity(self, entity_type, entity_key, entity_data):
    full_key = f"{entity_type}:{entity_key}"
    
    if full_key not in self.entities:
        # Create new entity
        self.entities[full_key] = {
            'id': entity_key,
            'type': entity_type,
            'properties': {**entity_data, 'id': entity_key}
        }
    else:
        # Merge properties
        existing = self.entities[full_key]
        for prop_key, prop_value in entity_data.items():
            if prop_key not in existing['properties']:
                existing['properties'][prop_key] = prop_value
    
    return self.entities[full_key]['id']
```

### `create_all_entities_first(df, schema, entity_store)`

**Purpose**: First pass - creates ALL entities before any relationships

**Flow**:
1. **Row Iteration**: Processes each row in DataFrame
2. **Entity Extraction**: For each entity type in schema:
   - Extracts relevant column data for this entity
   - Determines unique entity key
   - Adds/merges entity in store
3. **Deduplication**: EntityStore automatically handles duplicates
4. **Tracking**: Maintains counts of entities created

```python
def create_all_entities_first(df, schema, entity_store):
    for idx, row in df.iterrows():
        for entity_name, columns in schema.column_mappings.items():
            # Extract entity data from this row
            entity_data = extract_entity_data(row, columns, df.columns)
            if entity_data:
                # Determine unique key
                entity_key = determine_entity_key(entity_name, row, columns, df, schema)
                # Add to store (handles deduplication)
                entity_store.add_or_merge_entity(entity_name, entity_key, entity_data)
```

### `determine_entity_key(entity_name, row_data, columns, df, schema_hints)`

**Purpose**: Determines the best unique identifier for an entity

**Flow**:
1. **Schema Hint Check**: Uses suggested key column from schema if available
2. **Key Candidate Analysis**: Analyzes columns for uniqueness and ID patterns
3. **Priority Selection**: Prefers ID-like columns, then name columns
4. **Fallback Strategy**: Uses first non-null value or creates composite key
5. **Default Handling**: Creates unknown key if all else fails

```python
def determine_entity_key(entity_name, row_data, columns, df, schema_hints=None):
    # 1. Try schema hint
    if schema_hints and 'key_columns' in schema_hints:
        suggested_key = schema_hints['key_columns'].get(entity_name)
        if suggested_key and suggested_key in row_data and pd.notna(row_data[suggested_key]):
            return str(row_data[suggested_key])
    
    # 2. Try key candidates
    key_candidates = analyze_key_candidates(columns, df)
    for candidate in key_candidates:
        if candidate in row_data and pd.notna(row_data[candidate]):
            return str(row_data[candidate])
    
    # 3. Fallback strategies
    # ... (composite key, unknown key)
```

### `analyze_key_candidates(columns, df)`

**Purpose**: Analyzes columns to find best unique identifiers

**Flow**:
1. **Uniqueness Analysis**: Calculates uniqueness ratio for each column
2. **Completeness Analysis**: Calculates non-null ratio
3. **Pattern Bonus**: Adds scoring bonus for ID-like patterns
4. **Final Scoring**: Combines metrics with weights
5. **Ranking**: Returns columns sorted by score (best first)

```python
def analyze_key_candidates(columns, df):
    candidates = []
    
    for col in columns:
        unique_ratio = df[col].nunique() / len(df)
        completeness_ratio = 1 - (df[col].isnull().sum() / len(df))
        base_score = (unique_ratio * 0.7) + (completeness_ratio * 0.3)
        
        # Pattern bonus
        pattern_bonus = 0
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ['_id', 'id_', 'key', 'code']):
            pattern_bonus += 0.3
        elif 'name' in col_lower:
            pattern_bonus += 0.15
        
        final_score = min(base_score + pattern_bonus, 1.0)
        candidates.append((col, final_score))
    
    return [col for col, score in sorted(candidates, key=lambda x: x[1], reverse=True)]
```

### `create_schema_relationships(df, schema, entity_store)`

**Purpose**: Creates relationships between entities in the same row

**Flow**:
1. **Row Processing**: For each row, identifies all entities present
2. **Entity Lookup**: Gets entity IDs from EntityStore
3. **Relationship Creation**: For each relationship in schema:
   - Checks if both source and target entities exist in this row
   - Creates relationship record
   - Tracks relationship counts
4. **Same-Row Only**: Only creates relationships within single rows

### `create_reference_relationships(df, reference_columns, entity_store, schema)`

**Purpose**: Creates relationships from reference columns (cross-row references)

**Flow**:
1. **Row Iteration**: Processes each row for reference columns
2. **Reference Processing**: For each reference column value:
   - Identifies source entity in current row
   - Looks up target entity across all entities
   - Creates cross-row relationship
3. **Entity Lookup**: Uses EntityStore to find referenced entities
4. **Self-Loop Prevention**: Avoids creating relationships from entity to itself
5. **Failed Lookup Tracking**: Logs references that couldn't be resolved

```python
def create_reference_relationships(df, reference_columns, entity_store, schema):
    relationships = []
    
    for idx, row in df.iterrows():
        for col_name, ref_info in reference_columns.items():
            if pd.isna(row[col_name]):
                continue
                
            target_name = str(row[col_name]).strip()
            source_entity = ref_info['source_entity']
            target_entity = ref_info['target_entity']
            
            # Find source entity in this row
            source_key = determine_entity_key(source_entity, row, ...)
            source_full_key = f"{source_entity}:{source_key}"
            
            # Find target entity across all entities
            target_full_key = f"{target_entity}:{target_name}"
            
            if (source_full_key in entity_store.entities and 
                target_full_key in entity_store.entities):
                relationships.append({
                    'source_id': entity_store.entities[source_full_key]['id'],
                    'target_id': entity_store.entities[target_full_key]['id'],
                    'type': ref_info['relationship_type']
                })
    
    return relationships
```

## Phase 4: Neo4j Integration

### `connect_to_neo4j()`

**Purpose**: Establishes connection to Neo4j database

**Flow**:
1. **Driver Creation**: Uses environment variables for connection
2. **Authentication**: Applies username/password from environment
3. **Connection Return**: Returns driver object for session management

### `clear_database(driver)`

**Purpose**: Removes all existing data from Neo4j database

**Flow**:
1. **Session Creation**: Opens database session
2. **Delete Query**: Runs `MATCH (n) DETACH DELETE n`
3. **Cleanup**: Removes all nodes and relationships

### `create_constraints_from_schema(driver, schema)`

**Purpose**: Creates uniqueness constraints for entity IDs

**Flow**:
1. **Entity Iteration**: For each entity type in schema
2. **Label Sanitization**: Ensures valid Neo4j label names
3. **Constraint Creation**: Creates `UNIQUE` constraint on `id` property
4. **Conditional Creation**: Uses `IF NOT EXISTS` to avoid conflicts

```python
def create_constraints_from_schema(driver, schema):
    with driver.session(database=NEO4J_DATABASE) as session:
        for entity in schema.entities:
            entity_label = sanitize_label(entity)
            query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{entity_label}) REQUIRE n.id IS UNIQUE"
            session.run(query)
```

### `save_graph_documents_to_neo4j_batch(driver, graph_documents, batch_size)`

**Purpose**: Efficiently saves large datasets to Neo4j using batch processing

**Flow**:
1. **Node Collection**: Gathers all unique nodes from all graph documents
2. **Node Creation**: Processes nodes in batches:
   - Uses `MERGE` for upsert behavior
   - Sets all properties at once
   - Tracks creation counts by type
3. **Relationship Collection**: Gathers all unique relationships
4. **Relationship Creation**: Processes relationships in batches:
   - Matches source and target nodes
   - Uses `MERGE` to avoid duplicates
   - Sets relationship properties
5. **Statistics Logging**: Reports final counts and performance metrics

```python
def save_graph_documents_to_neo4j_batch(driver, graph_documents, batch_size=1000):
    # 1. Collect unique nodes
    unique_nodes = {}
    for doc in graph_documents:
        for node in doc.nodes:
            unique_nodes[node['id']] = node
    
    # 2. Create nodes in batches
    nodes_list = list(unique_nodes.values())
    with driver.session(database=NEO4J_DATABASE) as session:
        for i in range(0, len(nodes_list), batch_size):
            batch_nodes = nodes_list[i:i + batch_size]
            for node in batch_nodes:
                query = f"""
                MERGE (n:{sanitize_label(node['type'])} {{id: $id}})
                SET n += $properties
                """
                session.run(query, {
                    "id": node['properties']['id'], 
                    "properties": node['properties']
                })
    
    # 3. Create relationships similarly...
```

## Utility Functions

### `sanitize_label(label)` & `sanitize_property_key(key)`

**Purpose**: Ensures Neo4j compatibility for labels and property names

**Flow**:
1. **Null Handling**: Provides defaults for empty values
2. **Character Filtering**: Removes/replaces invalid characters
3. **Digit Prefix**: Handles labels starting with numbers
4. **Case Normalization**: Standardizes property key casing

### `deduplicate_relationships(relationships)`

**Purpose**: Removes duplicate relationships using composite keys

**Flow**:
1. **Key Generation**: Creates unique key from `source_id:type:target_id`
2. **Deduplication**: Keeps only first occurrence of each key
3. **List Return**: Returns deduplicated relationship list

## Error Handling and Fallbacks

### Fallback Schema Creation
When LLM extraction fails:
```python
def create_fallback_schema(df):
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
```

### Natural Language Parsing
When JSON extraction fails, the system falls back to natural language parsing using regex patterns to extract entities, relationships, and mappings from free-form text.

## Performance Considerations

### Batch Processing
- **Node Creation**: Processes nodes in configurable batches (default 1000)
- **Relationship Creation**: Handles relationships separately to manage memory
- **Memory Management**: Processes large datasets without loading everything into memory

### Database Optimization
- **Constraints**: Creates unique constraints before data loading
- **MERGE Operations**: Uses MERGE instead of CREATE to handle duplicates
- **Session Management**: Properly manages Neo4j sessions and connections

This comprehensive flow documentation shows how each function contributes to the overall pipeline, making it easier to understand, debug, and extend the system.
