# CSV to Neo4j Knowledge Graph Extraction

A Python script that automatically converts structured data (CSV/Excel files) into Neo4j knowledge graphs using large language models. Transforms structured tabular data into rich, interconnected graph databases with intelligent entity and relationship detection.

![Knowledge Graph Visualization](images/kg.PNG)

## Overview

This tool analyzes tabular data and automatically:
- **Identifies entities and relationships** within your data using LLM intelligence
- **Creates logical graph schemas** by understanding data structure and patterns  
- **Detects cross-references** between entities (e.g., "Person A knows Person B")
- **Builds Neo4j knowledge graphs** with proper entity deduplication
- **Handles complex relationship mappings** across different rows and tables

![Entity Linking Example](images/link.PNG)

## Key Features

- **Intelligent Schema Extraction**: Uses LLM to analyze data structure and suggest optimal entity-relationship models
- **Cross-Reference Detection**: Automatically finds columns that reference other entities
- **Entity Deduplication**: Ensures unique entities across the entire dataset
- **Relationship Discovery**: Creates both same-row and cross-row relationships
- **Constraint Management**: Automatically creates Neo4j constraints for data integrity
- **Batch Processing**: Handles large datasets efficiently

## Quick Start

### Requirements

Create a `.env` file with your configuration:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your_password>
NEO4J_DATABASE=neo4j
MODEL_PATH=<path_to_your_llm_model>
```
If you are connecting via Aura DB, Add this import:
`pip install graphdatascience`
Then add the following into the script and change your .env file as follows:
`from graphdatascience import GraphDataScience`
```
AURA_CONNECTION_URI = "neo4j+s://xxxxxxxx.databases.neo4j.io"
AURA_USERNAME = "neo4j"
AURA_PASSWORD = "..."

# Client instantiation
gds = GraphDataScience(
    AURA_CONNECTION_URI,
    auth=(AURA_USERNAME, AURA_PASSWORD),
    aura_ds=True
)
```

**Note**: Currently configured for Neo4j Desktop. For Neo4j Aura DB, update the URI and install `pip install graphdatascience`. For reference, https://neo4j.com/docs/aura/classic/aurads/connecting/python/

### Usage

**Command Line:**
```bash
python3 neo4j_graph.py your_data.csv --batch-size 2000
```

**Programmatic:**
```python
from script import process_file
process_file("your_data.csv", batch_size=2000)
```

## How It Works

The system follows a streamlined 4-phase pipeline:

```
CSV/Excel → Data Analysis → LLM Schema → Graph Build → Neo4j Storage
```

### Phase 1: Smart Data Loading
- **File Detection**: Automatically handles CSV and Excel formats
- **Quality Filtering**: Removes columns with >50% missing values
- **Data Profiling**: Analyzes column types, uniqueness, and patterns

### Phase 2: LLM-Powered Schema Extraction
- **Intelligent Analysis**: LLM examines your data structure and suggests entities
- **Relationship Discovery**: Identifies logical connections between different data elements
- **Key Column Detection**: Automatically finds the best unique identifiers
- **Schema Validation**: Ensures extracted schema matches your actual data

### Phase 3: Graph Construction
- **Entity Creation**: Builds unique entities across all rows with smart deduplication
- **Reference Detection**: Finds columns that reference other entities (like "Manager" pointing to other employees)
- **Relationship Mapping**: Creates two types of relationships:
  - **Same-row relationships**: Connections within a single record
  - **Cross-row relationships**: References between different records

### Phase 4: Neo4j Integration
- **Database Preparation**: Clears existing data and creates constraints
- **Batch Processing**: Efficiently uploads nodes and relationships
- **Performance Optimization**: Uses MERGE operations to handle duplicates

## Core Data Models

### Schema Structure
The LLM generates a structured schema containing:
- **Entities**: Main concepts in your data (Person, Company, Product)
- **Relationships**: How entities connect (WORKS_FOR, KNOWS, LOCATED_IN)
- **Column Mappings**: Which columns belong to which entities
- **Key Columns**: Best unique identifiers for each entity
- **Confidence Score**: How well the data fits a graph structure

### Graph Output
The final graph contains:
- **Nodes**: Unique entities with properties from your original data
- **Relationships**: Typed connections between entities
- **Metadata**: Processing statistics and quality metrics

## Key Algorithms

### Smart Entity Detection
The system identifies entities by:
1. Analyzing column patterns and data types
2. Looking for high-uniqueness columns as potential keys
3. Grouping related attributes together logically
4. Ensuring each entity represents a meaningful real-world concept

### Cross-Reference Discovery
Automatically detects reference columns by:
1. Comparing column values with entity identifiers
2. Calculating overlap ratios between columns
3. Identifying columns with >40% matches to entity keys
4. Converting column names to relationship types

### Entity Deduplication
Uses an intelligent EntityStore that:
1. Creates unique keys combining entity type and identifier
2. Merges additional properties when entities already exist
3. Ensures the same person/company/item appears only once
4. Maintains referential integrity across the entire dataset

## Example Workflow

Given a CSV with employee data:
```
Name,Company,Role,Manager,Location
Alice,TechCorp,Engineer,Bob,NYC
Bob,TechCorp,Manager,,NYC
Carol,DataInc,Analyst,David,LA
```

### LLM Schema Extraction
The system identifies:
- **Entities**: Person, Company, Location
- **Relationships**: WORKS_FOR, MANAGES, LOCATED_IN
- **Key Columns**: Name for Person, Company for Company
- **Reference Detection**: "Manager" column references other Person entities

### Graph Result
- **3 Person nodes**: Alice, Bob, Carol
- **2 Company nodes**: TechCorp, DataInc  
- **2 Location nodes**: NYC, LA
- **Relationships**: Alice-[WORKS_FOR]->TechCorp, Bob-[MANAGES]->Alice, etc.

## Performance & Scalability

### Efficient Processing
- **Batch Operations**: Configurable batch sizes for large datasets
- **Memory Management**: Processes data without loading everything into memory
- **Smart Filtering**: Removes low-quality columns automatically

### Database Optimization
- **Constraint Creation**: Automatic unique constraints for data integrity
- **MERGE Operations**: Handles duplicates efficiently
- **Session Management**: Proper connection handling and cleanup

## Error Handling & Reliability

### Robust Fallbacks
- **LLM Failure**: Creates generic schema if AI extraction fails
- **Invalid Schema**: Automatically validates and repairs extracted schemas
- **Missing References**: Logs failed lookups for debugging
- **Connection Issues**: Graceful error handling with proper cleanup

### Quality Assurance
- **Schema Validation**: Ensures column mappings match actual data
- **Relationship Verification**: Validates entity references exist
- **Confidence Scoring**: Provides quality metrics for extracted schemas

## Best Practices

### Data Preparation
- Use meaningful column names that clearly indicate content
- Ensure consistent naming conventions across your dataset
- Include identifier columns where possible (IDs, names, codes)
- Handle missing values appropriately before processing

### Optimization Tips
- Start with smaller datasets to test schema extraction
- Review LLM-generated schemas before processing large files
- Use appropriate batch sizes based on your system memory
- Monitor confidence scores to ensure quality results

## Limitations and possible future iterations
- Currently does not allow for user input to ensure entities and relationships generated from the LLM are accurate
- Limit the scope of data such that the schema can be predefined
- Fine tuning of model to improve accuracry
- integrate to allow LLMs to create Cypher queries as well

This script transforms structured into rich, queryable knowledge graphs that reveal hidden patterns and relationships in your data dynamically using LLMs. The automation of knowledge graph generation using LLM helps to reduce overhead and allow users not familiar with Cyber Query work with knowledge graphs.
