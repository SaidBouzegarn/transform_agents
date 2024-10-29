import os
from dotenv import load_dotenv, set_key
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_google_vertexai import ChatVertexAI
from langchain_community.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import subprocess
import logging
from langchain_core.documents import Document
from langchain.tools import BaseTool
from jinja2 import Environment, FileSystemLoader
import chromadb
from chromadb import Settings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
import uuid
import json
from langchain.prompts.chat import ChatPromptTemplate
from langchain.base_language import BaseLanguageModel
import requests
import re
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="List of at least 2 entities or entity relationships that represent the same object or real-world entity, but are not identical in their spelling and should be merged. The entity or relationship name to keep should be the first one in the list.",
        min_items=2
    )   

class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity but are not identical in their spelling and should be merged in the graph to avoid confusion"
    )

class GraphKnowledgeManager:
    def __init__(
        self, 
        name: str, 
        level: str,
        prompt_dir: str,
        aura_instance_id: str,  # Add this parameter
        aura_instance_name: str,  # Add this parameter
        neo4j_uri: str,  # Add this parameter
        neo4j_username: str,  # Add this parameter
        neo4j_password: str,  # Add this parameter
        llm_models: str = "gpt-4-turbo", 
        cypher_llm_model: str = "gpt-4",
        qa_llm_model: str = "gpt-3.5-turbo",
        cypher_llm_params: Dict[str, Any] = None,
        qa_llm_params: Dict[str, Any] = None,
        chain_verbose: bool = False,
        chain_callback_manager: Optional[Any] = None,
        chain_memory: Optional[Any] = None,
        similarity_threshold: float = 0.85,
        max_iterations: int = 5,
        execution_timeout: int = 30,
        max_retries: int = 3,
        return_intermediate_steps: bool = False,
        handle_retries: bool = True,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        strict_mode: bool = False,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        ignore_tool_usage: bool = False,
        **llm_params
    ):
        load_dotenv()
        self.name = name
        self.level = level
        self.prompt_dir = prompt_dir
        self.llm_models = llm_models
        self.llm_params = llm_params
        self.cypher_llm_model = cypher_llm_model
        self.qa_llm_model = qa_llm_model
        self.cypher_llm_params = cypher_llm_params if cypher_llm_params is not None else {}
        self.qa_llm_params = qa_llm_params if qa_llm_params is not None else {}
        self.chain_verbose = chain_verbose
        self.chain_callback_manager = chain_callback_manager
        self.chain_memory = chain_memory
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations
        self.execution_timeout = execution_timeout
        self.max_retries = max_retries
        self.return_intermediate_steps = return_intermediate_steps
        self.handle_retries = handle_retries
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self.node_properties = node_properties
        self.relationship_properties = relationship_properties
        self.ignore_tool_usage = ignore_tool_usage
        
        # Store Neo4j instance details
        self.aura_instance_id = aura_instance_id
        self.aura_instance_name = aura_instance_name
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        
        self.neo4j_graph = None
        self.jinja_env = Environment(loader=FileSystemLoader(os.path.join(self.prompt_dir)))
        self.graph_system_prompt = self._load_graph_system_prompt()
        self.schema = None
        
        self.ensure_connection()
        
        # Initialize GraphCypherQAChain with default values
        self.cypher_chain = None
        
        # Initialize LLMGraphTransformer        
        self.llm_transformer = None

    def _escape_single_brackets(self, text: str) -> str:
        """
        Replace single brackets with double brackets to escape them properly.
        Ignores already doubled brackets.
        """
        # Replace single { with {{ if not already {{
        text = re.sub(r'(?<!\{)\{(?!\{)', '{{', text)
        # Replace single } with }} if not already }}
        text = re.sub(r'(?<!\})\}(?!\})', '}}', text)
        return text

    def _create_prompt_template(self) -> ChatPromptTemplate:
        # First ensure schema is loaded
        if self.schema is None:
            self.schema = self._load_schema()
        
        # Format the schema for better readability
        formatted_schema = "Nodes:\n"
        for node in self.schema["nodes"]:
            formatted_schema += f"- {node['label']}: {', '.join(node['properties'])}\n"
        
        formatted_schema += "\nRelationships:\n"
        for rel in self.schema["relationships"]:
            formatted_schema += f"- {rel['type']}: {', '.join(rel['properties'])}\n"

        # Escape any single brackets in the formatted schema
        formatted_schema = self._escape_single_brackets(formatted_schema)

        messages = [
            SystemMessage(content=f"""You are an AI assistant for querying a Neo4j graph database about PEPFAR (President's Emergency Plan for AIDS Relief) and its impact. Translate the user's questions into Cypher queries.
Only provide the Cypher query without any explanations or additional text.
Ensure that the queries are optimized and follow best practices for graph databases.

The current database schema is:
{self.schema}

Use this schema information to construct your Cypher queries."""),
            HumanMessage(content="{input}")
        ]

        return ChatPromptTemplate.from_messages(messages)

    def _construct_llm(self, llm_name: str, llm_params: Dict[str, Any]) -> BaseLanguageModel:
        """Construct the appropriate LLM based on the input string and parameters."""
        OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        MISTRAL_MODELS = ["mistral-tiny", "mistral-small", "mistral-medium"]
        COHERE_MODELS = ["command", "command-light", "command-nightly"]
        GROQ_MODELS = ["llama2-70b-4096", "mixtral-8x7b-32768"]
        VERTEXAI_MODELS = ["chat-bison", "chat-bison-32k"]
        OLLAMA_MODELS = ["llama2", "mistral", "dolphin-phi"]
        NVIDIA_MODELS = ["mixtral-8x7b", "llama2-70b"]
        ANTHROPIC_MODELS = ["claude-2", "claude-instant-1"]
        FIREWORKS_MODELS = ["llama-v2-7b", "llama-v2-13b", "llama-v2-70b"]

        if llm_name in OPENAI_MODELS:
            return ChatOpenAI(model_name=llm_name, **llm_params)
        elif llm_name in MISTRAL_MODELS:
            return ChatMistralAI(model=llm_name, **llm_params)
        elif llm_name in COHERE_MODELS:
            return ChatCohere(model=llm_name, **llm_params)
        elif llm_name in GROQ_MODELS:
            return ChatGroq(model=llm_name, **llm_params)
        elif llm_name in VERTEXAI_MODELS:
            return ChatVertexAI(model_name=llm_name, **llm_params)
        elif llm_name in OLLAMA_MODELS:
            return ChatOllama(model=llm_name, **llm_params)
        elif llm_name in NVIDIA_MODELS:
            return ChatNVIDIA(model=llm_name, **llm_params)
        elif llm_name in ANTHROPIC_MODELS:
            return ChatAnthropic(model=llm_name, **llm_params)
        elif llm_name in FIREWORKS_MODELS:
            return ChatFireworks(model=llm_name, **llm_params)
        else:
            raise ValueError(f"Unsupported model: {llm_name}")

    def ensure_connection(self):
        """Connect to the existing Neo4j instance."""
        try:
            # Connect using Neo4jGraph for LangChain operations
            self.neo4j_graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password
            )
            logger.info(f"Connected to Neo4j Aura instance: {self.aura_instance_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j instance: {e}")
            raise

    def query_graph(self, question: str) -> Optional[Any]:
        if self.cypher_chain is None:
            self.cypher_chain = GraphCypherQAChain.from_llm(
                cypher_llm=self._construct_llm(self.cypher_llm_model, self.cypher_llm_params),
                qa_llm=self._construct_llm(self.qa_llm_model, self.qa_llm_params),
                graph=self.neo4j_graph,
                verbose=self.chain_verbose,
                callback_manager=self.chain_callback_manager,
                memory=self.chain_memory,
                prompt_template=self._create_prompt_template(),
                similarity_threshold=self.similarity_threshold,
                max_iterations=self.max_iterations,
                execution_timeout=self.execution_timeout,
                max_retries=self.max_retries,
                return_intermediate_steps=self.return_intermediate_steps,
                handle_retries=self.handle_retries,
                allow_dangerous_requests=True
            )        
        try:
            response = self.cypher_chain.invoke(question)
            logger.info(f"Query result: {response}")
            return response
        except Exception as e:
            logger.error(f"Error executing GraphCypherQAChain: {e}")
            return None

    def populate_knowledge_graph(self, texts: List[str], batch_size: int = 100):
        graph_prompt = ChatPromptTemplate.from_template(self.graph_system_prompt)

        if self.llm_transformer is None:
            self.llm_transformer = LLMGraphTransformer(
                llm=self._construct_llm(self.llm_models, self.llm_params),
                allowed_nodes=self.allowed_nodes,
                allowed_relationships=self.allowed_relationships,
                prompt=graph_prompt,
                strict_mode=self.strict_mode,
                node_properties=self.node_properties,
                relationship_properties=self.relationship_properties,
                ignore_tool_usage=self.ignore_tool_usage
            )
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            documents = [Document(page_content=text) for text in batch]
            graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
            
            self.neo4j_graph.add_graph_documents(graph_documents)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

    def delete_node_or_relationship(self, identifier: str):
        try:
            self.neo4j_graph.query(f"MATCH (n) WHERE n.id = '{identifier}' DETACH DELETE n")
            logger.info(f"Deleted node or relationship with identifier '{identifier}'")
        except Exception as e:
            logger.error(f"Error deleting node or relationship: {e}")

    def delete_database(self):
        try:
            with GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password)) as driver:
                with driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
            logger.info("Deleted all nodes and relationships in the database.")
        except Exception as e:
            logger.error(f"Error deleting database: {e}")

    def query_graph_tool(self) -> BaseTool:
        return BaseTool(
            name="query_graph",
            func=self.query_graph,
            description="Query the Neo4j graph database"
        )

    def populate_knowledge_graph_tool(self) -> BaseTool:
        return BaseTool.from_function(self.populate_knowledge_graph)

    def delete_node_or_relationship_tool(self) -> BaseTool:
        return BaseTool.from_function(self.delete_node_or_relationship)

    def delete_database_tool(self) -> BaseTool:
        return BaseTool.from_function(self.delete_database)

    def get_tools(self) -> List[BaseTool]:
        return [
            self.query_graph_tool(),
            self.populate_knowledge_graph_tool(),
            self.delete_node_or_relationship_tool(),
            self.delete_database_tool(),
        ]

    def _load_graph_system_prompt(self):
        template = self.jinja_env.get_template('graph_system_prompt.j2')
        return template.render()



    def _load_schema(self) -> dict:
        """Get the current schema from the Neo4j database."""
        try:
            # Get node labels and their properties
            nodes_query = """
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType, propertyName
            RETURN collect({
                label: nodeType,
                properties: collect(propertyName)
            }) as nodes
            """
            
            # Get relationship types and their properties
            rels_query = """
            CALL db.schema.relationshipTypeProperties()
            YIELD relationshipType, propertyName
            WITH relationshipType, collect(propertyName) as props
            MATCH (start)-[r:${relationshipType}]->(end)
            RETURN collect({
                type: relationshipType,
                properties: props,
                start_node: labels(start)[0],
                end_node: labels(end)[0]
            }) as relationships
            """
            
            nodes = self.neo4j_graph.query(nodes_query)
            relationships = self.neo4j_graph.query(rels_query)
            
            schema = {
                "nodes": nodes[0]["nodes"],
                "relationships": relationships[0]["relationships"]
            }
            
            logger.info("Successfully loaded schema from database")
            return schema
            
        except Exception as e:
            logger.error(f"Error loading schema from database: {e}")
            # Return a minimal default schema structure
            return {
                "nodes": [],
                "relationships": []
            }

    def disambiguate(self):
        """
        Resolve duplicate entities and relationships in the graph by identifying and merging
        nodes and relationships that represent the same concepts.
        """
        try:
            # First connect to the specific database
            logger.info(f"Starting disambiguation process for database: {self.aura_instance_name}")
            
            # Get all nodes
            nodes_query = """
            MATCH (n) 
            RETURN DISTINCT n.name as name, labels(n) as labels, 
            properties(n) as properties
            """
            nodes = self.neo4j_graph.query(nodes_query)
            
            # Get all relationships
            rels_query = """
            MATCH ()-[r]->() 
            RETURN DISTINCT type(r) as type, 
            startNode(r).name as start_name, 
            endNode(r).name as end_name,
            properties(r) as properties
            """
            relationships = self.neo4j_graph.query(rels_query)

            # Process nodes in batches
            batch_size = 15  # Adjust based on your needs
            node_batches = [nodes[i:i+batch_size] for i in range(0, len(nodes), batch_size)]
            
            for i, node_batch in enumerate(node_batches, 1):
                self._merge_similar_nodes(node_batch)
                logger.info(f"Processed node batch {i}/{len(node_batches)}")

            # Process relationships in batches
            rel_batches = [relationships[i:i+batch_size] for i in range(0, len(relationships), batch_size)]
            
            for i, rel_batch in enumerate(rel_batches, 1):
                self._merge_similar_relationships(rel_batch)
                logger.info(f"Processed relationship batch {i}/{len(rel_batches)}")

        except Exception as e:
            logger.error(f"Error during disambiguation: {e}")
            raise

    def _merge_similar_nodes(self, nodes):
        """Merge nodes that represent the same entity."""
        try:
            # Create a prompt for the LLM to identify similar nodes
            node_data = [
                {
                    "name": node["name"],
                    "labels": node["labels"],
                    "properties": node["properties"]
                }
                for node in nodes if node["name"]  # Filter out nodes without names
            ]
            
            if not node_data:
                return

            messages = [
                SystemMessage(content="""You are a data processing assistant specialized in identifying duplicate entities in Neo4j graphs.
                Your task is to analyze nodes and identify which ones represent the same real-world entity despite having different representations.

                Rules for identifying duplicates:
                1. Consider nodes with minor spelling variations or typographical differences as duplicates
                   Example: "John Smith" and "Jon Smith" might be the same person
                2. Consider nodes with different formats but same semantic meaning as duplicates
                   Example: "USA" and "United States of America" refer to the same country
                3. Consider nodes that refer to the same real-world entity as duplicates, even if described differently
                   Example: "NYC" and "New York City" refer to the same place
                4. Do NOT merge nodes if they represent:
                   - Different time periods or dates
                   - Different numerical values
                   - Different specific instances of similar things
                   Example: "Report 2023" and "Report 2024" should remain separate

                Return your response as a JSON array where each element is an array of duplicate nodes.
                The first node in each array should be the canonical form (the preferred version to keep).
                Only include nodes that have duplicates - ignore unique nodes.
                """),
                HumanMessage(content=f"Analyze these nodes for duplicates:\n{json.dumps(node_data, indent=2)}")
            ]

            # Get LLM response
            response = self.cypher_llm.invoke(messages)
            merge_groups = json.loads(response.content)

            # Merge similar nodes
            for group in merge_groups:
                if len(group) > 1:
                    primary = group[0]
                    for secondary in group[1:]:
                        merge_query = """
                        MATCH (primary {name: $primary_name}), (secondary {name: $secondary_name})
                        CALL apoc.merge.nodes([primary, secondary]) YIELD node
                        RETURN node
                        """
                        self.neo4j_graph.query(
                            merge_query,
                            {"primary_name": primary["name"], "secondary_name": secondary["name"]}
                        )
                        logger.info(f"Merged node '{secondary['name']}' into '{primary['name']}'")

        except Exception as e:
            logger.error(f"Error merging similar nodes: {e}")
            raise

    def _merge_similar_relationships(self, relationships):
        """Merge relationships that represent the same connection."""
        try:
            # Create a prompt for the LLM to identify similar relationships
            rel_data = [
                {
                    "type": rel["type"],
                    "start": rel["start_name"],
                    "end": rel["end_name"],
                    "properties": rel["properties"]
                }
                for rel in relationships
            ]
            
            if not rel_data:
                return

            messages = [
                SystemMessage(content="""
                    Identify relationships that represent the same connection but are written differently.
                    Return a list of groups where each group contains similar relationships.
                    The first relationship in each group should be the canonical form to keep.
                """),
                HumanMessage(content=f"Analyze these relationships:\n{json.dumps(rel_data, indent=2)}")
            ]

            # Get LLM response
            response = self.cypher_llm.invoke(messages)
            merge_groups = json.loads(response.content)

            # Merge similar relationships
            for group in merge_groups:
                if len(group) > 1:
                    primary = group[0]
                    for secondary in group[1:]:
                        merge_query = """
                        MATCH (s1 {name: $primary_start})-[r1:$primary_type]->(e1 {name: $primary_end}),
                              (s2 {name: $secondary_start})-[r2:$secondary_type]->(e2 {name: $secondary_end})
                        CALL apoc.merge.relationships([r1, r2]) YIELD rel
                        RETURN rel
                        """
                        self.neo4j_graph.query(
                            merge_query,
                            {
                                "primary_type": primary["type"],
                                "primary_start": primary["start"],
                                "primary_end": primary["end"],
                                "secondary_type": secondary["type"],
                                "secondary_start": secondary["start"],
                                "secondary_end": secondary["end"]
                            }
                        )
                        logger.info(f"Merged relationship '{secondary['type']}' into '{primary['type']}'")

        except Exception as e:
            logger.error(f"Error merging similar relationships: {e}")
            raise


class VectorDBManager:
    def __init__(
        self,
        name: str,
        prompt_dir: str,
        llm_models: str = "gpt-4-turbo",
    ):
        self.name = name
        self.prompt_dir = prompt_dir
        self.llm_models = llm_models
        self.vector_db = self._ensure_vector_db_exists_and_connect()
        self.embeddings = OpenAIEmbeddings()

    def _ensure_vector_db_exists_and_connect(self):
        try:
            persist_directory = f"./chroma_db_{self.name}"
            os.makedirs(persist_directory, exist_ok=True)
            vector_db = chromadb.Client(Settings(persist_directory=persist_directory))
            collection_name = f"{self.name}_collection"
            try:
                vector_db.get_collection(collection_name)
            except ValueError:
                vector_db.create_collection(collection_name)
            logger.info(f"Connected to Chroma vector database: {self.name}")
            return vector_db
        except Exception as e:
            logger.error(f"Failed to connect to Chroma vector database '{self.name}': {e}")
            raise

    def populate_vector_db(self, documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200):
        collection = self.vector_db.get_collection(f"{self.name}_collection")
        for doc in documents:
            embedding = self.embeddings.embed_query(doc['content'])
            collection.add(
                embeddings=[embedding],
                documents=[doc['content']],
                metadatas=[doc['metadata']]
            )
        logger.info(f"Populated vector database with {len(documents)} documents")

    def query_chroma_db(self, question: str, top_k: int = 5) -> Optional[str]:
        try:
            collection = self.vector_db.get_collection(f"{self.name}_collection")
            query_embedding = self.embeddings.embed_query(question)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            return results['documents'][0] if results['documents'] else None
        except Exception as e:
            logger.error(f"Error querying Chroma DB: {e}")
            return None

    def populate_vector_db_tool(self) -> BaseTool:
        return BaseTool.from_function(self.populate_vector_db)

    def query_vector_db_tool(self) -> BaseTool:
        return BaseTool.from_function(self.query_chroma_db)

    def get_tools(self) -> List[BaseTool]:
        return [
            self.populate_vector_db_tool(),
            self.query_vector_db_tool(),
        ]

# Test the code
def main():
    prompt_dir = "Data/prompts_test/level1/agent1"
    name = "agent1"
    gkm = None
    load_dotenv()

    aura_instance_id = os.getenv('AURA_INSTANCE_ID')
    aura_instance_name = os.getenv('AURA_INSTANCENAME')
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')   
    # Initialize GraphKnowledgeManager
    gkm = GraphKnowledgeManager(
        name=name,
        level="level1",
        prompt_dir=prompt_dir,
        temperature=0.2,
        max_tokens=4000,
        aura_instance_id=aura_instance_id,
        aura_instance_name=aura_instance_name,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,  
    )
    
    # Populate the knowledge graph with insights
    insights = [
        "PEPFAR, started in 2003, has saved over 25 million lives through antiretroviral treatments.",
        "Dr. Emily Kainne Dokubo, working for PEPFAR, emphasizes partnerships in addressing disparities in pediatric HIV treatment.",
    ]
    gkm.populate_knowledge_graph(insights, batch_size=25)
    
    # Query the graph database
    result = gkm.query_graph("How many lives has PEPFAR saved?")
    print(f"Answer from graph DB: {result}")
    

        
        # Optionally, clean up the database
        # gkm.delete_database()

if __name__ == "__main__":
    main()

