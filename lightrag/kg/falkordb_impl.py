import inspect
import os
import re
from dataclasses import dataclass
from typing import final
import configparser


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import logging
from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

import pipmaster as pm

if not pm.is_installed("networkx"):
    pm.install("networkx")

import networkx as nx


import asyncio
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


# Set FalkorDB logger level to ERROR to suppress warning logs
logging.getLogger("FalkorDB").setLevel(logging.ERROR)


@final
@dataclass
class FalkorDBStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}_falkordb_export.graphml"
        )

    @staticmethod
    def _knowledge_graph_to_nx(kg: KnowledgeGraph) -> nx.Graph:
        """Converts a KnowledgeGraph object to a networkx.Graph object."""
        nx_graph = nx.Graph()
        for node in kg.nodes:
            # Ensure properties are suitable for networkx attributes
            props = {k: str(v) if v is not None else "" for k, v in node.properties.items()}
            nx_graph.add_node(node.id, **props)
        for edge in kg.edges:
            props = {k: str(v) if v is not None else "" for k, v in edge.properties.items()}
            nx_graph.add_edge(edge.source, edge.target, **props)
        return nx_graph

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name: str):
        """Writes a networkx.Graph to a .graphml file."""
        logger.info(
            f"Writing graph to {file_name} with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        try:
            nx.write_graphml(graph, file_name)
        except Exception as e:
            logger.error(f"Error writing graph to {file_name}: {e}")

    async def initialize(self):
        REDIS_HOST = os.environ.get("REDIS_HOST", config.get("falkordb", "host", fallback="localhost"))
        REDIS_PORT = int(os.environ.get("REDIS_PORT", config.get("falkordb", "port", fallback=6379)))
        REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", config.get("falkordb", "password", fallback='falkordb'))
        REDIS_MAX_CONNECTIONS = int(os.environ.get("REDIS_MAX_CONNECTIONS", 
                                                 config.get("falkordb", "max_connections", fallback=16)))
        REDIS_TIMEOUT = float(os.environ.get("REDIS_TIMEOUT", 
                                           config.get("falkordb", "timeout", fallback=30.0)))
        
        # Use namespace as the graph name, sanitize it for FalkorDB compatibility
        GRAPH_NAME = re.sub(r"[^a-zA-Z0-9-]", "-", self.namespace)
        self._DATABASE = GRAPH_NAME
        
        try:
            # Create a Redis connection pool
            self.pool = BlockingConnectionPool(
                max_connections=REDIS_MAX_CONNECTIONS,
                timeout=REDIS_TIMEOUT,
                decode_responses=True
            )
            
            # Create the FalkorDB client
            self._driver = FalkorDB(connection_pool=self.pool,
                                    host=REDIS_HOST,
                                    port=REDIS_PORT,
                                    password=REDIS_PASSWORD)
            
            # Select the graph (creates it if it doesn't exist)
            graph = self._driver.select_graph(GRAPH_NAME)
            
            # Test the connection with a simple query
            test_result = await graph.query("MATCH (n) RETURN n LIMIT 0")
            if test_result and len(test_result.result_set) >= 0:
                logger.info(f"Connected to FalkorDB graph {GRAPH_NAME} at {REDIS_HOST}:{REDIS_PORT}")
            
            # Create index for base nodes on entity_id if it doesn't exist
            try:
                # Check if index exists - FalkorDB uses different syntax than Neo4j
                index_check = await graph.query("CALL db.indexes() YIELD types, label, properties, status")
                
                # Check if we have an index on entity_id for base nodes
                has_index = False
                if index_check and index_check.result_set:
                    for idx_record in index_check.result_set:
                        if idx_record[1] == 'base' and 'entity_id' in idx_record[2]:
                            has_index = True
                            break
                
                if not has_index:
                    # Create the index if it doesn't exist
                    await graph.query("CREATE INDEX FOR (n:base) ON (n.entity_id)")
                    logger.info(f"Created index for base nodes on entity_id in graph {GRAPH_NAME}")
            except Exception as e:
                # FalkorDB may have a different way to handle indexes, so log the error but continue
                logger.warning(f"Failed to create or check index: {str(e)}")
            
            # Connection successful
            return
                
        except Exception as e:
            logger.error(f"Error connecting to FalkorDB: {str(e)}")
            raise

    async def finalize(self):
        """Close the FalkorDB driver and release all resources"""
        if self.pool:
            # If using Redis connection pool, close it
            await self.pool.aclose()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure driver is closed when context manager exits"""
        await self.finalize()

    async def index_done_callback(self) -> None:
        # FalkorDB handles persistence automatically
        # Additionally, save a snapshot to GraphML
        logger.info(f"FalkorDB index_done_callback: Exporting graph {self.namespace} to GraphML.")
        try:
            # Fetch the entire graph
            # Using MAX_GRAPH_NODES to be consistent with get_knowledge_graph behavior
            kg_snapshot = await self.get_knowledge_graph(node_label="*", max_nodes=MAX_GRAPH_NODES)
            
            if kg_snapshot and (kg_snapshot.nodes or kg_snapshot.edges):
                # Convert to networkx.Graph
                nx_graph = FalkorDBStorage._knowledge_graph_to_nx(kg_snapshot)
                # Write to .graphml file
                FalkorDBStorage.write_nx_graph(nx_graph, self._graphml_xml_file)
                logger.info(f"Successfully exported graph {self.namespace} to {self._graphml_xml_file}")
            else:
                logger.info(f"Graph {self.namespace} is empty or could not be fetched. Skipping GraphML export.")
        except Exception as e:
            logger.error(f"Error during GraphML export in index_done_callback for {self.namespace}: {e}")

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given label exists in the database

        Args:
            node_id: Label of the node to check

        Returns:
            bool: True if node exists, False otherwise

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = "MATCH (n:base {entity_id: $entity_id}) RETURN count(n) > 0 AS node_exists"
            result = await graph.query(query, {'entity_id': node_id})
            
            if result and result.result_set:
                return result.result_set[0][0]
            return False
        except Exception as e:
            logger.error(f"Error checking node existence for {node_id}: {str(e)}")
            raise

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            bool: True if edge exists, False otherwise

        Raises:
            ValueError: If either node_id is invalid
            Exception: If there is an error executing the query
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = (
                "MATCH (a:base {entity_id: $source_entity_id})-[r]-(b:base {entity_id: $target_entity_id}) "
                "RETURN COUNT(r) > 0 AS edgeExists"
            )
            params = {
                'source_entity_id': source_node_id,
                'target_entity_id': target_node_id
            }
            result = await graph.query(query, params)
            
            if result and result.result_set:
                return result.result_set[0][0]
            return False
        except Exception as e:
            logger.error(
                f"Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
            )
            raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier, return only node properties

        Args:
            node_id: The node label to look up

        Returns:
            dict: Node properties if found
            None: If node not found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = "MATCH (n:base {entity_id: $entity_id}) RETURN n"
            result = await graph.query(query, {'entity_id': node_id})
            # print(f"Result: {result.result_set}") # User debug line, can be removed
            if result and len(result.result_set) > 0:
                if len(result.result_set) > 1:
                    logger.warning(
                        f"Multiple nodes found with label '{node_id}'. Using first node."
                    )
                
                node_object = result.result_set[0][0]
                node_dict = {}
                if hasattr(node_object, 'properties'):
                    node_dict = dict(node_object.properties)
                elif isinstance(node_object, dict): # Fallback if it's already a dict
                    node_dict = node_object
                else:
                    logger.error(f"Unexpected node object type for {node_id}: {type(node_object)}")
                    return None
                
                # Remove base label from labels list if it exists
                # FalkorDB nodes might store labels differently or not in node_dict['labels']
                # This part might need adjustment based on actual FalkorDB node structure if 'labels' isn't a property
                if "labels" in node_dict and isinstance(node_dict["labels"], list):
                    node_dict["labels"] = [
                        label for label in node_dict["labels"] if label != "base"
                    ]
                
                logger.debug(f"FalkorDB query node {query} return: {node_dict}")
                return node_dict
            return None
        except Exception as e:
            logger.error(f"Error getting node for {node_id}: {str(e)}")
            raise

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Retrieve multiple nodes in one query using UNWIND.

        Args:
            node_ids: List of node entity IDs to fetch.

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = """
            UNWIND $node_ids AS id
            MATCH (n:base {entity_id: id})
            RETURN n.entity_id AS entity_id, n
            """
            result = await graph.query(query, {'node_ids': node_ids})
            nodes = {}
            
            if result and result.result_set:
                for record in result.result_set:
                    entity_id = record[0]  # First column is entity_id
                    node_object = record[1]  # Second column is the node object
                    
                    node_dict = {}
                    if hasattr(node_object, 'properties'):
                        node_dict = dict(node_object.properties)
                    elif isinstance(node_object, dict): # Fallback if it's already a dict
                        node_dict = node_object
                    else:
                        logger.error(f"Unexpected node object type for {entity_id} in batch: {type(node_object)}")
                        continue # Skip this node or handle error as appropriate
                                        
                    # Remove the 'base' label if present in a 'labels' property
                    # Similar to get_node, this might need adjustment
                    if "labels" in node_dict and isinstance(node_dict["labels"], list):
                        node_dict["labels"] = [
                            label for label in node_dict["labels"] if label != "base"
                        ]
                    nodes[entity_id] = node_dict
            
            return nodes
        except Exception as e:
            logger.error(f"Error getting nodes batch: {str(e)}")
            raise

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of relationships) of a node with the given label.
        If multiple nodes have the same label, returns the degree of the first node.
        If no node is found, returns 0.

        Args:
            node_id: The label of the node

        Returns:
            int: The number of relationships the node has, or 0 if no node found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = """
                MATCH (n:base {entity_id: $entity_id})
                OPTIONAL MATCH (n)-[r]-()
                RETURN COUNT(r) AS degree
            """
            result = await graph.query(query, {'entity_id': node_id})
            
            if not result or not result.result_set:
                logger.warning(f"No node found with label '{node_id}'")
                return 0
                
            degree = result.result_set[0][0]  # First column of first row is degree
            logger.debug(f"FalkorDB query node degree for {node_id} return: {degree}")
            return degree
        except Exception as e:
            logger.error(f"Error getting node degree for {node_id}: {str(e)}")
            raise

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Retrieve the degree for multiple nodes in a single query using UNWIND.

        Args:
            node_ids: List of node labels (entity_id values) to look up.

        Returns:
            A dictionary mapping each node_id to its degree (number of relationships).
            If a node is not found, its degree will be set to 0.
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = """
                UNWIND $node_ids AS id
                MATCH (n:base {entity_id: id})
                RETURN n.entity_id AS entity_id, size((n)--()) AS degree
            """
            result = await graph.query(query, {'node_ids': node_ids})
            
            degrees = {}
            if result and result.result_set:
                for record in result.result_set:
                    entity_id = record[0]  # First column is entity_id
                    degree = record[1]     # Second column is degree
                    degrees[entity_id] = degree
            
            # For any node_id that did not return a record, set degree to 0
            for nid in node_ids:
                if nid not in degrees:
                    logger.warning(f"No node found with label '{nid}'")
                    degrees[nid] = 0
                    
            logger.debug(f"FalkorDB batch node degree query returned: {degrees}")
            return degrees
        except Exception as e:
            logger.error(f"Error getting node degrees batch: {str(e)}")
            raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes.

        Args:
            src_id: Label of the source node
            tgt_id: Label of the target node

        Returns:
            int: Sum of the degrees of both nodes
        """
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        return degrees

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """
        Calculate the combined degree for each edge (sum of the source and target node degrees)
        in batch using the already implemented node_degrees_batch.

        Args:
            edge_pairs: List of (src, tgt) tuples.

        Returns:
            A dictionary mapping each (src, tgt) tuple to the sum of their degrees.
        """
        # Collect unique node IDs from all edge pairs
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})

        # Get degrees for all nodes in one go
        degrees = await self.node_degrees_batch(list(unique_node_ids))

        # Sum up degrees for each edge pair
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            dict: Edge properties if found, default properties if not found or on error

        Raises:
            ValueError: If either node_id is invalid
            Exception: If there is an error executing the query
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = """
            MATCH (start:base {entity_id: $source_entity_id})-[r]-(end:base {entity_id: $target_entity_id})
            RETURN properties(r) as edge_properties
            """
            params = {
                'source_entity_id': source_node_id,
                'target_entity_id': target_node_id
            }
            
            result = await graph.query(query, params)
            
            if result and len(result.result_set) > 0:
                if len(result.result_set) > 1:
                    logger.warning(
                        f"Multiple edges found between '{source_node_id}' and '{target_node_id}'. Using first edge."
                    )
                
                try:
                    edge_result = dict(result.result_set[0][0])  # First column of first row contains edge properties
                    logger.debug(f"Result: {edge_result}")
                    
                    # Ensure required keys exist with defaults
                    required_keys = {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
                    for key, default_value in required_keys.items():
                        if key not in edge_result:
                            edge_result[key] = default_value
                            logger.warning(
                                f"Edge between {source_node_id} and {target_node_id} "
                                f"missing {key}, using default: {default_value}"
                            )

                    logger.debug(
                        f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_result}"
                    )
                    return edge_result
                except (KeyError, TypeError, ValueError) as e:
                    logger.error(
                        f"Error processing edge properties between {source_node_id} "
                        f"and {target_node_id}: {str(e)}"
                    )
                    # Return default edge properties on error
                    return {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }

            logger.debug(
                f"{inspect.currentframe().f_code.co_name}: No edge found between {source_node_id} and {target_node_id}"
            )
            # Return None when no edge found
            return None
                
        except Exception as e:
            logger.error(
                f"Error in get_edge between {source_node_id} and {target_node_id}: {str(e)}"
            )
            raise

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """
        Retrieve edge properties for multiple (src, tgt) pairs in one query.

        Args:
            pairs: List of dictionaries, e.g. [{"src": "node1", "tgt": "node2"}, ...]

        Returns:
            A dictionary mapping (src, tgt) tuples to their edge properties.
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = """
            UNWIND $pairs AS pair
            MATCH (start:base {entity_id: pair.src})-[r:DIRECTED]-(end:base {entity_id: pair.tgt})
            RETURN pair.src AS src_id, pair.tgt AS tgt_id, collect(properties(r)) AS edges
            """
            
            result = await graph.query(query, {'pairs': pairs})
            edges_dict = {}
            
            if result and result.result_set:
                for record in result.result_set:
                    src = record[0]      # First column is src_id
                    tgt = record[1]      # Second column is tgt_id
                    edges = record[2]    # Third column is edges collection
                    
                    if edges and len(edges) > 0:
                        edge_props = edges[0]  # Choose the first if multiple exist
                        # Ensure required keys exist with defaults
                        for key, default in {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }.items():
                            if key not in edge_props:
                                edge_props[key] = default
                        edges_dict[(src, tgt)] = edge_props
                    else:
                        # No edge found – set default edge properties
                        edges_dict[(src, tgt)] = {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }
            
            # Ensure all requested pairs have entries in the result
            for pair in pairs:
                if (pair['src'], pair['tgt']) not in edges_dict:
                    edges_dict[(pair['src'], pair['tgt'])] = {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
                    
            return edges_dict
        except Exception as e:
            logger.error(f"Error getting edges batch: {str(e)}")
            raise

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node identified by its label.

        Args:
            source_node_id: Label of the node to get edges for

        Returns:
            list[tuple[str, str]]: List of (source_label, target_label) tuples representing edges
            None: If no edges found

        Raises:
            ValueError: If source_node_id is invalid
            Exception: If there is an error executing the query
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = """MATCH (n:base {entity_id: $entity_id})
                    OPTIONAL MATCH (n)-[r]-(connected:base)
                    WHERE connected.entity_id IS NOT NULL
                    RETURN n, r, connected"""
                    
            result = await graph.query(query, {'entity_id': source_node_id})
            
            edges = []
            if result and result.result_set:
                for record in result.result_set:
                    source_node = record[0]      # First column is n
                    connected_node = record[2]   # Third column is connected node
                    
                    # Skip if either node is None
                    if not source_node or not connected_node:
                        continue
                        
                    source_label = None
                    if hasattr(source_node, 'properties') and isinstance(source_node.properties, dict):
                        source_label = source_node.properties.get("entity_id")
                    elif isinstance(source_node, dict): # Fallback if it's already a dict
                        source_label = source_node.get("entity_id")
                    
                    target_label = None
                    if hasattr(connected_node, 'properties') and isinstance(connected_node.properties, dict):
                        target_label = connected_node.properties.get("entity_id")
                    elif isinstance(connected_node, dict): # Fallback if it's already a dict
                        target_label = connected_node.get("entity_id")
                    
                    if source_label and target_label:
                        edges.append((source_label, target_label))
                        
            return edges
        except Exception as e:
            logger.error(f"Error in get_node_edges for {source_node_id}: {str(e)}")
            raise

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Batch retrieve edges for multiple nodes in one query using UNWIND.
        For each node, returns both outgoing and incoming edges to properly represent
        the undirected graph nature.

        Args:
            node_ids: List of node IDs (entity_id) for which to retrieve edges.

        Returns:
            A dictionary mapping each node ID to its list of edge tuples (source, target).
            For each node, the list includes both:
            - Outgoing edges: (queried_node, connected_node)
            - Incoming edges: (connected_node, queried_node)
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            # Query to get both outgoing and incoming edges
            query = """
                UNWIND $node_ids AS id
                MATCH (n:base {entity_id: id})
                OPTIONAL MATCH (n)-[r]-(connected:base)
                RETURN id AS queried_id, n.entity_id AS node_entity_id,
                       connected.entity_id AS connected_entity_id,
                       startNode(r).entity_id AS start_entity_id
            """
            
            result = await graph.query(query, {'node_ids': node_ids})
            
            # Initialize the dictionary with empty lists for each node ID
            edges_dict = {node_id: [] for node_id in node_ids}
            
            if result and result.result_set:
                for record in result.result_set:
                    queried_id = record[0]           # First column is queried_id
                    node_entity_id = record[1]       # Second column is node_entity_id
                    connected_entity_id = record[2]  # Third column is connected_entity_id
                    start_entity_id = record[3]      # Fourth column is start_entity_id
                    
                    # Skip if either node is None
                    if not node_entity_id or not connected_entity_id:
                        continue
                        
                    # Determine the actual direction of the edge
                    # If the start node is the queried node, it's an outgoing edge
                    # Otherwise, it's an incoming edge
                    if start_entity_id == node_entity_id:
                        # Outgoing edge: (queried_node -> connected_node)
                        edges_dict[queried_id].append((node_entity_id, connected_entity_id))
                    else:
                        # Incoming edge: (connected_node -> queried_node)
                        edges_dict[queried_id].append((connected_entity_id, node_entity_id))
                        
            return edges_dict
        except Exception as e:
            logger.error(f"Error getting nodes edges batch: {str(e)}")
            raise

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the FalkorDB database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        properties = node_data
        entity_type = properties["entity_type"]
        if "entity_id" not in properties:
            raise ValueError("FalkorDB: node properties must contain an 'entity_id' field")

        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = (
                """
                MERGE (n:base {entity_id: $entity_id})
                SET n += $properties
                SET n:`%s`
                """
                % entity_type
            )
            
            await graph.query(query, {
                'entity_id': node_id, 
                'properties': properties
            })
            
            logger.debug(
                f"Upserted node with entity_id '{node_id}' and properties: {properties}"
            )
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.
        Ensures both source and target nodes exist and are unique before creating the edge.
        Uses entity_id property to uniquely identify nodes.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge

        Raises:
            ValueError: If either source or target node does not exist or is not unique
        """
        try:
            edge_properties = edge_data
            graph = self._driver.select_graph(self._DATABASE)
            
            query = """
            MATCH (source:base {entity_id: $source_entity_id})
            WITH source
            MATCH (target:base {entity_id: $target_entity_id})
            MERGE (source)-[r:DIRECTED]-(target)
            SET r += $properties
            RETURN r, source, target
            """
            
            params = {
                'source_entity_id': source_node_id,
                'target_entity_id': target_node_id,
                'properties': edge_properties
            }
            
            result = await graph.query(query, params)
            
            if result and result.result_set and len(result.result_set) > 0:
                logger.debug(
                    f"Upserted edge from '{source_node_id}' to '{target_node_id}'"
                    f"with properties: {edge_properties}"
                )
        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maximum nodes to return by BFS, Defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()
        
        try:
            graph = self._driver.select_graph(self._DATABASE)
            
            if node_label == "*":
                # First check total node count to determine if graph is truncated
                count_query = "MATCH (n:base) RETURN count(n) as total"
                count_result = await graph.query(count_query)
                
                total_count = 0
                if count_result and count_result.result_set:
                    total_count = count_result.result_set[0][0]
                    
                    if total_count > max_nodes:
                        result.is_truncated = True
                        logger.info(
                            f"Graph truncated: {total_count} nodes found, limited to {max_nodes}"
                        )
                
                # Get nodes with highest degree, ordered by degree
                main_query = """
                MATCH (n:base)
                OPTIONAL MATCH (n)-[r]-()
                WITH n, COALESCE(count(r), 0) AS degree
                ORDER BY degree DESC
                LIMIT $max_nodes
                RETURN n, degree
                """
                
                main_result = await graph.query(main_query, {'max_nodes': max_nodes})
                
                if main_result and main_result.result_set:
                    # Process nodes first
                    for record in main_result.result_set:
                        node = record[0]  # First column is the node

                        degree = record[1]  # Second column is degree
                        
                        # Extract node properties from FalkorDB node object
                        node_id = None
                        node_properties = {}
                        
                        # FalkorDB nodes are dict-like objects
                        if hasattr(node, 'properties'):
                            node_properties = dict(node.properties)
                        else:
                            # Fallback: treat as dictionary
                            node_properties = dict(node) if node else {}
                        
                        node_id = node_properties.get("entity_id")
                        if not node_id:
                            continue
                            
                        if node_id not in seen_nodes:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=str(node_id),
                                    labels=[str(node_id)],
                                    properties=node_properties,
                                )
                            )
                            seen_nodes.add(node_id)
                
                # Then get edges between these nodes
                if seen_nodes:
                    edges_query = """
                    MATCH (a:base)-[r]-(b:base)
                    WHERE a.entity_id IN $node_ids AND b.entity_id IN $node_ids
                    RETURN r, a.entity_id AS source, b.entity_id AS target, ID(r) AS edge_id
                    """
                    
                    edges_result = await graph.query(edges_query, {'node_ids': list(seen_nodes)})
                    
                    if edges_result and edges_result.result_set:
                        for edge_rec in edges_result.result_set:
                            rel = edge_rec[0]          # First column is the relationship
                            source_id = edge_rec[1]    # Second column is source entity_id
                            target_id = edge_rec[2]    # Third column is target entity_id
                            edge_id = edge_rec[3]      # Fourth column is edge_id
                            
                            # Create a unique edge identifier by sorting source and target IDs
                            # to handle undirected edges consistently.
                            sorted_node_ids = sorted((str(source_id), str(target_id)))
                            edge_key = f"{sorted_node_ids[0]}_{sorted_node_ids[1]}_{edge_id}"
                            
                            if edge_key not in seen_edges:
                                # Extract edge properties
                                edge_properties = {}
                                if hasattr(rel, 'properties'):
                                    edge_properties = dict(rel.properties)
                                else:
                                    edge_properties = dict(rel) if rel else {}
                                
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_key,
                                        type=edge_properties.get("type", "DIRECTED"),
                                        source=str(source_id),
                                        target=str(target_id),
                                        properties=edge_properties,
                                    )
                                )
                                seen_edges.add(edge_key)
            else:
                # For specific node label, use the robust fallback implementation
                return await self._robust_fallback(node_label, max_depth, max_nodes)
                
        except Exception as e:
            logger.warning(f"Error in graph query: {str(e)}")
            if node_label != "*":
                logger.warning("FalkorDB: falling back to basic Cypher recursive search...")
                return await self._robust_fallback(node_label, max_depth, max_nodes)
            else:
                logger.warning("FalkorDB: Error with wildcard query, returning empty result")
                
        logger.info(
            f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def _robust_fallback(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """
        Fallback implementation for FalkorDB that implements BFS traversal
        using basic Cypher queries instead of specialized procedures.
        """
        from collections import deque

        result = KnowledgeGraph()
        visited_nodes = set()
        visited_edges = set()
        visited_edge_pairs = set()

        # Get the starting node's data
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = """
            MATCH (n:base {entity_id: $entity_id})
            RETURN ID(n) as node_id, n
            """
            
            node_result = await graph.query(query, {'entity_id': node_label})
            
            if not node_result or not node_result.result_set:
                return result
                
            # Extract node data
            node_record = node_result.result_set[0]
            internal_node_id = node_record[0]     # First column is internal node_id
            node_data = node_record[1]            # Second column is node data
            
            # Extract properties from FalkorDB node
            node_properties = {}
            if hasattr(node_data, 'properties'):
                node_properties = dict(node_data.properties)
            else:
                node_properties = dict(node_data) if node_data else {}
            
            entity_id = node_properties.get('entity_id', node_label)
            
            # Create initial KnowledgeGraphNode
            start_node = KnowledgeGraphNode(
                id=str(entity_id),
                labels=[str(entity_id)],
                properties=node_properties,
            )
            
            # Initialize queue for BFS with (node, edge, depth) tuples
            # edge is None for the starting node
            queue = deque([(start_node, None, 0)])
            
            # True BFS implementation using a queue
            while queue and len(visited_nodes) < max_nodes:
                # Dequeue the next node to process
                current_node, current_edge, current_depth = queue.popleft()
                
                # Skip if already visited or exceeds max depth
                if current_node.id in visited_nodes:
                    continue
                    
                if current_depth > max_depth:
                    logger.debug(
                        f"Skipping node at depth {current_depth} (max_depth: {max_depth})"
                    )
                    continue
                    
                # Add current node to result
                result.nodes.append(current_node)
                visited_nodes.add(current_node.id)
                
                # Add edge to result if it exists and not already added
                if current_edge and current_edge.id not in visited_edges:
                    result.edges.append(current_edge)
                    visited_edges.add(current_edge.id)
                    
                # Stop if we've reached the node limit
                if len(visited_nodes) >= max_nodes:
                    result.is_truncated = True
                    logger.info(
                        f"Graph truncated: breadth-first search limited to: {max_nodes} nodes"
                    )
                    break
                    
                # Get all edges and target nodes for the current node
                neighbor_query = """
                MATCH (a:base {entity_id: $entity_id})-[r]-(b:base)
                RETURN r, b, ID(r) as edge_id, b.entity_id as target_entity_id
                """
                
                neighbors_result = await graph.query(neighbor_query, {'entity_id': current_node.id})
                
                if neighbors_result and neighbors_result.result_set:
                    for record in neighbors_result.result_set:
                        rel = record[0]                 # First column is relationship
                        b_node = record[1]              # Second column is target node
                        edge_id = str(record[2])        # Third column is edge_id
                        target_entity_id = record[3]    # Fourth column is target entity_id
                        
                        if edge_id not in visited_edges and target_entity_id:
                            # Extract target node properties
                            target_properties = {}
                            if hasattr(b_node, 'properties'):
                                target_properties = dict(b_node.properties)
                            else:
                                target_properties = dict(b_node) if b_node else {}
                            
                            # Create KnowledgeGraphNode for target
                            target_node = KnowledgeGraphNode(
                                id=str(target_entity_id),
                                labels=[str(target_entity_id)],
                                properties=target_properties,
                            )
                            
                            # Extract edge properties
                            edge_properties = {}
                            if hasattr(rel, 'properties'):
                                edge_properties = dict(rel.properties)
                            else:
                                edge_properties = dict(rel) if rel else {}
                            
                            # Create KnowledgeGraphEdge
                            target_edge = KnowledgeGraphEdge(
                                id=f"{edge_id}",
                                type=edge_properties.get("type", "DIRECTED"),
                                source=str(current_node.id),
                                target=str(target_entity_id),
                                properties=edge_properties,
                            )
                            
                            # Sort source_id and target_id to ensure (A,B) and (B,A) are treated as the same edge
                            sorted_pair = tuple(sorted([current_node.id, target_entity_id]))
                            
                            # Check if the same edge already exists (considering undirectedness)
                            if sorted_pair not in visited_edge_pairs:
                                # Only add the edge if the target node is already in the result or will be added
                                if target_entity_id in visited_nodes or (
                                    target_entity_id not in visited_nodes
                                    and current_depth < max_depth
                                ):
                                    result.edges.append(target_edge)
                                    visited_edges.add(edge_id)
                                    visited_edge_pairs.add(sorted_pair)
                                    
                            # Only add unvisited nodes to the queue for further expansion
                            if target_entity_id not in visited_nodes:
                                # Only add to queue if we're not at max depth yet
                                if current_depth < max_depth:
                                    # Add node to queue with incremented depth
                                    # Edge is already added to result, so we pass None as edge
                                    queue.append((target_node, None, current_depth + 1))
                                else:
                                    # At max depth, we've already added the edge but we don't add the node
                                    logger.debug(
                                        f"Node {target_entity_id} beyond max depth {max_depth}, edge added but node not included"
                                    )
                            else:
                                # If target node already exists in result, we don't need to add it again
                                logger.debug(
                                    f"Node {target_entity_id} already visited, edge added but node not queued"
                                )
            
            logger.info(
                f"BFS subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
            )
            
        except Exception as e:
            logger.error(f"Error in _robust_fallback: {str(e)}")
            
        return result

    async def get_all_labels(self) -> list[str]:
        """
        Get all existing node labels in the database
        Returns:
            ["Person", "Company", ...]  # Alphabetically sorted label list
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = """
            MATCH (n:base)
            WHERE n.entity_id IS NOT NULL
            RETURN DISTINCT n.entity_id AS label
            ORDER BY label
            """
            
            result = await graph.query(query)
            labels = []
            
            if result and result.result_set:
                for record in result.result_set:
                    labels.append(record[0])  # First column is label
                    
            return labels
        except Exception as e:
            logger.error(f"Error getting all labels: {str(e)}")
            return []

    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified label

        Args:
            node_id: The label of the node to delete
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            query = """
            MATCH (n:base {entity_id: $entity_id})
            DETACH DELETE n
            """
            
            await graph.query(query, {'entity_id': node_id})
            logger.debug(f"Deleted node with label '{node_id}'")
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node labels to be deleted
        """
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:
            try:
                graph = self._driver.select_graph(self._DATABASE)
                query = """
                MATCH (source:base {entity_id: $source_entity_id})-[r]-(target:base {entity_id: $target_entity_id})
                DELETE r
                """
                
                params = {'source_entity_id': source, 'target_entity_id': target}
                await graph.query(query, params)
                logger.debug(f"Deleted edge from '{source}' to '{target}'")
            except Exception as e:
                logger.error(f"Error during edge deletion: {str(e)}")
                raise

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources

        This method will delete all nodes and relationships in the FalkorDB graph
        and also remove the exported .graphml file.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            graph = self._driver.select_graph(self._DATABASE)
            # Delete all nodes and relationships
            query = "MATCH (n) DETACH DELETE n"
            await graph.query(query)
            
            # Remove the exported .graphml file if it exists
            if os.path.exists(self._graphml_xml_file):
                os.remove(self._graphml_xml_file)
                logger.info(f"Removed exported GraphML file: {self._graphml_xml_file}")
            
            logger.info(f"Process {os.getpid()} drop FalkorDB graph {self._DATABASE}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping FalkorDB graph {self._DATABASE}: {e}")
            return {"status": "error", "message": str(e)}
