#!/usr/bin/env python
"""
General Graph Storage Test Program

The program selects the graph storage type to use based on the LIGHTRAG_GRAPH_STORAGE configuration in .env.
And test its basic and advanced operations.

Supported graph storage types include:
- NetworkXStorage
- Neo4JStorage
-PGGraphStorage
"""

import asyncio
import os
import sys
import importlib
import numpy as np
from dotenv import load_dotenv
from ascii_colors import ASCIIColors

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.types import KnowledgeGraph
from lightrag.kg import (
    STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
    verify_storage_implementation,
)
from lightrag.kg.shared_storage import initialize_share_data


# Simulated embedding function, returning a random vector
async def mock_embedding_func(texts):
    return np.random.rand(len(texts), 10) # Returns a 10-dimensional random vector


def check_env_file():
    """
    Checks if the .env file exists and warns if it does not exist
    Return True to continue execution, False to exit.
    """
    if not os.path.exists(".env"):
        warning_msg = "Warning: No .env file was found in the current directory, which may affect the loading of the storage configuration."
        ASCIIColors.yellow(warning_msg)

        # Check if running in an interactive terminal
        if sys.stdin.isatty():
            response = input("Do you want to continue? (yes/no): ")
            if response.lower() != "yes":
                ASCIIColors.red("Test program canceled")
                return False
    return True


async def initialize_graph_storage():
    """
    Initialize the corresponding graph storage instance according to the environment variables
    Returns the initialized storage instance
    """
    # Get the graph storage type from the environment variable
    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")

    # Verify that the storage type is valid
    try:
        verify_storage_implementation("GRAPH_STORAGE", graph_storage_type)
    except ValueError as e:
        ASCIIColors.red(f"Error: {str(e)}")
        ASCIIColors.yellow(
            f"Supported graph storage types: {', '.join(STORAGE_IMPLEMENTATIONS['GRAPH_STORAGE']['implementations'])}"
        )
        return None

    # Check required environment variables
    required_env_vars = STORAGE_ENV_REQUIREMENTS.get(graph_storage_type, [])
    missing_env_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_env_vars:
        ASCIIColors.red(
            f"Error: {graph_storage_type} requires the following environment variables, but they are not set: {', '.join(missing_env_vars)}"
        )
        return None

    # Dynamically import the corresponding modules
    module_path = STORAGES.get(graph_storage_type)
    if not module_path:
        ASCIIColors.red(f"Error: Module path not found for {graph_storage_type}")
        return None

    try:
        module = importlib.import_module(module_path, package="lightrag")
        storage_class = getattr(module, graph_storage_type)
    except (ImportError, AttributeError) as e:
        ASCIIColors.red(f"Error: import {graph_storage_type} failed: {str(e)}")
        return None

    # Initialize the storage instance
    global_config = {
        "embedding_batch_num": 10, # batch size
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.5 # Cosine similarity threshold
        },
        "working_dir": os.environ.get("WORKING_DIR", "./rag_storage"), # working directory
    }

    # If you use NetworkXStorage, you need to initialize shared_storage first
    if graph_storage_type == "NetworkXStorage":
        initialize_share_data() # Use single process mode

    try:
        storage = storage_class(
            namespace="test_graph",
            global_config=global_config,
            embedding_func=mock_embedding_func,
        )

        # Initialize the connection
        await storage.initialize()
        return storage
    except Exception as e:
        ASCIIColors.red(f"Error: Failed to initialize {graph_storage_type}: {str(e)}")
        return None


async def test_graph_basic(storage):
    """
    Test the basic operations of the graph database:
    1. Use upsert_node to insert two nodes
    2. Use upsert_edge to insert an edge connecting two nodes
    3. Use get_node to read a node
    4. Use get_edge to read an edge
    """
    try:
        # 1. Insert the first node
        node1_id = "Artificial Intelligence"
        node1_data = {
            "entity_id": node1_id,
            "description": "Artificial intelligence is a branch of computer science that seeks to understand the nature of intelligence and produce new intelligent machines that can respond in a manner similar to human intelligence.",
            "keywords": "AI, machine learning, deep learning",
            "entity_type": "Technology area",
        }
        print(f"Insert node 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 2. Insert the second node
        node2_id = "Machine Learning"
        node2_data = {
            "entity_id": node2_id,
            "description": "Machine learning is a branch of artificial intelligence that uses statistical methods to enable computer systems to learn without being explicitly programmed.",
            "keywords": "supervised learning, unsupervised learning, reinforcement learning",
            "entity_type": "Technology area",
        }
        print(f"Insert node 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 3. Insert connecting edges
        edge_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of machine learning",
        }
        print(f"Insert edge: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge_data)

        # 4. Read node attributes
        print(f"Read node attributes: {node1_id}")
        node1_props = await storage.get_node(node1_id)
        if node1_props:
            print(f"Successfully read node attributes: {node1_id}")
            print(f"Node description: {node1_props.get('description', 'No description')}")
            print(f"Node type: {node1_props.get('entity_type', 'no type')}")
            print(f"Node keywords: {node1_props.get('keywords', 'No keywords')}")
            # Verify that the returned attributes are correct
            assert (
                node1_props.get("entity_id") == node1_id
            ), f"Node ID mismatch: expected {node1_id}, actual {node1_props.get('entity_id')}"
            assert (
                node1_props.get("description") == node1_data["description"]
            ), "Node description does not match"
            assert (
                node1_props.get("entity_type") == node1_data["entity_type"]
            ), "Node type mismatch"
        else:
            print(f"Failed to read node attributes: {node1_id}")
            assert False, f"Failed to read node attribute: {node1_id}"

        # 5. Read edge attributes
        print(f"Read edge attributes: {node1_id} -> {node2_id}")
        edge_props = await storage.get_edge(node1_id, node2_id)
        if edge_props:
            print(f"Successfully read edge attributes: {node1_id} -> {node2_id}")
            print(f"Edge relationship: {edge_props.get('relationship', 'No relationship')}")
            print(f"Edge description: {edge_props.get('description', 'No description')}")
            print(f"Edge weight: {edge_props.get('weight', 'no weight')}")
            # Verify that the returned attributes are correct
            assert (
                edge_props.get("relationship") == edge_data["relationship"]
            ), "Edge relation mismatch"
            assert (
                edge_props.get("description") == edge_data["description"]
            ), "Edge description does not match"
            assert edge_props.get("weight") == edge_data["weight"], "edge weights do not match"
        else:
            print(f"Failed to read edge attributes: {node1_id} -> {node2_id}")
            assert False, f"Failed to read edge attribute: {node1_id} -> {node2_id}"

        # 5.1 Verify undirected graph properties - read reverse edge attributes
        print(f"Read reverse edge attributes: {node2_id} -> {node1_id}")
        reverse_edge_props = await storage.get_edge(node2_id, node1_id)
        if reverse_edge_props:
            print(f"Successfully read reverse edge attributes: {node2_id} -> {node1_id}")
            print(f"Reverse edge relationship: {reverse_edge_props.get('relationship', 'No relationship')}")
            print(f"Reverse edge description: {reverse_edge_props.get('description', 'No description')}")
            print(f"Reverse edge weight: {reverse_edge_props.get('weight', 'no weight')}")
            # Verify that the forward and reverse edge attributes are the same
            assert (
                edge_props == reverse_edge_props
            ), "The forward and reverse edge attributes are inconsistent, and the undirected graph feature verification failed"
            print("Undirected graph feature verification successful: forward and reverse edge attributes are consistent")
        else:
            print(f"Failed to read reverse edge attribute: {node2_id} -> {node1_id}")
            assert (
                False
            ), f"Failed to read reverse edge attribute: {node2_id} -> {node1_id}, undirected graph property validation failed"

        print("Basic test completed, data has been saved in the database")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during testing: {str(e)}")
        return False


async def test_graph_advanced(storage):
    """
    Test advanced operations of graph database:
    1. Use node_degree to get the degree of a node
    2. Use edge_degree to get the degree of the edge
    3. Use get_node_edges to get all the edges of a node
    4. Use get_all_labels to get all labels
    5. Use get_knowledge_graph to get the knowledge graph
    6. Use delete_node to delete a node
    7. Use remove_nodes to delete nodes in batches
    8. Remove edges using remove_edges
    9. Use drop to clean up data
    """
    try:
        # 1. Insert test data
        # Insert node 1: Artificial Intelligence
        node1_id = "Artificial Intelligence"
        node1_data = {
            "entity_id": node1_id,
            "description": "Artificial intelligence is a branch of computer science that seeks to understand the nature of intelligence and produce new intelligent machines that can respond in a manner similar to human intelligence.",
            "keywords": "AI, machine learning, deep learning",
            "entity_type": "Technology area",
        }
        print(f"Insert node 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # Insert node 2: machine learning
        node2_id = "Machine Learning"
        node2_data = {
            "entity_id": node2_id,
            "description": "Machine learning is a branch of artificial intelligence that uses statistical methods to enable computer systems to learn without being explicitly programmed.",
            "keywords": "supervised learning, unsupervised learning, reinforcement learning",
            "entity_type": "Technology area",
        }
        print(f"Insert node 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # Insert node 3: Deep learning
        node3_id = "Deep Learning"
        node3_data = {
            "entity_id": node3_id,
            "description": "Deep learning is a branch of machine learning that uses multi-layer neural networks to simulate the learning process of the human brain.",
            "keywords": "neural network,CNN,RNN",
            "entity_type": "Technology area",
        }
        print(f"Insert node 3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # Insert edge 1: Artificial Intelligence -> Machine Learning
        edge1_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of machine learning",
        }
        print(f"Insert edge 1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # Insert edge 2: Machine Learning -> Deep Learning
        edge2_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of machine learning includes the subfield of deep learning",
        }
        print(f"Insert edge 2: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 2. Test node_degree - Get the degree of the node
        print(f"== test node_degree: {node1_id}")
        node1_degree = await storage.node_degree(node1_id)
        print(f"The degree of node {node1_id}: {node1_degree}")
        assert node1_degree == 1, f"The degree of node {node1_id} should be 1, but is actually {node1_degree}"

        # 2.1 Test the degree of all nodes
        print("== Test the degree of all nodes")
        node2_degree = await storage.node_degree(node2_id)
        node3_degree = await storage.node_degree(node3_id)
        print(f"The degree of node {node2_id}: {node2_degree}")
        print(f"The degree of node {node3_id}: {node3_degree}")
        assert node2_degree == 2, f"The degree of node {node2_id} should be 2, but is actually {node2_degree}"
        assert node3_degree == 1, f"The degree of node {node3_id} should be 1, but is actually {node3_degree}"

        # 3. Test edge_degree - Get the degree of the edge
        print(f"== test edge_degree: {node1_id} -> {node2_id}")
        edge_degree = await storage.edge_degree(node1_id, node2_id)
        print(f"The degree of edge {node1_id} -> {node2_id}: {edge_degree}")
        assert (
            edge_degree == 3
        ), f"The degree of edge {node1_id} -> {node2_id} should be 3, but is actually {edge_degree}"

        # 3.1 Test the degree of the reverse edge - verify the undirected graph characteristics
        print(f"== Test the degree of the reverse edge: {node2_id} -> {node1_id}")
        reverse_edge_degree = await storage.edge_degree(node2_id, node1_id)
        print(f"The degree of the reverse edge {node2_id} -> {node1_id}: {reverse_edge_degree}")
        assert (
            edge_degree == reverse_edge_degree
        ), "The degrees of the forward and reverse edges are inconsistent, and the undirected graph feature verification failed"
        print("Undirected graph feature verification successful: the degrees of the forward and reverse edges are consistent")

        # 4. Test get_node_edges - Get all edges of a node
        print(f"== test get_node_edges: {node2_id}")
        node2_edges = await storage.get_node_edges(node2_id)
        print(f"All edges of node {node2_id}: {node2_edges}")
        assert (
            len(node2_edges) == 2
        ), f"Node {node2_id} should have 2 edges, but actually has {len(node2_edges)}"

        # 4.1 Verify the undirected graph characteristics of node edges
        print("== Verify the undirected graph characteristics of the node edge")
        # Check whether it contains connections with node1 and node3 (regardless of direction)
        has_connection_with_node1 = False
        has_connection_with_node3 = False
        for edge in node2_edges:
            # Check if there is a connection to node1 (regardless of direction)
            if (edge[0] == node1_id and edge[1] == node2_id) or (
                edge[0] == node2_id and edge[1] == node1_id
            ):
                has_connection_with_node1 = True
            # Check if there is a connection to node3 (regardless of direction)
            if (edge[0] == node2_id and edge[1] == node3_id) or (
                edge[0] == node3_id and edge[1] == node2_id
            ):
                has_connection_with_node3 = True

        assert (
            has_connection_with_node1
        ), f"The edge list of node {node2_id} should contain a connection with {node1_id}"
        assert (
            has_connection_with_node3
        ), f"The edge list of node {node2_id} should contain a connection with {node3_id}"
        print(f"Undirected graph feature verification successful: the edge list of node {node2_id} contains all related edges")

        # 5. Test get_all_labels - Get all labels
        print("== test get_all_labels")
        all_labels = await storage.get_all_labels()
        print(f"All labels: {all_labels}")
        assert len(all_labels) == 3, f"Expected 3 labels, actually {len(all_labels)}"
        assert node1_id in all_labels, f"{node1_id} should be in the label list"
        assert node2_id in all_labels, f"{node2_id} should be in the label list"
        assert node3_id in all_labels, f"{node3_id} should be in the label list"

        # 6. Test get_knowledge_graph - Get the knowledge graph
        print("== Test get_knowledge_graph")
        kg = await storage.get_knowledge_graph("*", max_depth=2, max_nodes=10)
        print(f"Number of knowledge graph nodes: {len(kg.nodes)}")
        print(f"Number of edges in the knowledge graph: {len(kg.edges)}")
        assert isinstance(kg, KnowledgeGraph), "The return result should be of KnowledgeGraph type"
        assert len(kg.nodes) == 3, f"The knowledge graph should have 3 nodes, but actually has {len(kg.nodes)}"
        assert len(kg.edges) == 2, f"The knowledge graph should have 2 edges, but actually has {len(kg.edges)}"

        # 7. Test delete_node - delete a node
        print(f"== test delete_node: {node3_id}")
        await storage.delete_node(node3_id)
        node3_props = await storage.get_node(node3_id)
        print(f"Query node properties after deletion {node3_id}: {node3_props}")
        assert node3_props is None, f"Node {node3_id} should have been deleted"

        # Re-insert node 3 for subsequent testing
        await storage.upsert_node(node3_id, node3_data)
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 8. Test remove_edges - remove edges
        print(f"== test remove_edges: {node2_id} -> {node3_id}")
        await storage.remove_edges([(node2_id, node3_id)])
        edge_props = await storage.get_edge(node2_id, node3_id)
        print(f"Query edge properties after deletion {node2_id} -> {node3_id}: {edge_props}")
        assert edge_props is None, f"edge {node2_id} -> {node3_id} should have been deleted"

        # 8.1 Verify the undirected graph property of deleting edges
        print(f"== Verify the undirected graph characteristics of deleting edges: {node3_id} -> {node2_id}")
        reverse_edge_props = await storage.get_edge(node3_id, node2_id)
        print(f"Query reverse edge properties after deletion {node3_id} -> {node2_id}: {reverse_edge_props}")
        assert (
            reverse_edge_props is None
        ), f"The reverse edge {node3_id} -> {node2_id} should also be deleted, and the undirected graph feature verification failed"
        print("Undirected graph feature verification is successful: after deleting an edge in one direction, the reverse edge is also deleted")

        # 9. Test remove_nodes - batch delete nodes
        print(f"== test remove_nodes: [{node2_id}, {node3_id}]")
        await storage.remove_nodes([node2_id, node3_id])
        node2_props = await storage.get_node(node2_id)
        node3_props = await storage.get_node(node3_id)
        print(f"Query node properties after deletion {node2_id}: {node2_props}")
        print(f"Query node properties after deletion {node3_id}: {node3_props}")
        assert node2_props is None, f"Node {node2_id} should have been deleted"
        assert node3_props is None, f"Node {node3_id} should have been deleted"

        print("\nAdvanced test completed")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during testing: {str(e)}")
        return False


async def test_graph_batch_operations(storage):
    """
    Test batch operations on graph database:
    1. Use get_nodes_batch to get the properties of multiple nodes in batches
    2. Use node_degrees_batch to get the degrees of multiple nodes in batches
    3. Use edge_degrees_batch to get the degrees of multiple edges in batches
    4. Use get_edges_batch to get the attributes of multiple edges in batches
    5. Use get_nodes_edges_batch to get all edges of multiple nodes in batches
    """
    try:
        # 1. Insert test data
        # Insert node 1: Artificial Intelligence
        node1_id = "Artificial Intelligence"
        node1_data = {
            "entity_id": node1_id,
            "description": "Artificial intelligence is a branch of computer science that seeks to understand the nature of intelligence and produce new intelligent machines that can respond in a manner similar to human intelligence.",
            "keywords": "AI, machine learning, deep learning",
            "entity_type": "Technology area",
        }
        print(f"Insert node 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # Insert node 2: machine learning
        node2_id = "Machine Learning"
        node2_data = {
            "entity_id": node2_id,
            "description": "Machine learning is a branch of artificial intelligence that uses statistical methods to enable computer systems to learn without being explicitly programmed.",
            "keywords": "supervised learning, unsupervised learning, reinforcement learning",
            "entity_type": "Technology area",
        }
        print(f"Insert node 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # Insert node 3: Deep learning
        node3_id = "Deep Learning"
        node3_data = {
            "entity_id": node3_id,
            "description": "Deep learning is a branch of machine learning that uses multi-layer neural networks to simulate the learning process of the human brain.",
            "keywords": "neural network,CNN,RNN",
            "entity_type": "Technology area",
        }
        print(f"Insert node 3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # Insert node 4: Natural language processing
        node4_id = "Natural Language Processing"
        node4_data = {
            "entity_id": node4_id,
            "description": "Natural language processing is a branch of artificial intelligence that focuses on enabling computers to understand and process human language.",
            "keywords": "NLP, text analysis, language model",
            "entity_type": "Technology area",
        }
        print(f"Insert node 4: {node4_id}")
        await storage.upsert_node(node4_id, node4_data)

        # Insert node 5: Computer Vision
        node5_id = "Computer Vision"
        node5_data = {
            "entity_id": node5_id,
            "description": "Computer vision is a branch of artificial intelligence that focuses on enabling computers to extract information from images or videos.",
            "keywords": "CV, image recognition, object detection",
            "entity_type": "Technology area",
        }
        print(f"Insert node 5: {node5_id}")
        await storage.upsert_node(node5_id, node5_data)

        # Insert edge 1: Artificial Intelligence -> Machine Learning
        edge1_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of machine learning",
        }
        print(f"Insert edge 1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # Insert edge 2: Machine Learning -> Deep Learning
        edge2_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of machine learning includes the subfield of deep learning",
        }
        print(f"Insert edge 2: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # Insert edge 3: Artificial Intelligence -> Natural Language Processing
        edge3_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of natural language processing",
        }
        print(f"Insert edge 3: {node1_id} -> {node4_id}")
        await storage.upsert_edge(node1_id, node4_id, edge3_data)

        # Insert edge 4: Artificial Intelligence -> Computer Vision
        edge4_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "The field of artificial intelligence includes the subfield of computer vision",
        }
        print(f"Insert edge 4: {node1_id} -> {node5_id}")
        await storage.upsert_edge(node1_id, node5_id, edge4_data)

        # Insert edge 5: Deep Learning -> Natural Language Processing
        edge5_data = {
            "relationship": "Applies to",
            "weight": 0.8,
            "description": "Deep learning technology applied in natural language processing",
        }
        print(f"Insert edge 5: {node3_id} -> {node4_id}")
        await storage.upsert_edge(node3_id, node4_id, edge5_data)

        # Insert edge 6: Deep Learning -> Computer Vision
        edge6_data = {
            "relationship": "Applies to",
            "weight": 0.8,
            "description": "Deep learning technology applied in computer vision",
        }
        print(f"Insert edge 6: {node3_id} -> {node5_id}")
        await storage.upsert_edge(node3_id, node5_id, edge6_data)

        # 2. Test get_nodes_batch - get the properties of multiple nodes in batches
        print("== test get_nodes_batch")
        node_ids = [node1_id, node2_id, node3_id]
        nodes_dict = await storage.get_nodes_batch(node_ids)
        print(f"Batch get node attribute results: {nodes_dict.keys()}")
        assert len(nodes_dict) == 3, f"should return 3 nodes, actually returns {len(nodes_dict)}"
        assert node1_id in nodes_dict, f"{node1_id} should be in the returned result"
        assert node2_id in nodes_dict, f"{node2_id} should be in the returned result"
        assert node3_id in nodes_dict, f"{node3_id} should be in the returned result"
        assert (
            nodes_dict[node1_id]["description"] == node1_data["description"]
        ), f"{node1_id} description does not match"
        assert (
            nodes_dict[node2_id]["description"] == node2_data["description"]
        ), f"{node2_id} description does not match"
        assert (
            nodes_dict[node3_id]["description"] == node3_data["description"]
        ), f"{node3_id} description does not match"

        # 3. Test node_degrees_batch - get the degrees of multiple nodes in batches
        print("== test node_degrees_batch")
        node_degrees = await storage.node_degrees_batch(node_ids)
        print(f"Batch get node degree results: {node_degrees}")
        assert (
            len(node_degrees) == 3
        ), f" should return the degrees of 3 nodes, but actually returns {len(node_degrees)}"
        assert node1_id in node_degrees, f"{node1_id} should be in the returned result"
        assert node2_id in node_degrees, f"{node2_id} should be in the returned result"
        assert node3_id in node_degrees, f"{node3_id} should be in the returned result"
        assert (
            node_degrees[node1_id] == 3
        ), f"{node1_id} degree should be 3, actual is {node_degrees[node1_id]}"
        assert (
            node_degrees[node2_id] == 2
        ), f"{node2_id} degree should be 2, actual is {node_degrees[node2_id]}"
        assert (
            node_degrees[node3_id] == 3
        ), f"{node3_id} degree should be 3, actually {node_degrees[node3_id]}"

        # 4. Test edge_degrees_batch - get the degrees of multiple edges in batches
        print("== test edge_degrees_batch")
        edges = [(node1_id, node2_id), (node2_id, node3_id), (node3_id, node4_id)]
        edge_degrees = await storage.edge_degrees_batch(edges)
        print(f"Batch get edge degree results: {edge_degrees}")
        assert (
            len(edge_degrees) == 3
        ), f" should return the degrees of 3 edges, but actually returns {len(edge_degrees)}"
        assert (
            node1_id,
            node2_id,
        ) in edge_degrees, f"The edge {node1_id} -> {node2_id} should be in the returned result"
        assert (
            node2_id,
            node3_id,
        ) in edge_degrees, f"The edge {node2_id} -> {node3_id} should be in the returned result"
        assert (
            node3_id,
            node4_id,
        ) in edge_degrees, f"The edge {node3_id} -> {node4_id} should be in the returned result"
        # Verify that the degree of the edge is correct (source node degree + target node degree)
        assert (
            edge_degrees[(node1_id, node2_id)] == 5
        ), f"The degree of edge {node1_id} -> {node2_id} should be 5, but is actually {edge_degrees[(node1_id, node2_id)]}"
        assert (
            edge_degrees[(node2_id, node3_id)] == 5
        ), f"The degree of edge {node2_id} -> {node3_id} should be 5, but is actually {edge_degrees[(node2_id, node3_id)]}"
        assert (
            edge_degrees[(node3_id, node4_id)] == 5
        ), f"The degree of edge {node3_id} -> {node4_id} should be 5, but is actually {edge_degrees[(node3_id, node4_id)]}"

        # 5. Test get_edges_batch - get the attributes of multiple edges in batches
        print("== Test get_edges_batch")
        # Convert a list of tuples to a Neo4j-style list of dictionaries
        edge_dicts = [{"src": src, "tgt": tgt} for src, tgt in edges]
        edges_dict = await storage.get_edges_batch(edge_dicts)
        print(f"Batch get edge attribute results: {edges_dict.keys()}")
        assert len(edges_dict) == 3, f"Should return 3 edge attributes, actually returns {len(edges_dict)}"
        assert (
            node1_id,
            node2_id,
        ) in edges_dict, f"The edge {node1_id} -> {node2_id} should be in the returned result"
        assert (
            node2_id,
            node3_id,
        ) in edges_dict, f"The edge {node2_id} -> {node3_id} should be in the returned result"
        assert (
            node3_id,
            node4_id,
        ) in edges_dict, f"The edge {node3_id} -> {node4_id} should be in the returned result"
        assert (
            edges_dict[(node1_id, node2_id)]["relationship"]
            == edge1_data["relationship"]
        ), f"Edge {node1_id} -> {node2_id} relationship does not match"
        assert (
            edges_dict[(node2_id, node3_id)]["relationship"]
            == edge2_data["relationship"]
        ), f"Edge {node2_id} -> {node3_id} relationship does not match"
        assert (
            edges_dict[(node3_id, node4_id)]["relationship"]
            == edge5_data["relationship"]
        ), f"Edge {node3_id} -> {node4_id} relationship does not match"

        # 5.1 Test batch acquisition of reverse edges - verify undirected graph characteristics
        print("== Test batch acquisition of reverse edges")
        # Create a dictionary list of reverse edges
        reverse_edge_dicts = [{"src": tgt, "tgt": src} for src, tgt in edges]
        reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)
        print(f"Batch get reverse edge attribute results: {reverse_edges_dict.keys()}")
        assert (
            len(reverse_edges_dict) == 3
        ), f" should return the attributes of 3 reverse edges, but actually returns {len(reverse_edges_dict)}"

        # Verify that the attributes of the forward and reverse edges are consistent
        for (src, tgt), props in edges_dict.items():
            assert (
                tgt,
                src,
            ) in reverse_edges_dict, f"The reverse edge {tgt} -> {src} should be in the returned result"
            assert (
                props == reverse_edges_dict[(tgt, src)]
            ), f"The attributes of edge {src} -> {tgt} and reverse edge {tgt} -> {src} are inconsistent"

        print("Undirected graph feature verification successful: the forward and reverse edge attributes obtained in batches are consistent")

        # 6. Test get_nodes_edges_batch - get all edges of multiple nodes in batches
        print("== test get_nodes_edges_batch")
        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node3_id])
        print(f"Batch get node edge results: {nodes_edges.keys()}")
        assert (
            len(nodes_edges) == 2
        ), f" should return the edges of 2 nodes, but actually returns {len(nodes_edges)}"
        assert node1_id in nodes_edges, f"{node1_id} should be in the returned result"
        assert node3_id in nodes_edges, f"{node3_id} should be in the returned result"
        assert (
            len(nodes_edges[node1_id]) == 3
        ), f"{node1_id} should have 3 edges, but actually has {len(nodes_edges[node1_id])}"
        assert (
            len(nodes_edges[node3_id]) == 3
        ), f"{node3_id} should have 3 edges, but actually has {len(nodes_edges[node3_id])}"

        # 6.1 Verify the undirected graph characteristics of batch acquisition of node edges
        print("== Verify the undirected graph characteristics of batch acquisition of node edges")

        # Check if the edge of node 1 contains all related edges (regardless of direction)
        node1_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if src == node1_id
        ]
        node1_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if tgt == node1_id
        ]
        print(f"Outgoing edges of node {node1_id}: {node1_outgoing_edges}")
        print(f"Incoming edges of node {node1_id}: {node1_incoming_edges}")

        # Check if edges to machine learning, natural language processing, and computer vision are included
        has_edge_to_node2 = any(tgt == node2_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node4 = any(tgt == node4_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node5 = any(tgt == node5_id for _, tgt in node1_outgoing_edges)

        assert has_edge_to_node2, f"The edge list of node {node1_id} should contain an edge to {node2_id}"
        assert has_edge_to_node4, f"The edge list of node {node1_id} should contain an edge to {node4_id}"
        assert has_edge_to_node5, f"The edge list of node {node1_id} should contain an edge to {node5_id}"

        # Check if the edge of node 3 contains all related edges (regardless of direction)
        node3_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if src == node3_id
        ]
        node3_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if tgt == node3_id
        ]
        print(f"Outgoing edges of node {node3_id}: {node3_outgoing_edges}")
        print(f"Incoming edges of node {node3_id}: {node3_incoming_edges}")

        # Check if connections to machine learning, natural language processing, and computer vision are included (ignore direction)
        has_connection_with_node2 = any(
            (src == node2_id and tgt == node3_id)
            or (src == node3_id and tgt == node2_id)
            for src, tgt in nodes_edges[node3_id]
        )
        has_connection_with_node4 = any(
            (src == node3_id and tgt == node4_id)
            or (src == node4_id and tgt == node3_id)
            for src, tgt in nodes_edges[node3_id]
        )
        has_connection_with_node5 = any(
            (src == node3_id and tgt == node5_id)
            or (src == node5_id and tgt == node3_id)
            for src, tgt in nodes_edges[node3_id]
        )

        assert (
            has_connection_with_node2
        ), f"The edge list of node {node3_id} should contain a connection with {node2_id}"
        assert (
            has_connection_with_node4
        ), f"The edge list of node {node3_id} should contain a connection with {node4_id}"
        assert (
            has_connection_with_node5
        ), f"The edge list of node {node3_id} should contain a connection with {node5_id}"

        print("Undirected graph feature verification successful: The node edges obtained in batches contain all related edges (regardless of direction)")

        print("\nBatch operation test completed")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during testing: {str(e)}")
        return False


async def test_graph_special_characters(storage):
    """
    Test the processing of special characters in the graph database:
    1. Test node names and descriptions contain single quotes, double quotes, and backslashes
    2. The description of the test edge contains single quotes, double quotes, and backslashes
    3. Verify that special characters are saved and retrieved correctly
    """
    try:
        # 1. Test for special characters in node names
        node1_id = "Node containing 'single quote'"
        node1_data = {
            "entity_id": node1_id,
            "description": "This description contains 'single quote', \"double quote\" and \\backslash",
            "keywords": "Special characters, quotation marks, escape",
            "entity_type": "Test Node",
        }
        print(f"Insert node 1 containing special characters: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 2. Double quotes in test node names
        node2_id = 'Node containing "double quotes"'
        node2_data = {
            "entity_id": node2_id,
            "description": "This description contains both 'single quotes' and \"double quotes\" and \\backslash\\ paths",
            "keywords": "Special characters, quotation marks, JSON",
            "entity_type": "Test Node",
        }
        print(f"Insert node 2 containing special characters: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 3. Test for backslashes in node names
        node3_id = "Node containing \\backslash\\"
        node3_data = {
            "entity_id": node3_id,
            "description": "This description contains the Windows path C:\\Program Files\\ and the escape characters \\n\\t",
            "keywords": "backslash, path, escape",
            "entity_type": "Test Node",
        }
        print(f"Insert node 3 containing special characters: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 4. Test special characters in edge descriptions
        edge1_data = {
            "relationship": "Special 'relationship'",
            "weight": 1.0,
            "description": "This edge description contains 'single quote', \"double quote\" and \\backslash",
        }
        print(f"Insert edge containing special characters: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 5. Test more complex special character combinations in edge descriptions
        edge2_data = {
            "relationship": 'Complex "relationship" \\ type',
            "weight": 0.8,
            "description": "Contains SQL injection attempt: SELECT * FROM users WHERE name='admin'--",
        }
        print(f"Insert edge containing complex special characters: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 6. Verify that node special characters are saved correctly
        print("\n== special characters for verification nodes")
        for node_id, original_data in [
            (node1_id, node1_data),
            (node2_id, node2_data),
            (node3_id, node3_data),
        ]:
            node_props = await storage.get_node(node_id)
            if node_props:
                print(f"Successfully read node: {node_id}")
                print(f"Node description: {node_props.get('description', 'No description')}")

                # Verify that the node ID is saved correctly
                assert (
                    node_props.get("entity_id") == node_id
                ), f"Node ID mismatch: expected {node_id}, actual {node_props.get('entity_id')}"

                # Verify that the description was saved correctly
                assert (
                    node_props.get("description") == original_data["description"]
                ), f"Node description mismatch: expected {original_data['description']}, actual {node_props.get('description')}"

                print(f"Node {node_id} Special character verification successful")
            else:
                print(f"Failed to read node attributes: {node_id}")
                assert False, f"Failed to read node attribute: {node_id}"

        # 7. Verify that special characters are saved correctly
        print("\n== special characters on verification side")
        edge1_props = await storage.get_edge(node1_id, node2_id)
        if edge1_props:
            print(f"Successfully read edge: {node1_id} -> {node2_id}")
            print(f"Edge relationship: {edge1_props.get('relationship', 'No relationship')}")
            print(f"Edge description: {edge1_props.get('description', 'No description')}")

            # Verify that the edge relationship is saved correctly
            assert (
                edge1_props.get("relationship") == edge1_data["relationship"]
            ), f"Edge relation mismatch: expected {edge1_data['relationship']}, actual {edge1_props.get('relationship')}"

            # Verify that the edge description is saved correctly
            assert (
                edge1_props.get("description") == edge1_data["description"]
            ), f"Edge description mismatch: expected {edge1_data['description']}, actual {edge1_props.get('description')}"

            print(f"Edge {node1_id} -> {node2_id} Special character verification successful")
        else:
            print(f"Failed to read edge attributes: {node1_id} -> {node2_id}")
            assert False, f"Failed to read edge attribute: {node1_id} -> {node2_id}"

        edge2_props = await storage.get_edge(node2_id, node3_id)
        if edge2_props:
            print(f"Successfully read edge: {node2_id} -> {node3_id}")
            print(f"Edge relationship: {edge2_props.get('relationship', 'No relationship')}")
            print(f"Edge description: {edge2_props.get('description', 'No description')}")

            # Verify that the edge relationship is saved correctly
            assert (
                edge2_props.get("relationship") == edge2_data["relationship"]
            ), f"Edge relation mismatch: expected {edge2_data['relationship']}, actual {edge2_props.get('relationship')}"

            # Verify that the edge description is saved correctly
            assert (
                edge2_props.get("description") == edge2_data["description"]
            ), f"Edge description mismatch: expected {edge2_data['description']}, actual {edge2_props.get('description')}"

            print(f"Edge {node2_id} -> {node3_id} Special character verification successful")
        else:
            print(f"Failed to read edge attributes: {node2_id} -> {node3_id}")
            assert False, f"Failed to read edge attribute: {node2_id} -> {node3_id}"

        print("\nSpecial character test completed, data saved in the database")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during testing: {str(e)}")
        return False


async def test_graph_undirected_property(storage):
    """
    Specifically test the undirected graph characteristics of graph storage:
    1. Verify that after inserting an edge in one direction, the reverse query can obtain the same result
    2. Verify that the attributes of the edge are consistent in forward and reverse queries
    3. Verify that after deleting an edge in one direction, the edge in the other direction is also deleted
    4. Verify the undirected graph characteristics in batch operations
    """
    try:
        # 1. Insert test data
        # Insert node 1: Computer Science
        node1_id = "Computer Science"
        node1_data = {
            "entity_id": node1_id,
            "description": "Computer science is the study of computers and their applications.",
            "keywords": "computer, science, technology",
            "entity_type": "Subject",
        }
        print(f"Insert node 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # Insert node 2: data structure
        node2_id = "data structure"
        node2_data = {
            "entity_id": node2_id,
            "description": "Data structure is a fundamental concept in computer science that is used to organize and store data.",
            "keywords": "data, structure, organization",
            "entity_type": "Concept",
        }
        print(f"Insert node 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # Insert node 3: Algorithm
        node3_id = "algorithm"
        node3_data = {
            "entity_id": node3_id,
            "description": "Algorithms are the steps and methods to solve problems.",
            "keywords": "algorithm, steps, methods",
            "entity_type": "Concept",
        }
        print(f"Insert node 3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 2. Test the undirected graph characteristics after inserting edges
        print("\n== Test the undirected graph characteristics after inserting edges")

        # Insert edge 1: Computer Science -> Data Structure
        edge1_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "Computer science includes the concept of data structure",
        }
        print(f"Insert edge 1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # Verify forward query
        forward_edge = await storage.get_edge(node1_id, node2_id)
        print(f"Forward edge attribute: {forward_edge}")
        assert forward_edge is not None, f"Failed to read forward edge attribute: {node1_id} -> {node2_id}"

        # Verify reverse query
        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(f"Reverse edge attribute: {reverse_edge}")
        assert reverse_edge is not None, f"Failed to read reverse edge attribute: {node2_id} -> {node1_id}"

        # Verify that the forward and reverse edge attributes are consistent
        assert (
            forward_edge == reverse_edge
        ), "The forward and reverse edge attributes are inconsistent, and the undirected graph feature verification failed"
        print("Undirected graph feature verification successful: forward and reverse edge attributes are consistent")

        # 3. Test the undirected graph properties of edge degrees
        print("\n== Undirected graph properties of testing edge degrees")

        # Insert edge 2: Computer Science -> Algorithms
        edge2_data = {
            "relationship": "includes",
            "weight": 1.0,
            "description": "Computer science includes the concept of algorithms",
        }
        print(f"Insert edge 2: {node1_id} -> {node3_id}")
        await storage.upsert_edge(node1_id, node3_id, edge2_data)

        # Verify the degrees of forward and reverse edges
        forward_degree = await storage.edge_degree(node1_id, node2_id)
        reverse_degree = await storage.edge_degree(node2_id, node1_id)
        print(f"The degree of the forward edge {node1_id} -> {node2_id}: {forward_degree}")
        print(f"The degree of the reverse edge {node2_id} -> {node1_id}: {reverse_degree}")
        assert (
            forward_degree == reverse_degree
        ), "The degrees of the forward and reverse edges are inconsistent, and the undirected graph feature verification failed"
        print("Undirected graph feature verification successful: the degrees of forward and reverse edges are consistent")

        # 4. Test the undirected graph feature of deleting edges
        print("\n== Test the undirected graph characteristics of deleting edges")

        # Delete the forward edge
        print(f"Delete edge: {node1_id} -> {node2_id}")
        await storage.remove_edges([(node1_id, node2_id)])

        # Verify that the forward edge is deleted
        forward_edge = await storage.get_edge(node1_id, node2_id)
        print(f"Query the forward edge attribute after deletion {node1_id} -> {node2_id}: {forward_edge}")
        assert forward_edge is None, f"edge {node1_id} -> {node2_id} should have been deleted"

        # Verify that the reverse edge is also deleted
        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(f"Query reverse edge attribute after deletion {node2_id} -> {node1_id}: {reverse_edge}")
        assert (
            reverse_edge is None
        ), f"The reverse edge {node2_id} -> {node1_id} should also be deleted, and the undirected graph feature verification failed"
        print("Undirected graph feature verification is successful: after deleting an edge in one direction, the reverse edge is also deleted")

        # 5. Test the undirected graph feature in batch operation
        print("\n== Test the undirected graph feature in batch operation")

        # Reinsert edges
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # Get edge attributes in batches
        edge_dicts = [
            {"src": node1_id, "tgt": node2_id},
            {"src": node1_id, "tgt": node3_id},
        ]
        reverse_edge_dicts = [
            {"src": node2_id, "tgt": node1_id},
            {"src": node3_id, "tgt": node1_id},
        ]

        edges_dict = await storage.get_edges_batch(edge_dicts)
        reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)

        print(f"Batch get forward edge attribute results: {edges_dict.keys()}")
        print(f"Batch get reverse edge attribute results: {reverse_edges_dict.keys()}")

        # Verify that the attributes of the forward and reverse edges are consistent
        for (src, tgt), props in edges_dict.items():
            assert (
                tgt,
                src,
            ) in reverse_edges_dict, f"The reverse edge {tgt} -> {src} should be in the returned result"
            assert (
                props == reverse_edges_dict[(tgt, src)]
            ), f"The attributes of edge {src} -> {tgt} and reverse edge {tgt} -> {src} are inconsistent"

        print("Undirected graph feature verification successful: the forward and reverse edge attributes obtained in batches are consistent")

        # 6. Test the undirected graph feature of batch acquisition of node edges
        print("\n== Test the undirected graph characteristics of batch acquisition of node edges")

        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node2_id])
        print(f"Batch get node edge results: {nodes_edges.keys()}")

        # Check if the edge of node 1 contains all related edges (regardless of direction)
        node1_edges = nodes_edges[node1_id]
        node2_edges = nodes_edges[node2_id]

        # Check if node 1 has edges to nodes 2 and 3
        has_edge_to_node2 = any(
            (src == node1_id and tgt == node2_id) for src, tgt in node1_edges
        )
        has_edge_to_node3 = any(
            (src == node1_id and tgt == node3_id) for src, tgt in node1_edges
        )

        assert has_edge_to_node2, f"The edge list of node {node1_id} should contain an edge to {node2_id}"
        assert has_edge_to_node3, f"The edge list of node {node1_id} should contain an edge to {node3_id}"

        # Check if node 2 has an edge to node 1
        has_edge_to_node1 = any(
            (src == node2_id and tgt == node1_id)
            or (src == node1_id and tgt == node2_id)
            for src, tgt in node2_edges
        )
        assert (
            has_edge_to_node1
        ), f"The edge list of node {node2_id} should contain a connection with {node1_id}"

        print("Undirected graph feature verification successful: The node edges obtained in batches contain all related edges (regardless of direction)")

        print("\nUndirected graph feature test completed")
        return True

    except Exception as e:
        ASCIIColors.red(f"An error occurred during testing: {str(e)}")
        return False


async def main():
    """Main function"""
    # Display the program title
    ASCIIColors.cyan("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║ General Graph Storage Test Program ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Check .env file
    if not check_env_file():
        return

    # Load environment variables
    load_dotenv(dotenv_path=".env", override=False)

    # Get the graph storage type
    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    ASCIIColors.magenta(f"\nCurrently configured graph storage type: {graph_storage_type}")
    ASCIIColors.white(
        f"Supported graph storage types: {', '.join(STORAGE_IMPLEMENTATIONS['GRAPH_STORAGE']['implementations'])}"
    )

    # Initialize the storage instance
    storage = await initialize_graph_storage()
    if not storage:
        ASCIIColors.red("Failed to initialize the storage instance, test program exits")
        return

    try:
        # Display test options
        ASCIIColors.yellow("\nPlease select the test type:")
        ASCIIColors.white("1. Basic tests (insertion and reading of nodes and edges)")
        ASCIIColors.white("2. Advanced tests (degree, label, knowledge graph, deletion operations, etc.)")
        ASCIIColors.white("3. Batch operation test (batch acquisition of node, edge attributes and degrees, etc.)")
        ASCIIColors.white("4. Undirected graph feature test (verify the stored undirected graph features)")
        ASCIIColors.white("5. Special character test (verify special characters such as single quotes, double quotes and backslash)")
        ASCIIColors.white("6. All tests")

        choice = input("\nPlease enter your choice (1/2/3/4/5/6): ")

        # Clean up the data before running the test
        if choice in ["1", "2", "3", "4", "5", "6"]:
            ASCIIColors.yellow("\nClean data before executing the test...")
            await storage.drop()
            ASCIIColors.green("Data cleaning completed\n")

        if choice == "1":
            await test_graph_basic(storage)
        elif choice == "2":
            await test_graph_advanced(storage)
        elif choice == "3":
            await test_graph_batch_operations(storage)
        elif choice == "4":
            await test_graph_undirected_property(storage)
        elif choice == "5":
            await test_graph_special_characters(storage)
        elif choice == "6":
            ASCIIColors.cyan("\n=== Start basic test===")
            basic_result = await test_graph_basic(storage)

            if basic_result:
                ASCIIColors.cyan("\n=== Start advanced test===")
                advanced_result = await test_graph_advanced(storage)

                if advanced_result:
                    ASCIIColors.cyan("\n=== Start batch operation test===")
                    batch_result = await test_graph_batch_operations(storage)

                    if batch_result:
                        ASCIIColors.cyan("\n=== Start undirected graph feature test===")
                        undirected_result = await test_graph_undirected_property(
                            storage
                        )

                        if undirected_result:
                            ASCIIColors.cyan("\n=== Start special character test===")
                            await test_graph_special_characters(storage)
        else:
            ASCIIColors.red("Invalid option")

    finally:
        # Close the connection
        if storage:
            await storage.finalize()
            ASCIIColors.green("\nStorage connection closed")


if __name__ == "__main__":
    asyncio.run(main())