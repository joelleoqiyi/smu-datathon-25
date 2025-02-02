import Graph from "graphology";
import Sigma from "sigma";
import axios from "axios";
import forceAtlas2 from "graphology-layout-forceatlas2";

// Create a graph instance
const graph = new Graph();

// Function to Assign Default Positions
function assignNodePosition() {
    return {
        x: Math.random() * 500, 
        y: Math.random() * 500,
        size: 10,
        color: "blue"
    };
}

// Track Selected Nodes
let selectedNodes = [];

// Load JSON Data
async function loadGraphData() {
    try {
        const response = await axios.get("/extracted_relationships_multiple.json");
        const data = response.data;

        console.log("Data loaded successfully.");

        data.forEach((excerpt) => {
            excerpt.relationships.forEach((relation) => {
                const head = relation.head;
                const tail = relation.tail;
                let relationshipDescriptions = relation.relationships.map(rel => rel.relationship).join("; ");

                relation.relationships.forEach((rel) => {
                    if (rel.relationship_type !== "null") {
                        if (!graph.hasNode(head)) {
                            graph.addNode(head, {
                                label: head,
                                ...assignNodePosition()
                            });
                        }
                        if (!graph.hasNode(tail)) {
                            graph.addNode(tail, {
                                label: tail,
                                ...assignNodePosition()
                            });
                        }

                        if (!graph.hasEdge(head, tail)) {
                            graph.addEdge(head, tail, {
                                label: relationshipDescriptions,  // Store all relationships as label
                                size: 2,
                                color: "purple",
                                hidden: false 
                            });
                        }
                    }
                });
            });
        });

        console.log("Graph created successfully.");

        // Apply ForceAtlas2 Layout for Better Spacing
        console.log("Applying ForceAtlas2 Layout...");
        forceAtlas2.assign(graph, {
            iterations: 300,
            settings: {
                gravity: 1,
                linLogMode: true
            }
        });
        console.log("Layout Applied.");

        // Initialize Sigma for Visualization
        setupSigma();
    } catch (error) {
        console.error("Error loading JSON data:", error);
    }
}

// Function to Setup Sigma.js
function setupSigma() {
    const container = document.getElementById("container");
    const sigmaInstance = new Sigma(graph, container);

    // Handle Click Event on Nodes
    sigmaInstance.on("clickNode", ({ node }) => {
        if (selectedNodes.length < 2) {
            selectedNodes.push(node);
        } else {
            // ✅ Reset previous selections
            selectedNodes.forEach((n) => graph.setNodeAttribute(n, "color", "blue"));
            selectedNodes = [node]; // Start new selection
        }

        // ✅ Highlight Selected Nodes
        selectedNodes.forEach((n) => graph.setNodeAttribute(n, "color", "yellow"));

        console.log(`🔹 Selected Nodes: ${selectedNodes}`);
        updateGraphView(sigmaInstance);
    });
}

// ✅ Function to Show Relationships in Text Box
function updateGraphView(sigmaInstance) {
    const relationshipInfo = document.getElementById("relationship-info");
    if (selectedNodes.length < 2) {
        graph.forEachEdge((edge) => graph.setEdgeAttribute(edge, "hidden", false)); // Show all edges
        relationshipInfo.textContent = "Select two nodes to see their relationships.";
    } else {
        const [nodeA, nodeB] = selectedNodes;
        let relationshipsText = "";

        graph.forEachEdge((edge, attributes, source, target) => {
            const isMatch = (source === nodeA && target === nodeB) || (source === nodeB && target === nodeA);
            graph.setEdgeAttribute(edge, "hidden", !isMatch);

            if (isMatch) {
                relationshipsText += `🔹 ${attributes.label}\n`;
            }
        });

        if (relationshipsText) {
            relationshipInfo.innerHTML = `<strong>Relationships between ${nodeA} and ${nodeB}:</strong><br>${relationshipsText}`;
        } else {
            relationshipInfo.innerHTML = `<strong>No direct relationships found between ${nodeA} and ${nodeB}.</strong>`;
        }
    }

    sigmaInstance.refresh();
}

// Load Graph When DOM is Ready
document.addEventListener("DOMContentLoaded", loadGraphData);