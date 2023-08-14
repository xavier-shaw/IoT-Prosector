import React, { useEffect, useState, useMemo, useCallback, useRef, forwardRef, useImperativeHandle } from "react";
import ReactFlow, { Background, Controls, useEdgesState, useNodesState, applyNodeChanges, applyEdgeChanges, Panel, useReactFlow, ReactFlowProvider, addEdge, getIncomers, getOutgoers, getConnectedEdges } from 'reactflow';
import Dagre from 'dagre';
import 'reactflow/dist/style.css';
import { cloneDeep } from 'lodash';
import ExploreNode from "./ExploreNode";
import AnnotateNode from "./AnnotateNode";
import ExploreEdge from "./ExploreEdge";
import AnnotateEdge from "./AnnotateEdge";
import SystemNode from "./SystemNode";
import ModeNode from "./ModeNode";
import { MarkerType } from "reactflow";
import { v4 as uuidv4 } from "uuid";
import GroupNode from "./GroupNode";

const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));

const getLayoutedElements = (nodes, edges, options) => {
    g.setGraph({ rankdir: options.direction, ranksep: 150 });

    edges.forEach((edge) => g.setEdge(edge.source, edge.target));
    nodes.forEach((node) => g.setNode(node.id, node));

    Dagre.layout(g);

    return {
        nodes: nodes.map((node) => {
            const { x, y } = g.node(node.id);

            return { ...node, position: { x, y } };
        }),
        edges,
    };
};

const NodeChart = forwardRef((props, ref) => {
    return (
        <ReactFlowProvider>
            <FlowChart {...props} ref={ref} />
        </ReactFlowProvider>
    )
})

const FlowChart = forwardRef((props, ref) => {
    let { board, step } = props;
    const reactFlowWrapper = useRef(null);
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const { setViewport } = useReactFlow();
    // this ref stores the current dragged node
    const dragRef = useRef(null);
    // target is the node that the node is dragged over
    const [target, setTarget] = useState(null);
    const nodeTypes_explore = useMemo(() => ({ stateNode: ExploreNode, systemNode: SystemNode, modeNode: ModeNode, groupNode: GroupNode }), []);
    const nodeTypes_annotate = useMemo(() => ({ stateNode: AnnotateNode, systemNode: SystemNode, modeNode: ModeNode, groupNode: GroupNode }), []);
    const edgeTypes_explore = useMemo(() => ({ transitionEdge: ExploreEdge }), []);
    const edgeTypes_annotate = useMemo(() => ({ transitionEdge: AnnotateEdge }), []);

    useEffect(() => {
        if (board.hasOwnProperty("chart")) {
            console.log("flow chart", board.chart);
            setNodes([...board.chart.nodes]);
            setEdges([...board.chart.edges]);
            setViewport({ ...board.chart.viewport });
        }
    }, [board]);

    useImperativeHandle(ref, () => ({
        updateAnnotation
    }))

    const updateAnnotation = () => {
        const flowObj = reactFlowInstance.toObject();
        return flowObj;
    };

    const onLayout = useCallback(() => {
        const layouted = getLayoutedElements(nodes, edges, { direction: "LR" });
        setNodes([...layouted.nodes]);
        setEdges([...layouted.edges]);
    },
        [nodes, edges]
    );

    const addNewNode = (position, type) => {
        let zIndex = 0;
        let nodeStyle = {};
        let nodeData = { label: `${type} node` };
        if (type === "systemNode") {
            zIndex = 0;
            nodeStyle = {
                width: "800px",
                height: "400px",
                borderWidth: "5px",
                borderStyle: "solid",
                borderColor: "#cfc098",
                padding: "10px",
                borderRadius: "10px",
                backgroundColor: "lightgrey",
            }
        }
        else if (type === "modeNode") {
            zIndex = 1;
            nodeStyle = {
                width: "400px",
                height: "250px",
                borderWidth: "3px",
                borderStyle: "solid",
                borderColor: "#bd938c",
                padding: "10px",
                borderRadius: "10px",
                backgroundColor: "lightgrey",
            }
        }
        else if (type === "groupNode") {
            zIndex = 2;
            nodeData["subNodes"] = [];
            nodeStyle = {
                width: "250px",
                height: "150px",
                borderWidth: "3px",
                borderStyle: "solid",
                borderColor: "#ec6681",
                padding: "10px",
                borderRadius: "10px",
                backgroundColor: "lightgrey",
            }
        }
        else {
            zIndex = 3;
            nodeStyle = {
                width: "150px",
                height: "80px",
                borderWidth: "1px",
                borderStyle: "solid",
                borderColor: "#6d8ee0",
                padding: "10px",
                borderRadius: "10px",
                backgroundColor: "lightgrey",
                display: "flex",
                justifyContent: "center",
                alignItems: "center"
            }
        };

        const newNode = {
            id: "node_" + uuidv4(),
            type: type,
            position: position,
            positionAbsolute: position,
            data: nodeData,
            style: nodeStyle,
            zIndex: zIndex
        };

        return newNode;
    };

    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event) => {
            event.preventDefault();

            const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
            const type = event.dataTransfer.getData('application/reactflow');

            // check if the dropped element is valid
            if (typeof type === 'undefined' || !type) {
                return;
            }

            const position = reactFlowInstance.project({
                x: event.clientX - reactFlowBounds.left,
                y: event.clientY - reactFlowBounds.top,
            });

            const newNode = addNewNode(position, type);

            setNodes((nds) => nds.concat(newNode));
        },
        [reactFlowInstance]
    );

    const onNodeDragStart = (evt, node) => {
        dragRef.current = node;
    };

    const onNodeDrag = (evt, node) => {
        // calculate the center point of the node from position and dimensions
        const centerX = node.positionAbsolute.x + node.width / 2;
        const centerY = node.positionAbsolute.y + node.height / 2;

        let nodes_sort = nodes.sort((a, b) => b.zIndex - a.zIndex);

        // find a node where the center point is inside
        const targetNode = nodes_sort.find(
            (n) =>
                centerX > n.positionAbsolute.x &&
                centerX < n.positionAbsolute.x + n.width &&
                centerY > n.positionAbsolute.y &&
                centerY < n.positionAbsolute.y + n.height &&
                n.id !== node.id // this is needed, otherwise we would always find the dragged node
        );

        if (targetNode?.zIndex < node.zIndex) {
            setTarget(targetNode);
        }
        else {
            setTarget(null);
        }
    };

    const onNodeDragStop = (evt, node) => {
        let newNodes = [...nodes];
        let allEdges = [...edges];
        if (target?.type === "groupNode") {
            let nodeIdx = node.id.split("_")[1];
            newNodes = newNodes.filter((n) => n.id !== node.id && n.id !== target.id);
            target.data.subNodes = [...target.data.subNodes, node];
            const incomers = getIncomers(node, nodes, edges);
            const outgoers = getOutgoers(node, nodes, edges);
            let connectedEdges = getConnectedEdges([node], edges);
            const remainEdges = allEdges.filter((edge) => !connectedEdges.includes(edge));
            connectedEdges = connectedEdges.map((edge) => {
                if (edge.source === node.id) {
                    let targetIdx = edge.target.split("_")[1];
                    edge.source = target.id;
                    edge.id = "edge_" + nodeIdx + "-" + targetIdx;
                }
                else {
                    let sourceIdx = edge.source.split("_")[1];
                    edge.target = target.id;
                    edge.id = "edge_" + sourceIdx + "-" + nodeIdx;
                };
                return edge;
            });
            newNodes.push(target);
            console.log(target);
            setNodes(newNodes);
            setEdges([...remainEdges, ...connectedEdges]);
        }
        else {
            newNodes.map((n) => {
                if (n.id === node.id) {
                    if (target) {
                        n.parentNode = target.id;
                        // n.extent = "parent";
                        n.position = { x: node.positionAbsolute.x - target.positionAbsolute.x, y: node.positionAbsolute.y - target.positionAbsolute.y }
                    }
                    else {
                        n.parentNode = "";
                        n.position = { x: node.positionAbsolute.x, y: node.positionAbsolute.y };
                    }
                }
                return n;
            })
        }

        setNodes(newNodes);
        setTarget(null);
        dragRef.current = null;
    };

    useEffect(() => {
        setNodes((nodes) =>
            nodes.map((node) => {
                if (node.id === target?.id) {
                    node.style = { ...node.style, backgroundColor: "lightyellow" };
                } else {
                    node.style = { ...node.style, backgroundColor: "lightgrey" };
                }

                return node;
            })
        );
    }, [target]);

    const onConnect = useCallback((params) => {
        let prevIdx = params.source.split("_")[1];
        let idx = params.target.split("_")[1];
        let newEdge = {
            id: "edge_" + prevIdx + "-" + idx,
            type: "transitionEdge",
            source: params.source,
            target: params.target,
            markerEnd: {
                type: MarkerType.ArrowClosed,
                width: 30,
                height: 30,
                color: '#FF0072',
            },
            data: {
                label: "action (" + prevIdx + "->" + idx + ")"
            },
            zIndex: 2
        }
        setEdges((eds) => addEdge(newEdge, eds))
    }, []);

    return (
        <div style={{ width: '100%', height: '100%' }} ref={reactFlowWrapper}>
            {step === 0 &&
                <ReactFlow
                    nodeTypes={nodeTypes_explore}
                    edgeTypes={edgeTypes_explore}
                    nodes={nodes}
                    edges={edges}
                    onInit={setReactFlowInstance}
                    fitView
                >
                    <Panel position="top-right">
                        <button onClick={onLayout}>Layout</button>
                    </Panel>
                    <Background />
                    <Controls />
                </ReactFlow>
            }
            {step === 1 &&
                <ReactFlow
                    nodeTypes={nodeTypes_annotate}
                    edgeTypes={edgeTypes_annotate}
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onInit={setReactFlowInstance}
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    onNodeDragStart={onNodeDragStart}
                    onNodeDrag={onNodeDrag}
                    onNodeDragStop={onNodeDragStop}
                    onConnect={onConnect}
                    fitView
                >
                    <Panel position="top-right">
                        <button onClick={onLayout}>Layout</button>
                    </Panel>
                    <Background />
                    <Controls />
                </ReactFlow>
            }
        </div>
    )
});


export default NodeChart;