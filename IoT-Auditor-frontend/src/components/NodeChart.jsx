import React, { useEffect, useState, useMemo, useCallback, useRef, forwardRef, useImperativeHandle } from "react";
import ReactFlow, { Background, Controls, useEdgesState, useNodesState, Panel, useReactFlow, ReactFlowProvider, addEdge, getIncomers, getOutgoers, getConnectedEdges } from 'reactflow';
import Dagre from 'dagre';
import 'reactflow/dist/style.css';
import { cloneDeep } from 'lodash';
import ExploreNode from "./ExploreNode";
import AnnotateNode from "./AnnotateNode";
import ExploreEdge from "./ExploreEdge";
import AnnotateEdge from "./AnnotateEdge";
import ModeNode from "./ModeNode";
import { MarkerType } from "reactflow";
import { v4 as uuidv4 } from "uuid";
import "./NodeChart.css";
import { childNodeMarginY, childNodeoffsetX, childNodeoffsetY, highlightColor, modeNodeStyle, nodeMarginX, nodeMarginY, nodeOffsetX, stateNodeStyle } from "../shared/chartStyle";
import axios from "axios";

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
    let { board, chart, setChart, step, setChartSelection } = props;
    const reactFlowWrapper = useRef(null);
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const dragRef = useRef(null);
    const [onDragging, setOnDragging] = useState(false);
    const [target, setTarget] = useState(null);
    const nodeTypes_explore = useMemo(() => ({ stateNode: ExploreNode, modeNode: ModeNode }), []);
    const nodeTypes_annotate = useMemo(() => ({ stateNode: AnnotateNode, modeNode: ModeNode }), []);
    const edgeTypes_explore = useMemo(() => ({ transitionEdge: ExploreEdge }), []);
    const edgeTypes_annotate = useMemo(() => ({ transitionEdge: AnnotateEdge }), []);

    useEffect(() => {
        if (chart.hasOwnProperty("nodes")) {
            setNodes([...chart.nodes]);
            setEdges([...chart.edges]);
        }
    }, [chart]);

    useImperativeHandle(ref, () => ({
        updateAnnotation,
        collageStates
    }));

    const collageStates = async () => {
        await axios
            .post(window.HARDWARE_ADDRESS + "/collage", {
                device: board.title,
                nodes: nodes
            })
            .then((resp) => {
                let group_cnt = resp.data.group_cnt;
                let node_collage_dict = resp.data.node_collage_dict;
                let newNodes = [...nodes];
                let newEdges = [...edges];
                for (let index = 1; index <= group_cnt; index++) {
                    let collage_node = addNewNode({ x: nodeMarginX + nodeOffsetX * index, y: nodeMarginY }, "modeNode");
                    collage_node.data.label += " " + index;
                    for (const [nid, cid] of Object.entries(node_collage_dict)) {
                        if (cid === index) {
                            let node = newNodes.find((n) => n.id === nid);
                            collage_node.data.children = [...collage_node.data.children, node.id];
                            node.parentNode = collage_node.id;
                            node.position = { x: childNodeoffsetX, y: childNodeMarginY + (collage_node.data.children.length - 1) * childNodeoffsetY };
                            if (collage_node.data.children.length > 1) {
                                collage_node.style = { ...collage_node.style, height: parseInt(collage_node.style.height.slice(0, -2)) + childNodeoffsetY + "px" };
                            }
                        }
                    };
                    newEdges = hiddenInsideEdges(newEdges, collage_node.data.children);
                    newNodes.push(collage_node);
                };

                setNodes(newNodes);
                setEdges(newEdges);
            })
    };

    const updateAnnotation = () => {
        const newChart = reactFlowInstance.toObject();
        setChart(newChart);
        console.log("update annotation")
        return newChart;
    };

    const onNodeClick = (evt, node) => {
        setChartSelection(node);
    }

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
        let nodeData = { label: "Group State"};

        zIndex = 1;
        nodeData["children"] = [];
        nodeStyle = modeNodeStyle;

        const newNode = {
            id: uuidv4(),
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
        console.log("start")
        dragRef.current = node;
    };

    const onNodeDrag = (evt, node) => {
        setOnDragging(true);
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
                n.zIndex < node.zIndex
        );

        if (targetNode) {
            setTarget(targetNode);
        }
        else {
            setTarget(null);
        }
    };

    const onNodeDragStop = (evt, node) => {
        if (!onDragging) {
            dragRef.current = null;
            return;
        }

        let newNodes = [...nodes];
        let newEdges = [...edges];

        if (target) {
            newNodes.map((n) => {
                if (n.id === node.id) {
                    n.parentNode = target.id;
                    let parent = newNodes.find((e) => e.id === target.id);
                    if (!parent.data.children.includes(n.id)) {
                        newEdges = revealOwnEdges(newEdges, node.id);
                        parent.data.children = [...parent.data.children, n.id];
                        n.position = { x: childNodeoffsetX, y: childNodeMarginY + (parent.data.children.length - 1) * childNodeoffsetY };
                        if (parent.data.children.length > 1) {
                            parent.style = { ...parent.style, height: parseInt(parent.style.height.slice(0, -2)) + childNodeoffsetY + "px" };
                            newEdges = hiddenInsideEdges(newEdges, parent.data.children);
                        }
                    }
                    else {
                        n.position = { x: childNodeoffsetX, y: childNodeMarginY + parent.data.children.indexOf(n.id) * childNodeoffsetY };
                    }
                }
                return n;
            })
        }
        else {
            newNodes.map((n) => {
                if (n.id === node.id) {
                    if (n.parentNode) {
                        let parent = newNodes.find((e) => e.id === n.parentNode);
                        parent.data.children = parent.data.children.filter((e) => e !== n.id);
                        if (parent.data.children.length > 0) {
                            parent.style = { ...parent.style, height: parseInt(parent.style.height.slice(0, -2)) - childNodeoffsetY + "px" };
                            for (const child of parent.data.children) {
                                let childNode = newNodes.find((e) => e.id === child);
                                childNode.position = { x: childNodeoffsetX, y: childNodeMarginY + parent.data.children.indexOf(child) * childNodeoffsetY };
                            }
                        }
                    }
                    n.parentNode = null;
                    n.position = { x: node.positionAbsolute.x, y: node.positionAbsolute.y };
                    newEdges = revealOwnEdges(newEdges, node.id);
                }
                return n;
            })
        }

        setNodes(newNodes);
        setEdges(newEdges);
        setTarget(null);
        setOnDragging(false);
        dragRef.current = null;
    };

    const revealOwnEdges = (edges, idx) => {
        edges.map((e) => {
            if (e.source === idx || e.target === idx) {
                e.hidden = false;
            }

            return e;
        });
        return edges;
    };

    const hiddenInsideEdges = (edges, children) => {
        edges.map((e) => {
            if (children.includes(e.source) && children.includes(e.target)) {
                e.hidden = true;
            }

            return e;
        });

        return edges;
    };

    useEffect(() => {
        setNodes((nodes) =>
            nodes.map((node) => {
                if (node.id === target?.id) {
                    node.style = { ...node.style, backgroundColor: highlightColor };
                } else {
                    let color;
                    switch (node.type) {
                        case "modeNode":
                            color = "#bfd7ff";
                            break;
                        default:
                            color = "#788bff";
                            break;
                    }
                    node.style = { ...node.style, backgroundColor: color };
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
            style: {
                strokeWidth: 2,
                stroke: '#000000',
            },
            data: {
                label: "action (" + prevIdx + "->" + idx + ")"
            },
            zIndex: 4
        }
        setEdges((eds) => addEdge(newEdge, eds))
    }, []);

    const onDragStart = (event, nodeType) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.effectAllowed = 'move';
    };

    return (
        <div style={{ width: '100%', height: '100%', backgroundColor: "white" }} ref={reactFlowWrapper}>
            {step === 0 &&
                <ReactFlow
                    nodeTypes={nodeTypes_explore}
                    edgeTypes={edgeTypes_explore}
                    nodes={nodes}
                    edges={edges}
                    onInit={setReactFlowInstance}
                    fitView
                >
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
                    onNodeClick={onNodeClick}
                    onConnect={onConnect}
                    fitView
                >
                    <Panel position="top-right">
                        <div className='mode-node-div' onDragStart={(event) => onDragStart(event, 'modeNode')} draggable>
                            Group State
                        </div>
                    </Panel>
                    <Background />
                    <Controls />
                </ReactFlow>
            }
            {step === 2 &&
                <ReactFlow
                    nodeTypes={nodeTypes_explore}
                    edgeTypes={edgeTypes_explore}
                    nodes={nodes}
                    edges={edges}
                    onInit={setReactFlowInstance}
                    fitView
                >
                    <Background />
                    <Controls />
                </ReactFlow>
            }
        </div>
    )
});


export default NodeChart;