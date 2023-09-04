import React, { useEffect, useState, useMemo, useCallback, useRef, forwardRef, useImperativeHandle } from "react";
import ReactFlow, { Background, Controls, useEdgesState, useNodesState, Panel, useReactFlow, ReactFlowProvider, addEdge, getIncomers, getOutgoers, getConnectedEdges } from 'reactflow';
import Dagre from 'dagre';
import 'reactflow/dist/style.css';
import { cloneDeep } from 'lodash';
import ExploreNode from "./ExploreNode";
import AnnotateNode from "./AnnotateNode";
import ExploreEdge from "./ExploreEdge";
import AnnotateEdge from "./AnnotateEdge";
import SemanticNode from "./SemanticNode";
import { MarkerType } from "reactflow";
import { v4 as uuidv4 } from "uuid";
import "./NodeChart.css";
import { nodeOffsetX, nodeOffsetY, layoutRowNum, childNodeMarginY, childNodeoffsetX, childNodeoffsetY, highlightColor, semanticNodeStyle, semanticNodeMarginX, semanticNodeMarginY, semanticNodeOffsetX, stateNodeStyle, combinedNodeMarginX, combinedNodeMarginY, combinedNodeOffsetX, childSemanticNodeOffsetX, childSemanticNodeOffsetY, childNodeMarginX, combinedNodeStyle, childSemanticNodeMarginX, childSemanticNodeMarginY, offWidth, offHeight } from "../shared/chartStyle";
import axios from "axios";
import CombinedNode from "./CombinedNode";
import { max } from "d3";

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
    let { board, chart, setChart, step, setChartSelection, updateMatrix } = props;
    const reactFlowWrapper = useRef(null);
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const dragRef = useRef(null);
    const [onDragging, setOnDragging] = useState(false);
    const [target, setTarget] = useState(null);
    const [autoLayoutMode, setAutoLayoutMode] = useState(true);
    const nodeTypes_explore = useMemo(() => ({ stateNode: ExploreNode, semanticNode: SemanticNode, combinedNode: CombinedNode }), []);
    const nodeTypes_annotate = useMemo(() => ({ stateNode: AnnotateNode, semanticNode: SemanticNode, combinedNode: CombinedNode }), []);
    const edgeTypes_explore = useMemo(() => ({ transitionEdge: ExploreEdge }), []);
    const edgeTypes_annotate = useMemo(() => ({ transitionEdge: AnnotateEdge }), []);

    useEffect(() => {
        if (chart.hasOwnProperty("nodes")) {
            setNodes([...chart.nodes]);
            setEdges([...chart.edges]);
        }
    }, [chart]);

    useEffect(() => {
        if (step === 1) {
            updateMatrix(nodes);
        }
    }, [step]);

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
                console.log(resp)
                let semantic_group_cnt = resp.data.semantic_group_cnt;
                let semantic_collage_dict = resp.data.semantic_collage_dict;
                let combined_group_cnt = resp.data.combined_group_cnt;
                let combined_collage_dict = resp.data.combined_collage_dict;

                let newNodes = [...nodes];
                let newEdges = [...edges];
                let semanticNodes = [];

                for (let index = 0; index < semantic_group_cnt; index++) {
                    let semanticNode = createNewNode({ x: semanticNodeMarginX + semanticNodeOffsetX * index, y: semanticNodeMarginY }, "semanticNode");
                    semanticNode.data.label += " " + index;
                    for (const [nid, cid] of Object.entries(semantic_collage_dict)) {
                        if (cid === index) {
                            let node = newNodes.find((n) => n.id === nid);
                            semanticNode.data.children = [...semanticNode.data.children, node.id];
                            node.parentNode = semanticNode.id;
                            node.position = { x: childNodeoffsetX, y: childNodeMarginY + (semanticNode.data.children.length - 1) * childNodeoffsetY };
                            if (semanticNode.data.children.length > 1) {
                                semanticNode.style = { ...semanticNode.style, height: parseInt(semanticNode.style.height.slice(0, -2)) + childNodeoffsetY + "px" };
                            }
                        }
                    };
                    newEdges = hiddenInsideEdges(newEdges, semanticNode.data.children);
                    newNodes.push(semanticNode);
                    semanticNodes.push(semanticNode);
                };

                let prevCombinedNode = null;
                for (let index = 0; index < combined_group_cnt; index++) {
                    let combinedNodePosition;
                    if (prevCombinedNode) {
                        combinedNodePosition = {
                            x: prevCombinedNode.position.x + parseInt(prevCombinedNode.style.width.slice(0, -2)) + combinedNodeOffsetX,
                            y: combinedNodeMarginY
                        }
                    }
                    else {
                        combinedNodePosition = {
                            x: combinedNodeMarginX,
                            y: combinedNodeMarginY
                        }
                    }

                    let combinedNode = createNewNode(combinedNodePosition, "combinedNode");
                    combinedNode.data.label += " " + index;
                    let completedSids = [];
                    for (const [nid, cid] of Object.entries(combined_collage_dict)) {
                        if (cid === index) {
                            let sidx = semantic_collage_dict[nid];
                            if (completedSids.indexOf(sidx) !== -1) {
                                continue;
                            }
                            completedSids.push(sidx);
                            let semanticNode = semanticNodes[sidx];
                            combinedNode.data.children.push(semanticNode.id);
                            semanticNode.parentNode = combinedNode.id;
                            semanticNode.position = {
                                x: childSemanticNodeMarginX + (combinedNode.data.children.length - 1) * childSemanticNodeOffsetX,
                                y: childSemanticNodeOffsetY
                            };
                            semanticNode.positionAbsolute = {
                                x: combinedNode.position.x + semanticNode.position.x,
                                y: combinedNode.position.y + combinedNode.position.y
                            }
                        }
                    };

                    combinedNode.style = { ...combinedNode.style, width: changeWidth(combinedNode), height: changeHeight(newNodes, combinedNode) };
                    newNodes.push(combinedNode);
                    prevCombinedNode = combinedNode;
                }

                autoLayout(newNodes, true);
                setEdges(newEdges);
            })
    };

    const changeWidth = (parentNode, LAYOUT = false) => {
        if (parentNode.type === "semanticNode") {
            return semanticNodeStyle.width;
        }
        else if (parentNode.type === "combinedNode") {
            let childCnt = parentNode.data.children.length;
            if (childCnt <= 1) {
                return combinedNodeStyle.width;
            }
            else {
                if (childCnt < layoutRowNum) {
                    return (childSemanticNodeMarginX + childCnt * childSemanticNodeOffsetX - offWidth) + "px";
                }
                else {
                    return (childSemanticNodeMarginX + layoutRowNum * childSemanticNodeOffsetX - offWidth) + "px";
                }
            }
        }
    };

    const changeHeight = (nodes, parentNode) => {
        let maxHeight = 0;

        if (parentNode.data.children.length <= 1) {
            if (parentNode.type === "semanticNode") {
                return semanticNodeStyle.height;
            }
        }

        if (parentNode.type === "semanticNode") {
            maxHeight = childNodeMarginY + parentNode.data.children.length * childNodeoffsetY - offHeight;
        }
        else if (parentNode.type === "combinedNode") {
            for (const child_id of parentNode.data.children) {
                let child = nodes.find((n) => n.id === child_id);
                let childBottom = child.position.y + parseInt(child.style.height.slice(0, -2));
                if (childBottom > maxHeight) {
                    maxHeight = childBottom;
                }
            };
            maxHeight += childNodeMarginY;
        }

        return maxHeight + "px";
    }

    const updateAnnotation = () => {
        const newChart = reactFlowInstance.toObject();
        setChart(newChart);
        console.log("update annotation")
        return newChart;
    };

    const onNodeClick = (evt, node) => {
        setChartSelection(node);
    };

    const changeLayoutMode = () => {
        setAutoLayoutMode((prev) => (!prev));
    };

    const autoLayout = (nodes, needUpdate = false) => {
        switch (step) {
            case 0:
                onInteractionLayout(nodes);
                break;
            case 1:
                onCollageLayout(nodes);
            default:
                break;
        }

        if (needUpdate) {
            updateMatrix(nodes);
        }

        console.log(nodes);
    };

    const onInteractionLayout = (nodes) => {
        let newNodes = [...nodes];
        let nextRowY = 0;

        newNodes.map((node, index) => {
            if (index % layoutRowNum === 0 && index !== 0) {
                for (let i = index - 1; i >= index - layoutRowNum; i--) {
                    let prevNode = newNodes[i];
                    if (prevNode.position.y + parseInt(prevNode.style.height.slice(0, -2)) > nextRowY) {
                        nextRowY = prevNode.position.y + parseInt(prevNode.style.height.slice(0, -2));
                    };
                }
            }
            node.position = { x: nodeOffsetX * (index % layoutRowNum), y: nextRowY + nodeOffsetY }
            return node;
        })

        setNodes(newNodes);
    }

    const onCollageLayout = (nodes) => {
        let newNodes = [...nodes];
        let prevNode = null;
        newNodes.map((node, idx) => {
            if (autoLayoutMode) {
                if (!node.parentNode) {
                    if (prevNode) {
                        node.position = { x: prevNode.position.x + parseInt(prevNode.style.width.slice(0, -2)) + combinedNodeOffsetX, y: prevNode.position.y }
                        node.positionAbsolute = { x: prevNode.position.x + parseInt(prevNode.style.width.slice(0, -2)) + combinedNodeOffsetX, y: prevNode.position.y }
                    }
                    prevNode = node;
                }
            }
            
            if (node.type === "combinedNode") {
                let children = node.data.children;
                let nextRowY = 0;
                let rowCnt = 0;
                for (let index = 0; index < children.length; index++) {
                    const childId = children[index];
                    let child = newNodes.find((n) => n.id === childId);
                    rowCnt = Math.floor(index / layoutRowNum);
                    if (index % layoutRowNum === 0 && index !== 0) {
                        for (let i = index - 1; i >= index - layoutRowNum; i--) {
                            let prevChildNodeId = children[i];
                            let prevChildNode = newNodes.find((n) => n.id === prevChildNodeId);
                            let prevChildBottom = prevChildNode.position.y + parseInt(prevChildNode.style.height.slice(0, -2))
                            if (prevChildBottom > nextRowY) {
                                nextRowY = prevChildBottom;
                            };
                        }
                    }

                    if (rowCnt === 0) {
                        child.position = { x: childSemanticNodeMarginX + childSemanticNodeOffsetX * (index % layoutRowNum), y: nextRowY + childSemanticNodeMarginY }
                    }
                    else {
                        child.position = { x: childSemanticNodeMarginX + childSemanticNodeOffsetX * (index % layoutRowNum), y: nextRowY + nodeOffsetY }
                    }
                    child.positionAbsolute = { x: node.positionAbsolute.x + child.position.x, y: node.positionAbsolute.y + child.position.y }
                };

                node.style = { ...node.style, width: changeWidth(node, true), height: changeHeight(newNodes, node) };
            };

            return node;
        });

        setNodes(newNodes);
    }

    const createNewNode = (position, type) => {
        let zIndex;
        let nodeStyle;
        let nodeData = { label: "", children: [] };

        switch (type) {
            case "semanticNode":
                zIndex = 1;
                nodeData.label = "Semantic Node"
                nodeStyle = semanticNodeStyle;
                break;
            case "combinedNode":
                zIndex = 0;
                nodeData.label = "Combined Node"
                nodeStyle = combinedNodeStyle;
                break;
            default:
                break;
        }

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

            const newNode = createNewNode(position, type);

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
        let needUpdate = false;

        if (target) {
            newNodes.map((n) => {
                if (n.id === node.id) {
                    if (n.parentNode && n.parentNode !== target.id) {
                        needUpdate = true;
                        removeFromParentNode(newNodes, n);
                    };

                    n.parentNode = target.id;
                    let parent = newNodes.find((e) => e.id === target.id);
                    if (!parent.data.children.includes(n.id)) {
                        needUpdate = true;
                        newEdges = revealOwnEdges(newEdges, node.id);
                        parent.data.children = [...parent.data.children, n.id];
                        if (parent.type === "semanticNode") {
                            n.position = { x: childNodeoffsetX, y: childNodeMarginY + (parent.data.children.length - 1) * childNodeoffsetY };
                        }
                        else if (parent.type === "combinedNode") {
                            n.position = {
                                x: childSemanticNodeMarginX + (parent.data.children.length - 1) * childSemanticNodeOffsetX,
                                y: childSemanticNodeOffsetY
                            }
                        }
                        parent.style = { ...parent.style, width: changeWidth(parent), height: changeHeight(newNodes, parent) };
                        if (parent.parentNode) {
                            let grandParent = newNodes.find((n) => n.id === parent.parentNode);
                            grandParent.style = { ...grandParent.style, width: changeWidth(grandParent), height: changeHeight(newNodes, grandParent) };
                        }
                        newEdges = hiddenInsideEdges(newEdges, parent.data.children);
                    }
                    else {
                        if (parent.type === "semanticNode") {
                            n.position = { x: childNodeoffsetX, y: childNodeMarginY + parent.data.children.indexOf(n.id) * childNodeoffsetY };
                        }
                        else if (parent.type === "combinedNode") {
                            n.position = { x: childSemanticNodeMarginX + parent.data.children.indexOf(n.id) * childSemanticNodeOffsetX, y: childSemanticNodeOffsetY };
                        }
                    }
                }
                return n;
            })
        }
        else {
            newNodes.map((n) => {
                if (n.id === node.id) {
                    if (n.parentNode) {
                        needUpdate = true;
                        removeFromParentNode(newNodes, n);
                    }
                    n.position = { x: node.positionAbsolute.x, y: node.positionAbsolute.y };
                    newEdges = revealOwnEdges(newEdges, node.id);
                }
                return n;
            })
        }

        autoLayout(newNodes, needUpdate);
        setEdges(newEdges);
        setTarget(null);
        setOnDragging(false);
        dragRef.current = null;
    };

    const removeFromParentNode = (newNodes, n) => {
        let parent = newNodes.find((e) => e.id === n.parentNode);
        parent.data.children = parent.data.children.filter((e) => e !== n.id);
        parent.style = { ...parent.style, width: changeWidth(parent), height: changeHeight(newNodes, parent) };
        if (parent.parentNode) {
            let grandParent = newNodes.find((n) => n.id === parent.parentNode);
            grandParent.style = { ...grandParent.style, width: changeWidth(grandParent), height: changeHeight(newNodes, grandParent) };
        }
        for (const child of parent.data.children) {
            let childNode = newNodes.find((e) => e.id === child);
            if (parent.type === "semanticNode") {
                childNode.position = { x: childNodeoffsetX, y: childNodeMarginY + parent.data.children.indexOf(child) * childNodeoffsetY };
            }
            else if (parent.type === "combinedNode") {
                childNode.position = { x: childSemanticNodeMarginX + parent.data.children.indexOf(child) * childSemanticNodeOffsetX, y: childSemanticNodeOffsetY };
            }
        }

        n.parentNode = null;
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
                        case "combinedNode":
                            color = combinedNodeStyle.backgroundColor;
                            break;
                        case "semanticNode":
                            color = semanticNodeStyle.backgroundColor;
                            break;
                        default:
                            color = stateNodeStyle.backgroundColor;
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
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onInit={setReactFlowInstance}
                    fitView
                >
                    <Panel position="top-right">
                        <button onClick={() => { autoLayout(nodes) }}>Layout</button>
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
                    onNodeClick={onNodeClick}
                    onConnect={onConnect}
                    fitView
                >
                    <Panel position="top-right">
                        <button onClick={changeLayoutMode}>{autoLayoutMode ? "Auto" : "Manual"} Layout</button>
                        <div className='mode-node-div' onDragStart={(event) => onDragStart(event, 'semanticNode')} draggable>
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