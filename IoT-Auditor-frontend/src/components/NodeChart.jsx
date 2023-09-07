import React, { useEffect, useState, useMemo, useCallback, useRef, forwardRef, useImperativeHandle } from "react";
import ReactFlow, { Background, Controls, useStore, useEdgesState, useNodesState, Panel, useReactFlow, ReactFlowProvider, addEdge, getIncomers, getOutgoers, getConnectedEdges } from 'reactflow';
import * as d3 from "d3";
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
import { nodeOffsetX, nodeOffsetY, layoutRowNum, childNodeMarginY, childNodeoffsetX, childNodeoffsetY, highlightColor, semanticNodeStyle, semanticNodeMarginX, semanticNodeMarginY, semanticNodeOffsetX, stateNodeStyle, combinedNodeMarginX, combinedNodeMarginY, combinedNodeOffsetX, childSemanticNodeOffsetX, childSemanticNodeOffsetY, childNodeMarginX, combinedNodeStyle, childSemanticNodeMarginX, childSemanticNodeMarginY, offWidth, offHeight, displayNodeStyle, groupZIndex, edgeZIndex, selectedColor, customColors } from "../shared/chartStyle";
import axios from "axios";
import ELK from 'elkjs/lib/elk.bundled.js';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle } from "@mui/material";
import DisplayNode from "./DisplayNode";
import DisplayEdge from "./DisplayEdge";

const elk = new ELK();

const useLayoutedElements = () => {
    const { getNodes, setNodes, getEdges, fitView } = useReactFlow();
    const defaultOptions = {
        'elk.algorithm': 'layered',
        'elk.layered.spacing.nodeNodeBetweenLayers': 100,
        'elk.spacing.nodeNode': 80,
    };

    const getLayoutedElements = useCallback((options) => {
        const layoutOptions = { ...defaultOptions, ...options };
        let nodes = getNodes();
        let filter_nodes = nodes.filter((n) => !n.parentNode);

        const graph = {
            id: 'root',
            layoutOptions: layoutOptions,
            children: filter_nodes,
            edges: getEdges(),
        };

        elk.layout(graph).then(({ children }) => {
            nodes = nodes.map((n) => {
                let node = children.find((e) => e.id === n.id);
                if (node) {
                    n.position = { x: node.x, y: node.y };
                }

                return n;
            });

            console.log("new nodes", nodes);
            setNodes(nodes);
        });
    }, []);

    return { getLayoutedElements };
};

const NodeChart = forwardRef((props, ref) => {
    return (
        <ReactFlowProvider>
            <FlowChart {...props} ref={ref} />
        </ReactFlowProvider>
    )
})

const FlowChart = forwardRef((props, ref) => {
    let { board, chart, setChart, step, chartSelection, setChartSelection, updateMatrix } = props;
    const { getLayoutedElements } = useLayoutedElements();
    const { fitView } = useReactFlow();
    const reactFlowWrapper = useRef(null);
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [displayNodes, setDisplayNodes, onDisplayNodesChange] = useNodesState([]);
    const [displayEdges, setDisplayEdges, onDisplayEdgesChange] = useEdgesState([]);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const dragRef = useRef(null);
    const [onDragging, setOnDragging] = useState(false);
    const [target, setTarget] = useState(null);
    const [autoLayoutMode, setAutoLayoutMode] = useState(true);
    const [semanticHints, setSemanticHints] = useState({});
    const [dataHints, setDataHints] = useState({});
    const [preview, setPreview] = useState(false);
    const [openRepresentDialog, setOpenRepresentDialog] = useState(false);
    const [representNode, setRepresentNode] = useState(null);
    const nodeTypes_explore = useMemo(() => ({ stateNode: AnnotateNode, semanticNode: SemanticNode }), []);
    const nodeTypes_annotate = useMemo(() => ({ stateNode: ExploreNode, semanticNode: SemanticNode }), []);
    const nodeTypes_verify = useMemo(() => ({ stateNode: DisplayNode, semanticNode: DisplayNode }), []);
    const edgeTypes_explore = useMemo(() => ({ transitionEdge: ExploreEdge }), []);
    const edgeTypes_annotate = useMemo(() => ({ transitionEdge: ExploreEdge }), []);
    const edgeTypes_verify = useMemo(() => ({ transitionEdge: DisplayEdge }), []);

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
        else if (step === 2) {
            generateFinalChart();
        }
    }, [step]);

    useImperativeHandle(ref, () => ({
        updateAnnotation,
        collageStates,
        showSemanticHints,
        showDataHints,
        hideSemanticHints,
        hideDataHints,
        previewChart
    }));

    const onLayout = (newNodes, newEdges) => {
        if (autoLayoutMode) {
            getLayoutedElements({
                'elk.algorithm': 'org.eclipse.elk.force',
                // 'elk.algorithm': 'org.eclipse.elk.radial"
                // 'elk.algorithm': 'layered', 'elk.direction': 'RIGHT'
                // 'elk.algorithm': 'layered', 'elk.direction': 'DOWN'
            });

            window.requestAnimationFrame(() => {
                fitView();
            });
        }
        else {
            layout(newNodes, newEdges, false);
        };

        setAutoLayoutMode((prev) => (!prev));
    };

    const collageStates = async () => {
        await axios
            .post(window.HARDWARE_ADDRESS + "/collage", {
                device: board.title,
                nodes: nodes
            })
            .then((resp) => {
                let action_group_count = resp.data.action_group_count;
                let action_collage_dict = resp.data.action_collage_dict;
                let semantic_group_cnt = resp.data.semantic_group_cnt;
                let semantic_collage_dict = resp.data.semantic_collage_dict;
                let combined_group_cnt = resp.data.combined_group_cnt;
                let combined_collage_dict = resp.data.combined_collage_dict;

                let newNodes = [...nodes];
                let newEdges = [...edges];

                newNodes = newNodes.map((n) => {
                    n.style = stateNodeStyle;
                    return n;
                })

                for (let index = 0; index < action_group_count; index++) {
                    let semanticNode = createNewNode({ x: semanticNodeMarginX + semanticNodeOffsetX * index, y: semanticNodeMarginY }, "semanticNode");
                    semanticNode.data.label += " " + index;
                    for (const [nid, cid] of Object.entries(action_collage_dict)) {
                        if (cid === index) {
                            let node = newNodes.find((n) => n.id === nid);
                            semanticNode.data.children.push(node.id);
                            node.parentNode = semanticNode.id;
                            node.position = { x: childNodeoffsetX, y: childNodeMarginY + (semanticNode.data.children.length - 1) * childNodeoffsetY };
                            node.positionAbsolute = {
                                x: semanticNode.positionAbsolute.x + node.position.x,
                                y: semanticNode.positionAbsolute.y + node.position.y
                            };
                        }
                    };
                    semanticNode.style = { ...semanticNode.style, height: changeHeight(newNodes, semanticNode) };
                    semanticNode.height = parseInt(semanticNode.style.height.slice(0, -2));
                    if (semanticNode.data.children.length == 1) {
                        let child = newNodes.find((n) => n.id === semanticNode.data.children[0]);
                        child.parentNode = null;
                    }
                    else {
                        newNodes.push(semanticNode);
                    }
                };

                let semanticHints = {};
                for (let index = 0; index < semantic_group_cnt; index++) {
                    semanticHints[index] = [];
                    for (const [nid, cid] of Object.entries(semantic_collage_dict)) {
                        if (cid === index) {
                            semanticHints[index].push(nid);
                        }
                    };
                }

                let dataHints = {};
                for (let index = 0; index < combined_group_cnt; index++) {
                    dataHints[index] = [];
                    for (const [nid, cid] of Object.entries(combined_collage_dict)) {
                        if (cid === index) {
                            dataHints[index].push(nid);
                        }
                    }
                }

                layout(newNodes, newEdges, false);
                newEdges = hiddenChildEdges(newNodes, newEdges);
                updateMatrix(newNodes);
                setChart((prevChart) => ({ ...prevChart, nodes: newNodes, edges: newEdges }));
                setEdges(newEdges);
                setSemanticHints(semanticHints);
                setDataHints(dataHints);
                setChartSelection(null);
            })
    };

    const changeHeight = (nodes, parentNode) => {
        let maxHeight = 0;

        for (const child_id of parentNode.data.children) {
            let child = nodes.find((n) => n.id === child_id);
            let childBottom = child.position.y + parseInt(child.style.height.slice(0, -2));
            if (childBottom > maxHeight) {
                maxHeight = childBottom;
            }
        };
        maxHeight += childNodeMarginY;

        return maxHeight + "px";
    }

    const updateAnnotation = () => {
        const newChart = reactFlowInstance.toObject();
        setChart(newChart);
        console.log("update annotation");
        return newChart;
    };

    const onNodeClick = (evt, node) => {
        const color = d3.scaleOrdinal()
            .domain(nodes.filter((n) => n.type === "stateNode").map(d => d.id))
            .range(customColors);

        let newNodes = [...nodes];

        if (chartSelection?.type === "stateNode") {
            newNodes = newNodes.map((n) => {
                if (n.id === chartSelection.id) {
                    n.style.backgroundColor = "#F7E2E1";
                }

                return n;
            })
        }

        if (node.type === "stateNode" && node.id !== chartSelection?.id) {
            newNodes = newNodes.map((n) => {
                if (n.id === node.id) {
                    n.style.backgroundColor = color(n.id);
                }

                return n;
            });
            setChartSelection(node);
        }
        else {
            setChartSelection(null);
        }

        setNodes(newNodes);
    };

    const changeLayoutMode = () => {
        setAutoLayoutMode((prev) => (!prev));
    };

    const layout = (newNodes, newEdges, preview) => {
        let nextRowY = 0;
        let index = 0;
        let layoutNodes = [];
        newNodes = newNodes.map((node) => {
            // if (node.type === "stateNode") {
            //     node.data.representative = node.data.label;
            // }
            if (!node.parentNode) {
                if (index % layoutRowNum === 0 && index !== 0) {
                    for (let i = index - 1; i >= index - layoutRowNum; i--) {
                        let prevNode = layoutNodes[i];
                        if (prevNode.position.y + parseInt(prevNode.style.height.slice(0, -2)) > nextRowY) {
                            nextRowY = prevNode.position.y + parseInt(prevNode.style.height.slice(0, -2));
                        };
                    }
                }
                node.position = { x: nodeOffsetX * (index % layoutRowNum), y: nextRowY + nodeOffsetY };
                node.positionAbsolute = { x: nodeOffsetX * (index % layoutRowNum), y: nextRowY + nodeOffsetY };
                if (node.data.children?.length > 0 && !preview) {
                    for (const childId of node.data.children) {
                        let child = newNodes.find((n) => n.id === childId);
                        child.positionAbsolute = {
                            x: node.positionAbsolute.x + child.position.x,
                            y: node.positionAbsolute.y + child.position.y
                        }
                    }
                }
                layoutNodes.push(node);
                index += 1;
            };

            return node;
        })

        if (preview) {
            setDisplayNodes(newNodes);
            setDisplayEdges(newEdges);
        }
        else {
            setNodes(newNodes);
            setEdges(newEdges);
        }
    };

    const insideLayout = (nodes) => {
        nodes = nodes.map((n) => {
            if (!n.parentNode) {
                return n;
            }
            else {
                let parent = nodes.find((nd) => nd.id === n.parentNode);
                n.position = { x: childNodeoffsetX, y: childNodeMarginY + parent.data.children.indexOf(n.id) * childNodeoffsetY };
                return n;
            }
        });

        return nodes;
    };

    const showSemanticHints = (node) => {
        let newNodes = [...nodes];
        for (const children of Object.values(semanticHints)) {
            if (children.includes(node.id)) {
                for (const nid of children) {
                    let semanticNode = newNodes.find((n) => n.id === nid);
                    semanticNode.style = { ...semanticNode.style, animation: "wiggle 1s infinite" }
                }
            }
        }
        setNodes(newNodes);
    };

    const hideSemanticHints = (node) => {
        let newNodes = [...nodes];
        for (const children of Object.values(semanticHints)) {
            if (children.includes(node.id)) {
                for (const nid of children) {
                    let semanticNode = newNodes.find((n) => n.id === nid);
                    semanticNode.style = { ...semanticNode.style, animation: "" }
                }
            }
        }
        setNodes(newNodes);
    };

    const showDataHints = (node) => {
        let newNodes = [...nodes];
        for (const children of Object.values(dataHints)) {
            if (children.includes(node.id)) {
                for (const nid of children) {
                    console.log("data", nid);
                    let dataNode = newNodes.find((n) => n.id === nid);
                    dataNode.style = { ...dataNode.style, animation: "wiggle 1s infinite" }
                }
            }
        }
        setNodes(newNodes);
    };

    const hideDataHints = (node) => {
        let newNodes = [...nodes];
        for (const children of Object.values(dataHints)) {
            if (children.includes(node.id)) {
                for (const nid of children) {
                    let dataNode = newNodes.find((n) => n.id === nid);
                    dataNode.style = { ...dataNode.style, animation: "" }
                }
            }
        };
        setNodes(newNodes);
    };

    const previewChart = () => {
        if (!preview) {
            generateFinalChart();
            setPreview(true);
        }
        else {
            setPreview(false);
        }
    };

    const generateFinalChart = () => {
        let newNodes = [];
        let newEdges = [];

        for (const node of nodes) {
            if (!node.parentNode) {
                let label = "";
                if (node.data.representative) {
                    label = node.data.representative;
                }
                else {
                    for (const childId of node.data.children) {
                        let child = nodes.find((n) => n.id === childId);
                        label += child.data.label.split(" ")[0] + ", ";
                    }
                    label = label.slice(0, -2);
                };

                let newNode = { ...node, data: { ...node.data, representative: label }, style: displayNodeStyle };
                newNodes.push(newNode);
            }
        };
        
        let edgeSet = {};
        for (const edge of edges) {
            let uniqueId = edge.source + "-" + edge.target;
            if (!edgeSet.hasOwnProperty(uniqueId)) {
                edgeSet[uniqueId] = edge;
                edge.data.actions = [edge.data.label];
            }
            else {
                if (!edgeSet[uniqueId].data.actions.includes(edge.data.label)) {
                    edgeSet[uniqueId].data.actions.push(edge.data.label);                    
                }
            }
        };
        
        for (const [key, value] of Object.entries(edgeSet)) {
            newEdges.push(value);
        }

        layout(newNodes, newEdges, true);
    };

    const createNewNode = (position, type) => {
        let zIndex;
        let nodeStyle;
        let nodeData = { label: "", children: [] };

        switch (type) {
            case "semanticNode":
                zIndex = groupZIndex;
                nodeData.label = "State Group"
                nodeStyle = semanticNodeStyle;
                break;
            default:
                break;
        }

        const newNode = {
            width: parseInt(nodeStyle.width.slice(0, -2)),
            height: parseInt(nodeStyle.height.slice(0, -2)),
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
            newNodes = newNodes.map((n) => {
                if (n.id === node.id) {
                    if (n.parentNode && n.parentNode !== target.id) {
                        needUpdate = true;
                        newEdges = revealOwnEdges(newEdges, node);
                        let parent = newNodes.find((e) => e.id === n.parentNode);
                        parent.data.children = parent.data.children.filter((e) => e !== n.id);
                        parent.style = { ...parent.style, height: changeHeight(newNodes, parent) };
                        n.parentNode = null;
                    };

                    n.parentNode = target.id;
                    let parent = newNodes.find((e) => e.id === target.id);
                    if (!parent.data.children.includes(n.id)) {
                        needUpdate = true;
                        parent.data.children.push(n.id);
                        n.position = { x: childNodeoffsetX, y: childNodeMarginY + (parent.data.children.length - 1) * childNodeoffsetY };
                        parent.style = { ...parent.style, height: changeHeight(newNodes, parent) };
                    }
                    else {
                        n.position = { x: childNodeoffsetX, y: childNodeMarginY + parent.data.children.indexOf(n.id) * childNodeoffsetY };
                    }
                }
                return n;
            })
        }
        else {
            newNodes = newNodes.map((n) => {
                if (n.id === node.id) {
                    if (n.parentNode) {
                        needUpdate = true;
                        newEdges = revealOwnEdges(newEdges, node);
                        let parent = newNodes.find((e) => e.id === n.parentNode);
                        parent.data.children = parent.data.children.filter((e) => e !== n.id);
                        parent.style.height = changeHeight(newNodes, parent);
                        n.parentNode = null;
                    }
                    n.position = node.positionAbsolute;
                }
                return n;
            })
            newNodes = newNodes.filter((n) => !n.data.children || n.data.children?.length > 0);
        }

        newNodes = insideLayout(newNodes);
        newEdges = hiddenChildEdges(newNodes, newEdges);
        if (needUpdate) {
            updateMatrix(newNodes);
            setChart((prevChart) => ({ ...prevChart, nodes: newNodes, edges: newEdges }));
        }
        setNodes(newNodes);
        setEdges(newEdges);
        setTarget(null);
        setOnDragging(false);
        dragRef.current = null;
    };

    const onNodeContextMenu = (event, node) => {
        event.preventDefault();
        if (node.parentNode) {
            setRepresentNode(node);
            setOpenRepresentDialog(true);
        }
    };

    const onConfirmRepresentativeNode = () => {
        let newNodes = [...nodes];
        let parent = newNodes.find((n) => n.data.children?.includes(representNode.id));
        parent.data.representative = representNode.data.label;
        setNodes(newNodes);
        onCloseDialog();
    };

    const onCloseDialog = () => {
        setRepresentNode(null);
        setOpenRepresentDialog(false);
    };

    const revealOwnEdges = (edges, node) => {
        edges = edges.map((e) => {
            if (e.originalSource === node.id) {
                e.source = node.id;
                e.sourceHanlde = null;
                e.hidden = false;
            }
            else if (e.originalTarget === node.id) {
                e.target = node.id;
                e.targetHandle = null;
                e.hidden = false;
            }

            return e;
        });

        return edges;
    };

    const hiddenChildEdges = (nodes, edges) => {
        edges = edges.map((e) => {
            let srcNode = nodes.find((n) => n.id === e.source);
            let dstNode = nodes.find((n) => n.id === e.target);
            let srcNodeParent = nodes.find((n) => n.id === srcNode.parentNode);
            let dstNodeParent = nodes.find((n) => n.id === dstNode.parentNode);

            e = {
                ...e,
                hidden: (srcNode.id === dstNode.id) || (srcNodeParent && dstNodeParent && (srcNodeParent === dstNodeParent)) ? true : false,
                source: srcNodeParent ? srcNodeParent.id : srcNode.id,
                target: dstNodeParent ? dstNodeParent.id : dstNode.id,
                sourceHandle: "source-" + e.originalSource,
                targetHandle: "target-" + e.originalTarget,
                animated: true
            };

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
                        <button onClick={() => layout(nodes, edges, false)}>Layout</button>
                    </Panel>
                    <Background />
                    <Controls />
                </ReactFlow>
            }
            {step === 1 && !preview &&
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
                    onNodeContextMenu={onNodeContextMenu}
                    fitView
                >
                    <Panel position="top-right">
                        <button onClick={() => onLayout(nodes, edges)}>Layout</button>
                        <div className='mode-node-div' onDragStart={(event) => onDragStart(event, 'semanticNode')} draggable>
                            State Group
                        </div>
                    </Panel>
                    <Background />
                    <Controls />
                </ReactFlow>
            }
            {(step === 2 || preview === true) &&
                <ReactFlow
                    nodeTypes={nodeTypes_verify}
                    edgeTypes={edgeTypes_verify}
                    nodes={displayNodes}
                    edges={displayEdges}
                    onNodesChange={onDisplayNodesChange}
                    onEdgesChange={onDisplayEdgesChange}
                    onInit={setReactFlowInstance}
                    fitView
                >
                    <Background />
                    <Controls />
                </ReactFlow>
            }

            <Dialog open={openRepresentDialog}>
                <DialogTitle>Set As Representative</DialogTitle>
                <DialogContent>
                    Are you sure to set Node {representNode?.data.label} as the Representative Node of this group?
                </DialogContent>
                <DialogActions>
                    <Button variant="outlined" color="error" onClick={onCloseDialog}>Cancel</Button>
                    <Button variant="outlined" color="primary" onClick={onConfirmRepresentativeNode}>Confirm</Button>
                </DialogActions>
            </Dialog>
        </div>
    )
});


export default NodeChart;