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
import 'reactflow/dist/style.css';
import { nodeOffsetX, nodeOffsetY, layoutRowNum, childNodeMarginY, childNodeoffsetX, childNodeoffsetY, highlightColor, semanticNodeStyle, semanticNodeMarginX, semanticNodeMarginY, semanticNodeOffsetX, stateNodeStyle, combinedNodeMarginX, combinedNodeMarginY, combinedNodeOffsetX, childSemanticNodeOffsetX, childSemanticNodeOffsetY, childNodeMarginX, combinedNodeStyle, childSemanticNodeMarginX, childSemanticNodeMarginY, offWidth, offHeight, displayNodeStyle, groupZIndex, edgeZIndex, selectedColor, customColors, stateZIndex, colorPalette } from "../shared/chartStyle";
import axios from "axios";
import { Button, Dialog, DialogActions, DialogContent, DialogTitle } from "@mui/material";
import DisplayNode from "./DisplayNode";
import DisplayEdge from "./DisplayEdge";
import FloatingEdge from "./FloatingEdge";
import FloatingConnectionLine from "./FloatingConnectionLine";
import * as htmlToImage from 'html-to-image'

const NodeChart = forwardRef((props, ref) => {
    return (
        <ReactFlowProvider>
            <FlowChart {...props} ref={ref} />
        </ReactFlowProvider>
    )
})

const FlowChart = forwardRef((props, ref) => {
    let { board, chart, setChart, step, chartSelection, setChartSelection, updateMatrix } = props;
    const reactFlowWrapper = useRef(null);
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [displayNodes, setDisplayNodes, onDisplayNodesChange] = useNodesState([]);
    const [displayEdges, setDisplayEdges, onDisplayEdgesChange] = useEdgesState([]);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const dragRef = useRef(null);
    const [onDragging, setOnDragging] = useState(false);
    const [target, setTarget] = useState(null);
    const [semanticHints, setSemanticHints] = useState({});
    const [dataHints, setDataHints] = useState({});
    const [preview, setPreview] = useState(false);
    const [openRepresentDialog, setOpenRepresentDialog] = useState(false);
    const [representNode, setRepresentNode] = useState(null);
    const nodeTypes_explore = useMemo(() => ({ stateNode: AnnotateNode, semanticNode: SemanticNode }), []);
    const nodeTypes_annotate = useMemo(() => ({ stateNode: ExploreNode, semanticNode: SemanticNode }), []);
    const nodeTypes_verify = useMemo(() => ({ stateNode: DisplayNode, semanticNode: DisplayNode }), []);
    // const edgeTypes_explore = useMemo(() => ({ transitionEdge: FloatingEdge }), []);
    // const edgeTypes_annotate = useMemo(() => ({ transitionEdge: FloatingEdge }), []);
    // const edgeTypes_verify = useMemo(() => ({ transitionEdge: FloatingEdge }), []);
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
        [newNodes, newEdges] = layout(newNodes, newEdges, false);
        setNodes(newNodes);
        setEdges(newEdges);
        console.log("new edges", newEdges)
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
                // let semantic_group_cnt = resp.data.semantic_group_cnt;
                // let semantic_collage_dict = resp.data.semantic_collage_dict;
                // let combined_group_cnt = resp.data.combined_group_cnt;
                // let combined_collage_dict = resp.data.combined_collage_dict;

                let newNodes = [...nodes];
                let newEdges = [...edges];

                // newNodes = newNodes.map((n) => {
                //     // n.style = stateNodeStyle;
                //     n.zIndex = stateZIndex;
                //     return n;
                // })

                for (let index = 0; index < action_group_count; index++) {
                    let semanticNode = createNewNode({ x: semanticNodeMarginX + semanticNodeOffsetX * index, y: semanticNodeMarginY }, "semanticNode");
                    semanticNode.data.label = "Group (";
                    for (const [nid, cid] of Object.entries(action_collage_dict)) {
                        if (cid === index) {
                            let node = newNodes.find((n) => n.id === nid);
                            semanticNode.data.children.push(node.id);
                            semanticNode.data.label += node.data.label.split(" ")[0] + ",";
                            node.parentNode = semanticNode.id;
                            node.position = { x: childNodeoffsetX, y: childNodeMarginY + (semanticNode.data.children.length - 1) * childNodeoffsetY };
                            node.positionAbsolute = {
                                x: semanticNode.positionAbsolute.x + node.position.x,
                                y: semanticNode.positionAbsolute.y + node.position.y
                            };
                        }
                    };
                    semanticNode.data.label = semanticNode.data.label.slice(0, -1) + ")";
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

                [newNodes, newEdges] = layout(newNodes, newEdges, false);
                newNodes = updateGroups(newNodes);
                newEdges = hiddenChildEdges(newNodes, newEdges);
                updateMatrix(newNodes);
                setChart((prevChart) => ({ ...prevChart, nodes: newNodes, edges: newEdges }));
                setNodes(newNodes);
                setEdges(newEdges);
                // setSemanticHints(semanticHints);
                // setDataHints(dataHints);
                setChartSelection(null);
                console.log("collage nodes", newNodes);
                console.log("collage edges", newEdges);
            })
    };

    const changeHeight = (nodes, parentNode) => {
        let newHeight = childNodeMarginY + parentNode.data.children.length * childNodeoffsetY;
        return newHeight + "px";
    }

    const updateAnnotation = () => {
        const newChart = reactFlowInstance.toObject();
        setChart(newChart);
        console.log("update annotation");
        return newChart;
    };

    const onNodeClick = (evt, node) => {
        const color = d3.scaleOrdinal()
            .domain(nodes.map(d => d.id))
            .range(colorPalette);

        let newNodes = [...nodes];

        if (chartSelection?.type === "stateNode") {
            newNodes = newNodes.map((n) => {
                if (n.id === chartSelection.id) {
                    n.style.backgroundColor = stateNodeStyle.backgroundColor;
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

    const layout = (newNodes, newEdges, preview) => {
        let nextRowY = 0;
        let index = 0;
        let layoutNodes = [];

        newEdges = newEdges.map((edge) => {
            edge.animated = false;
            edge.markerEnd = {
                type: MarkerType.ArrowClosed,
                width: 30,
                height: 30,
                color: '#FF0072',
            }
            edge.zIndex = edgeZIndex;
            return edge;
        });

        newNodes = newNodes.map((node) => {
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
                node.positionAbsolute = node.position;
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

        return [newNodes, newEdges];
    };

    const updateGroups = (nodes) => {
        const colors = d3.scaleOrdinal()
            .domain(nodes.map(d => d.id))
            .range(colorPalette);

        nodes = nodes.map((n) => {
            if (n.type === "semanticNode") {
                let label = "Group (";
                for (const childId of n.data.children) {
                    let child = nodes.find((n1) => n1.id === childId);
                    child.position = { x: childNodeoffsetX, y: childNodeMarginY + n.data.children.indexOf(childId) * childNodeoffsetY };
                    child.positionAbsolute = { x: n.positionAbsolute.x + child.position.x, y: n.positionAbsolute.y + child.position.y };
                    label += child.data.label.split(" ")[0] + ",";
                }
                label = label.slice(0, -1) + ")";
                n.data.label = label;
                n.style = {...n.style, backgroundColor: colors(n.id)};
                return n;
            }
            else if (n.type === "stateNode") {
                if (n.parentNode && n.id !== chartSelection?.id) {
                    n.style = {...n.style, backgroundColor: stateNodeStyle.backgroundColor};
                }
                else {
                    n.style = {...n.style, backgroundColor: colors(n.id)}
                }
                // let parent = nodes.find((nd) => nd.id === n.parentNode);
                // n.position = { x: childNodeoffsetX, y: childNodeMarginY + parent.data.children.indexOf(n.id) * childNodeoffsetY };
                // n.positionAbsolute = { x: parent.positionAbsolute.x + n.position.x, y: parent.positionAbsolute.y + n.position.y };
                
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
                    label = node.data.label;
                };

                let newNode = { ...node, data: { ...node.data, representative: label }, style: displayNodeStyle };
                newNodes.push(newNode);
            }
        };

        let edgeSet = {};
        let pos_edges = edges.filter((e) => !e.hidden);
        for (const edge of pos_edges) {
            let uniqueId = edge.source + "-" + edge.target + "-" + edge.data.label;
            if (!edgeSet.hasOwnProperty(uniqueId)) {
                edgeSet[uniqueId] = edge;
                edge.data.actions = edge.data.label;
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

        // [newNodes, newEdges] = layout(newNodes, newEdges, true);
        setDisplayNodes(newNodes);
        setDisplayEdges(newEdges);
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
        console.log(centerX + ", " + centerY)
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
        console.log("target", target)
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
                    n.positionAbsolute = node.positionAbsolute;
                }
                return n;
            })
            newNodes = newNodes.filter((n) => !n.data.children || n.data.children?.length > 0);
        }

        newNodes = updateGroups(newNodes);
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
        parent.data.children = parent.data.children.filter((c) => c !== representNode.id);
        parent.data.children.unshift(representNode.id);
        
        newNodes = updateGroups(newNodes);
        let newEdges = hiddenChildEdges(newNodes, edges);
        setNodes(newNodes);
        setEdges(newEdges);
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
                sourceHandle: "source-" + (srcNodeParent? srcNodeParent.data.children.indexOf(e.originalSource): srcNode.data.children?.indexOf(e.originalSource)),
                targetHandle: "target-" + (dstNodeParent? dstNodeParent.data.children.indexOf(e.originalTarget): dstNode.data.children?.indexOf(e.originalTarget)),
            };

            return e;
        });

        return edges;
    };

    useEffect(() => {
            const colors = d3.scaleOrdinal()
            .domain(nodes.map(d => d.id))
            .range(colorPalette);

            setNodes((nodes) =>
                nodes.map((node) => {
                    if (node.id === target?.id) {
                        node.style = { ...node.style, backgroundColor: highlightColor };
                    } else {
                        let color;
                        if (!node.parentNode || node.id === chartSelection?.id) {
                            color = colors(node.id);
                        }
                        else{
                            color = stateNodeStyle.backgroundColor;
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

    async function exportDiagram() {
        try {
            // Get the DOM element
            let elements = document.getElementsByClassName('flow')[0];
            // Convert to SVG
            const svgContent = await htmlToImage.toSvg(elements);
            const svgElement = decodeURIComponent(svgContent.replace("data:image/svg+xml;charset=utf-8,", "").trim());
            // Open new window
            const newWindow = open();
            // Safer version of document.write
            document.write=function(s){
                var scripts = document.getElementsByTagName('script');
                var lastScript = scripts[scripts.length-1];
                lastScript.insertAdjacentHTML("beforebegin", s);
            }
            // Write our page content to the newly opened page
            newWindow.document.write(
                `<html>
                    <head>
                        <title>React Flow PDF</title>
                        <style>
                            body {
                                width: ${"1200px"};
                                height: ${"1000px"};
                                margin: auto
                            }
                            .container {
                                background: #393D43;
                                text-align: center;
                                height: 100%;
                                width: 100%;
                            }
                            
                            @page {
                                size: 29.7cm, 21cm
                                margin:0 !important;
                            }
                            @media print {
                                * {
                                    -webkit-print-color-adjust: exact !important;
                                    color-adjust: exact !important;
                                }
                                .container {
                                    background: none;
                                }
                            }
                        </style>
                    </head>
                    <body>
                        <div class='container'>
                            <div class='svg-container'>
                                ${svgElement}
                            </div>
                            
                            <script>
                                document.close();
                                window.print();
                            </script>
                        </div>
                    </body>
                </html>`
            )
        } catch (error) {
            console.log(error);
        }
    }

    return (
        <div style={{ width: '100%', height: '100%', backgroundColor: "white" }} ref={reactFlowWrapper}>
            {step === 0 &&
                <ReactFlow
                    className="flow"
                    nodeTypes={nodeTypes_explore}
                    edgeTypes={edgeTypes_explore}
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onInit={setReactFlowInstance}
                    connectionLineComponent={FloatingConnectionLine}
                    fitView
                >
                    <Panel position="top-right">
                        <button onClick={() => onLayout(nodes, edges)}>Layout</button>
                        <button onClick={exportDiagram}>Export</button>
                    </Panel>
                    <Background />
                    <Controls />
                </ReactFlow>
            }
            {step === 1 && !preview &&
                <ReactFlow
                    className="flow"
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
                    connectionLineComponent={FloatingConnectionLine}
                    fitView
                >
                    <Panel position="top-right">
                        <button onClick={() => onLayout(nodes, edges)}>Layout</button>
                        <button onClick={exportDiagram}>Export</button>
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
                    className="flow"
                    nodeTypes={nodeTypes_verify}
                    edgeTypes={edgeTypes_verify}
                    nodes={displayNodes}
                    edges={displayEdges}
                    onNodesChange={onDisplayNodesChange}
                    onEdgesChange={onDisplayEdgesChange}
                    onInit={setReactFlowInstance}
                    fitView
                >
                    <Panel position="top-right">
                        {/* <button onClick={() => onLayout(nodes, edges)}>Layout</button> */}
                        <button onClick={exportDiagram}>Export</button>
                    </Panel>
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