import React, { useEffect, useState, useMemo, useCallback, useRef, forwardRef, useImperativeHandle } from "react";
import ReactFlow, { Background, Controls, useEdgesState, useNodesState, applyNodeChanges, applyEdgeChanges } from 'reactflow';
import 'reactflow/dist/style.css';
import ExploreNode from "./ExploreNode";
import AnnotateNode from "./AnnotateNode";
import ExploreEdge from "./ExploreEdge";
import AnnotateEdge from "./AnnotateEdge";

const NodeChart = forwardRef((props, ref) => {
    let { step, states, setStates, transitions, setTransitions } = props;
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    const nodeTypes_explore = useMemo(() => ({ stateNode: ExploreNode }), []);
    const nodeTypes_annotate = useMemo(() => ({ stateNode: AnnotateNode }), []);
    const edgeTypes_explore = useMemo(() => ({ transitionEdge: ExploreEdge }), []);
    const edgeTypes_annotate = useMemo(() => ({ transitionEdge: AnnotateEdge }), []);

    useEffect(() => {
        console.log("states", states);
        setNodes(states);
    }, [states]);

    useEffect(() => {
        console.log("transitions", transitions);
        setEdges(transitions);
    }, [transitions]);

    useImperativeHandle(ref, () => ({
        updateAnnotation
    }))

    const updateAnnotation = () => {
        setStates(nodes);
        setTransitions(transitions);
    };

    return (
        <div style={{ width: '100%', height: '100%' }}>
            {step === 0 &&
                <ReactFlow
                    nodeTypes={nodeTypes_explore}
                    edgeTypes={edgeTypes_explore}
                    nodes={nodes}
                    edges={edges}
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