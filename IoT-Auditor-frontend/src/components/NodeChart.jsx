import React, { useEffect, useState, useMemo, useCallback, useRef, forwardRef, useImperativeHandle } from "react";
import ReactFlow, { Background, Controls, useEdgesState, useNodesState, applyNodeChanges, applyEdgeChanges } from 'reactflow';
import 'reactflow/dist/style.css';
import StateNode from "./StateNode";
import TransitionEdge from "./TransitionEdge";
import { Button } from "@mui/material";

const NodeChart = forwardRef((props, ref) => {
    let { states, setStates, transitions, setTransitions } = props;
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    const nodeTypes = useMemo(() => ({ stateNode: StateNode }), []);
    const edgeTypes = useMemo(() => ({ transitionEdge: TransitionEdge }), []);

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
            <ReactFlow
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                fitView
            >
                <Background />
                <Controls />
            </ReactFlow>
        </div>
    )
});

export default NodeChart;