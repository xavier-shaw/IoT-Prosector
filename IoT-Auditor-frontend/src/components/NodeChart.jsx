import React, { useEffect, useState, useMemo, useCallback, useRef } from "react";
import ReactFlow, { Background, Controls, useEdgesState, useNodesState, applyNodeChanges, applyEdgeChanges } from 'reactflow';
import 'reactflow/dist/style.css';
import StateNode from "./StateNode";
import TransitionEdge from "./TransitionEdge";

export default function NodeChart(props) {
    let { board, setBoard } = props;
    let ref = useRef({});
    const [states, setStates, onStateChange] = useNodesState([]);
    const [transitions, setTransitions, onTransitionChange] = useEdgesState([]);

    const nodeTypes = useMemo(() => ({ stateNode: StateNode }), []);
    const edgeTypes = useMemo(() => ({ transitionEdge: TransitionEdge }), []);

    useEffect(() => {
        if (board.hasOwnProperty("data")) {
            if (board.data.hasOwnProperty("statesDict")) {
                let states = [];
                for (const [idx, state] of Object.entries(board.data.statesDict)) {
                    states.push(state);
                };
                console.log("here set state", states);
                setStates(states);
                ref.current.states = states;
            };

            if (board.data.hasOwnProperty("transitionsDict")) {
                let transitions = [];
                for (const [idx, transition] of Object.entries(board.data.transitionsDict)) {
                    transitions.push(transition);
                };
                setTransitions(transitions);
            }
        }
    }, [board]);

    useEffect(() => {
        ref.current.states = states;
        console.log(states)
    }, [states]);

    useEffect(() => {
        ref.current.transitions = transitions;
    }, [transitions]);

    return (
        <div style={{ width: '100%', height: '100%' }}>
            <ReactFlow
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                nodes={states}
                edges={transitions}
                onNodesChange={onStateChange}
                onEdgesChange={onTransitionChange}
            >
                <Background />
                <Controls />
            </ReactFlow>
        </div>
    )
}