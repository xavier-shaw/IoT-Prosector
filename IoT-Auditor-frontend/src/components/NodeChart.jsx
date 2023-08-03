import React, { useEffect, useState } from "react";
import ReactFlow, { Background, useEdgesState, useNodesState } from 'reactflow';
import 'reactflow/dist/style.css';

export default function NodeChart(props) {
    let { board, setBoard } = props;

    const [states, setStates] = useNodesState([]);
    const [transitions, setTransitions] = useEdgesState([]);

    useEffect(() => {
        if (board.hasOwnProperty("data")) {
            if (board.data.hasOwnProperty("statesDict")) {
                let states = [];
                for (const [idx, state] of Object.entries(board.data.statesDict)) {
                    states.push(state);
                };
                setStates(states);
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

    return (
        <div style={{ width: '100%', height: '100%' }}>
            <ReactFlow nodes={states} edges={transitions}>
                <Background/>
            </ReactFlow>
        </div>
    )
}