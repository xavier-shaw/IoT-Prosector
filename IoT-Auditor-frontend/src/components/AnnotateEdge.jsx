import { Chip, TextField } from '@mui/material';
import React, { useState } from 'react';
import { getBezierPath, EdgeLabelRenderer, BaseEdge, getSmoothStepPath } from 'reactflow';

export default function AnnotateEdge(props) {
    let { sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition, 
        markerEnd,
        style,
        id, data } = props;
    const l = data.label;
    let [transitionName, setTransitionName] = useState(l);
    let [editable, setEditable] = useState(false);

    const [edgePath, labelX, labelY] = getBezierPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition
    });

    const handleTextChange = (event) => {
        data.label = event.target.value;
        setTransitionName(event.target.value);
    };

    const handleChipClick = () => {
        setEditable(true);
    };

    const handleTextBlur = () => {
        setEditable(false);
    }

    return (
        <>
            <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style}/>
            <EdgeLabelRenderer>
                <div
                    id={id + "_label"}
                    style={{
                        position: 'absolute',
                        transform: `translate(-50%, -50%) translate(${labelX + (targetX - labelX) / 2}px,${labelY + (targetY - labelY) / 2}px)`,
                        fontSize: 14,
                        fontWeight: 'bold',
                        pointerEvents: 'all',
                        backgroundColor: "#f4a261",
                        borderRadius: 10,
                        zIndex: 4
                    }}
                    className="nodrag nopan"
                >
                    {editable ?
                        <TextField style={{borderColor: 'black'}} size='small' value={transitionName} onChange={handleTextChange} onBlur={handleTextBlur} autoFocus/>
                        :
                        <Chip label={transitionName} onClick={handleChipClick} />}
                </div>
            </EdgeLabelRenderer>
        </>
    );
};