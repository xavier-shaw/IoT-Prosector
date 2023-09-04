import { Chip, TextField } from '@mui/material';
import React, { useEffect, useState } from 'react';
import { getBezierPath, EdgeLabelRenderer, BaseEdge, getSmoothStepPath } from 'reactflow';
import { getEdgeStyle } from '../shared/chartStyle';

export default function ExploreEdge(props) {
    let { sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition, 
        markerEnd,
        style,
        id, data } = props;

    const [edgePath, labelX, labelY] = getSmoothStepPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition
    });

    return (
        <>
            <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style}/>
            <EdgeLabelRenderer>
                <div
                    id={id + "_label"}
                    style={getEdgeStyle(labelX, labelY)}
                    className="nodrag nopan"
                >
                    <Chip label={data.label}/>
                </div>
            </EdgeLabelRenderer>
        </>
    );
};