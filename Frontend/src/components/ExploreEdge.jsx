import { Chip, TextField } from '@mui/material';
import React, { useEffect, useState } from 'react';
import { getBezierPath, EdgeLabelRenderer, BaseEdge, getSmoothStepPath, useNodes, MarkerType } from 'reactflow';
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

    const [edgePath, labelX, labelY] = getBezierPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition
    });

    return (
        <>
            <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={{...style, strokeWidth: 4}} />
            <EdgeLabelRenderer>
                <div
                    id={id + "_label"}
                    style={getEdgeStyle(labelX, labelY)}
                    className="nodrag nopan" 
                >
                    <p style={{ margin: 0, fontSize: 28, fontFamily: "Times New Roman" }}>{data.label}</p>
                    {/* <Chip color="secondary"  label={data.label} /> */}
                </div>
            </EdgeLabelRenderer>
        </>
    )
};