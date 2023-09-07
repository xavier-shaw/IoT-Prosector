import { Chip, TextField } from '@mui/material';
import React, { useState } from 'react';
import { getBezierPath, EdgeLabelRenderer, BaseEdge, getSmoothStepPath } from 'reactflow';
import { getEdgeStyle } from '../shared/chartStyle';
import { MarkerType } from "reactflow";
const markerStyle = {
    type: MarkerType.ArrowClosed,
    width: 30,
    height: 30,
    color: '#FF0072',
}

export default function DisplayEdge(props) {
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
            <BaseEdge id={id} path={edgePath} markerEnd={markerStyle} style={style} />
            <EdgeLabelRenderer>
                <div
                    id={id + "_label"}
                    style={getEdgeStyle(labelX, labelY)}
                    className="nodrag nopan"
                >
                    <Chip color="secondary" label={data.actions} />
                </div>
            </EdgeLabelRenderer>
        </>
    );
};