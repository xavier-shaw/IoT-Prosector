import { useCallback } from 'react';
import { useStore, getBezierPath, EdgeLabelRenderer, BaseEdge } from 'reactflow';
import { getEdgeParams } from './utils.js';
import { getEdgeStyle } from '../shared/chartStyle.js';
import { Chip } from '@mui/material';

function FloatingEdge({ id, source, target, markerEnd, style, data }) {
    const sourceNode = useStore(useCallback((store) => store.nodeInternals.get(source), [source]));
    const targetNode = useStore(useCallback((store) => store.nodeInternals.get(target), [target]));

    if (!sourceNode || !targetNode) {
        return null;
    }

    const { sx, sy, tx, ty, sourcePos, targetPos } = getEdgeParams(sourceNode, targetNode);

    const [edgePath, centerX, centerY] = getBezierPath({
        sourceX: sx,
        sourceY: sy,
        sourcePosition: sourcePos,
        targetPosition: targetPos,
        targetX: tx,
        targetY: ty,
    });

    return (
        // <path
        //   id={id}
        //   className="react-flow__edge-path"
        //   d={edgePath}
        //   markerEnd={markerEnd}
        //   style={style}
        // />
        <>
            <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
            <EdgeLabelRenderer>
                <div
                    id={id + "_label"}
                    style={getEdgeStyle(centerX, centerY)}
                    className="nodrag nopan"
                >
                    <Chip color="secondary" label={data.label} />
                </div>
            </EdgeLabelRenderer>
        </>
    );
}

export default FloatingEdge;
