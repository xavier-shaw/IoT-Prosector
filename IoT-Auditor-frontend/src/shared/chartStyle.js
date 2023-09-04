export const stateNodeStyle = {
    width: "150px",
    height: "80px",
    borderWidth: "1px",
    borderStyle: "solid",
    padding: "10px",
    borderRadius: "10px",
    backgroundColor: "#788bff",
    display: "flex",
    justifyContent: "center",
    alignItems: "center"
}

export const semanticNodeStyle = {
    width: "250px",
    height: "160px",
    borderWidth: "3px",
    borderStyle: "solid",
    padding: "10px",
    borderRadius: "10px",
    backgroundColor: "#F7E2E1",
}

export const combinedNodeStyle = {
    width: "400px",
    height: "200px",
    borderWidth: "3px",
    borderStyle: "solid",
    padding: "10px",
    borderRadius: "10px",
    backgroundColor: "#E2EEEC",
}

export const getEdgeStyle = (labelX, labelY) => {
    return {
        position: 'absolute',
        transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
        fontSize: 18,
        fontWeight: 'bold',
        pointerEvents: 'all',
        backgroundColor: "#f4a261",
        borderRadius: 10,
        zIndex: 4
    }
}

export const nodeOffsetX = 400;
export const nodeOffsetY = 100;

export const childNodeoffsetX = 50;
export const childNodeoffsetY = 100;
export const childNodeMarginY = 60;
export const childNodeMarginX = 50;

export const semanticNodeMarginX = 10;
export const semanticNodeMarginY = 10;
export const semanticNodeOffsetX = 400;

export const combinedNodeMarginX = 10;
export const combinedNodeMarginY = 10;
export const combinedNodeOffsetX = 200;

export const childSemanticNodeMarginX = 80;
export const childSemanticNodeMarginY = 60;
export const childSemanticNodeOffsetX = 400;
export const childSemanticNodeOffsetY = 80;

export const offWidth = 60;
export const offHeight = 0;

export const layoutRowNum = 3;

export const highlightColor = "lightyellow";
