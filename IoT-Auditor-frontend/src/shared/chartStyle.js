export const groupZIndex = 2002;
export const stateZIndex = 3003;
export const edgeZIndex = 1;
export const labelZIndex = 1001;

export const stateNodeStyle = {
    width: "120px",
    height: "80px",
    borderWidth: "1px",
    borderStyle: "solid",
    padding: "10px",
    borderRadius: "10px",
    backgroundColor: "#F7E2E1",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    zIndex: stateZIndex
}

export const semanticNodeStyle = {
    width: "190px",
    height: "160px",
    borderWidth: "3px",
    borderStyle: "solid",
    padding: "10px",
    borderRadius: "10px",
    backgroundColor: "lightgrey",
    zIndex: groupZIndex
}

export const displayNodeStyle = {
    width: "200px",
    height: "100px",
    borderWidth: "3px",
    borderStyle: "solid",
    padding: "10px",
    borderRadius: "10px",
    backgroundColor: "#F7E2E1",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    zIndex: groupZIndex
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
        padding: 10,
        borderRadius: 10,
        zIndex: labelZIndex
    }
}

export const nodeOffsetX = 400;
export const nodeOffsetY = 100;

export const childNodeoffsetX = 35;
export const childNodeoffsetY = 100;
export const childNodeMarginY = 20;
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

export const stateColor = "blue";
export const siblingColor = "skyblue";
export const noneColor = "lightgrey";

export const selectedColor = "#4361ee";

export const customColors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
'#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
'#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'];

// export const customColors = [
//     "#001219", "#005F73", "#0A9396", "#94D2BD", "#E9D8A6", "#EE9B00", "#CA6702", "#BB3E03", "#AE2012", "#9B2226",
//     "#582F0E", "#936639", "#A4AC86", "#656D4A", "#414833", "#023e8a", "#0096c7", "#ef476f", "#7b2cbf", "#4d908e"
// ]

export const displayHandleMargin = 5;
export const displayHandleOffset = 40;