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
    backgroundColor: "white",
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

export const nodeOffsetX = 300;
export const nodeOffsetY = 150;

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
export const noneColor = "black";

export const selectedColor = "#4361ee";

export const customColors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
'#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
'#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'];

export const colorPalette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#9edae5", "#dbdb8d", 
    "#c7c7c7", "#c49c94", "#f7b6d2", "#c5b0d5", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#fdd0a2", "#dadaeb", "#c7dbd2", "#b0e57c",
    "#ff9896", "#9edae5", "#c5d0e6", "#fbebd6", "#fdae6b", "#e6550d",
    "#fdae6b", "#a1d99b", "#e7ba52", "#6baed6", "#637939", "#8ca252"
  ];
  

export const displayHandleMargin = 5;
export const displayHandleOffset = 40;