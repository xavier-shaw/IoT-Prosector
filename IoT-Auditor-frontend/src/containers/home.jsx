import { React, useState, useEffect } from "react";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";
import { Link } from 'react-router-dom';
import Typography from '@mui/material/Typography';
import { Button, Checkbox, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, FormGroup, Popover, Radio, RadioGroup } from "@mui/material";
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import LinkOffIcon from '@mui/icons-material/LinkOff';
import LinkIcon from '@mui/icons-material/Link';
import WidgetsIcon from '@mui/icons-material/Widgets';
import "./home.css"

export default function Home(props) {
    const [boards, setBoards] = useState([]);
    const [availablePorts, setAvailablePorts] = useState([]);
    const [connectedPort, setConnectedPort] = useState(null);
    const [selectedFunctions, setSelectedFunctions] = useState({
        "recording": false, "visualization": false, "verification": false
    })
    const [openConnectionDialog, setOpenConnectionDialog] = useState(false);
    const [openFunctionDialog, setOpenFunctionDialog] = useState(false);
    const [portSelection, setPortSelection] = useState(null);
    const [functionSelections, setFunctionSelections] = useState({});

    const functions = ["recording", "visualization", "verification"];

    useEffect(() => {
        // Side effect code goes here
        // It will run after the component renders
        getBoards();
        getConnectedPort();
        // Optional cleanup function
        return () => {
            // Cleanup code goes here
            // It will run before the component is removed from the DOM
        };
    }, []); // Dependency array

    function getBoards() {
        axios
            .get(window.BACKEND_ADDRESS + "/boards")
            .then((resp) => {
                console.log("Get boards from db", resp);
                setBoards(resp.data)
            })
    };

    function getConnectedPort() {
        axios
            .get(window.HARDWARE_ADDRESS + "/getConnectedPort")
            .then((resp) => {
                console.log(resp)
                setConnectedPort(resp.data.connected_port)
            })
    }

    function handleClickManageConnection() {
        axios
            .get(window.HARDWARE_ADDRESS + "/getAvailableSensingPorts")
            .then((resp) => {
                console.log("Available Ports", resp.data.available_ports);
                setAvailablePorts(resp.data.available_ports);
                setPortSelection(connectedPort);
                setOpenConnectionDialog(true);
            })
    }

    const handleClickManageFunctions = () => {
        setFunctionSelections(selectedFunctions);
        setOpenFunctionDialog(true);
    }

    const handlePortSelectionChange = (event) => {
        setPortSelection(event.target.value)
    }

    const handleFunctionSelectionChange = (func) => {
        console.log(functionSelections);
        console.log(func)
        console.log(functionSelections[func])
        setFunctionSelections(prev => ({ ...prev, [func]: !prev[func] }))
    }

    const handleCloseConnectionDialog = () => {
        setPortSelection(null);
        setOpenConnectionDialog(false);
    }

    const handleCloseFunctionDialog = () => {
        setOpenFunctionDialog(false);
    }

    const handleConfirmConnectionDialog = () => {
        axios
            .get(window.HARDWARE_ADDRESS + "/connectPort/", {
                params: {
                    port: portSelection
                }
            })
            .then((resp) => {
                console.log(resp);
                setConnectedPort(portSelection);
                handleCloseConnectionDialog();
            })
    }

    const handleConfirmFunctionDialog = () => {
        setSelectedFunctions(functionSelections);
        handleCloseFunctionDialog();
    }

    function createBoard() {
        let newId = uuidv4();
        let curboard = {
            _id: newId
        };

        axios
            .post(window.BACKEND_ADDRESS + "/board", curboard)
            .then(response => {
                console.log("successed with response message:", response);
                window.location.href = 'board/' + newId;
            })
            .catch(error => {
                console.log("failed with error message:", error);
            });
    }


    return (
        <div className="home">
            <Typography variant="h1">
                IoT Auditor
            </Typography>
            <Typography variant="h6" gutterBottom>
                (Connect with a sensing port to obtain sensing data)
            </Typography>

            <Button variant={connectedPort ? "contained" : "outlined"} onClick={handleClickManageConnection} startIcon={<LinkIcon />}>
                Manage Port Connection
            </Button>
            <Button className="ms-3" variant="outlined" color="secondary" onClick={handleClickManageFunctions} startIcon={<WidgetsIcon />}>
                Manage Functions
            </Button>

            <Dialog
                open={openConnectionDialog}
            >
                <DialogTitle>
                    Manage Port Connection
                </DialogTitle>
                <DialogContent>
                    <RadioGroup
                        value={portSelection}
                        onChange={handlePortSelectionChange}
                    >
                        {availablePorts.map(port => (
                            <FormControlLabel key={port} value={port} control={<Radio />} label={port} />
                        ))}
                    </RadioGroup>
                </DialogContent>
                <DialogActions>
                    <Button color="error" onClick={handleCloseConnectionDialog}>Cancel</Button>
                    <Button onClick={handleConfirmConnectionDialog}>Confirm</Button>
                </DialogActions>
            </Dialog>

            <Dialog
                open={openFunctionDialog}
            >
                <DialogTitle>
                    Manage Functions
                </DialogTitle>
                <DialogContent>
                    <FormGroup>
                        {functions.map(func => (
                            <FormControlLabel
                                key={func}
                                control={
                                    <Checkbox
                                        checked={functionSelections[func]}
                                        onChange={() => handleFunctionSelectionChange(func)}
                                        inputProps={{ 'aria-label': 'controlled' }}
                                    />
                                }
                                label={func}
                            />
                        ))}
                    </FormGroup>
                </DialogContent>
                <DialogActions>
                    <Button color="error" onClick={handleCloseFunctionDialog}>Cancel</Button>
                    <Button onClick={handleConfirmFunctionDialog}>Confirm</Button>
                </DialogActions>
            </Dialog>
            <div className="storyexamplecontainer">
                <div className="leancontainer">
                    <p className="storyexamples">Boards</p>
                    <div className="row">
                        <div className="col-sm-12 text-start">
                            <div className='homeBoardList'>
                                {boards.map(board => (
                                    <div className='homeBoardStory' key={`row-${board._id}`}>
                                        <Link className='homeStoryLink'
                                            to={`board/${board._id}`}>
                                            {board.title}
                                        </Link>
                                    </div>
                                ))}
                                <div
                                    className="homeBoardStory"
                                    style={{ background: "whitesmoke", boxShadow: "0px 0px", justifyItems: "center" }}
                                >
                                    <Button variant="contained" startIcon={<AddCircleOutlineIcon />} onClick={createBoard}>
                                        New Board
                                    </Button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}