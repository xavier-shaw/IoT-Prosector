import React from "react";
import { useState } from "react";
import { Button, Paper, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Dialog, DialogTitle, DialogActions, DialogContent } from "@mui/material";
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import PlayCircleFilledIcon from '@mui/icons-material/PlayCircleFilled';
import { memo } from "react";

function InstructionTable(props) {
    const { instructions, setInstructions, addAction, status } = props;
    const [newRow, setNewRow] = useState({
        function: '',
        interaction: '',
        image: ''
    });
    const [imagePreview, setImagePreview] = useState(null);
    const [editingCell, setEditingCell] = useState(null);
    const [openDeleteDialog, setOpenDeleteDialog] = useState(false);

    const handleCellClick = (rowIndex, columnName) => {
        setEditingCell({ index: rowIndex, columnName: columnName });
    }

    const handleCellBlur = () => {
        setEditingCell(null);
    }

    const handleInputChange = (e, index) => {
        const { name, value } = e.target;
        if (index !== undefined) {
            const newInstructions = [...instructions];
            newInstructions[index][name] = value;
            setInstructions(newInstructions);
        } else {
            setNewRow(prevState => ({
                ...prevState,
                [name]: value
            }));
        }
    }

    const handleImageChange = (e, index) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                if (index !== undefined) {
                    const newInstructions = [...instructions];
                    newInstructions[index].image = reader.result;
                    setInstructions(newInstructions);
                } else {
                    setImagePreview(reader.result);
                    setNewRow(prevState => ({
                        ...prevState,
                        image: reader.result
                    }));
                }
            }
            reader.readAsDataURL(file);
        }
    }

    const handleAddRow = () => {
        if (!newRow.function.trim() || !newRow.interaction.trim()) {
            alert('Function and Interaction cannot be empty!');
            return;
        }

        setInstructions([...instructions, newRow]);
        setNewRow({
            function: '',
            interaction: '',
            image: ''
        });
        setImagePreview(null);
    }

    const handleDeleteRow = () => {
        const updatedInstructions = [...instructions];
        updatedInstructions.splice(editingCell.index, 1);
        setInstructions(updatedInstructions);
        setOpenDeleteDialog(false);
    };

    const handleClickAction = (index) => {
        addAction(instructions[index].interaction);
    };

    return (
        <div>
            <TableContainer component={Paper} sx={{ maxWidth: "100%" }}>
                <Table stickyHeader>
                    <TableHead>
                        <TableRow>
                            <TableCell align="left" sx={{ fontFamily: "Times New Roman", fontSize: 30, fontWeight: "bold" }}>Function</TableCell>
                            <TableCell align="left" sx={{ fontFamily: "Times New Roman", fontSize: 30, fontWeight: "bold" }}>Interaction</TableCell>
                            <TableCell align="left" sx={{ fontFamily: "Times New Roman", fontSize: 30, fontWeight: "bold" }}>Image</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {instructions.map((instruction, index) => (
                            <TableRow key={index} >
                                <TableCell sx={{ fontFamily: "Times New Roman", fontSize: 26 }} onClick={() => handleCellClick(index, 'function')}>
                                    {editingCell?.index === index && editingCell?.columnName === 'function' ?
                                        <TextField
                                            name="function"
                                            size="small"
                                            value={instruction.function}
                                            onChange={(e) => handleInputChange(e, index)}
                                            onBlur={handleCellBlur}
                                            autoFocus
                                        /> :
                                        instruction.function
                                    }
                                </TableCell>
                                <TableCell sx={{ fontFamily: "Times New Roman", fontSize: 26 }} onClick={() => handleCellClick(index, 'interaction')}>
                                    {editingCell?.index === index && editingCell?.columnName === 'interaction' ?
                                        <TextField
                                            name="interaction"
                                            size="small"
                                            value={instruction.interaction}
                                            onChange={(e) => handleInputChange(e, index)}
                                            onBlur={handleCellBlur}
                                            autoFocus
                                        /> :
                                        instruction.interaction
                                    }
                                </TableCell>
                                <TableCell>
                                    <input
                                        style={{ display: 'none' }} // This hides the default file input
                                        id={`file-upload-${index}`} // unique id for each input
                                        type="file"
                                        accept="image/*"
                                        onChange={e => handleImageChange(e, index)}
                                    />
                                    {instruction.image ? (
                                        <label htmlFor={`file-upload-${index}`}>
                                            {/* Clicking on this image will now trigger the file input */}
                                            <img src={instruction.image} alt="Uploaded" width="130" height="130" style={{ cursor: 'pointer' }} />
                                        </label>
                                    ) : (
                                        <label htmlFor={`file-upload-${index}`}>
                                            <Button component="span" color="primary" startIcon={<CloudUploadIcon />} />
                                        </label>
                                    )}
                                </TableCell>
                                <TableCell onClick={() => handleCellClick(index, 'operation')}>
                                    <Button color="primary" size="large" onClick={() => handleClickAction(index)} disabled={status !== "choose action"} startIcon={<PlayCircleFilledIcon fontSize="large" />} />
                                    <Button color="error" size="large" onClick={() => setOpenDeleteDialog(true)} startIcon={<DeleteIcon />} />
                                </TableCell>
                            </TableRow>
                        ))}

                        <TableRow>
                            <TableCell>
                                <TextField
                                    name="function"
                                    size="small"
                                    value={newRow.function}
                                    onChange={handleInputChange}
                                    placeholder="Function"
                                />
                            </TableCell>
                            <TableCell>
                                <TextField
                                    name="interaction"
                                    size="small"
                                    value={newRow.interaction}
                                    onChange={handleInputChange}
                                    placeholder="Interaction"
                                />
                            </TableCell>
                            <TableCell>
                                <input
                                    style={{ display: 'none' }}
                                    id="new-file-upload"
                                    type="file"
                                    accept="image/*"
                                    onChange={handleImageChange}
                                />
                                {imagePreview ? (
                                    <label htmlFor="new-file-upload">
                                        <img src={imagePreview} alt="Preview" width="100" height="100" style={{ cursor: 'pointer' }} />
                                    </label>
                                ) : (
                                    <label htmlFor="new-file-upload">
                                        <Button component="span" color="primary" startIcon={<CloudUploadIcon />} />
                                    </label>
                                )}
                            </TableCell>
                            <TableCell>
                                <Button onClick={handleAddRow} startIcon={<AddCircleIcon />} />
                            </TableCell>
                        </TableRow>
                    </TableBody>
                </Table>
            </TableContainer>

            <Dialog open={openDeleteDialog} onClose={() => setOpenDeleteDialog(false)}>
                <DialogTitle>Are you sure to delete this instruction?</DialogTitle>
                <DialogActions>
                    <Button color="primary" variant="outlined" onClick={() => setOpenDeleteDialog(false)}>Cancel</Button>
                    <Button color="error" variant="contained" onClick={handleDeleteRow}>Delete</Button>
                </DialogActions>
            </Dialog>

        </div>
    )
}

export default memo(InstructionTable);