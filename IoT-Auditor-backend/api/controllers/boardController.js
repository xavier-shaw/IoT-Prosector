var Board = require('../models/boardModel');
var selectedFunctions = {
    "recording": false, "visualization": false, "verification": false
}

exports.get_selected_functions = function (req, res) {
    res.json(selectedFunctions)
};

exports.update_selected_functions = function (req, res) {
    selectedFunctions = req.body;
    res.json(selectedFunctions);
};

exports.create_board = function (req, res) {
    var new_board = new Board(req.body);
    new_board.save().then((board) => {
        res.json(board);
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};

exports.get_all_boards = function (req, res) {
    Board.find().then((boards) => {
        res.json(boards);
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};

exports.get_board_by_id = function (req, res) {
    Board.find({ "_id": req.params.boardId })
        .then((board) => {
            res.json(board);
        })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};

exports.save_board = function (req, res) {
    Board.findByIdAndUpdate(req.body.boardId, req.body.updates, { new: true })
        .then((updatedBoard) => {
            if (updatedBoard) {
                res.json(updatedBoard);
            } else {
                console.log('Board not found');
            }
        })
}
