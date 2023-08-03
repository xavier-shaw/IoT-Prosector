module.exports = function (app) {
    var board = require('../controllers/boardController');

    app.route('/api/board')
        .post(board.create_board);

    app.route('/api/boards/:boardId')
        .get(board.get_board_by_id);

    app.route('/api/boards/saveBoard')
        .post(board.save_board)
};

