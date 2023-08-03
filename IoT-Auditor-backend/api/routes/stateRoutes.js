module.exports = function (app) {
    var state = require('../controllers/stateController');

    app.route('/api/states/:device')
        .get(state.get_all_states_by_device);
};

