module.exports = function (app) {
    var shared = require('../controllers/sharedController');

    app.route('/api/shared/start')
        .get(shared.start_sensing);

    app.route('/api/shared/stop')
        .get(shared.stop_sensing);
};

