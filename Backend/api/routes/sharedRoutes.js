module.exports = function (app) {
    var shared = require('../controllers/sharedController');

    app.route('/api/shared/start/:stage')
        .get(shared.start_sensing);

    app.route('/api/shared/stop')
        .get(shared.stop_sensing);

    app.route('/api/shared/update/:device')
        .get(shared.update_device)
};

