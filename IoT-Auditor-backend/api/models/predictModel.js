var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var PredictSchema = new Schema({
  _id: {
    type: String,
  },
  idx: {
    type: String,
  },
  time: {
    type: Number,
  },
  device: {
    type: String,
  }
});

module.exports = mongoose.model('predictstates', PredictSchema);