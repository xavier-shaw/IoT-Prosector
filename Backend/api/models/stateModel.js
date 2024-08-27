var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var StateSchema = new Schema({
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

module.exports = mongoose.model('iotstates', StateSchema);