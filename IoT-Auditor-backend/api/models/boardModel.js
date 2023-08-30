var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var BoardSchema = new Schema({
  _id: {
    type: String,
  },
  title: {
    type: String,
    default: 'untitled'
  },
  created_date: {
    type: Date,
    default: Date.now
  },
  data: {
    type: Object,
    default: {instructions: []},
  },
  chart: {
    type: Object,
    default: {nodes: [], edges: []}
  }
});

module.exports = mongoose.model('boards', BoardSchema);