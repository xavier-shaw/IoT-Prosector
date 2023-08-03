var express = require('express'),
    path = require('path'),
    app = express(),
    cors = require('cors'),
    port = process.env.PORT || 9990,
    mongoose = require('mongoose'),
    bodyParser = require('body-parser');

// mongoose instance connection url connection
mongoose.Promise = global.Promise;
mongoose.connect("mongodb+srv://haojian:xwBVZV7fG8rjDKD@cluster0-f8w36.mongodb.net/iotdb", {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => {
  console.log("connected to database - iotdb.")
})
.catch(err => console.error("connect to db failed.", err));

app.use(cors());

app.use(bodyParser.urlencoded({limit: '10mb', extended: true}));
app.use(bodyParser.json({limit: '10mb', extended: true}));

var boardRoutes = require('./api/routes/boardRoutes'); //importing route
boardRoutes(app); //register the route

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

