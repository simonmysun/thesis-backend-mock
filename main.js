const express = require('express');
const app = express();
const http = require('http');
const server = http.createServer(app);
const { Server } = require('socket.io');
const { createNoise2D } = require('simplex-noise');

const io = new Server(server,{
  cors: {
    origin: 'http://localhost:3000'
  }
});

const port = 3001;

const noise2D = createNoise2D();

function hashFnv32a(str, asString, seed) {
  let i;
  let l;
  let hval = (seed === undefined) ? 0x811c9dc5 : seed;
  for (i = 0, l = str.length; i < l; i++) {
    hval ^= str.charCodeAt(i);
    hval += (hval << 1) + (hval << 4) + (hval << 7) + (hval << 8) + (hval << 24);
  }
  if(asString){
    // Convert to 8 digit hex string
    return ("0000000" + (hval >>> 0).toString(16)).slice(-8);
  }
  return hval >>> 0;
}

// app.use(express.static('public'));

app.get('/', (req, res) => {
  res.send('<h1>It works</h1>');
});

io.on('connection', (socket) => {
  console.log(`client connected`);
  socket.on('client hello', (deviceID) => {
    console.log('client subscribed to #update:' + deviceID)
    socket.intervalHandler = setInterval(((socket, length) => (_ => {
      const arr = Array.from({ length: length }).map((x, i) => {
        let noise = noise2D(hashFnv32a(deviceID) % 1024 + i / length * 2, new Date().valueOf() % 1000000 / 10000);
        noise = noise ** 2;
        noise = noise < 0.5 ? 8 * noise * noise * noise * noise : 1 - Math.pow(-2 * noise + 2, 4) / 2;
        return noise;
      });
      socket.emit('update:' + deviceID, JSON.stringify(arr));
    }))(socket, hashFnv32a(deviceID) % 16 + 16), 150);
  });
  socket.on('client goodbye', (deviceID) => {
    console.log('client unsubscribed to #update:' + deviceID)
    clearInterval(socket.intervalHandler);
  });
  socket.on('disconnect', () => {
    console.log('client disconnected');
    clearInterval(socket.intervalHandler);
  });
});

server.listen(port, () => {
  console.log(`listening on *:${port}`);
});

