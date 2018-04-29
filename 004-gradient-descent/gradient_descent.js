// adjust training set size

const M = 10;

// generate random training set

const DATA = [];

const getRandomIntFromInterval = (min, max) =>
  Math.floor(Math.random() * (max - min + 1) + min);

const createRandomPortlandHouse = () => ({
  squareMeter: getRandomIntFromInterval(0, 100),
  price: getRandomIntFromInterval(0, 100),
});

for (let i = 0; i < M; i++) {
  DATA.push(createRandomPortlandHouse());
}

const _x = DATA.map(date => date.squareMeter);
const _y = DATA.map(date => date.price);

// linear regression and gradient descent

const LEARNING_RATE = 0.0003;

let thetaOne = 0;
let thetaZero = 0;

const hypothesis = x => thetaZero + thetaOne * x;

const learn = (x, y, alpha) => {
  let thetaZeroSum = 0;
  let thetaOneSum = 0;

  for (let i = 0; i < M; i++) {
    thetaZeroSum += hypothesis(x[i]) - y[i];
    thetaOneSum += (hypothesis(x[i]) - y[i]) * x[i];
  }

  thetaZero = thetaZero - (alpha / M) * thetaZeroSum;
  thetaOne = thetaOne - (alpha / M) * thetaOneSum;
}

const MAX_ITER = 1500;
for (let i = 0; i < MAX_ITER; i++) {
  learn(_x, _y, LEARNING_RATE);
  console.log('thetaZero ', thetaZero, 'thetaOne', thetaOne)
}
